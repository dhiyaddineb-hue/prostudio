#!/usr/bin/env python3
"""One-command ProStudio dubbing pipeline.

Stages:
  1) optional Demucs speech/background separation
  2) optional pyannote speaker diarization manifest
  3) Seed-VC enhancement using the cleaned reference
  4) conditional Wav2Lip face gate and final remux

The script intentionally keeps fallbacks safe: if diarization or source
separation is unavailable, it does not invent speaker identities or copy the
original dialogue back into the result. Translation/TTS is supplied as
``--dubbed-audio`` so this coordinator can be used with any upstream TTS engine
(VoxCPM, Edge-TTS, XTTS, or a human-approved track).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from lip_sync.run import detect_faces, fit_audio, mux_audio, probe, run_wav2lip
from youtube_auto_dub.speaker_diarization import annotate_segments
from youtube_auto_dub.source_separation import separate_dialogue_background


def call(cmd: list[str]) -> float:
    start = time.monotonic()
    subprocess.run(cmd, check=True)
    return time.monotonic() - start


def main() -> None:
    p = argparse.ArgumentParser(description="Run the complete ProStudio media pipeline")
    p.add_argument("--video", type=Path, required=True, help="Original video")
    p.add_argument("--dubbed-audio", type=Path, required=True,
                   help="Final translated/TTS audio before Seed-VC")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, default=Path("output/pipeline-work"))
    p.add_argument("--seed-script", type=Path, default=Path("scripts/seed_vc_enhance.py"))
    p.add_argument("--seed-space", default="phuoc2005/seed-vc")
    p.add_argument("--diffusion-steps", type=int, default=40)
    p.add_argument("--demucs", action="store_true", help="Enable neural source separation")
    p.add_argument("--diarize", action="store_true", help="Enable pyannote diarization")
    p.add_argument("--keep-background", action="store_true", help="Keep separated music/effects")
    p.add_argument("--lip-sync", action="store_true", help="Run Wav2Lip when a face is visible")
    p.add_argument("--wav2lip-repo", type=Path)
    p.add_argument("--wav2lip-checkpoint", type=Path)
    p.add_argument("--offset-ms", type=int, default=0)
    p.add_argument("--min-face-coverage", type=float, default=0.20)
    p.add_argument("--min-face-ratio", type=float, default=0.008)
    args = p.parse_args()
    for path in (args.video, args.dubbed_audio, args.seed_script):
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")
    if args.lip_sync and (not args.wav2lip_repo or not args.wav2lip_checkpoint):
        raise SystemExit("--lip-sync requires --wav2lip-repo and --wav2lip-checkpoint")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"video": str(args.video), "output": str(args.output), "stages": {}}

    # Stage 1: neural separation is used for the reference/background report.
    separated = None
    if args.demucs:
        started = time.monotonic()
        separated = separate_dialogue_background(args.video, args.work_dir / "sources")
        report["stages"]["source_separation"] = {
            "requested": True, "success": bool(separated),
            "runtime_s": round(time.monotonic() - started, 3),
            "speech": str(separated[0]) if separated else None,
            "background": str(separated[1]) if separated else None,
        }
    else:
        report["stages"]["source_separation"] = {"requested": False}

    # Stage 2: diarization produces a persistent manifest for downstream TTS.
    if args.diarize:
        speech_audio = separated[0] if separated else args.video
        duration = probe(speech_audio, "duration")
        raw = [{"start": 0.0, "end": duration, "text": "", "confidence": 1.0}]
        annotated = annotate_segments(speech_audio, raw)
        diarization_path = args.work_dir / "diarization.json"
        diarization_path.write_text(json.dumps(annotated, indent=2), encoding="utf-8")
        report["stages"]["diarization"] = {
            "requested": True, "manifest": str(diarization_path),
            "speaker_count": len({x.get("speaker") for x in annotated if x.get("speaker")}),
        }
    else:
        report["stages"]["diarization"] = {"requested": False}

    # Stage 3: Seed-VC. It performs its own cleaned-reference preparation and
    # can invoke Demucs again when requested, keeping the CLI contract stable.
    seed_output = args.work_dir / "seedvc-enhanced.mp4"
    seed_cmd = [sys.executable, str(args.seed_script), "--original", str(args.video),
                "--dubbed", str(args.dubbed_audio), "--output", str(seed_output),
                "--space", args.seed_space, "--diffusion-steps", str(args.diffusion_steps)]
    if args.keep_background:
        seed_cmd.append("--keep-background")
    if args.demucs:
        seed_cmd.append("--separate-sources")
    report["stages"]["seed_vc"] = {"runtime_s": round(call(seed_cmd), 3), "output": str(seed_output)}

    # Stage 4: audio fit and optional lip-sync. Wav2Lip gets the same final
    # audio used for remuxing, so the output cannot silently retain old audio.
    final_audio = args.work_dir / "final-audio.wav"
    timing = fit_audio(args.dubbed_audio, probe(args.video, "duration"), final_audio, args.offset_ms)
    report["stages"]["timing"] = timing
    if not args.lip_sync:
        mux_audio(seed_output, final_audio, args.output)
        report["stages"]["lip_sync"] = {"requested": False, "applied": False}
    else:
        face = detect_faces(seed_output)
        if face["coverage"] < args.min_face_coverage or face["max_face_ratio"] < args.min_face_ratio:
            mux_audio(seed_output, final_audio, args.output)
            report["stages"]["lip_sync"] = {"requested": True, "applied": False,
                                              "reason": "face_gate", "face": face}
        else:
            raw_lipsync = args.work_dir / "wav2lip-raw.mp4"
            run_wav2lip(args.wav2lip_repo, args.wav2lip_checkpoint, seed_output, final_audio,
                        raw_lipsync)
            mux_audio(raw_lipsync, final_audio, args.output)
            report["stages"]["lip_sync"] = {"requested": True, "applied": True, "face": face}
    report["status"] = "ok"
    report_path = args.output.with_suffix(".pipeline.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
