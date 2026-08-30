#!/usr/bin/env python3
"""Run Wav2Lip and Diff2Lip on identical inputs and write a comparison report.

Diff2Lip's upstream repository uses a research script whose flags can change;
therefore this runner accepts a command template instead of guessing its CLI.
Supported placeholders are {video}, {audio}, {output}, {repo}, and {checkpoint}.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

from run import detect_faces, fit_audio, mux_audio, probe, run_wav2lip


def execute_template(template: str, values: dict[str, str]) -> float:
    command = shlex.split(template.format(**values))
    started = time.monotonic()
    subprocess.run(command, check=True)
    return time.monotonic() - started


def main() -> None:
    p = argparse.ArgumentParser(description="Compare Wav2Lip and Diff2Lip")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--wav2lip-repo", type=Path, required=True)
    p.add_argument("--wav2lip-checkpoint", type=Path, required=True)
    p.add_argument("--diff2lip-command", required=True,
                   help="Quoted command template with {video} {audio} {output} {repo} {checkpoint}")
    p.add_argument("--diff2lip-repo", type=Path, required=True)
    p.add_argument("--diff2lip-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--min-coverage", type=float, default=0.20)
    p.add_argument("--min-face-ratio", type=float, default=0.008)
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--offset-ms", type=int, default=0)
    p.add_argument("--nosmooth", action="store_true")
    args = p.parse_args()
    for path in (args.video, args.audio, args.wav2lip_repo, args.wav2lip_checkpoint,
                 args.diff2lip_repo, args.diff2lip_checkpoint):
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_duration = probe(args.video, "duration")
    face = detect_faces(args.video, max(1, args.sample_every))
    report = {"video": str(args.video), "video_duration": video_duration,
              "face_gate": face, "candidates": {}}
    if face["coverage"] < args.min_coverage or face["max_face_ratio"] < args.min_face_ratio:
        report["status"] = "skipped_face_gate"
        report["reason"] = "No sufficiently visible face was detected"
        (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    fitted = args.output_dir / "comparison-audio.wav"
    timing = fit_audio(args.audio, video_duration, fitted, args.offset_ms)
    report["timing"] = timing

    wav_raw = args.output_dir / "wav2lip-raw.mp4"
    wav_final = args.output_dir / "wav2lip.mp4"
    started = time.monotonic()
    run_wav2lip(args.wav2lip_repo, args.wav2lip_checkpoint, args.video, fitted, wav_raw,
                nosmooth=args.nosmooth)
    wav_elapsed = time.monotonic() - started
    mux_audio(wav_raw, fitted, wav_final)
    report["candidates"]["wav2lip"] = {"output": str(wav_final), "runtime_s": round(wav_elapsed, 3),
                                        "duration": probe(wav_final, "duration")}

    diff_raw = args.output_dir / "diff2lip-raw.mp4"
    values = {"video": str(args.video), "audio": str(fitted), "output": str(diff_raw),
              "repo": str(args.diff2lip_repo), "checkpoint": str(args.diff2lip_checkpoint)}
    diff_elapsed = execute_template(args.diff2lip_command, values)
    if not diff_raw.exists():
        raise RuntimeError("Diff2Lip command completed but did not create its {output} file")
    diff_final = args.output_dir / "diff2lip.mp4"
    mux_audio(diff_raw, fitted, diff_final)
    report["candidates"]["diff2lip"] = {"output": str(diff_final), "runtime_s": round(diff_elapsed, 3),
                                         "duration": probe(diff_final, "duration")}
    report["status"] = "ok"
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
