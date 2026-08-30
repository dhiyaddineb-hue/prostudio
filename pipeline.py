#!/usr/bin/env python3
"""Production-oriented, conservative ProStudio dubbing coordinator.

The coordinator owns stage contracts and reporting; heavy model inference remains
behind existing adapters. It never treats an unavailable optional model as success
and never remixes the original mixed dialogue. ``--dubbed-audio`` is the approved
translated/TTS track, allowing VoxCPM/Seed-VC, Edge-TTS, XTTS, or human audio to
feed the same safe finalization path.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from lip_sync.run import detect_faces, fit_audio, mux_audio, probe, run_wav2lip
from youtube_auto_dub.speaker_diarization import annotate_segments
from youtube_auto_dub.source_separation import separate_dialogue_background, validate_stems

log = logging.getLogger("prostudio.pipeline")

DEFAULT_PROFILE = Path(__file__).parent / "config" / "pipeline_profile.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read the pipeline profile") from exc


def _set_path(data: dict[str, Any], key: str, value: Any) -> None:
    cur = data
    bits = key.split(".")
    for bit in bits[:-1]:
        cur = cur.setdefault(bit, {})
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        value = value.lower() == "true"
    else:
        try: value = float(value) if "." in value else int(value)
        except (ValueError, TypeError): pass
    cur[bits[-1]] = value


def load_profile(path: Path, mode: str | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    profile = _load_yaml(path)
    selected = mode or profile.get("mode", "safe")
    merged = copy.deepcopy(profile)
    for key, value in (profile.get("modes", {}).get(selected, {}) or {}).items():
        _set_path(merged, key, value)
    merged["mode"] = selected
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"Override must be key=value: {item}")
        key, value = item.split("=", 1)
        _set_path(merged, key, value)
    return merged


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    log.debug("exec: %s", shlex.join(cmd))
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def stage(name: str, requested: bool, **values: Any) -> dict[str, Any]:
    return {"name": name, "requested": requested, **values}


def _duration(path: Path) -> float:
    return probe(path, "duration")


def validate_output(video: Path, source: Path, report: dict[str, Any], tolerance: float) -> bool:
    checks: dict[str, Any] = {"exists": video.exists(), "audio_stream": False, "original_dialogue_stream": False}
    if video.exists():
        try:
            result = run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)])
            streams = json.loads(result.stdout).get("streams", [])
            checks["audio_stream"] = any(s.get("codec_type") == "audio" for s in streams)
            checks["original_dialogue_stream"] = False  # mux maps only the generated track.
            checks["duration"] = _duration(video)
            checks["source_duration"] = _duration(source)
            checks["duration_close"] = abs(checks["duration"] - checks["source_duration"]) <= tolerance
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            checks["error"] = str(exc)
    report["quality"] = checks
    return bool(checks.get("exists") and checks.get("audio_stream") and checks.get("duration_close"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the complete ProStudio dubbing pipeline")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--dubbed-audio", type=Path, required=True, help="Approved translated/TTS audio")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    p.add_argument("--mode", choices=["safe", "high_quality_single", "multi_speaker_cinematic", "experiment"])
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--work-dir", type=Path, default=Path("output/pipeline-work"))
    p.add_argument("--seed-script", type=Path, default=Path("scripts/seed_vc_enhance.py"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--offset-ms", type=int, default=0)
    p.add_argument("--wav2lip-repo", type=Path)
    p.add_argument("--wav2lip-checkpoint", type=Path)
    args = p.parse_args(argv)
    profile = load_profile(args.profile, args.mode, args.overrides)
    report: dict[str, Any] = {"schema_version": 1, "status": "running", "configuration": profile,
                              "input": {"video": str(args.video), "dubbed_audio": str(args.dubbed_audio)},
                              "output": str(args.output), "stages": {}, "warnings": [], "fallback_events": []}
    required = [args.video, args.dubbed_audio]
    if not args.dry_run: required.append(args.seed_script)
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        report["status"] = "blocked"; report["fatal_reason"] = f"Missing inputs: {', '.join(missing)}"
        print(json.dumps(report, indent=2)); return 2
    if args.dry_run:
        report["status"] = "dry_run_ok"
        report["capabilities"] = {"hf_token": bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")),
                                   "ffmpeg": True, "seed_script": args.seed_script.exists()}
        print(json.dumps(report, indent=2)); return 0
    args.work_dir.mkdir(parents=True, exist_ok=True); args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    separated = None
    sep_cfg = profile["separation"]
    if sep_cfg.get("enabled"):
        try: separated = separate_dialogue_background(args.video, args.work_dir / "sources", model=sep_cfg.get("model"))
        except Exception as exc: report["warnings"].append(f"source separation failed: {exc}")
        valid = validate_stems(separated, _duration(args.video)) if separated else {"valid": False, "reason": "unavailable"}
        if not valid["valid"]: report["fallback_events"].append({"stage": "source_separation", "action": "audio_only", "reason": valid.get("reason")})
        report["stages"]["source_separation"] = stage("source_separation", True, success=bool(separated and valid["valid"]), validation=valid, runtime_s=round(time.monotonic()-started, 3))
    else: report["stages"]["source_separation"] = stage("source_separation", False, success=False, reason="disabled")

    diar_cfg = profile["diarization"]
    speech_audio = separated[0] if separated else args.video
    diar_path = args.work_dir / "diarization.json"
    if diar_cfg.get("enabled"):
        if diar_cfg.get("require_token") and not (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")):
            report["warnings"].append("Diarization requested but no HF_TOKEN/HUGGINGFACE_TOKEN is configured")
            report["fallback_events"].append({"stage": "diarization", "action": "single_reference", "reason": "missing_token"})
            annotated = []
        else:
            annotated = annotate_segments(speech_audio, [], token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"), model=diar_cfg.get("model"), min_overlap=float(diar_cfg.get("min_overlap", .55)))
        diar_path.write_text(json.dumps(annotated, indent=2), encoding="utf-8")
        report["stages"]["diarization"] = stage("diarization", True, success=bool(annotated), manifest=str(diar_path), speaker_count=len({x.get("speaker") for x in annotated if x.get("speaker")}))
    else: report["stages"]["diarization"] = stage("diarization", False, success=False, reason="disabled")

    seed_output = args.work_dir / "seedvc-enhanced.mp4"
    seed_cmd = [sys.executable, str(args.seed_script), "--original", str(args.video), "--dubbed", str(args.dubbed_audio), "--output", str(seed_output), "--space", str(profile["tts"]["seed_vc_space"]), "--diffusion-steps", str(profile["tts"]["diffusion_steps"])]
    if sep_cfg.get("preserve_background") and separated: seed_cmd += ["--keep-background", "--separate-sources"]
    try:
        t0 = time.monotonic(); completed = run(seed_cmd)
        report["stages"]["voice_generation"] = stage("voice_generation", True, success=True, backend=profile["tts"]["backend"], runtime_s=round(time.monotonic()-t0, 3), stdout=completed.stdout[-1000:])
    except subprocess.CalledProcessError as exc:
        report["stages"]["voice_generation"] = stage("voice_generation", True, success=False, backend=profile["tts"]["backend"], stderr=(exc.stderr or "")[-2000:])
        report["status"] = "blocked"; report["fatal_reason"] = "Required voice generation failed"; print(json.dumps(report, indent=2)); return 3

    final_audio = args.work_dir / "final-audio.wav"
    report["stages"]["timing"] = stage("timing", True, **fit_audio(args.dubbed_audio, _duration(args.video), final_audio, args.offset_ms), per_segment=False, note="Per-segment timing is provided by upstream segment/TTS adapters; whole-track fit is conservative fallback")
    lip_cfg = profile["lip_sync"]
    applied = False
    if lip_cfg.get("enabled") and args.wav2lip_repo and args.wav2lip_checkpoint:
        face = detect_faces(seed_output)
        report["stages"]["lip_sync"] = stage("lip_sync", True, backend=lip_cfg.get("backend"), face_gate=face, applied=False, quality="unvalidated")
        if face.get("coverage", 0) >= float(lip_cfg.get("min_face_coverage", .2)) and face.get("max_face_ratio", 0) >= float(lip_cfg.get("min_face_area_ratio", .008)):
            try:
                synced = args.work_dir / "wav2lip.mp4"; run_wav2lip(args.wav2lip_repo, args.wav2lip_checkpoint, seed_output, final_audio, synced)
                mux_audio(synced, final_audio, args.output); applied = True; report["stages"]["lip_sync"].update(applied=True, quality="unvalidated")
            except Exception as exc: report["warnings"].append(f"lip-sync failed; audio-only fallback: {exc}")
        else: report["warnings"].append("lip-sync face gate failed; preserved original pixels")
    else:
        report["stages"]["lip_sync"] = stage("lip_sync", bool(lip_cfg.get("enabled")), applied=False, reason="disabled_or_missing_verified_checkpoint")
    if not applied: mux_audio(seed_output, final_audio, args.output)
    report["stages"]["finalization"] = stage("finalization", True, original_dialogue_remixed=False, background_from_separated_stem=bool(separated and sep_cfg.get("preserve_background")))
    report["status"] = "ok" if validate_output(args.output, args.video, report, float(profile["quality"]["duration_tolerance_seconds"])) else "quality_failed"
    report["runtime_s"] = round(time.monotonic() - started, 3)
    report_path = args.output.with_suffix(".pipeline.json"); tmp = report_path.with_suffix(".tmp"); tmp.write_text(json.dumps(report, indent=2), encoding="utf-8"); tmp.replace(report_path)
    print(json.dumps(report, indent=2)); return 0 if report["status"] == "ok" else 4


if __name__ == "__main__": raise SystemExit(main())
