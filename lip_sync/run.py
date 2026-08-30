#!/usr/bin/env python3
"""Conditional, shot-aware Wav2Lip runner.

The runner edits pixels only when a single sufficiently large face is visible in
stable sampled frames. It always muxes the supplied generated audio afterwards,
so the original dialogue can never become the lip-sync audio source.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def probe(path: Path, entry: str) -> float:
    result = run(["ffprobe", "-v", "error", "-show_entries", f"format={entry}", "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(result.stdout.strip())


def detect_faces(video: Path, sample_every: int = 5, min_face_ratio: float = 0.008) -> dict:
    """Return visibility, single-face stability, size, and ambiguity metrics."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for the face gate") from exc
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video}")
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    frames = sampled = visible = single = multi = 0; max_ratio = 0.0; ratios: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        if frames % max(1, sample_every) == 0:
            sampled += 1; gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            if len(faces):
                visible += 1
                if len(faces) == 1: single += 1
                if len(faces) > 1: multi += 1
                h, w = gray.shape[:2]
                ratio = max((fw * fh) / float(w * h) for _, _, fw, fh in faces)
                ratios.append(ratio); max_ratio = max(max_ratio, ratio)
        frames += 1
    cap.release(); sampled = max(1, sampled)
    return {"frames": frames, "sampled": sampled, "visible": visible, "single_face": single,
            "multiple_face": multi, "coverage": visible / sampled, "single_face_coverage": single / sampled,
            "max_face_ratio": max_ratio, "mean_face_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
            "ambiguous": multi / sampled > 0.10, "min_face_ratio": min_face_ratio}


def atempo_chain(factor: float) -> str:
    if factor <= 0: raise ValueError("tempo factor must be positive")
    parts: list[str] = []
    while factor > 2: parts.append("atempo=2.0"); factor /= 2
    while factor < .5: parts.append("atempo=0.5"); factor /= .5
    parts.append(f"atempo={factor:.7f}"); return ",".join(parts)


def fit_audio(audio: Path, video_duration: float, dest: Path, offset_ms: int = 0, min_factor: float = .69, max_factor: float = 1.45) -> dict:
    """Fit audio without silently accepting an extreme speech-rate change."""
    audio_duration = probe(audio, "duration")
    if audio_duration <= 0 or video_duration <= 0: raise RuntimeError("Audio/video duration is unavailable")
    factor = audio_duration / video_duration
    if not min_factor <= factor <= max_factor:
        raise RuntimeError(f"audio duration requires unsafe tempo factor {factor:.3f}; rewrite/regenerate first")
    filters = ([f"adelay={max(0, offset_ms)}:all=1"] if offset_ms else []) + [atempo_chain(factor), "apad", f"atrim=duration={video_duration:.6f}", "asetpts=PTS-STARTPTS"]
    run(["ffmpeg", "-y", "-i", str(audio), "-af", ",".join(filters), "-ar", "24000", "-ac", "1", str(dest)])
    return {"audio_duration": audio_duration, "video_duration": video_duration, "tempo_factor": factor, "offset_ms": offset_ms, "safe": True}


def mux_audio(video: Path, audio: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    run(["ffmpeg", "-y", "-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(temp)])
    temp.replace(output)


def run_wav2lip(repo: Path, checkpoint: Path, video: Path, audio: Path, output: Path, resize_factor: int = 1, nosmooth: bool = False) -> None:
    inference = repo / "inference.py"
    if not inference.exists(): raise RuntimeError(f"Wav2Lip inference.py not found in {repo}")
    cmd = [sys.executable, str(inference), "--checkpoint_path", str(checkpoint), "--face", str(video), "--audio", str(audio), "--outfile", str(output), "--resize_factor", str(resize_factor)]
    if nosmooth: cmd.append("--nosmooth")
    run(cmd)
    if not output.exists() or output.stat().st_size == 0: raise RuntimeError("Wav2Lip produced no output")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True); p.add_argument("--audio", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wav2lip-repo", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--min-coverage", type=float, default=.20); p.add_argument("--min-single-face-coverage", type=float, default=.15); p.add_argument("--min-face-ratio", type=float, default=.008); p.add_argument("--sample-every", type=int, default=5); p.add_argument("--offset-ms", type=int, default=0); p.add_argument("--resize-factor", type=int, default=1); p.add_argument("--nosmooth", action="store_true")
    args = p.parse_args()
    for path in (args.video, args.audio, args.wav2lip_repo, args.checkpoint):
        if not path.exists(): raise SystemExit(f"Missing input: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prostudio-lipsync-") as td:
        work = Path(td); duration = probe(args.video, "duration"); face = detect_faces(args.video, args.sample_every, args.min_face_ratio)
        gate = (face["coverage"] >= args.min_coverage and face["single_face_coverage"] >= args.min_single_face_coverage and face["max_face_ratio"] >= args.min_face_ratio and not face["ambiguous"])
        fitted = work / "fitted.wav"; timing = fit_audio(args.audio, duration, fitted, args.offset_ms)
        result = {"ok": True, "lip_sync": False, "face": face, "gate_passed": gate, "timing": timing, "quality": "unvalidated"}
        if not gate:
            mux_audio(args.video, fitted, args.output); result["reason"] = "face_gate"
        else:
            synced = work / "wav2lip.mp4"; run_wav2lip(args.wav2lip_repo, args.checkpoint, args.video, fitted, synced, args.resize_factor, args.nosmooth); mux_audio(synced, fitted, args.output); result.update(lip_sync=True, quality="unvalidated")
        result["output"] = str(args.output); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
