#!/usr/bin/env python3
"""Conditional lip-sync runner for ProStudio.

The script refuses to edit a video when no sufficiently visible face is found.
It first makes the final dubbed audio fit the video duration, then delegates
mouth editing to a checked-out Wav2Lip repository. The original audio is never
copied into the output.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def probe(path: Path, entry: str) -> float:
    r = run(["ffprobe", "-v", "error", "-show_entries", f"format={entry}",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(r.stdout.strip())


def detect_faces(video: Path, sample_every: int = 5) -> dict:
    """Sample frames and return conservative face visibility statistics."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for the face gate") from exc
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    cascade = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    frames = visible = 0
    max_ratio = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frames % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                             minSize=(80, 80))
            if len(faces):
                visible += 1
                h, w = gray.shape[:2]
                max_ratio = max(max_ratio, max((fw * fh) / float(w * h)
                                               for _, _, fw, fh in faces))
        frames += 1
    cap.release()
    sampled = max(1, (frames + sample_every - 1) // sample_every)
    return {"frames": frames, "sampled": sampled, "visible": visible,
            "coverage": visible / sampled, "max_face_ratio": max_ratio}


def atempo_chain(factor: float) -> str:
    # factor is audio_duration / target_duration. atempo > 1 speeds audio up.
    parts: list[str] = []
    while factor > 2.0:
        parts.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        parts.append("atempo=0.5")
        factor /= 0.5
    parts.append(f"atempo={factor:.7f}")
    return ",".join(parts)


def fit_audio(audio: Path, video_duration: float, dest: Path, offset_ms: int = 0) -> dict:
    """Fit final dubbed audio to video length without truncating its tail."""
    audio_duration = probe(audio, "duration")
    if audio_duration <= 0 or video_duration <= 0:
        raise RuntimeError("Audio/video duration is unavailable")
    factor = audio_duration / video_duration
    filters = []
    if offset_ms:
        filters.append(f"adelay={max(0, offset_ms)}:all=1")
    filters.append(atempo_chain(factor))
    filters += ["apad", f"atrim=duration={video_duration:.6f}", "asetpts=PTS-STARTPTS"]
    run(["ffmpeg", "-y", "-i", str(audio), "-af", ",".join(filters),
         "-ar", "24000", "-ac", "1", str(dest)])
    return {"audio_duration": audio_duration, "video_duration": video_duration,
            "tempo_factor": factor, "offset_ms": offset_ms}


def mux_audio(video: Path, audio: Path, output: Path) -> None:
    run(["ffmpeg", "-y", "-i", str(video), "-i", str(audio),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "192k", "-shortest", str(output)])


def run_wav2lip(repo: Path, checkpoint: Path, video: Path, audio: Path, output: Path,
                resize_factor: int = 1, nosmooth: bool = False) -> None:
    inference = repo / "inference.py"
    if not inference.exists():
        raise RuntimeError(f"Wav2Lip inference.py not found in {repo}")
    cmd = [sys.executable, str(inference), "--checkpoint_path", str(checkpoint),
           "--face", str(video), "--audio", str(audio), "--outfile", str(output),
           "--resize_factor", str(resize_factor)]
    if nosmooth:
        cmd.append("--nosmooth")
    run(cmd)


def main() -> None:
    p = argparse.ArgumentParser(description="Conditionally lip-sync a dubbed video")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wav2lip-repo", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--min-coverage", type=float, default=0.20)
    p.add_argument("--min-face-ratio", type=float, default=0.008)
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--offset-ms", type=int, default=0)
    p.add_argument("--resize-factor", type=int, default=1)
    p.add_argument("--nosmooth", action="store_true")
    p.add_argument("--allow-no-face-check", action="store_true",
                    help="Only for controlled tests; never recommended in production")
    args = p.parse_args()
    for path in (args.video, args.audio, args.wav2lip_repo, args.checkpoint):
        if not path.exists():
            raise SystemExit(f"Missing input: {path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prostudio-lipsync-") as td:
        work = Path(td)
        video_duration = probe(args.video, "duration")
        face = {"skipped": True} if args.allow_no_face_check else detect_faces(
            args.video, max(1, args.sample_every)
        )
        if not args.allow_no_face_check and (
            face["coverage"] < args.min_coverage or
            face["max_face_ratio"] < args.min_face_ratio
        ):
            # Safe output: preserve the video but replace its audio only.
            fitted = work / "fitted.wav"
            timing = fit_audio(args.audio, video_duration, fitted, args.offset_ms)
            mux_audio(args.video, fitted, args.output)
            print(json.dumps({"ok": True, "lip_sync": False, "reason": "face_gate",
                              "face": face, "timing": timing}, indent=2))
            return
        fitted = work / "fitted.wav"
        timing = fit_audio(args.audio, video_duration, fitted, args.offset_ms)
        synced = work / "wav2lip.mp4"
        run_wav2lip(args.wav2lip_repo, args.checkpoint, args.video, fitted, synced,
                    args.resize_factor, args.nosmooth)
        mux_audio(synced, fitted, args.output)
        print(json.dumps({"ok": True, "lip_sync": True, "face": face,
                          "timing": timing, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
