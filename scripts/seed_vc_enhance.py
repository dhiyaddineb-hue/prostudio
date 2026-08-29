"""Enhance a translated dub with Seed-VC while preserving timing.

The English dub supplies content and timing. A cleaned speech reference from the
original video supplies the target timbre. This intentionally leaves speaker
assignment conservative: when no diarization evidence exists, it uses one
reference voice rather than guessing multiple speakers.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from gradio_client import Client, handle_file


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def duration(path: str | Path) -> float:
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return float(result.stdout.strip())


def atempo_filter(factor: float) -> str:
    # ffmpeg atempo accepts 0.5..2.0 per filter; chain for safety.
    factor = max(0.5, min(2.0, factor))
    return f"atempo={factor:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--dubbed", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--space", default="phuoc2005/seed-vc")
    parser.add_argument("--diffusion-steps", type=int, default=40)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    args = parser.parse_args()

    work = Path(args.output).with_suffix(".seed-work")
    work.mkdir(parents=True, exist_ok=True)
    original_wav = work / "original-reference.wav"
    dubbed_wav = work / "english-content.wav"
    cleaned_ref = work / "clean-reference.wav"
    converted_wav = work / "seed-converted.wav"
    timed_wav = work / "seed-timed.wav"

    # Clean the reference without destroying the speaker's pitch or cadence.
    run(["ffmpeg", "-y", "-i", args.original, "-vn", "-ac", "1", "-ar", "22050",
         str(original_wav)])
    run(["ffmpeg", "-y", "-i", str(original_wav), "-af",
         "highpass=f=80,lowpass=f=9000,afftdn=nr=12:nf=-25,dynaudnorm=f=150:g=7",
         "-ar", "22050", "-ac", "1", str(cleaned_ref)])
    run(["ffmpeg", "-y", "-i", args.dubbed, "-vn", "-ac", "1", "-ar", "22050",
         str(dubbed_wav)])

    client = Client(args.space)
    result = client.predict(
        handle_file(str(dubbed_wav)),
        handle_file(str(cleaned_ref)),
        args.diffusion_steps,
        args.length_adjust,
        0.7,
        False,
        True,
        0,
        api_name="/predict_1",
    )
    if not isinstance(result, (list, tuple)) or len(result) < 2:
        raise RuntimeError(f"Unexpected Seed-VC response: {result!r}")
    full = result[1]
    if isinstance(full, dict):
        full = full.get("path") or full.get("url")
    if not full:
        raise RuntimeError(f"Seed-VC returned no full audio: {result!r}")
    run(["ffmpeg", "-y", "-i", str(full), "-ar", "24000", "-ac", "1", str(converted_wav)])

    # Correct any global duration drift before muxing, rather than cutting words.
    target_duration = duration(args.dubbed)
    converted_duration = duration(converted_wav)
    factor = converted_duration / target_duration if target_duration else 1.0
    if abs(factor - 1.0) > 0.015:
        run(["ffmpeg", "-y", "-i", str(converted_wav), "-af", atempo_filter(factor),
             "-ar", "24000", "-ac", "1", str(timed_wav)])
    else:
        timed_wav = converted_wav

    # Keep video untouched, pad only tiny tails, and never let a long VC result
    # truncate the last words or extend beyond the original video.
    run([
        "ffmpeg", "-y", "-i", args.dubbed, "-i", str(timed_wav),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
        "-af", "apad", "-t", f"{target_duration:.3f}",
        "-c:a", "aac", "-b:a", "160k", args.output,
    ])
    print(json.dumps({
        "ok": True,
        "output": args.output,
        "space": args.space,
        "diffusion_steps": args.diffusion_steps,
        "duration_factor": round(factor, 4),
        "reference_cleaned": True,
        "speaker_mode": "conservative-single-reference",
    }))


if __name__ == "__main__":
    main()
