"""Use the public Seed-VC Space to transfer the source speaker's voice onto a dub.

The dub audio supplies English content and timing; the original audio supplies the
reference timbre. The converted WAV is then muxed back into the original video.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from gradio_client import Client, handle_file


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--dubbed", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--space", default="phuoc2005/seed-vc")
    args = parser.parse_args()

    work = Path(args.output).with_suffix(".seed-work")
    work.mkdir(parents=True, exist_ok=True)
    original_wav = work / "original-reference.wav"
    dubbed_wav = work / "english-content.wav"
    converted_wav = work / "seed-converted.wav"

    run(["ffmpeg", "-y", "-i", args.original, "-vn", "-ac", "1", "-ar", "22050", str(original_wav)])
    run(["ffmpeg", "-y", "-i", args.dubbed, "-vn", "-ac", "1", "-ar", "22050", str(dubbed_wav)])

    client = Client(args.space)
    result = client.predict(
        handle_file(str(dubbed_wav)),
        handle_file(str(original_wav)),
        25,
        1.0,
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

    # Keep the video stream untouched and fit the converted track to its duration.
    run([
        "ffmpeg", "-y", "-i", args.dubbed, "-i", str(converted_wav),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
        "-af", "apad", "-shortest", "-c:a", "aac", "-b:a", "128k", args.output,
    ])
    print(json.dumps({"ok": True, "output": args.output, "space": args.space}))


if __name__ == "__main__":
    main()
