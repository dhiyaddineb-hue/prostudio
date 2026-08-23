#!/usr/bin/env python3
"""Build a ready-to-play Arabic dubbed demo video."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from youtube_auto_dub.ffmpeg_bin import ensure_ffmpeg_on_path, ffmpeg_exe
from youtube_auto_dub.local_tts import speak_local
from youtube_auto_dub.pipeline_args import build_args
from youtube_auto_dub.core import run

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
OUTPUT = ROOT / "output"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

EN_SCRIPT = (
    "Welcome to ProStudio. "
    "This short film shows automatic video dubbing. "
    "First we transcribe the speech. "
    "Then we translate the meaning into Arabic. "
    "Finally we generate a new voice and sync it with the picture."
)


def make_source() -> Path:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    ensure_ffmpeg_on_path()
    narration = SAMPLES / "en_narration.wav"
    speak_local(EN_SCRIPT, narration, lang="en", gender="male")
    video = SAMPLES / "prostudio_en.mp4"
    card = SAMPLES / "title_card.png"
    ff = ffmpeg_exe()
    cmd = [
        ff, "-y",
        "-loop", "1", "-i", str(card),
        "-i", str(narration),
        "-shortest",
        "-vf", "scale=1280:720,format=yuv420p",
        "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-r", "25",
        str(video),
    ]
    import subprocess
    subprocess.run(cmd, check=True)
    return video


async def main() -> None:
    source = make_source()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    args = build_args(
        str(source),
        lang="ar",
        mode="both",
        gender="male",
        model="tiny",
        bg_music=False,
        output_dir=str(OUTPUT),
        transcript=EN_SCRIPT,
        source_lang="en",
    )
    out = await run(args)
    final = OUTPUT / "ProStudio_Arabic_Demo.mp4"
    shutil.copy2(out, final)
    srt = Path(str(out)).with_suffix(".srt")
    temp_srt = ROOT / "temp" / "subtitles.srt"
    if temp_srt.exists():
        shutil.copy2(temp_srt, OUTPUT / "ProStudio_Arabic_Demo.srt")
    elif srt.exists():
        shutil.copy2(srt, OUTPUT / "ProStudio_Arabic_Demo.srt")
    print("DEMO", final, final.stat().st_size)


if __name__ == "__main__":
    asyncio.run(main())
