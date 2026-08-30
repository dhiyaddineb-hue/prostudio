"""Optional neural source separation for dialogue/background preservation."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from youtube_auto_dub.ffmpeg_bin import ffmpeg_exe

log = logging.getLogger(__name__)


def separate_dialogue_background(source: Path, work: Path) -> tuple[Path, Path] | None:
    """Return (speech, background) from Demucs when installed and successful.

    Demucs is trained for vocal/accompaniment separation rather than cinema
    dialogue, so callers must still measure contamination and use fallback when
    the result is not usable.
    """
    if shutil.which("demucs") is None:
        log.warning("Demucs is not installed; neural source separation skipped")
        return None
    out = work / "demucs"
    out.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["demucs", "--two-stems=vocals", "-n", os.environ.get("YAD_DEMUCS_MODEL", "htdemucs"),
             "--out", str(out), str(source)],
            check=True,
            capture_output=True,
            text=True,
        )
        stem_dir = out / os.environ.get("YAD_DEMUCS_MODEL", "htdemucs") / source.stem
        speech = stem_dir / "vocals.wav"
        background = stem_dir / "no_vocals.wav"
        if speech.exists() and background.exists():
            return speech, background
    except Exception as exc:
        log.warning("Demucs separation failed; using fallback: %s", exc)
    return None


def convert_mono(path: Path, dest: Path, rate: int = 24000) -> Path:
    subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(path), "-ar", str(rate), "-ac", "1", str(dest)],
        check=True,
        capture_output=True,
    )
    return dest
