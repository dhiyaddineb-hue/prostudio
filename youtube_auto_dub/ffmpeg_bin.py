"""Locate a working FFmpeg binary (system PATH or imageio-ffmpeg fallback)."""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    env = os.environ.get("FFMPEG_BINARY") or os.environ.get("YAD_FFMPEG")
    if env and Path(env).exists():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).exists():
            return bundled
    except Exception:
        pass
    raise RuntimeError(
        "FFmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`."
    )


@lru_cache(maxsize=1)
def ffprobe_exe() -> str:
    env = os.environ.get("FFPROBE_BINARY") or os.environ.get("YAD_FFPROBE")
    if env and Path(env).exists():
        return env
    found = shutil.which("ffprobe")
    if found:
        return found
    # imageio-ffmpeg only ships ffmpeg; callers should tolerate this
    raise FileNotFoundError("ffprobe not found")


def ensure_ffmpeg_on_path() -> str:
    """Make sure `ffmpeg` is callable as a bare command for child processes."""
    exe = ffmpeg_exe()
    if shutil.which("ffmpeg"):
        return exe
    bindir = Path(exe).parent
    path = os.environ.get("PATH", "")
    if str(bindir) not in path.split(os.pathsep):
        os.environ["PATH"] = f"{bindir}{os.pathsep}{path}"
    alias = bindir / "ffmpeg"
    if not alias.exists() and Path(exe).exists() and alias.parent.exists():
        try:
            alias.symlink_to(exe)
        except OSError:
            pass
    return exe
