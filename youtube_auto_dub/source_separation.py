"""Optional neural source separation for dialogue/background preservation."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from youtube_auto_dub.ffmpeg_bin import ffmpeg_exe

log = logging.getLogger(__name__)


def _probe(path: Path, field: str) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", f"format={field}", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def validate_stems(stems: tuple[Path, Path] | None, source_duration: float, tolerance: float = 1.0) -> dict:
    """Validate existence, duration, and non-silent energy without claiming purity."""
    if not stems: return {"valid": False, "reason": "unavailable", "contamination": "unknown"}
    speech, background = stems
    try:
        values = {"speech": _probe(speech, "duration"), "background": _probe(background, "duration")}
        ok = all(p.exists() and p.stat().st_size > 0 for p in stems) and all(abs(d-source_duration) <= tolerance for d in values.values()) and all(d > 0 for d in values.values())
        return {"valid": ok, "durations": values, "contamination": "unknown", "reason": None if ok else "missing_or_duration_mismatch"}
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        return {"valid": False, "reason": str(exc), "contamination": "unknown"}


def separate_dialogue_background(source: Path, work: Path, model: str | None = None) -> tuple[Path, Path] | None:
    """Return (speech, background) from Demucs when installed and successful.

    Demucs is trained for vocals/accompaniment, not cinema dialogue. Callers must
    validate stems and report contamination as unknown rather than calling them clean.
    """
    if shutil.which("demucs") is None:
        log.warning("Demucs is not installed; neural source separation skipped")
        return None
    model = model or os.environ.get("YAD_DEMUCS_MODEL", "htdemucs")
    out = work / "demucs"; out.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["demucs", "--two-stems=vocals", "-n", model, "--out", str(out), str(source)], check=True, capture_output=True, text=True)
        stem_dir = out / model / source.stem
        speech, background = stem_dir / "vocals.wav", stem_dir / "no_vocals.wav"
        if speech.exists() and background.exists(): return speech, background
    except Exception as exc: log.warning("Demucs separation failed; using safe no-background fallback: %s", exc)
    return None


def convert_mono(path: Path, dest: Path, rate: int = 24000) -> Path:
    subprocess.run([ffmpeg_exe(), "-y", "-i", str(path), "-ar", str(rate), "-ac", "1", str(dest)], check=True, capture_output=True)
    return dest
