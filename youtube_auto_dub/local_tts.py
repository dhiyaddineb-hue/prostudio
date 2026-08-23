"""Offline neural-style fallback TTS via eSpeak NG."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from youtube_auto_dub.ffmpeg_bin import ffmpeg_exe
from youtube_auto_dub.models import SR_TTS

_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_BIN = _ROOT / ".local" / "bin" / "espeak-ng"
_LOCAL_DATA = _ROOT / ".local" / "share" / "espeak-ng-data"

VOICE_BY_LANG = {
    "ar": "ar",
    "en": "en-us",
    "fr": "fr",
    "es": "es",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "ru": "ru",
    "tr": "tr",
    "hi": "hi",
    "ja": "ja",
    "zh": "cmn",
    "ko": "ko",
    "nl": "nl",
    "pl": "pl",
    "uk": "uk",
    "fa": "fa",
    "ur": "ur",
}


def espeak_bin() -> str:
    if _LOCAL_BIN.exists():
        return str(_LOCAL_BIN)
    found = shutil.which("espeak-ng") or shutil.which("espeak")
    if not found:
        raise RuntimeError("espeak-ng is not installed")
    return found


def _env() -> dict:
    env = os.environ.copy()
    if _LOCAL_DATA.exists():
        env["ESPEAK_DATA_PATH"] = str(_LOCAL_DATA)
    local_bin = str(_ROOT / ".local" / "bin")
    env["PATH"] = f"{local_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def speak_local(text: str, dest: Path, lang: str = "ar", gender: str = "male") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw = dest.with_name(dest.stem + "_espeak.wav")
    voice = VOICE_BY_LANG.get(lang, lang or "en")
    if gender == "female":
        voice = f"{voice}+f3"
        pitch, speed = "55", "128"
    else:
        voice = f"{voice}+m3"
        pitch, speed = "38", "122"
    cmd = [
        espeak_bin(),
        "-v", voice,
        "-s", speed,
        "-p", pitch,
        "-a", "160",
        "-w", str(raw),
        text or ".",
    ]
    subprocess.run(cmd, check=True, capture_output=True, env=_env())
    subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(raw), "-ar", str(SR_TTS), "-ac", "1", str(dest)],
        check=True, capture_output=True,
    )
    raw.unlink(missing_ok=True)
    return dest
