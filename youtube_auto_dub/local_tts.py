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

# libespeak-ng constants (speak_lib.h)
_AUDIO_OUTPUT_RETRIEVAL = 1
_ESPEAKCHARS_UTF8 = 1
_RATE = 1
_PITCH = 3

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


def espeak_bin() -> str | None:
    if _LOCAL_BIN.exists():
        return str(_LOCAL_BIN)
    return shutil.which("espeak-ng") or shutil.which("espeak")


def _synth_via_library(text: str, dest: Path, voice: str, speed: int, pitch: int) -> bool:
    """Synthesize with the libespeak-ng shared object shipped by espeakng-loader.

    Distros without an ``espeak-ng`` executable (slim containers, CI images)
    can still dub offline through the library. Returns False when unavailable.
    """
    try:
        import array
        import ctypes
        import wave

        import espeakng_loader
    except ImportError:
        return False

    try:
        lib = ctypes.CDLL(espeakng_loader.get_library_path())
        samples: list[int] = []

        callback = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p
        )

        def _collect(wav, count, _events):
            if wav and count > 0:
                samples.extend(wav[i] for i in range(count))
            return 0

        keep = callback(_collect)
        rate = lib.espeak_Initialize(
            _AUDIO_OUTPUT_RETRIEVAL, 0, espeakng_loader.get_data_path().encode(), 0
        )
        if rate <= 0:
            return False
        lib.espeak_SetSynthCallback(keep)
        if lib.espeak_SetVoiceByName(voice.encode()) != 0:
            # Fall back to the bare language code without the +mN/+fN variant.
            if lib.espeak_SetVoiceByName(voice.split("+")[0].encode()) != 0:
                return False
        lib.espeak_SetParameter(_RATE, speed, 0)
        lib.espeak_SetParameter(_PITCH, pitch, 0)

        payload = (text or ".").encode("utf-8")
        lib.espeak_Synth(
            payload, len(payload) + 1, 0, 1, 0, _ESPEAKCHARS_UTF8, None, None
        )
        lib.espeak_Synchronize()
        if not samples:
            return False

        dest.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(dest), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(array.array("h", samples).tobytes())
        return True
    except Exception:
        return False


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
    binary = espeak_bin()
    if binary:
        cmd = [
            binary,
            "-v", voice,
            "-s", speed,
            "-p", pitch,
            "-a", "160",
            "-w", str(raw),
            text or ".",
        ]
        subprocess.run(cmd, check=True, capture_output=True, env=_env())
    elif not _synth_via_library(text, raw, voice, int(speed), int(pitch)):
        raise RuntimeError(
            "espeak-ng is not installed (install the binary or `pip install espeakng-loader`)"
        )

    subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(raw), "-ar", str(SR_TTS), "-ac", "1", str(dest)],
        check=True, capture_output=True,
    )
    raw.unlink(missing_ok=True)
    return dest
