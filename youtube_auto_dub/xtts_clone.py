"""Voice cloning with XTTS-v2, which actually speaks Arabic.

F5-TTS was the wrong engine for this job. Its vocabulary is English/Chinese: it
holds 25 Arabic glyphs, is missing ``ا`` — the commonest letter in the language
— and drops 21% of this script before synthesis, which is why the output was
not recognisably Arabic.

XTTS-v2 lists ``ar`` as a supported language and ships dedicated Arabic text
handling (abbreviation expansion, number-to-words, punctuation). Its tokenizer
covers 100% of this script with 554 Arabic tokens, 495 of them multi-character
subwords rather than bare letters.

It clones from a short reference clip of the target speaker. Unlike F5 it does
*not* require a transcript of that clip, which removes the second failure of
the earlier attempt: previously the reference was the actor speaking English
while the reference text was the Arabic translation, and the mismatch made the
output slur into itself.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SR = 24000

# XTTS clones from a few seconds; much more brings no gain and slows loading.
REF_MIN_SEC = 3.0
REF_MAX_SEC = 12.0

_MODEL = None
_LOAD_FAILED = False


def load_model(device: str = "cpu"):
    """Load XTTS-v2 once. Returns None when unavailable, never raises."""
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _LOAD_FAILED:
        return None
    try:
        # Coqui's licence prompt blocks non-interactive runs unless agreed.
        import os

        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from TTS.api import TTS

        _MODEL = TTS(MODEL_NAME).to(device)
    except Exception as exc:
        _LOAD_FAILED = True
        log.warning("XTTS-v2 unavailable (%s): %s", type(exc).__name__, exc)
        return None
    return _MODEL


def available(device: str = "cpu") -> bool:
    return load_model(device) is not None


def write_reference(
    stem: np.ndarray,
    sr: int,
    start: float,
    end: float,
    dest: Path,
) -> Optional[Path]:
    """Save a normalised reference clip of the speaker.

    No transcript is needed — that is the point of using XTTS here.
    """
    import soundfile as sf

    clip = stem[int(max(start, 0) * sr): int(end * sr)]
    if clip.size < int(REF_MIN_SEC * sr * 0.5):
        return None
    peak = float(np.max(np.abs(clip)) or 0.0)
    if peak < 1e-4:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, (clip / peak * 0.89).astype(np.float32), sr)
    return dest


def clone_speak(
    text: str,
    reference: Path,
    dest: Path,
    language: str = "ar",
    device: str = "cpu",
) -> bool:
    """Speak ``text`` in the reference speaker's voice. False if unavailable."""
    model = load_model(device)
    if model is None or not Path(reference).exists() or not (text or "").strip():
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        model.tts_to_file(
            text=text.strip(),
            speaker_wav=str(reference),
            language=language,
            file_path=str(dest),
        )
        return dest.exists() and dest.stat().st_size > 1000
    except Exception as exc:
        log.warning("XTTS clone failed for %r: %s", text[:40], exc)
        return False


def coverage(texts: list[str]) -> tuple[int, int]:
    """How many Arabic characters the tokenizer can represent.

    Used as a pre-flight check so a run cannot silently mangle the script the
    way F5-TTS did.
    """
    model = load_model()
    if model is None:
        return (0, 0)
    try:
        tok = model.synthesizer.tts_model.tokenizer
        vocab = set(tok.tokenizer.get_vocab().keys())
    except Exception:
        return (0, 0)

    letters = [c for t in texts for c in t if "\u0600" <= c <= "\u06ff"]
    known = [c for c in letters if c in vocab]
    return (len(known), len(letters))
