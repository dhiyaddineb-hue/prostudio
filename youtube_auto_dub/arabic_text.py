"""Arabic text preparation for speech synthesis.

Undiacritised Arabic is ambiguous: the same letters can be several different
words, and a synthesiser has to guess. Adding the short vowels removes that
guesswork and audibly improves pronunciation.

The diacritiser ships inside the ``piper-tts`` wheel, so this works with no
network access — unlike every other Arabic model I could find, which live on
HuggingFace.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Optional

log = logging.getLogger(__name__)

# Short vowels and sukun/shadda: present means the text is already diacritised.
_HARAKAT = re.compile(r"[\u064b-\u0652\u0670]")
_ARABIC = re.compile(r"[\u0600-\u06ff]")


def is_arabic(text: str) -> bool:
    return bool(text) and bool(_ARABIC.search(text))


def diacritic_ratio(text: str) -> float:
    """Share of Arabic letters that already carry a vowel mark."""
    letters = _ARABIC.findall(text or "")
    if not letters:
        return 0.0
    return len(_HARAKAT.findall(text)) / len(letters)


@lru_cache(maxsize=1)
def _diacritiser():
    """Load the bundled model once, or None when piper-tts is absent."""
    try:
        from piper.tashkeel import TashkeelDiacritizer

        return TashkeelDiacritizer()
    except Exception as exc:
        log.info("Arabic diacritiser unavailable (%s)", type(exc).__name__)
        return None


def available() -> bool:
    return _diacritiser() is not None


def add_diacritics(text: str, min_ratio: float = 0.15) -> str:
    """Vowel-mark Arabic ``text`` for synthesis.

    Returns the input unchanged when the text is not Arabic, is already
    diacritised, or the model is missing — this must never be the reason a dub
    fails.
    """
    if not is_arabic(text):
        return text
    if diacritic_ratio(text) >= min_ratio:
        return text  # already marked up by hand

    model = _diacritiser()
    if model is None:
        return text
    try:
        out = model.diacritize(text)
    except Exception as exc:
        log.warning("Diacritisation failed for %r: %s", text[:40], exc)
        return text

    # Trust the result only if it merely added vowel marks. If the letters
    # themselves changed, the model rewrote the line rather than marking it up,
    # and speaking that would put words in the actor's mouth.
    if _HARAKAT.sub("", out).replace(" ", "") != text.replace(" ", ""):
        log.warning("Diacritiser altered the text; keeping the original")
        return text
    return out or text


def prepare(text: str) -> str:
    """Everything a line needs before it is handed to a synthesiser."""
    return add_diacritics((text or "").strip())
