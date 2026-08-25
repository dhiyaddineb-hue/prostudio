"""Studio voice bank — pre-rendered professional neural takes.

Edge-TTS needs Microsoft's endpoint and eSpeak sounds robotic. When a line has
an approved studio take on disk we use it verbatim, which is the only way to
get a genuinely professional read without network access at dub time.

Takes live in ``samples/voices/`` and are indexed by a normalised hash of the
spoken text, so re-running a dub reuses the same approved audio.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Optional

from youtube_auto_dub.ffmpeg_bin import ffmpeg_exe
from youtube_auto_dub.models import SR_TTS

ROOT = Path(__file__).resolve().parent.parent
VOICE_DIR = ROOT / "samples" / "voices"
INDEX_PATH = VOICE_DIR / "index.json"

# Arabic diacritics and tatweel carry no phonetic weight for lookup purposes.
_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u0640]")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Collapse a line to a stable lookup key."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _DIACRITICS.sub("", text)
    # Unify alef/ya/ta-marbuta variants that differ only in orthography.
    for src, dst in (("أإآٱ", "ا"), ("ى", "ي"), ("ة", "ه")):
        text = text.translate({ord(c): dst for c in src})
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip().lower()


def key_for(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:16]


def load_index() -> dict:
    if INDEX_PATH.exists():
        try:
            return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_index(index: dict) -> None:
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def register(text: str, audio: Path, voice: str = "studio") -> str:
    """Record an approved take so later dubs of the same line reuse it."""
    index = load_index()
    entry_key = key_for(text)
    index[entry_key] = {
        "text": text,
        "file": Path(audio).name,
        "voice": voice,
    }
    save_index(index)
    return entry_key


def lookup(text: str) -> Optional[Path]:
    """Return the approved take for ``text``, if one exists."""
    entry = load_index().get(key_for(text))
    if not entry:
        return None
    path = VOICE_DIR / entry["file"]
    return path if path.exists() else None


def speak_studio(text: str, dest: Path) -> bool:
    """Render ``text`` from the voice bank into ``dest``.

    Returns False when the line has no approved take, letting the caller fall
    back to Edge-TTS or eSpeak.
    """
    source = lookup(text)
    if not source:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(source),
         "-ar", str(SR_TTS), "-ac", "1", str(dest)],
        check=True, capture_output=True,
    )
    return dest.exists() and dest.stat().st_size > 0


def coverage(lines: list[str]) -> tuple[int, int]:
    """How many of ``lines`` already have approved takes."""
    have = sum(1 for line in lines if lookup(line))
    return have, len(lines)
