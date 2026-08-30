"""Fully offline speech recognition via PocketSphinx.

Whisper is the better recogniser, but it downloads its weights on first use.
Where that download is impossible — an air-gapped box, a locked-down CI runner,
this sandbox — the studio previously had no automatic transcription at all and
forced the user to paste a script by hand.

PocketSphinx ships a complete English acoustic model *inside its PyPI wheel*
(~37 MB), so `pip install pocketsphinx` is all it needs. It is markedly less
accurate than Whisper and English-only, so it is offered as a clearly labelled
fallback rather than a silent substitute.
"""

from __future__ import annotations

import logging
import os
import subprocess
import wave
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from youtube_auto_dub.ffmpeg_bin import ffmpeg_exe

log = logging.getLogger(__name__)

SR = 16000
# Filler tokens PocketSphinx emits that are not words.
_NOISE = {"<s>", "</s>", "<sil>", "[SPEECH]", "[NOISE]", "<unk>"}


@lru_cache(maxsize=1)
def available() -> bool:
    """True when PocketSphinx and its bundled English model are present."""
    try:
        from pocketsphinx import get_model_path
    except ImportError:
        return False
    try:
        base = Path(get_model_path()) / "en-us"
        return (base / "en-us").is_dir() and (base / "cmudict-en-us.dict").exists()
    except Exception:
        return False


def _to_wav(src: Path) -> Path:
    """Decode to the 16 kHz mono PCM PocketSphinx expects."""
    dst = src.with_name(src.stem + "_ps16k.wav")
    subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(src), "-ar", str(SR), "-ac", "1",
         "-c:a", "pcm_s16le", str(dst)],
        check=True, capture_output=True,
    )
    return dst


def _decoder():
    from pocketsphinx import Config, Decoder, get_model_path

    base = Path(get_model_path()) / "en-us"
    config = Config()
    config.set_string("-hmm", str(base / "en-us"))
    config.set_string("-lm", str(base / "en-us.lm.bin"))
    config.set_string("-dict", str(base / "cmudict-en-us.dict"))
    config.set_string("-logfn", os.devnull)
    return Decoder(config)


def transcribe_offline(
    audio: Path,
    max_gap: float = 0.6,
    max_duration: float = 8.0,
) -> List[dict]:
    """Transcribe ``audio`` into timed segments, entirely offline.

    Words are grouped into utterances on silence, mirroring the shape the rest
    of the pipeline expects from Whisper: ``{"start", "end", "text"}``.
    """
    if not available():
        raise RuntimeError(
            "PocketSphinx is not installed (pip install pocketsphinx)."
        )

    wav = audio if audio.suffix == ".wav" else _to_wav(audio)
    decoder = _decoder()
    with wave.open(str(wav), "rb") as handle:
        frames = handle.readframes(handle.getnframes())

    decoder.start_utt()
    decoder.process_raw(frames, False, True)
    decoder.end_utt()

    words = [
        {"word": s.word, "start": s.start_frame / 100.0, "end": s.end_frame / 100.0}
        for s in decoder.seg()
        if s.word not in _NOISE
    ]
    if not words:
        return []

    segments: List[dict] = []
    current = [words[0]]
    for word in words[1:]:
        gap = word["start"] - current[-1]["end"]
        span = word["end"] - current[0]["start"]
        if gap > max_gap or span > max_duration:
            segments.append(_pack(current))
            current = [word]
        else:
            current.append(word)
    segments.append(_pack(current))
    return [s for s in segments if s["text"]]


def _pack(words: List[dict]) -> dict:
    # PocketSphinx marks pronunciation variants as "word(2)".
    text = " ".join(w["word"].split("(")[0] for w in words).strip()
    return {"start": words[0]["start"], "end": words[-1]["end"], "text": text}


def transcript_text(audio: Path) -> Optional[str]:
    """Plain transcript, or None when nothing was recognised."""
    try:
        segments = transcribe_offline(audio)
    except Exception as exc:
        log.warning("Offline ASR failed: %s", exc)
        return None
    return " ".join(s["text"] for s in segments).strip() or None
