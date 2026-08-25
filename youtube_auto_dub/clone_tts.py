"""Neural voice cloning via F5-TTS, when its weights are actually available.

F5-TTS clones a speaker from a few seconds of reference audio plus the text of
that reference. Here the reference is cut straight from the separated centre
stem, so a dubbed line is spoken in the original actor's own voice.

The weights live on HuggingFace. Where that host is unreachable this module
reports ``available() is False`` and the caller falls back to the acoustic
conversion in ``voice_profile`` — the pipeline never hard-fails on it.

Searching GitHub for weights instead
------------------------------------
GitHub *is* reachable here when HuggingFace is not, and a few projects commit
their model weights straight into the repo, which codeload.github.com will then
serve as a tarball. That route works — ``ramishi/vocale-tts-cli`` ships 186 MB
of voice-cloning ONNX and downloads fine.

It does not solve Arabic. Its tokenizer holds 4000 pieces, of which 66 contain
Arabic script and only 21 are longer than a single character, so Arabic is
spelled out letter by letter: the model is English. The mainstream engines
(F5-TTS, OpenVoice, Coqui) commit no weights at all, and every Arabic project
found — including nipponjo/tts_arabic — hosts its weights on HuggingFace.

Note also that GitHub *Release* assets are not a way around this: they redirect
to release-assets.githubusercontent.com, which is blocked even through an
authenticated ``gh api`` stream. Only in-repo files reachable via codeload work.

If you do point this at a cloning model, mind the licence: the vocale weights
above are CC BY-NC 4.0 and may not be used commercially.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_MODEL = None
_LOAD_FAILED = False

# A clone needs enough clean speech to characterise the voice, but F5 degrades
# on very long references.
REF_MIN_SEC = 3.0
REF_MAX_SEC = 12.0


@dataclass
class CloneRef:
    """A reference clip plus the words spoken in it."""

    audio_path: Path
    text: str

    def valid(self) -> bool:
        return bool(self.text.strip()) and self.audio_path.exists()


def available() -> bool:
    """True when F5-TTS is importable *and* its weights can be loaded."""
    return load_model() is not None


def load_model():
    """Load F5-TTS once. Returns None when unusable (offline, missing, OOM)."""
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _LOAD_FAILED:
        return None
    try:
        from f5_tts.api import F5TTS
    except ImportError:
        _LOAD_FAILED = True
        log.info("f5-tts not installed; cloning disabled")
        return None
    try:
        _MODEL = F5TTS()
    except Exception as exc:  # network, cache miss, corrupt download
        _LOAD_FAILED = True
        log.warning("F5-TTS weights unavailable (%s); cloning disabled", type(exc).__name__)
        return None
    return _MODEL


def pick_reference(
    stem: np.ndarray,
    sr: int,
    start: float,
    end: float,
    text: str,
    dest: Path,
) -> Optional[CloneRef]:
    """Cut the cleanest speech window in ``start..end`` to clone from.

    Picks the loudest contiguous slice, since on a separated stem the quiet
    parts are mostly leftover score rather than the actor.
    """
    import soundfile as sf

    a, b = int(max(start, 0) * sr), int(end * sr)
    seg = stem[a:b]
    span = len(seg) / sr
    if span < REF_MIN_SEC or not text.strip():
        return None

    want = int(min(span, REF_MAX_SEC) * sr)
    if len(seg) > want:
        step = int(sr * 0.25)
        best, best_energy = 0, -1.0
        for off in range(0, len(seg) - want + 1, step):
            energy = float(np.mean(seg[off:off + want] ** 2))
            if energy > best_energy:
                best, best_energy = off, energy
        seg = seg[best:best + want]

    peak = float(np.max(np.abs(seg)) or 0.0)
    if peak < 1e-4:
        return None
    seg = (seg / peak) * 0.89

    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, seg.astype(np.float32), sr)
    return CloneRef(audio_path=dest, text=text.strip())


def clone_speak(text: str, ref: CloneRef, dest: Path, sr: int = 24000) -> bool:
    """Speak ``text`` in the reference speaker's voice. False if unavailable."""
    model = load_model()
    if model is None or not ref.valid():
        return False
    try:
        import soundfile as sf

        wav, out_sr, _ = model.infer(
            ref_file=str(ref.audio_path),
            ref_text=ref.text,
            gen_text=text,
            remove_silence=True,
        )
        wav = np.asarray(wav, dtype=np.float32)
        if wav.size == 0 or not np.isfinite(wav).all():
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dest, wav, int(out_sr or sr))
        return dest.exists() and dest.stat().st_size > 0
    except Exception as exc:
        log.warning("clone failed for %r: %s", text[:40], exc)
        return False
