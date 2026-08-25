"""Cloning must degrade gracefully when weights are unreachable."""

import numpy as np
import soundfile as sf

from youtube_auto_dub import clone_tts
from youtube_auto_dub.clone_tts import CloneRef, clone_speak, pick_reference

SR = 24000


def _speechish(dur: float, sr: int = SR, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    sig = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 8))
    return (sig * amp).astype(np.float32)


def test_available_never_raises():
    assert isinstance(clone_tts.available(), bool)


def test_pick_reference_rejects_short_windows(tmp_path):
    stem = _speechish(1.0)
    assert pick_reference(stem, SR, 0.0, 1.0, "مرحبا", tmp_path / "r.wav") is None


def test_pick_reference_rejects_empty_text(tmp_path):
    stem = _speechish(6.0)
    assert pick_reference(stem, SR, 0.0, 6.0, "   ", tmp_path / "r.wav") is None


def test_pick_reference_rejects_silence(tmp_path):
    stem = np.zeros(int(SR * 6), dtype=np.float32)
    assert pick_reference(stem, SR, 0.0, 6.0, "مرحبا", tmp_path / "r.wav") is None


def test_pick_reference_writes_a_capped_clip(tmp_path):
    stem = _speechish(30.0)
    ref = pick_reference(stem, SR, 0.0, 30.0, "نص مرجعي", tmp_path / "r.wav")
    assert ref is not None and ref.valid()
    data, sr = sf.read(str(ref.audio_path))
    assert sr == SR
    assert len(data) / sr <= clone_tts.REF_MAX_SEC + 0.1
    assert float(np.max(np.abs(data))) <= 0.95


def test_pick_reference_prefers_the_loudest_window(tmp_path):
    """The quiet half of a separated stem is mostly leftover score."""
    quiet = _speechish(8.0, amp=0.02)
    loud = _speechish(8.0, amp=0.5)
    stem = np.concatenate([quiet, loud])
    ref = pick_reference(stem, SR, 0.0, 16.0, "نص", tmp_path / "r.wav")
    assert ref is not None
    data, _ = sf.read(str(ref.audio_path))
    # normalised, but the chosen slice should come from the energetic half
    assert float(np.sqrt(np.mean(data ** 2))) > 0.1


def test_clone_speak_returns_false_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(clone_tts, "load_model", lambda: None)
    ref = CloneRef(audio_path=tmp_path / "missing.wav", text="نص")
    assert clone_speak("مرحبا", ref, tmp_path / "out.wav") is False


def test_clone_speak_rejects_invalid_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(clone_tts, "load_model", lambda: object())
    ref = CloneRef(audio_path=tmp_path / "nope.wav", text="نص")
    assert clone_speak("مرحبا", ref, tmp_path / "out.wav") is False
