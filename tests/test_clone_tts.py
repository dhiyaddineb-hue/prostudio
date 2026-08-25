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


def test_merge_runs_joins_consecutive_cues():
    """Single captions are too short to clone from; merged runs are not."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cp", "scripts/clone_project.py"
    )
    src = open("scripts/clone_project.py").read()
    ns = {"np": np, "REF_MIN_SEC": 3.0, "SR": 44100}
    exec(compile(src[src.index("def merge_runs"):src.index("def main(")], "x", "exec"), ns)

    # Three consecutive lines, none of them 3s on its own.
    windows = [(2.4, 4.7), (5.0, 7.2), (7.7, 9.0)]
    runs = ns["merge_runs"](windows)
    assert len(runs) == 1
    assert runs[0][1] - runs[0][0] > 6.0

    # A real pause must still split the runs.
    spaced = [(0.0, 2.0), (30.0, 32.0)]
    assert len(ns["merge_runs"](spaced)) == 2


def test_pick_reference_falls_back_to_the_longest_window():
    """A short reference still beats refusing to clone."""
    import importlib.util

    src = open("scripts/clone_project.py").read()
    ns = {"np": np, "REF_MIN_SEC": 3.0, "SR": 44100}
    exec(compile(src[src.index("def merge_runs"):src.index("def main(")], "x", "exec"), ns)

    stem = (np.random.randn(44100 * 10) * 0.1).astype(np.float32)
    short = [(1.0, 3.5)]  # under REF_MIN_SEC
    win = ns["pick_reference_window"](stem, short, 8.0)
    assert win is not None

    # Anything genuinely tiny is still rejected.
    assert ns["pick_reference_window"](stem, [(1.0, 1.4)], 8.0) is None
