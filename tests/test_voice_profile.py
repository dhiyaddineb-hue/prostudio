"""Voice measurement must be honest, and conversion must move the voice."""

import numpy as np
import pytest

from youtube_auto_dub.voice_profile import VoiceProfile, convert, measure

parselmouth = pytest.importorskip("parselmouth")

SR = 22050


def _voice(f0: float, dur: float = 1.6, sr: int = SR) -> np.ndarray:
    """A crude glottal-ish source: harmonics with a decaying spectrum."""
    t = np.arange(int(sr * dur)) / sr
    sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    # light amplitude modulation so pitch tracking has something to lock onto
    sig *= 1.0 + 0.1 * np.sin(2 * np.pi * 4.0 * t)
    return (sig * 0.2).astype(np.float32)


def test_measure_recovers_known_pitch():
    prof = measure(_voice(150.0), SR, focus=False)
    assert prof is not None
    assert abs(prof.f0_median - 150.0) / 150.0 < 0.08


def test_measure_returns_none_on_silence():
    assert measure(np.zeros(SR, dtype=np.float32), SR, focus=False) is None


def test_measure_returns_none_when_too_short():
    assert measure(np.zeros(100, dtype=np.float32), SR, focus=False) is None


def test_reliability_rejects_wide_spread():
    """A wide F0 spread means music, not a speaker."""
    steady = VoiceProfile(200.0, 10.0, 600, 1700, 2500, 0.8, 120)
    noisy = VoiceProfile(200.0, 160.0, 600, 1700, 2500, 0.8, 120)
    sparse = VoiceProfile(200.0, 10.0, 600, 1700, 2500, 0.8, 5)
    assert steady.reliable is True
    assert noisy.reliable is False
    assert sparse.reliable is False


def test_convert_moves_pitch_toward_target():
    low = _voice(120.0)
    src = measure(low, SR, focus=False)
    tgt = VoiceProfile(170.0, 8.0, src.f1, src.f2, src.f3, 0.9, 200)
    out, report = convert(low, SR, src, tgt)
    got = measure(out, SR, focus=False)
    assert got is not None
    assert abs(got.f0_median - 170.0) < abs(src.f0_median - 170.0)
    assert report["pitch_semitones"] > 0


def test_convert_preserves_duration_and_stays_finite():
    sig = _voice(140.0)
    src = measure(sig, SR, focus=False)
    tgt = VoiceProfile(185.0, 8.0, src.f1, src.f2, src.f3, 0.9, 200)
    out, _ = convert(sig, SR, src, tgt)
    assert np.isfinite(out).all()
    assert abs(len(out) - len(sig)) / len(sig) < 0.05


def test_convert_clamps_extreme_shifts():
    """A wild target must not be followed off a cliff."""
    sig = _voice(110.0)
    src = measure(sig, SR, focus=False)
    tgt = VoiceProfile(400.0, 8.0, 3000, 3200, 3400, 0.9, 200)
    _, report = convert(sig, SR, src, tgt, max_semitones=4.0, max_formant_ratio=0.18)
    assert abs(report["pitch_semitones"]) <= 4.0
    assert 0.82 <= report["formant_ratio"] <= 1.18


def test_convert_is_a_noop_without_profiles():
    sig = _voice(150.0)
    out, report = convert(sig, SR, None, None)
    assert out is sig
    assert report["pitch_semitones"] == 0.0
