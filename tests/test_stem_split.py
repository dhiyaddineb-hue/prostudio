"""Centre/side separation must be lossless and actually isolate the centre."""

import numpy as np

from youtube_auto_dub.stem_split import split_center


def _tone(freq: float, dur: float = 1.0, sr: int = 44100) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    return (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)


def test_reconstruction_is_lossless():
    """voice + music must add back up to the mono original."""
    sr = 44100
    centre = _tone(180, sr=sr)
    wide = _tone(700, sr=sr)
    left = centre + wide
    right = centre - wide
    voice, music = split_center(left, right, sr)
    mono = (left + right) / 2.0
    err = float(np.sqrt(np.mean((mono - (voice + music)) ** 2)))
    assert err < 1e-5


def test_centre_content_lands_in_the_voice_stem():
    """A hard-panned tone must not dominate the dialogue stem."""
    sr = 44100
    centre = _tone(200, sr=sr)      # in the voice band, phantom centre
    wide = _tone(3000, sr=sr)       # out of band and hard panned

    left = centre + wide
    right = centre - wide
    voice, _ = split_center(left, right, sr)

    def energy_at(sig, freq):
        spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        freqs = np.fft.rfftfreq(len(sig), 1 / sr)
        band = (freqs > freq - 30) & (freqs < freq + 30)
        return float(spec[band].max())

    assert energy_at(voice, 200) > energy_at(voice, 3000) * 5


def test_silence_stays_silent():
    sr = 44100
    zeros = np.zeros(sr, dtype=np.float32)
    voice, music = split_center(zeros, zeros, sr)
    assert float(np.max(np.abs(voice))) < 1e-6
    assert float(np.max(np.abs(music))) < 1e-6


def test_output_length_matches_input():
    sr = 44100
    left = _tone(150, 0.7, sr)
    right = _tone(150, 0.7, sr)
    voice, music = split_center(left, right, sr)
    assert len(voice) == len(left)
    assert len(music) == len(left)
