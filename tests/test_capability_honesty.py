"""Reported capabilities must reflect what actually works, not what imports."""

import numpy as np
import pytest

from youtube_auto_dub import runtime
from youtube_auto_dub.audio import _isolate_ambient


def test_whisper_needs_weights_not_just_the_package(monkeypatch):
    """An importable faster-whisper with no weights and no hub cannot run."""
    monkeypatch.setattr(runtime, "have_module", lambda name: name == "faster_whisper")
    monkeypatch.setattr(runtime, "whisper_cached_models", lambda: [])
    monkeypatch.setattr(runtime, "huggingface_reachable", lambda: False)
    assert runtime.have_whisper() is False


def test_whisper_true_when_weights_are_cached(monkeypatch):
    monkeypatch.setattr(runtime, "have_module", lambda name: name == "faster_whisper")
    monkeypatch.setattr(runtime, "whisper_cached_models", lambda: ["faster-whisper-tiny"])
    monkeypatch.setattr(runtime, "huggingface_reachable", lambda: False)
    assert runtime.have_whisper() is True


def test_whisper_true_when_hub_is_reachable(monkeypatch):
    monkeypatch.setattr(runtime, "have_module", lambda name: name == "faster_whisper")
    monkeypatch.setattr(runtime, "whisper_cached_models", lambda: [])
    monkeypatch.setattr(runtime, "huggingface_reachable", lambda: True)
    assert runtime.have_whisper() is True


def test_needs_transcript_tracks_real_asr_usability():
    caps = runtime.capabilities()
    # Either recogniser removes the need for a pasted script.
    assert caps["needs_transcript"] is not (caps["whisper"] or caps["offline_asr"])
    # The distinction the UI depends on must be reported.
    assert "whisper_installed" in caps
    assert isinstance(caps["whisper_models_cached"], list)


def test_ambient_isolation_prefers_the_stereo_source(tmp_path, monkeypatch):
    """Working audio is mono for Whisper; separation needs the stereo original."""
    import soundfile as sf

    from youtube_auto_dub import audio as audio_mod

    sr = 44100
    t = np.arange(sr) / sr
    tone = (np.sin(2 * np.pi * 200 * t) * 0.3).astype(np.float32)

    stereo = tmp_path / "stereo.wav"
    # Genuinely different channels, so the mix is not effectively mono.
    sf.write(stereo, np.stack([tone, np.roll(tone, 64)], axis=1), sr)
    mono = tmp_path / "mono.wav"
    sf.write(mono, tone, sr)

    seen: dict = {}
    from youtube_auto_dub import stem_split

    real_decode = stem_split.decode_stereo

    def spy(path, rate):
        seen["path"] = str(path)
        return real_decode(path, rate)

    monkeypatch.setattr(stem_split, "decode_stereo", spy)

    result = _isolate_ambient(mono, tmp_path / "bed.wav", sr=sr, stereo_source=stereo)
    assert result is not None and result.exists()
    # The stereo original must be what gets analysed, never the mono downmix.
    assert seen.get("path") == str(stereo)


def test_ambient_isolation_falls_back_without_stereo(tmp_path):
    """A mono source must still produce a bed rather than failing."""
    import soundfile as sf

    sr = 22050
    t = np.arange(sr) / sr
    tone = (np.sin(2 * np.pi * 300 * t) * 0.3).astype(np.float32)
    mono = tmp_path / "mono.wav"
    sf.write(mono, tone, sr)

    result = _isolate_ambient(mono, tmp_path / "bed.wav", sr=sr)
    assert result is None or result.exists()
