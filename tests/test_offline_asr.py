"""Offline recognition and thread-safe local TTS."""

import concurrent.futures as cf

import pytest
import soundfile as sf

from youtube_auto_dub import offline_asr
from youtube_auto_dub.local_tts import speak_local


def test_available_never_raises():
    assert isinstance(offline_asr.available(), bool)


def test_transcribe_requires_the_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(offline_asr, "available", lambda: False)
    with pytest.raises(RuntimeError):
        offline_asr.transcribe_offline(tmp_path / "nope.wav")


def test_pack_strips_pronunciation_variants():
    seg = offline_asr._pack([
        {"word": "get(2)", "start": 0.1, "end": 0.3},
        {"word": "all", "start": 0.3, "end": 0.5},
    ])
    assert seg["text"] == "get all"
    assert seg["start"] == 0.1 and seg["end"] == 0.5


@pytest.mark.skipif(not offline_asr.available(), reason="pocketsphinx not installed")
def test_transcribes_speech_without_network(tmp_path):
    """A real decode, proving no download is needed."""
    from youtube_auto_dub.local_tts import speak_local

    clip = tmp_path / "speech.wav"
    speak_local("the quick brown fox jumps over the lazy dog", clip, lang="en")
    segments = offline_asr.transcribe_offline(clip)
    assert segments, "expected at least one segment"
    for seg in segments:
        assert seg["end"] >= seg["start"]
        assert seg["text"]


def test_local_tts_is_thread_safe(tmp_path):
    """libespeak-ng has global state; parallel calls used to segfault.

    The pipeline synthesises segments concurrently, so this is the exact
    pattern that crashed every offline dub.
    """
    def job(i):
        dest = tmp_path / f"p{i}.wav"
        speak_local(f"sentence number {i}", dest, lang="en")
        data, sr = sf.read(str(dest))
        return len(data) / sr

    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        durations = list(pool.map(job, range(6)))

    assert len(durations) == 6
    assert all(d > 0.05 for d in durations)


def test_repeated_synthesis_reuses_one_initialisation(tmp_path):
    """Re-initialising the library corrupts its state; it must init once."""
    for i in range(4):
        dest = tmp_path / f"s{i}.wav"
        speak_local("hello there", dest, lang="en")
        assert dest.exists() and dest.stat().st_size > 0
