"""Regression tests: Whisper must judge the audio that is actually delivered.

Run #150 of the Napoleon project failed on chunk 3 because ``transcribe`` reused a
``*_16k.wav`` copy restored from the checkpoint archive (0.31 s, produced when the
phrase window was still 0.34 s) instead of the regenerated 1.0 s take. Every later
repair was rejected without ever being heard. These tests pin the fixes.
"""
from __future__ import annotations

import ast
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/resumable_smart_dub.py"
SPEECH = ROOT / "youtube_auto_dub/speech.py"


def _ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _write_wav(path: Path, seconds: float, rate: int, freq: float = 220.0) -> None:
    t = np.arange(int(seconds * rate)) / rate
    pcm = (0.4 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def _wav_seconds(path: Path) -> tuple[float, int]:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate(), handle.getframerate()


def _extract(source: Path, names: set[str], namespace: dict) -> dict:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def test_transcribe_no_longer_trusts_an_existing_16k_copy():
    text = SPEECH.read_text(encoding="utf-8")
    body = text.split("def transcribe(", 1)[1]
    assert "if not wav.exists()" not in body
    assert "wav = prepare_asr_audio(audio)" in body
    helper = text.split("def prepare_asr_audio(", 1)[1].split("def transcribe(", 1)[0]
    assert "os.replace(tmp, wav)" in helper
    assert '"_16k.wav"' in text


def test_prepare_asr_audio_rebuilds_a_stale_16k_copy(tmp_path):
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        pytest.skip("ffmpeg is not available")
    import os

    namespace = {
        "Path": Path, "os": os, "subprocess": subprocess, "SR_WHISPER": 16000,
        "ffmpeg_exe": lambda: ffmpeg,
    }
    _extract(SPEECH, {"asr_cache_path", "prepare_asr_audio", "_is_16k_mono"}, namespace)
    namespace["_is_16k_mono"] = lambda path: False  # the 24 kHz take always needs a copy

    take = tmp_path / "delivery.fitted.wav"
    _write_wav(take, 1.0, 24000)
    stale = tmp_path / "delivery.fitted_16k.wav"
    _write_wav(stale, 0.31, 16000)  # what a restored checkpoint used to hand to Whisper

    result = namespace["prepare_asr_audio"](take)

    assert result == stale
    seconds, rate = _wav_seconds(result)
    assert rate == 16000
    assert abs(seconds - 1.0) < 0.02
    assert not list(tmp_path.glob("*.tmp-*")), "temporary conversion files must not linger"


def test_prepare_asr_audio_returns_a_16k_mono_wav_untouched(tmp_path):
    import os

    namespace = {"Path": Path, "os": os, "subprocess": subprocess, "SR_WHISPER": 16000, "ffmpeg_exe": lambda: "ffmpeg"}
    _extract(SPEECH, {"asr_cache_path", "prepare_asr_audio", "_is_16k_mono"}, namespace)
    namespace["_is_16k_mono"] = lambda path: True
    ready = tmp_path / "probe.wav"
    _write_wav(ready, 0.5, 16000)
    assert namespace["prepare_asr_audio"](ready) == ready
    assert not (tmp_path / "probe_16k.wav").exists()


def test_checkpoint_archives_exclude_derived_asr_caches():
    text = SCRIPT.read_text(encoding="utf-8")
    upload = text.split("def upload_tree(", 1)[1].split("def upload_chunk(", 1)[0]
    assert "is_derived_asr_cache(path)" in upload
    namespace = _extract(SCRIPT, {"is_derived_asr_cache"}, {"Path": Path})
    is_cache = namespace["is_derived_asr_cache"]
    assert is_cache(Path("chunks/0003/delivery.fitted_16k.wav")) is True
    assert is_cache(Path("chunks/0003/content-retry-2.synced_16k.wav")) is True
    assert is_cache(Path("chunks/0003/delivery.fitted_16k.tmp-4242.wav")) is True
    for kept in ("delivery.fitted.wav", "dubbed.mp4", "status.json", "content-retry-2.edge.mp3", "generated.wav"):
        assert is_cache(Path("chunks/0003") / kept) is False, kept


def test_edge_retry_output_is_normalised_to_pcm_wav(tmp_path):
    text = SCRIPT.read_text(encoding="utf-8")
    edge_branch = text.split("await speak_edge(", 1)[1].split('content_retry_mode="edge_exact_short_phrase"', 1)[0]
    assert "retry_raw = ensure_pcm_wav(retry_raw)" in edge_branch

    ffmpeg = _ffmpeg()
    if not ffmpeg:
        pytest.skip("ffmpeg is not available")

    def run(cmd, *, check=True, capture=True):
        cmd = [ffmpeg if cmd[0] == "ffmpeg" else cmd[0], *cmd[1:]]
        return subprocess.run(cmd, check=check, capture_output=capture, text=True)

    namespace = _extract(SCRIPT, {"ensure_pcm_wav"}, {"Path": Path, "shutil": shutil, "run": run, "SR_TTS": 24000, "RuntimeError": RuntimeError})
    ensure_pcm_wav = namespace["ensure_pcm_wav"]

    # A genuine WAV is returned untouched.
    wav = tmp_path / "content-retry-1.wav"
    _write_wav(wav, 0.4, 24000)
    before = wav.read_bytes()
    assert ensure_pcm_wav(wav) == wav and wav.read_bytes() == before

    # Edge-TTS writes MPEG audio even when asked for a .wav path.
    source = tmp_path / "tone.wav"
    _write_wav(source, 0.6, 24000)
    disguised = tmp_path / "content-retry-2.wav"
    probe = subprocess.run([ffmpeg, "-y", "-i", str(source), "-f", "mp3", str(disguised)], capture_output=True, text=True)
    if probe.returncode != 0 or not disguised.exists():
        pytest.skip("ffmpeg build cannot encode MP3")
    assert disguised.read_bytes()[:4] != b"RIFF"

    result = ensure_pcm_wav(disguised)

    assert result == disguised
    assert disguised.read_bytes()[:4] == b"RIFF"
    seconds, rate = _wav_seconds(disguised)
    assert rate == 24000 and abs(seconds - 0.6) < 0.1
    assert (tmp_path / "content-retry-2.edge.mp3").exists(), "original Edge bytes are preserved, not deleted"


def test_completed_silence_chunks_are_not_re_rendered_on_resume():
    text = SCRIPT.read_text(encoding="utf-8")
    render = text.split("failures: list[int] = []", 1)[1]
    block = render.split("content_current = (", 1)[1].split("complete = store.completed_file(index)", 1)[0]
    assert "or non_speech" in block
    assert 'or not chunk.get("source_text")' in block
    assert 'bool((chunk.get("content_validation") or {}).get("ok"))' in block
