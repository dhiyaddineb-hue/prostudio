"""Tests for the Vosk transcription path.

PocketSphinx was the only recogniser that worked offline here, and its output
was close to unusable — "trolls many types of art which are reacting to what
the removing of awful thing" for "ninety five percent of the population are
reacting to life". Getting a real recogniser working meant getting a model into
a sandbox where every model host is blocked, so these pin the two things that
would silently undo that: the guards that stop a Git LFS pointer being mistaken
for weights, and the shape of the output the dub pipeline consumes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from youtube_auto_dub import vosk_asr

SPEC = importlib.util.spec_from_file_location(
    "fetch_asr_model",
    Path(__file__).resolve().parent.parent / "scripts" / "fetch_asr_model.py",
)
fetch_model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch_model)


def test_an_lfs_pointer_is_recognised_as_not_being_a_model(tmp_path):
    # This is what GitHub serves for any LFS-stored file, and it is exactly
    # what every Whisper model on GitHub turned out to be.
    pointer = tmp_path / "final.mdl"
    pointer.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c\n"
        b"size 147964211\n"
    )
    assert fetch_model.looks_like_lfs_pointer(pointer)


def test_real_weights_are_not_mistaken_for_a_pointer(tmp_path):
    real = tmp_path / "final.mdl"
    real.write_bytes(b"\x00B<TransitionModel> " + b"\x00" * 5000)
    assert not fetch_model.looks_like_lfs_pointer(real)


def test_a_large_text_file_is_not_a_pointer(tmp_path):
    # Pointers are always tiny; size alone is checked first so a big file that
    # happens to start with "version " is not rejected.
    big = tmp_path / "notes.txt"
    big.write_bytes(b"version " + b"x" * 2000)
    assert not fetch_model.looks_like_lfs_pointer(big)


def test_model_presence_requires_real_bytes(tmp_path, monkeypatch):
    fake = tmp_path / "model"
    (fake / "am").mkdir(parents=True)
    monkeypatch.setattr(vosk_asr, "MODEL_DIR", fake)

    assert not vosk_asr.model_present()

    (fake / "am" / "final.mdl").write_bytes(b"tiny")
    assert not vosk_asr.model_present(), "a 4-byte file is not a model"

    (fake / "am" / "final.mdl").write_bytes(b"\x00" * 2_000_000)
    assert vosk_asr.model_present()


def test_transcribe_refuses_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(vosk_asr, "MODEL_DIR", tmp_path / "absent")
    with pytest.raises(RuntimeError, match="no Vosk model"):
        vosk_asr.transcribe(tmp_path / "audio.wav")


def test_collect_builds_segments_with_word_timings():
    # Word timings are the point: they let a line be cut where a phrase really
    # ends rather than at a guess from text length.
    out = []
    vosk_asr._collect(out, json.dumps({
        "text": "hello there world",
        "result": [
            {"word": "hello", "start": 1.0, "end": 1.4},
            {"word": "there", "start": 1.45, "end": 1.8},
            {"word": "world", "start": 1.9, "end": 2.35},
        ],
    }))
    assert len(out) == 1
    seg = out[0]
    assert seg["start"] == 1.0 and seg["end"] == 2.35
    assert seg["text"] == "hello there world"
    assert [w["w"] for w in seg["words"]] == ["hello", "there", "world"]


def test_collect_skips_empty_and_malformed_results():
    out = []
    vosk_asr._collect(out, json.dumps({"text": ""}))
    vosk_asr._collect(out, json.dumps({"text": "   "}))
    vosk_asr._collect(out, "not json at all")
    assert out == []


def test_collect_handles_text_without_word_timings():
    # SetWords(False) returns text with no "result" array; that must not crash.
    out = []
    vosk_asr._collect(out, json.dumps({"text": "some words"}))
    assert out[0]["text"] == "some words"
    assert out[0]["start"] is None and out[0]["words"] == []


def test_required_files_cover_the_parts_vosk_actually_loads():
    # A partial extraction that still "succeeds" would fail later at runtime,
    # so the fetcher checks for each piece Kaldi needs.
    assert "am/final.mdl" in fetch_model.REQUIRED
    assert any("HCLr" in name for name in fetch_model.REQUIRED)
    assert any("conf" in name for name in fetch_model.REQUIRED)
