import json
from pathlib import Path

import pytest

from youtube_auto_dub.content_validation import is_non_speech_text, normalize_tokens, validate_spoken_content, word_timing_report
from youtube_auto_dub.voice_profiles import ENGINES, REFERENCE_MODES, VOICE_CONVERSIONS, load_voice_profiles, template_for_speakers


def test_every_character_gets_all_independent_options(tmp_path):
    document = {
        "speakers": {
            "SPEAKER_00": {"reference_mode": "source", "tts_engine": "xtts", "voice_conversion": "seed-vc", "approved": True},
            "SPEAKER_01": {"reference_mode": "synthetic", "tts_engine": "edge", "voice": "en-US-AndrewMultilingualNeural", "voice_conversion": "none", "approved": True},
        }
    }
    path = tmp_path / "voices.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    profiles = load_voice_profiles(path, ["SPEAKER_00", "SPEAKER_01"], require_approval=True)
    assert profiles["SPEAKER_00"]["tts_engine"] == "xtts"
    assert profiles["SPEAKER_01"]["tts_engine"] == "edge"
    assert profiles["SPEAKER_00"] is not profiles["SPEAKER_01"]
    assert set(ENGINES) == {"xtts", "voxcpm", "qwen", "edge"}
    assert set(REFERENCE_MODES) == {"source", "custom", "synthetic"}
    assert set(VOICE_CONVERSIONS) == {"seed-vc", "none"}


def test_custom_reference_must_exist_and_unapproved_profile_is_refused(tmp_path):
    missing = tmp_path / "missing.wav"
    path = tmp_path / "voices.json"
    path.write_text(json.dumps({"speakers": {"SPEAKER_00": {"reference_mode": "custom", "reference_path": str(missing), "approved": True}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="custom reference is missing"):
        load_voice_profiles(path, ["SPEAKER_00"], require_approval=True)
    path.write_text(json.dumps({"speakers": {"SPEAKER_00": {"reference_mode": "source", "approved": False}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="not approved"):
        load_voice_profiles(path, ["SPEAKER_00"], require_approval=True)


def test_template_contains_full_controls_for_each_character():
    template = template_for_speakers(["A", "B"])
    for speaker in ("A", "B"):
        profile = template["speakers"][speaker]
        assert {"reference_mode", "reference_path", "tts_engine", "voice", "voice_conversion", "style", "gender", "approved"} <= set(profile)


def test_content_validator_detects_missing_phrase():
    expected = "the quick brown fox jumps over the lazy dog near the river"
    good = "The quick brown fox jumps over the lazy dog near the river."
    missing = "the quick fox near the river"
    assert validate_spoken_content(expected, good)["ok"] is True
    result = validate_spoken_content(expected, missing)
    assert result["ok"] is False
    assert result["recall"] < 0.70
    assert "brown" in result["missing_words"]


def test_word_timing_report_records_each_observed_word():
    words = [
        {"word": "hello", "start": 0.0, "end": 0.4},
        {"word": "world", "start": 0.5, "end": 1.0},
    ]
    result = word_timing_report(words, 4.0, 5.0)
    assert result["ok"] is True
    assert result["word_count"] == 2
    assert result["words"][0]["actual_start"] == 4.0
    assert result["words"][-1]["ideal_end"] == 5.0


def test_non_speech_labels_are_not_synthesized():
    assert is_non_speech_text("Music") is True
    assert is_non_speech_text("[MUSIC]") is True
    assert is_non_speech_text("موسيقى") is True
    assert is_non_speech_text("music begins now") is False
