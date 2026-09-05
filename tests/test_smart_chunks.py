import json
from pathlib import Path

import pytest

from youtube_auto_dub.smart_chunks import CheckpointStore, plan_smart_chunks, sha256_file


def words_segment(words):
    return {
        "start": words[0][1],
        "end": words[-1][2],
        "text": " ".join(item[0] for item in words),
        "confidence": 0.9,
        "words": [{"word": token, "start": start, "end": end} for token, start, end in words],
    }


def test_smart_chunks_are_contiguous_and_never_exceed_ten_seconds():
    raw = [words_segment([
        ("This", 0.4, 1.0), ("ends.", 1.1, 3.1),
        ("A", 3.5, 4.0), ("long", 4.1, 6.0), ("sentence", 6.1, 8.9),
        ("continues", 9.1, 11.8), ("safely.", 12.0, 14.4),
        ("Final", 15.0, 17.0), ("words", 17.1, 19.5),
    ])]
    chunks = plan_smart_chunks(raw, 23.0, max_seconds=10.0, target_seconds=8.0, min_seconds=2.5)
    assert chunks[0]["start"] == 0.0
    assert chunks[-1]["end"] == 23.0
    assert all(chunk["duration"] <= 10.0 for chunk in chunks)
    assert all(chunks[i]["end"] == chunks[i + 1]["start"] for i in range(len(chunks) - 1))
    observed = [word_id for chunk in chunks for word_id in chunk["word_ids"]]
    assert observed == list(range(9))


def test_sentence_and_pause_are_preferred_over_random_hard_cut():
    raw = [words_segment([
        ("First", 0.3, 1.2), ("sentence.", 1.3, 4.2),
        ("Second", 4.9, 6.0), ("keeps", 6.1, 7.0), ("going", 7.1, 8.7),
        ("more", 8.8, 9.5), ("later", 9.6, 11.0),
    ])]
    chunks = plan_smart_chunks(raw, 12.0, max_seconds=10.0, target_seconds=8.0, min_seconds=2.5)
    assert chunks[0]["end"] == 4.2
    assert chunks[0]["cut_reason"] == "sentence_end"
    assert chunks[0]["source_text"] == "First sentence."


def test_long_silence_is_covered_without_inventing_words():
    raw = [words_segment([("hello", 21.0, 22.0)])]
    chunks = plan_smart_chunks(raw, 25.0, max_seconds=10.0, target_seconds=8.0, min_seconds=2.5)
    assert chunks[0]["start"] == 0.0 and chunks[0]["end"] == 10.0
    assert chunks[1]["start"] == 10.0 and chunks[1]["end"] == 20.0
    assert chunks[0]["source_text"] == ""
    assert chunks[1]["source_text"] == ""
    assert "hello" in [chunk["source_text"] for chunk in chunks]


def test_checkpoint_resume_skips_only_valid_completed_chunk(tmp_path):
    plans = plan_smart_chunks([words_segment([("hello.", 0.2, 2.0)])], 3.0)
    source_file = tmp_path / "source.mp4"
    source_file.write_bytes(b"source")
    store = CheckpointStore(tmp_path / "checkpoint")
    source = {"path": source_file.name, "sha256": sha256_file(source_file), "duration": 3.0}
    config = {"target": "en", "max": 10}
    store.initialize(source=source, config=config, chunks=plans)
    dubbed = store.chunk_dir(0) / "dubbed.mp4"
    dubbed.write_bytes(b"x" * 2048)
    store.update_chunk(0, status="completed", dubbed_sha256=sha256_file(dubbed))

    resumed = CheckpointStore(tmp_path / "checkpoint")
    resumed.initialize(source=source, config=config, chunks=plans)
    assert resumed.completed_file(0) == dubbed
    dubbed.write_bytes(b"corrupted")
    assert resumed.completed_file(0) is None


def test_checkpoint_never_authorizes_cleanup(tmp_path):
    plans = plan_smart_chunks([words_segment([("hello", 0.2, 1.0)])], 2.0)
    source = {"path": "source.mp4", "sha256": "abc", "duration": 2.0}
    store = CheckpointStore(tmp_path / "checkpoint")
    store.initialize(source=source, config={"target": "en"}, chunks=plans)
    store.mark_state("completed_waiting_for_cleanup_approval", cleanup_authorized=False)
    assert store.summary()["cleanup_authorized"] is False
    assert json.loads(store.manifest_path.read_text())["cleanup_authorized"] is False


def test_non_speech_labels_and_short_speech_do_not_create_orphan_chunks():
    raw = [
        {"start": 0.0, "end": 15.0, "text": "music", "confidence": 0.2},
        words_segment([("hello", 20.0, 20.5), ("again.", 31.0, 32.0)]),
    ]
    chunks = plan_smart_chunks(raw, 35.0, max_seconds=10.0, target_seconds=8.0, min_seconds=2.5)
    assert all(chunk["duration"] <= 10.0 for chunk in chunks)
    assert all(chunk["duration"] >= 0.5 for chunk in chunks)
    assert not any("music" in chunk["source_text"].lower() for chunk in chunks)
    observed_text = " ".join(chunk["source_text"] for chunk in chunks)
    assert "hello" in observed_text and "again." in observed_text


def test_speaker_change_forces_a_safe_chunk_boundary():
    raw = [{
        "start": 0.0, "end": 7.0, "text": "one two three four", "confidence": 0.9,
        "words": [
            {"word": "one", "start": 0.2, "end": 1.0, "speaker": "SPEAKER_00"},
            {"word": "two", "start": 1.1, "end": 2.0, "speaker": "SPEAKER_00"},
            {"word": "three", "start": 2.2, "end": 3.0, "speaker": "SPEAKER_01"},
            {"word": "four", "start": 3.1, "end": 4.0, "speaker": "SPEAKER_01"},
        ],
    }]
    chunks = plan_smart_chunks(raw, 7.0, max_seconds=10.0, target_seconds=8.0, min_seconds=2.5)
    assert chunks[0]["end"] == 2.0
    assert chunks[0]["cut_reason"] == "speaker_change"
    assert chunks[0]["speaker"] == "SPEAKER_00"
    assert chunks[1]["speaker"] == "SPEAKER_01"
    for chunk in chunks:
        assert len({word["speaker"] for word in chunk["source_words"] if word.get("speaker")}) <= 1


def test_every_tiny_chunk_has_an_independent_stage_checklist(tmp_path):
    from youtube_auto_dub.smart_chunks import CheckpointStore, STAGE_ORDER
    store = CheckpointStore(tmp_path / "project")
    chunk = {
        "index": 0, "start": 0.0, "end": 0.35, "speech_start": 0.02, "speech_end": 0.34,
        "source_text": "hello", "translated_text": "hello", "word_ids": [0], "word_count": 1, "status": "pending",
    }
    store.initialize(source={"sha256": "abc"}, config={"engine": "test"}, chunks=[chunk])
    assert tuple(store.chunk(0)["checklist"]) == STAGE_ORDER
    assert all(store.stage(0, name)["state"] in {"pending", "success", "skipped", "failed"} for name in STAGE_ORDER)


def test_stage_success_reuses_valid_output_and_rejects_corruption(tmp_path):
    from youtube_auto_dub.smart_chunks import CheckpointStore
    store = CheckpointStore(tmp_path / "project")
    chunk = {"index": 0, "start": 0.0, "end": 1.0, "source_text": "a", "word_ids": [0], "word_count": 1}
    store.initialize(source={"sha256": "abc"}, config={}, chunks=[chunk])
    output = store.chunk_dir(0) / "voice.wav"
    output.write_bytes(b"a" * 2048)
    store.mark_stage(0, "tts", "success", output=output, input_hash="input-a")
    assert store.stage_valid(0, "tts", output, input_hash="input-a") is True
    output.write_bytes(b"b" * 2048)
    assert store.stage_valid(0, "tts", output, input_hash="input-a") is False


def test_invalidation_starts_at_failed_stage_only(tmp_path):
    from youtube_auto_dub.smart_chunks import CheckpointStore, STAGE_ORDER
    store = CheckpointStore(tmp_path / "project")
    chunk = {"index": 0, "start": 0.0, "end": 1.0, "source_text": "a", "word_ids": [0], "word_count": 1}
    store.initialize(source={"sha256": "abc"}, config={}, chunks=[chunk])
    for name in STAGE_ORDER:
        store.mark_stage(0, name, "success")
    store.invalidate_from(0, "timing_fit", "timing policy changed")
    assert store.stage(0, "analysis")["state"] == "success"
    assert store.stage(0, "translation")["state"] == "success"
    assert store.stage(0, "tts")["state"] == "success"
    assert store.stage(0, "seed_vc")["state"] == "success"
    assert all(store.stage(0, name)["state"] == "pending" for name in STAGE_ORDER[4:])
