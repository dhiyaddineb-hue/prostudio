"""The dub must not fall silent where the speaker keeps talking.

Run #155 (the 33 s short) published a dub with a 4.0 s hole at 25.7-29.7 s: the
speech stem was as loud there as everywhere else, but Whisper's transcript had
no words between 25.74 s and 29.71 s, so nothing was translated or voiced. The
gap (3.97 s) also slipped under the 4.0 s whole-file retry threshold and the
quality gate, which judged coverage on the full mix, saw nothing wrong.

These tests pin the energy-based coverage check, the targeted re-transcription
of wordless spans, the report field, and the new quality-gate check.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/resumable_smart_dub.py"
SR = 24000


def _ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _extract(source: Path, names: set[str], namespace: dict) -> dict:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def _namespace(tmp_path: Path) -> dict:
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        pytest.skip("ffmpeg is not available")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    link = bin_dir / "ffmpeg"
    if not link.exists():
        try:
            link.symlink_to(ffmpeg)
        except OSError:
            shutil.copy2(ffmpeg, link)
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    import sys

    namespace = {"Path": Path, "np": np, "subprocess": subprocess, "sys": sys, "SR_TTS": SR, "transcribe": None}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    constants = [
        n for n in tree.body
        if isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets)
    ]
    exec(compile(ast.Module(body=constants, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    _extract(SCRIPT, {
        "run", "decode_mono", "word_spans", "untrusted_word_spans", "find_uncovered_speech",
        "recover_uncovered_speech", "summarize_speech_coverage", "slice_audio",
    }, namespace)
    return namespace


def _speech_like(seconds: float, level: float = 0.3) -> np.ndarray:
    """A buzzy tone burst with an envelope, loud like speech on the stem."""
    t = np.arange(int(seconds * SR)) / SR
    tone = np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 360 * t) + 0.25 * np.sin(2 * np.pi * 720 * t)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t) ** 2
    return (level * tone * envelope).astype(np.float32)


def _timeline(total: float, bursts: list[tuple[float, float]], noise: float = 0.002) -> np.ndarray:
    rng = np.random.default_rng(7)
    audio = (rng.standard_normal(int(total * SR)) * noise).astype(np.float32)
    for start, end in bursts:
        clip = _speech_like(end - start)
        audio[int(start * SR):int(start * SR) + len(clip)] = clip
    return audio


def _write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def _words(*spans: tuple[float, float]) -> list[dict]:
    words = [{"word": f" w{i}", "start": s, "end": e} for i, (s, e) in enumerate(spans)]
    return [{"start": spans[0][0], "end": spans[-1][1], "text": "x", "confidence": 0.9, "words": words}]


def test_loud_wordless_span_is_found_and_music_free_silence_is_not(tmp_path):
    ns = _namespace(tmp_path)
    # Speech at 1-3 s (transcribed), 5-8.5 s (NOT transcribed), 10-11 s (transcribed); quiet elsewhere.
    audio = _timeline(12.0, [(1.0, 3.0), (5.0, 8.5), (10.0, 11.0)])
    raw = _words((1.0, 1.5), (1.5, 2.2), (2.2, 3.0)) + _words((10.0, 10.5), (10.5, 11.0))
    spans = ns["find_uncovered_speech"](audio, raw, 12.0)
    assert len(spans) == 1
    assert abs(spans[0]["start"] - 5.0) < 0.1 and abs(spans[0]["end"] - 8.5) < 0.1
    assert spans[0]["seconds"] > 3.3 and spans[0]["median_db"] > spans[0]["threshold_db"]
    # Fully transcribed speech reports nothing.
    full = raw + _words((5.0, 5.6), (5.6, 6.3), (6.3, 7.0), (7.0, 7.8), (7.8, 8.5))
    assert ns["find_uncovered_speech"](audio, full, 12.0) == []
    # Short wordless blips (< 0.8 s) are ignored; they are breaths, laughs, or ASR word-edge slop.
    blip = _timeline(6.0, [(1.0, 2.0), (3.0, 3.5)])
    assert ns["find_uncovered_speech"](blip, _words((1.0, 2.0)), 6.0) == []


def test_threshold_adapts_to_the_speakers_level(tmp_path):
    ns = _namespace(tmp_path)
    quiet = _timeline(8.0, [(1.0, 3.0), (4.0, 6.0)]) * 0.05  # -26 dB below the loud fixture
    raw = _words((1.0, 2.0), (2.0, 3.0))
    spans = ns["find_uncovered_speech"](quiet, raw, 8.0)
    assert len(spans) == 1 and abs(spans[0]["start"] - 4.0) < 0.1 and abs(spans[0]["end"] - 6.0) < 0.1
    assert spans[0]["threshold_db"] <= -25.0


def test_recovery_transcribes_only_the_hole_and_offsets_the_words(tmp_path):
    ns = _namespace(tmp_path)
    audio = _timeline(12.0, [(1.0, 3.0), (5.0, 8.5), (10.0, 11.0)])
    source = tmp_path / "speech.wav"
    _write_wav(source, audio)
    raw = _words((1.0, 1.5), (1.5, 2.2), (2.2, 3.0)) + _words((10.0, 10.5), (10.5, 11.0))
    spans = ns["find_uncovered_speech"](source, raw, 12.0)
    calls = []

    def fake_transcribe(path, model_name, device, language, use_vad):
        calls.append({"path": Path(path), "model": model_name, "language": language, "use_vad": use_vad})
        with wave.open(str(path), "rb") as handle:
            seconds = handle.getnframes() / handle.getframerate()
        # Whisper reports clip-relative times: two words, plus one that leaks past the slice end.
        return ([{
            "start": 0.4, "end": seconds, "text": "found phrase", "confidence": 0.8, "no_speech_prob": 0.1,
            "words": [{"word": " found", "start": 0.4, "end": 1.6}, {"word": " phrase", "start": 1.7, "end": 3.2},
                      {"word": " leak", "start": seconds + 0.5, "end": seconds + 0.9}],
        }], "ar")

    merged, count = ns["recover_uncovered_speech"](
        source, raw, spans, duration=12.0, work_dir=tmp_path / "recovery",
        model_name="medium", device="cpu", language="ar", transcribe_fn=fake_transcribe,
    )
    assert count == 2 and len(calls) == 1
    assert calls[0]["use_vad"] is False and calls[0]["model"] == "medium" and calls[0]["language"] == "ar"
    with wave.open(str(calls[0]["path"]), "rb") as handle:
        clip_seconds = handle.getnframes() / handle.getframerate()
    # Padded by 0.35 s but stopped before the neighbouring transcribed words (3.0 s and 10.0 s).
    assert 3.4 <= clip_seconds <= 4.4
    recovered = [seg for seg in merged if seg.get("recovered")]
    assert len(recovered) == 1
    words = recovered[0]["words"]
    assert [w["word"].strip() for w in words] == ["found", "phrase"]
    assert 4.9 <= words[0]["start"] <= 5.2 and words[-1]["end"] <= 8.9
    assert recovered[0]["text"] == "found phrase"
    # Merged timeline stays ordered and the original segments are untouched.
    starts = [float(seg["start"]) for seg in merged]
    assert starts == sorted(starts) and len(merged) == 3
    assert ns["find_uncovered_speech"](source, merged, 12.0) == []


def test_recovery_survives_a_failing_transcriber(tmp_path):
    ns = _namespace(tmp_path)
    audio = _timeline(8.0, [(1.0, 2.0), (4.0, 6.0)])
    source = tmp_path / "speech.wav"
    _write_wav(source, audio)
    raw = _words((1.0, 2.0))
    spans = ns["find_uncovered_speech"](source, raw, 8.0)

    def broken(*args, **kwargs):
        raise RuntimeError("model unavailable")

    merged, count = ns["recover_uncovered_speech"](
        source, raw, spans, duration=8.0, work_dir=tmp_path / "recovery",
        model_name="medium", device="cpu", language="ar", transcribe_fn=broken,
    )
    assert merged == raw and count == 0


def test_coverage_summary_for_reports(tmp_path):
    ns = _namespace(tmp_path)
    assert ns["summarize_speech_coverage"](None) == {"measured": False}
    summary = ns["summarize_speech_coverage"]({
        "spans_before": [{"start": 25.74, "end": 29.71, "seconds": 3.97}],
        "recovered_words": 0,
        "spans_after": [{"start": 25.74, "end": 29.71, "seconds": 3.97}],
    })
    assert summary["measured"] and summary["before_seconds"] == 3.97 and summary["longest_after_seconds"] == 3.97


def test_analysis_wires_recovery_before_caching_and_reports_it():
    text = SCRIPT.read_text(encoding="utf-8")
    fresh = text.split("forced = None if args.source_lang in", 1)[1].split("if not reference.exists():", 1)[0]
    assert "find_uncovered_speech(working_speech, raw, source_duration)" in fresh
    assert "recover_uncovered_speech(" in fresh
    assert '"speech_coverage": speech_coverage' in fresh
    assert fresh.index("recover_uncovered_speech(") < fresh.index("atomic_write_json(asr_path, raw)")
    # Cached analyses (existing projects) are never re-planned: they only read the stored measurement.
    cached = text.split("if asr_path.exists() and meta_path.exists():", 1)[1].split("else:", 1)[0]
    assert 'speech_coverage = analysis_meta.get("speech_coverage")' in cached
    assert "recover_uncovered_speech(" not in cached
    assert '"uncovered_speech": summarize_speech_coverage(speech_coverage)' in text


def test_quality_gate_checks_uncovered_speech_only_when_measured():
    spec = importlib.util.spec_from_file_location("quality_under_test", ROOT / "scripts/validate_dub_quality.py")
    quality = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quality)
    ok, info = quality.uncovered_speech_check({"asr_timeline": {"coverage": 0.9}}, 2.0)
    assert ok and info == {"measured": False}
    measured = {"asr_timeline": {"uncovered_speech": {
        "measured": True, "before_seconds": 3.97, "after_seconds": 3.97, "longest_after_seconds": 3.97,
        "recovered_words": 0, "spans_after": [{"start": 25.74, "end": 29.71, "seconds": 3.97}],
    }}}
    ok, info = quality.uncovered_speech_check(measured, 2.0)
    assert not ok and info["longest_after_seconds"] == 3.97 and info["limit_seconds"] == 2.0
    ok, _info = quality.uncovered_speech_check(measured, 3.5)
    assert not ok
    recovered = json.loads(json.dumps(measured))
    recovered["asr_timeline"]["uncovered_speech"].update({"after_seconds": 0.0, "longest_after_seconds": 0.0, "spans_after": [], "recovered_words": 9})
    assert quality.uncovered_speech_check(recovered, 1.0)[0]
    text = (ROOT / "scripts/validate_dub_quality.py").read_text(encoding="utf-8")
    assert '"speech_covered_by_transcript": speech_covered' in text
    for policy, limit in (("safe", 3.5), ("balanced", 2.0), ("strict", 1.0)):
        assert f'"{policy}":{{' in text and f'"uncovered_speech":{limit}' in text.split(f'"{policy}":{{', 1)[1].split("}", 1)[0]


def test_a_token_stretched_over_seconds_is_not_trusted_as_coverage(tmp_path):
    ns = _namespace(tmp_path)
    audio = _timeline(12.0, [(1.0, 3.0), (5.0, 9.0), (10.0, 11.0)])
    # Run #156: Whisper "heard" one token spanning the whole 4 s passage.
    raw = _words((1.0, 1.5), (1.5, 2.2), (2.2, 3.0)) + _words((5.0, 9.0)) + _words((10.0, 10.5), (10.5, 11.0))
    assert ns["untrusted_word_spans"](raw) == [(5.0, 9.0)]
    assert (5.0, 9.0) not in ns["word_spans"](raw, trusted_only=True)
    assert (5.0, 9.0) in ns["word_spans"](raw)
    spans = ns["find_uncovered_speech"](audio, raw, 12.0)
    assert len(spans) == 1 and abs(spans[0]["start"] - 5.0) < 0.1 and abs(spans[0]["end"] - 9.0) < 0.1
    # A sparse segment (few words over a long stretch) is treated the same way.
    sparse = _words((5.0, 5.4), (8.4, 9.0))
    assert ns["untrusted_word_spans"](sparse) == [(5.0, 5.4), (8.4, 9.0)]
    # Normal speech density and word lengths are trusted.
    assert ns["untrusted_word_spans"](_words((1.0, 1.5), (1.5, 2.2), (2.2, 3.0))) == []


def test_recovery_replaces_the_stretched_token_with_real_words(tmp_path):
    ns = _namespace(tmp_path)
    audio = _timeline(12.0, [(1.0, 3.0), (5.0, 9.0), (10.0, 11.0)])
    source = tmp_path / "speech.wav"
    _write_wav(source, audio)
    raw = _words((1.0, 1.5), (1.5, 2.2), (2.2, 3.0)) + _words((5.0, 9.0)) + _words((10.0, 10.5), (10.5, 11.0))
    raw[1]["text"] = "بي"
    spans = ns["find_uncovered_speech"](source, raw, 12.0)

    def fake_transcribe(path, model_name, device, language, use_vad):
        return ([{
            "start": 0.3, "end": 3.9, "text": "real words here", "confidence": 0.8, "no_speech_prob": 0.1,
            "words": [{"word": " real", "start": 0.3, "end": 1.2}, {"word": " words", "start": 1.3, "end": 2.4},
                      {"word": " here", "start": 2.5, "end": 3.9}],
        }], "ar")

    merged, count = ns["recover_uncovered_speech"](
        source, raw, spans, duration=12.0, work_dir=tmp_path / "recovery",
        model_name="medium", device="cpu", language="ar", transcribe_fn=fake_transcribe,
    )
    assert count == 3
    texts = [seg["text"] for seg in merged]
    assert "بي" not in texts and "real words here" in texts
    assert len(merged) == 3 and [float(s["start"]) for s in merged] == sorted(float(s["start"]) for s in merged)
    assert ns["untrusted_word_spans"](merged) == []
    assert ns["find_uncovered_speech"](source, merged, 12.0) == []

    # If the re-transcription yields nothing, the original token is kept (never make it worse).
    merged2, count2 = ns["recover_uncovered_speech"](
        source, raw, spans, duration=12.0, work_dir=tmp_path / "recovery2",
        model_name="medium", device="cpu", language="ar", transcribe_fn=lambda *a, **k: ([], "ar"),
    )
    assert count2 == 0 and merged2 == raw
