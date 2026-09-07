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
    for kept in ("delivery.fitted.wav", "dubbed.mp4", "status.json", "content-retry-2.part1.wav", "generated.wav"):
        assert is_cache(Path("chunks/0003") / kept) is False, kept


def test_completed_silence_chunks_are_not_re_rendered_on_resume():
    text = SCRIPT.read_text(encoding="utf-8")
    render = text.split("failures: list[int] = []", 1)[1]
    block = render.split("content_current = (", 1)[1].split("complete = store.completed_file(index)", 1)[0]
    assert "or non_speech" in block
    assert 'or not chunk.get("source_text")' in block
    assert 'bool((chunk.get("content_validation") or {}).get("ok"))' in block



# ── Final assembly: frame-accurate concatenation ─────────────────────────────

def _decode_frame_count(ffmpeg: str, path: Path) -> int:
    import re

    result = subprocess.run(
        [ffmpeg, "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    counts = re.findall(r"frame=\s*(\d+)", result.stderr)
    assert counts, result.stderr[-800:]
    return int(counts[-1])


def _napoleon_like_plan(fps: float, n: int = 40) -> list[dict]:
    """Chunk boundaries that fall between frames, like word-timed cuts do."""
    import random

    rng = random.Random(7)
    plan, cursor = [], 0.0
    for index in range(n):
        end = round(cursor + rng.choice([7.98, 7.96, 9.94, 2.92, 8.0, 5.67, 3.5]), 2)
        plan.append({"index": index, "start": cursor, "end": end})
        cursor = end
    return plan


def test_frame_counts_drop_exactly_the_straddling_duplicates():
    namespace = _extract(SCRIPT, {"frame_aligned_frame_counts"}, {"math": __import__("math"), "Path": Path})
    counts_for = namespace["frame_aligned_frame_counts"]
    import math

    fps = 24.0
    plan = _napoleon_like_plan(fps)
    # render_chunk cuts with -ss/-t: a chunk holds ceil(duration*fps) frames,
    # so it can carry the frame that also opens the next chunk.
    held = [math.ceil((c["end"] - c["start"]) * fps - 1e-6) for c in plan]
    counts = counts_for(plan, fps, held)

    cursor = 0
    for chunk, kept, available in zip(plan, counts, held):
        assert 1 <= kept <= available
        cursor += kept
        # every boundary lands on the first source frame of the next chunk
        assert cursor == math.ceil(chunk["end"] * fps - 1e-6)
    assert sum(counts) == math.ceil(plan[-1]["end"] * fps - 1e-6)
    assert sum(held) - sum(counts) > 0, "the synthetic plan must contain straddling frames"

    # Chunks that hold exactly their own frames are never trimmed.
    exact = [math.ceil(c["end"] * fps - 1e-6) - math.ceil(c["start"] * fps - 1e-6) for c in plan]
    assert counts_for(plan, fps, exact) == exact


def test_final_assembly_is_frame_accurate_at_the_source_frame_rate():
    text = SCRIPT.read_text(encoding="utf-8")
    body = text.split("def concatenate_chunks_frame_accurate(", 1)[1].split("def concatenate_chunks(", 1)[0]
    assert "trim=end_frame={frames}" in body
    assert "atrim=end={seconds:.6f}" in body
    assert '"-r", fps_text, "-fps_mode", "cfr"' in body
    assert "concat=n={len(frame_counts)}:v=1:a=1" in body
    call_site = text.split('final = concatenate_chunks(', 1)[1].split("final_duration = ffprobe_duration(final)", 1)[0]
    assert "frame_rate=frame_rate" in call_site and "held_frames=held_frames" in call_site
    assert "probe_video_frames(path) for path in chunk_files" in text
    # the stream-copy path is kept only as a fallback
    wrapper = text.split("def concatenate_chunks(", 1)[1].split("def is_derived_asr_cache(", 1)[0]
    assert wrapper.index("concatenate_chunks_frame_accurate(") < wrapper.index('"-c", "copy"')


def test_frame_accurate_assembly_produces_exact_frame_total(tmp_path):
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        pytest.skip("ffmpeg is not available")

    def run(cmd, *, check=True, capture=True):
        cmd = [ffmpeg if cmd[0] == "ffmpeg" else cmd[0], *cmd[1:]]
        return subprocess.run(cmd, check=check, capture_output=capture, text=True)

    import math
    import sys

    namespace = _extract(
        SCRIPT, {"frame_aligned_frame_counts", "concatenate_chunks_frame_accurate", "concatenate_chunks"},
        {
            "Path": Path, "math": math, "subprocess": subprocess, "sys": sys, "run": run,
            "PUBLISH_SIZE_BUDGET_BYTES": 85 * 1000 * 1000, "PUBLISH_SIZE_HARD_LIMIT_BYTES": 95 * 1000 * 1000,
        },
    )
    fps = 24.0
    clips = []
    for index, frames in enumerate((48, 47)):  # 2.000 s and 1.958 s of 24 fps video
        clip = tmp_path / f"chunk-{index}.mp4"
        probe = subprocess.run([
            ffmpeg, "-y", "-f", "lavfi", "-i", f"testsrc=size=64x48:rate=24", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-frames:v", str(frames), "-t", f"{frames / fps:.6f}", "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-ac", "2", str(clip),
        ], capture_output=True, text=True)
        if probe.returncode != 0:
            pytest.skip("ffmpeg build lacks lavfi/libx264/aac")
        assert _decode_frame_count(ffmpeg, clip) == frames
        clips.append(clip)

    # Plan: 1.98 s = 47.52 frames, so the first chunk's 48th frame (pts 1.958 s) is
    # still inside its window and is kept; the second chunk (3.9 s = 93.6 frames)
    # would end 1.4 frames past its boundary, so its straddling frame is left out.
    plan = [{"start": 0.0, "end": 1.98}, {"start": 1.98, "end": 3.9}]
    destination = tmp_path / "final-dub.mp4"
    result = namespace["concatenate_chunks"](
        clips, destination, chunks=plan, frame_rate=(fps, "24"), held_frames=[48, 47],
    )
    assert result == destination and destination.exists()
    assert namespace["frame_aligned_frame_counts"](plan, fps, [48, 47]) == [48, 46]
    assert _decode_frame_count(ffmpeg, destination) == 94  # == ceil(3.9 * 24)
    assert (tmp_path / "final-dub.concat.filter").exists()


def test_skipped_complete_chunks_are_counted_as_completed():
    """Run #152 assembled all 158 chunks but the workflow's completion check
    failed with by_status={'completed': 157, 'non_speech': 1}: pass 1 relabels a
    finished music chunk every run, and skipping it left that label in place."""
    text = SCRIPT.read_text(encoding="utf-8")
    render = text.split("failures: list[int] = []", 1)[1]
    skip = render.split("if complete and profile_current and content_current", 1)[1].split("continue", 1)[0]
    assert 'if chunk.get("status") != "completed":' in skip
    assert 'store.update_chunk(index, status="completed", error=None)' in skip



# ── Quality gate: speech-aware silence checks, publish size budget ────────────

QUALITY = ROOT / "scripts/validate_dub_quality.py"


def _quality_namespace() -> dict:
    import re

    from youtube_auto_dub.content_validation import is_non_speech_text

    namespace = {"re": re, "is_non_speech_text": is_non_speech_text}
    return _extract(QUALITY, {"segment_text", "speech_segments", "merged_spans", "silence_over_speech", "timeline_metrics"}, namespace)


def test_quality_gate_ignores_music_cues_and_judges_silence_against_speech():
    """Run #153: the film opens with 38.9 s of music. The speech-only dub is
    silent there by design, yet the gate failed silence_not_added (38.9 s vs a
    1.1 s longest source silence) and segment_gaps (the music cue counted as a
    speech span, leaving a 26.5 s 'internal' gap)."""
    ns = _quality_namespace()
    segments = [
        {"start": 11.02, "end": 12.4, "source_text": "موسيقى"},
        {"start": 38.92, "end": 47.9, "source_text": "ونبدأ"},
        {"start": 48.3, "end": 55.9, "source_text": "كلام"},
        {"start": 60.0, "end": 66.0, "source_text": "كلام آخر"},
    ]
    spoken = ns["speech_segments"](segments)
    assert [s["start"] for s in spoken] == [38.92, 48.3, 60.0]
    metrics = ns["timeline_metrics"](spoken, 70.0)
    assert metrics["leading_gap"] == 38.92
    assert round(metrics["internal_max_gap"], 2) == 4.1

    spans = ns["merged_spans"](spoken, 70.0)
    intro_only = [{"start": 0.0, "end": 38.92, "duration": 38.92}]
    assert ns["silence_over_speech"](intro_only, spans) < 1e-6
    missing_line = [{"start": 47.0, "end": 56.5, "duration": 9.5}]
    assert round(ns["silence_over_speech"](missing_line, spans), 2) == 8.5  # 0.9 s + 7.6 s of planned speech


def test_quality_gate_uses_speech_aware_silence_only_for_speech_only_dubs():
    text = QUALITY.read_text(encoding="utf-8")
    assert 'speech_only_dub=background_preserved is False' in text
    assert '"silence_not_added": (final_over_speech if speech_only_dub else final_long) <= max(3.0,source_long+limits["extra_silence"])' in text
    assert 'tm=timeline_metrics(spoken,source_dur)' in text
    assert '"segments_present": len(spoken) > 0' in text
    # the smart pipeline records how the dub was mixed so the gate can decide
    script = SCRIPT.read_text(encoding="utf-8")
    assert '"background_preserved": bool(args.preserve_background)' in script


def test_final_encode_stays_below_github_file_limit():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "PUBLISH_SIZE_BUDGET_BYTES = 85 * 1000 * 1000" in text
    assert "PUBLISH_SIZE_HARD_LIMIT_BYTES = 95 * 1000 * 1000" in text
    body = text.split("def concatenate_chunks_frame_accurate(", 1)[1].split("def concatenate_chunks(", 1)[0]
    assert '"-maxrate", f"{maxrate_kbps}k", "-bufsize", f"{2 * maxrate_kbps}k"' in body
    assert "audio_kbps = 192 if duration <= 600 else 128" in body
    # 20-minute film: the cap must leave the total under the hard limit
    duration = 1208.8
    budget_kbps = 85 * 1000 * 1000 * 8 / duration / 1000
    maxrate_kbps = int(max(300, min(6000, budget_kbps - 128)))
    worst_case_bytes = (maxrate_kbps + 128) * 1000 * duration / 8
    assert worst_case_bytes < 95 * 1000 * 1000
    # 25-second clip: the cap is far above what crf 20 produces, so nothing changes
    assert int(max(300, min(6000, 85 * 1000 * 1000 * 8 / 25.5 / 1000 - 192))) == 6000



# ── Delivery stage is a checkpoint too: assembled film and language verdict ──

LANGUAGE = ROOT / "scripts/validate_dub_language.py"


def test_assembled_final_is_reused_when_delivery_is_unchanged(tmp_path):
    import hashlib

    namespace = _extract(SCRIPT, {"reusable_final"}, {"Path": Path, "sha256_file": lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()})
    reusable = namespace["reusable_final"]
    final = tmp_path / "final-dub.mp4"
    final.write_bytes(b"film" * 1000)
    digest = hashlib.sha256(final.read_bytes()).hexdigest()
    chunks = ["a" * 64, "b" * 64]

    keyed = {"final_sha256": digest, "delivery_key": "k1", "delivery_chunks": chunks, "state": "completed_waiting_for_cleanup_approval"}
    assert reusable(keyed, "k1", chunks, final) is True
    assert reusable(keyed, "k2", chunks, final) is False, "different assembly parameters must re-encode"
    assert reusable(keyed, "k1", ["a" * 64, "c" * 64], final) is False, "a rebuilt chunk must re-encode"
    assert reusable({**keyed, "final_sha256": "0" * 64}, "k1", chunks, final) is False, "sha mismatch"

    legacy = {"final_sha256": digest, "state": "completed_waiting_for_cleanup_approval"}
    assert reusable(legacy, "k1", chunks, final) is True
    assert reusable(legacy, "k1", chunks, final, chunks_changed=True) is False
    assert reusable({**legacy, "state": "failed_resumable"}, "k1", chunks, final) is False
    assert reusable(legacy, "k1", ["a" * 64, ""], final) is False, "a chunk without a recorded render hash"

    text = SCRIPT.read_text(encoding="utf-8")
    assert 'cached_final = mirror.cached_asset("final-dub.mp4")' in text
    assert "if not reused_final:\n        mirror.upload_final(final)" in text
    assert "checkpoint_release_tag=mirror.tag" in text
    assert "delivery_chunks=delivery_chunks" in text


def _fake_gh(tmp_path: Path, stored: dict[str, str]) -> Path:
    """A stand-in for the gh CLI that serves/records release assets from a folder."""
    store = tmp_path / "release-assets"
    store.mkdir(exist_ok=True)
    for name, body in stored.items():
        (store / name).write_text(body, encoding="utf-8")
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    script = bindir / "gh"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import shutil, sys\n"
        "from pathlib import Path\n"
        f"store = Path({str(store)!r})\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['release', 'download']:\n"
        "    pattern = args[args.index('--pattern') + 1]; target = Path(args[args.index('--dir') + 1])\n"
        "    src = store / pattern\n"
        "    if not src.exists(): sys.exit(1)\n"
        "    target.mkdir(parents=True, exist_ok=True); shutil.copy2(src, target / pattern); sys.exit(0)\n"
        "if args[:2] == ['release', 'upload']:\n"
        "    shutil.copy2(args[3], store / Path(args[3]).name); sys.exit(0)\n"
        "sys.exit(2)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return bindir


def test_language_verdict_is_reused_only_for_the_identical_file(tmp_path, monkeypatch):
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import tempfile

    namespace = _extract(
        LANGUAGE, {"sha256_file", "checkpoint_cache", "_gh_available", "fetch_cached_report", "store_cached_report"},
        {"Path": Path, "hashlib": hashlib, "json": json, "os": os, "shutil": shutil, "subprocess": subprocess, "tempfile": tempfile},
    )
    video = tmp_path / "final-dub.mp4"
    video.write_bytes(b"dub" * 5000)
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    manifest = tmp_path / "checkpoint-manifest.json"
    manifest.write_text(json.dumps({"checkpoint_release_tag": "checkpoint-demo", "final_sha256": digest}), encoding="utf-8")

    tag, asset, seen = namespace["checkpoint_cache"](manifest, video)
    assert (tag, asset, seen) == ("checkpoint-demo", f"language-{digest[:16]}.json", digest)
    assert namespace["checkpoint_cache"](None, video)[0] is None

    good = {"expected": "en", "detected": "en", "valid": True, "video_sha256": digest}
    stale = {**good, "video_sha256": "f" * 64}
    bindir = _fake_gh(tmp_path, {asset: json.dumps(good), "language-stale.json": json.dumps(stale)})
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("GH_TOKEN", "test-token")
    assert namespace["_gh_available"]() is True

    assert namespace["fetch_cached_report"]("checkpoint-demo", asset, digest, "en") == good
    assert namespace["fetch_cached_report"]("checkpoint-demo", asset, digest, "fr") is None, "different target language"
    assert namespace["fetch_cached_report"]("checkpoint-demo", "language-stale.json", digest, "en") is None, "verdict for another file"
    assert namespace["fetch_cached_report"]("checkpoint-demo", "language-missing.json", digest, "en") is None

    fresh = {**good, "probability": 0.99}
    namespace["store_cached_report"]("checkpoint-demo", "language-new.json", fresh)
    assert json.loads((tmp_path / "release-assets" / "language-new.json").read_text(encoding="utf-8")) == fresh

    monkeypatch.delenv("GH_TOKEN")
    assert namespace["fetch_cached_report"]("checkpoint-demo", asset, digest, "en") is None, "no token, no cache"


def test_language_step_is_wired_to_the_checkpoint_cache():
    text = LANGUAGE.read_text(encoding="utf-8")
    assert 'p.add_argument("--manifest"' in text
    assert 'cached["reused_from_checkpoint"] = True' in text
    assert '"video_sha256": video_digest' in text
    assert "if valid and cache_tag:\n        store_cached_report(cache_tag, cache_asset, report)" in text
    workflow = (ROOT / ".github/workflows/dub.yml").read_text(encoding="utf-8")
    step = workflow.split("- name: Validate final target language", 1)[1].split("- name: Enforce professional quality gate", 1)[0]
    assert "GH_TOKEN: ${{ github.token }}" in step
    assert "--manifest output/checkpoint-manifest.json" in step
