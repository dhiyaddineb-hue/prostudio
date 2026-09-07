#!/usr/bin/env python3
"""Resume-safe, speech-aware <=10s dubbing pipeline.

The full source is analysed once. Each smart chunk is translated, synthesized,
rendered, validated, and mirrored to a draft GitHub Release immediately. The
final video is concatenated only when every chunk is complete. Nothing is ever
deleted here; cleanup requires the separate approval workflow.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Running ``python scripts/resumable_smart_dub.py`` makes scripts/ sys.path[0].
# Add the repository root before importing the local youtube_auto_dub package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from youtube_auto_dub.emotion import infer_emotion
from youtube_auto_dub.content_validation import is_non_speech_text, normalize_tokens, validate_spoken_content, word_timing_report
from youtube_auto_dub.voice_profiles import load_voice_profiles, template_for_speakers
from youtube_auto_dub.googlev4 import GoogleTranslator
from youtube_auto_dub.llm_translate import (
    CONTINUATION_CUTS,
    LLMTranslateConfig,
    LLMTranslator,
    TranslationAPIError,
    TranslationConfigError,
)
from youtube_auto_dub.models import SR_TTS
from youtube_auto_dub.runtime import pick_device
from youtube_auto_dub.smart_chunks import (
    CheckpointStore,
    atomic_write_json,
    plan_smart_chunks,
    safe_project_id,
    sha256_file,
    stable_hash,
)
from youtube_auto_dub.source_separation import separate_dialogue_background, validate_stems
from youtube_auto_dub.speaker_diarization import annotate_segments
from youtube_auto_dub.speech import transcribe
from youtube_auto_dub import xtts_clone
from youtube_auto_dub.voxcpm_tts import speak_voxcpm
from youtube_auto_dub.youtube import load_source


# GitHub refuses files above 100 MB; the published dub must stay well below it.
PUBLISH_SIZE_BUDGET_BYTES = 85 * 1000 * 1000
PUBLISH_SIZE_HARD_LIMIT_BYTES = 95 * 1000 * 1000

# Speech that is shorter than its window is slowed down to fill it, but never
# below this tempo: slower than ~0.85x sounds drugged, so the remainder of the
# window stays silent instead.
SLOW_TEMPO_FLOOR = 0.85
# Boundary fades on the voice track. Where the planner had to cut inside a
# sentence the speech continues in the next chunk, so only a click guard is
# applied there; a full fade would dent every mid-sentence seam.
EDGE_FADE_SECONDS = 0.015
DECLICK_FADE_SECONDS = 0.003
# A TTS take far longer than its text warrants (repeats, runaway pauses, a
# crawling read) is regenerated instead of being crushed into the window.
TTS_WORDS_PER_SECOND = 2.7
TTS_LONG_TAKE_RATIO = 1.7
TTS_LONG_TAKE_SLACK_SECONDS = 0.8


def run(cmd: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def ffprobe_duration(path: Path) -> float:
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return float(result.stdout.strip())


def timeline_metrics(raw: list[dict], duration: float) -> tuple[float, float]:
    spans = sorted(
        (max(0.0, float(item.get("start", 0.0))), min(duration, float(item.get("end", 0.0))))
        for item in raw if float(item.get("end", 0.0)) > float(item.get("start", 0.0))
    )
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    if not merged or duration <= 0:
        return 0.0, duration
    coverage = sum(end - start for start, end in merged) / duration
    gaps = [merged[0][0], max(0.0, duration - merged[-1][1])]
    gaps.extend(merged[index + 1][0] - merged[index][1] for index in range(len(merged) - 1))
    return coverage, max(gaps + [0.0])


def decode_mono(path: Path, rate: int = SR_TTS) -> np.ndarray:
    """Decode any media file to mono float32 samples at ``rate`` via ffmpeg."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32)


# A real word does not last 1.5 s, and real speech does not run at fewer than
# 0.7 words per second for two seconds. Whisper produces both when it gives up
# on a passage: one token stretched over the whole span (run #156 heard "بي"
# for 3.98 s) or a near-empty segment. Such words are not trusted as coverage.
MAX_TRUSTED_WORD_SECONDS = 1.5
MIN_TRUSTED_WORDS_PER_SECOND = 0.7
MIN_DENSITY_SEGMENT_SECONDS = 2.0


def untrusted_word_spans(raw: list[dict]) -> list[tuple[float, float]]:
    """Word intervals that look like ASR give-ups rather than speech."""
    spans: list[tuple[float, float]] = []
    for segment in raw:
        words = [w for w in (segment.get("words") or []) if float(w["end"]) > float(w["start"])]
        if not words:
            continue
        seg_start, seg_end = float(words[0]["start"]), float(words[-1]["end"])
        seg_seconds = seg_end - seg_start
        sparse = seg_seconds >= MIN_DENSITY_SEGMENT_SECONDS and len(words) / seg_seconds < MIN_TRUSTED_WORDS_PER_SECOND
        for w in words:
            if sparse or float(w["end"]) - float(w["start"]) > MAX_TRUSTED_WORD_SECONDS:
                spans.append((float(w["start"]), float(w["end"])))
    return sorted(spans)


def word_spans(raw: list[dict], *, trusted_only: bool = False) -> list[tuple[float, float]]:
    untrusted = set(untrusted_word_spans(raw)) if trusted_only else set()
    spans: list[tuple[float, float]] = []
    for segment in raw:
        words = segment.get("words") or []
        if words:
            spans.extend(
                (float(w["start"]), float(w["end"])) for w in words
                if float(w["end"]) > float(w["start"]) and (float(w["start"]), float(w["end"])) not in untrusted
            )
        elif float(segment.get("end", 0.0)) > float(segment.get("start", 0.0)) and str(segment.get("text", "")).strip():
            spans.append((float(segment["start"]), float(segment["end"])))
    return sorted(spans)


def find_uncovered_speech(
    audio: Path | np.ndarray, raw: list[dict], duration: float, *,
    rate: int = SR_TTS, frame_ms: int = 20, min_seconds: float = 0.8, pad_seconds: float = 0.15,
    bridge_seconds: float = 0.25, level_margin_db: float = 12.0,
) -> list[dict]:
    """Loud stretches of the speech track that the transcript left without words.

    Whisper occasionally drops a whole phrase (its VAD or decoder gives up), or
    stretches a single token over it, and the dub then says nothing — or
    something else — exactly there, because nothing real was translated. This
    compares the speech-stem energy with the timeline of *trusted* words (see
    ``untrusted_word_spans``): frames louder than the transcribed speech minus
    ``level_margin_db`` that fall outside every trusted word (padded by
    ``pad_seconds``) are collected into spans of at least ``min_seconds``.
    Music-only passages are quiet on the speech stem, so they do not register.
    """
    samples = decode_mono(audio, rate) if not isinstance(audio, np.ndarray) else audio
    frame = max(1, int(rate * frame_ms / 1000))
    count = len(samples) // frame
    if count == 0:
        return []
    rms = np.sqrt(np.mean(samples[:count * frame].reshape(count, frame).astype(np.float64) ** 2, axis=1) + 1e-12)
    level_db = 20.0 * np.log10(rms + 1e-9)
    times = (np.arange(count) + 0.5) * frame / rate
    covered = np.zeros(count, dtype=bool)
    for start, end in word_spans(raw, trusted_only=True):
        lo = int(max(0.0, start - pad_seconds) * rate // frame)
        hi = int(min(duration, end + pad_seconds) * rate // frame) + 1
        covered[max(0, lo):min(count, hi)] = True
    if covered.any():
        speech_level = float(np.median(level_db[covered]))
        threshold = float(min(-25.0, max(-45.0, speech_level - level_margin_db)))
    else:
        threshold = -35.0
    loud_uncovered = (level_db > threshold) & ~covered
    spans: list[list[float]] = []
    for index in np.flatnonzero(loud_uncovered):
        t0 = float(index * frame / rate)
        t1 = float((index + 1) * frame / rate)
        if spans and t0 - spans[-1][1] <= bridge_seconds:
            spans[-1][1] = t1
        else:
            spans.append([t0, t1])
    result = []
    for t0, t1 in spans:
        if t1 - t0 + 1e-9 < min_seconds:
            continue
        mask = (times >= t0) & (times < t1)
        result.append({
            "start": round(t0, 3), "end": round(min(t1, duration), 3), "seconds": round(min(t1, duration) - t0, 3),
            "median_db": round(float(np.median(level_db[mask])), 1) if mask.any() else None,
            "threshold_db": round(threshold, 1),
        })
    return result


def recover_uncovered_speech(
    audio: Path, raw: list[dict], spans: list[dict], *, duration: float, work_dir: Path,
    model_name: str, device: str, language: str | None, transcribe_fn=None,
) -> tuple[list[dict], int]:
    """Re-transcribe each uncovered span on its own (no VAD) and merge the words.

    The slice is padded by 0.35 s but never reaches into neighbouring
    transcribed words, so nothing already recognised is heard twice. Recovered
    segments are tagged ``recovered: true``; a span that still yields nothing
    stays in the report for the quality gate.
    """
    transcribe_fn = transcribe_fn or transcribe
    existing = word_spans(raw, trusted_only=True)
    untrusted = set(untrusted_word_spans(raw))
    work_dir.mkdir(parents=True, exist_ok=True)
    recovered: list[dict] = []
    replaced_ranges: list[tuple[float, float]] = []
    for number, span in enumerate(spans):
        start, end = float(span["start"]), float(span["end"])
        previous_end = max([e for _s, e in existing if e <= start + 1e-6] + [0.0])
        next_start = min([s for s, _e in existing if s >= end - 1e-6] + [duration])
        lo = max(0.0, start - 0.35, previous_end)
        hi = min(duration, end + 0.35, next_start)
        if hi - lo < 0.4:
            continue
        clip = work_dir / f"span-{number:02d}.wav"
        slice_audio(audio, lo, hi - lo, clip)
        try:
            segments, _language = transcribe_fn(clip, model_name=model_name, device=device, language=language, use_vad=False)
        except Exception as exc:  # noqa: BLE001 - recovery must never break the analysis
            print(f"ASR recovery for {lo:.2f}-{hi:.2f}s failed: {exc}", file=sys.stderr)
            continue
        for segment in segments:
            words = []
            for word in segment.get("words") or []:
                w_start, w_end = float(word["start"]) + lo, float(word["end"]) + lo
                middle = (w_start + w_end) / 2
                if middle < lo or middle > hi:
                    continue
                if any(s - 0.05 <= middle <= e + 0.05 for s, e in existing):
                    continue
                words.append({**word, "start": round(w_start, 3), "end": round(min(w_end, hi), 3)})
            text = " ".join(str(w["word"]).strip() for w in words).strip()
            if not words or not text:
                continue
            recovered.append({
                "start": words[0]["start"], "end": words[-1]["end"], "text": text,
                "confidence": float(segment.get("confidence", 0.0)),
                "no_speech_prob": float(segment.get("no_speech_prob", 0.0)),
                "words": words, "recovered": True,
            })
            replaced_ranges.append((lo, hi))
    if not recovered:
        return raw, 0
    # A give-up token that the recovery just re-heard properly must not stay in
    # the transcript beside the real words, or it would be translated too.
    cleaned: list[dict] = []
    for segment in raw:
        words = segment.get("words") or []
        if not words or not untrusted:
            cleaned.append(segment)
            continue
        kept = [
            w for w in words
            if (float(w["start"]), float(w["end"])) not in untrusted
            or not any(lo - 1e-6 <= (float(w["start"]) + float(w["end"])) / 2 <= hi + 1e-6 for lo, hi in replaced_ranges)
        ]
        if len(kept) == len(words):
            cleaned.append(segment)
        elif kept:
            cleaned.append({
                **segment, "words": kept, "start": kept[0]["start"], "end": kept[-1]["end"],
                "text": " ".join(str(w["word"]).strip() for w in kept).strip(),
                "recovery_replaced_words": len(words) - len(kept),
            })
        # a segment whose only words were give-up tokens disappears entirely
    merged = sorted(cleaned + recovered, key=lambda item: float(item["start"]))
    return merged, sum(len(item["words"]) for item in recovered)


def summarize_speech_coverage(report: dict | None) -> dict:
    if not report:
        return {"measured": False}
    before = report.get("spans_before") or []
    after = report.get("spans_after") or []
    return {
        "measured": True,
        "before_seconds": round(sum(float(s["seconds"]) for s in before), 3),
        "after_seconds": round(sum(float(s["seconds"]) for s in after), 3),
        "longest_after_seconds": round(max([float(s["seconds"]) for s in after] + [0.0]), 3),
        "recovered_words": int(report.get("recovered_words") or 0),
        "spans_after": after,
    }


def atempo_filter(speed: float) -> str:
    values: list[float] = []
    remaining = max(float(speed), 0.01)
    while remaining > 2.0:
        values.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        values.append(0.5)
        remaining /= 0.5
    values.append(remaining)
    return ",".join(f"atempo={value:.8f}" for value in values)


def trim_generated(source: Path, destination: Path) -> Path:
    run([
        "ffmpeg", "-y", "-i", str(source), "-af",
        "silenceremove=start_periods=1:start_duration=0.06:start_threshold=-45dB,"
        "areverse,silenceremove=start_periods=1:start_duration=0.10:start_threshold=-45dB,areverse",
        "-ar", str(SR_TTS), "-ac", "1", str(destination),
    ])
    if not destination.exists() or destination.stat().st_size < 1024:
        raise RuntimeError("trimmed speech is empty")
    return destination


def fit_without_cutting(source: Path, destination: Path, budget: float) -> tuple[Path, float, float]:
    """Fit complete speech to an exact sample budget using tempo only.

    Codec padding and sample-rate rounding can add a few milliseconds even when
    ffprobe reports a matching duration. Compare real decoded sample counts and
    apply a tiny additional atempo correction; never slice or discard samples.
    """
    if budget <= 0.10:
        raise RuntimeError(f"invalid speech budget: {budget:.3f}s")
    source_info = sf.info(str(source))
    source_rate = int(source_info.samplerate or SR_TTS)
    actual = source_info.frames / max(source_rate, 1)
    budget_samples = max(1, int(round(budget * SR_TTS)))
    speed = max((actual * SR_TTS) / budget_samples, 1.0)
    if speed <= 1.0:
        shutil.copy2(source, destination)
    else:
        run([
            "ffmpeg", "-y", "-i", str(source), "-filter:a", atempo_filter(speed * 1.002),
            "-ar", str(SR_TTS), "-ac", "1", str(destination),
        ])

    # atempo/filter rounding is codec-dependent. Correct again if needed, still
    # by changing tempo only. Three passes are ample for sub-frame differences.
    for attempt in range(8):
        info = sf.info(str(destination))
        frames = int(info.frames)
        rate = int(info.samplerate or SR_TTS)
        normalized_frames = int(round(frames * SR_TTS / max(rate, 1)))
        if normalized_frames <= budget_samples:
            fitted = normalized_frames / SR_TTS
            return destination, actual, fitted
        correction = normalized_frames / budget_samples
        corrected = destination.with_name(f"{destination.stem}.corrected-{attempt}.wav")
        run([
            "ffmpeg", "-y", "-i", str(destination), "-filter:a", atempo_filter(correction * 1.006),
            "-ar", str(SR_TTS), "-ac", "1", str(corrected),
        ])
        os.replace(corrected, destination)
    final_info = sf.info(str(destination))
    final_frames = int(round(final_info.frames * SR_TTS / max(int(final_info.samplerate or SR_TTS), 1)))
    raise RuntimeError(f"complete speech does not fit chunk after tempo correction: {final_frames} > {budget_samples} samples")


def match_duration_without_cutting(source: Path, destination: Path, target_duration: float) -> tuple[Path, float, float]:
    """Match the source speech window in both directions using tempo only.

    Short translated speech is slowed to end with the original phrase; long
    speech is accelerated. No atrim, sample slicing, or text shortening occurs.
    """
    if target_duration <= 0.10:
        raise RuntimeError(f"invalid sync target: {target_duration:.3f}s")
    info = sf.info(str(source))
    actual = info.frames / max(int(info.samplerate or SR_TTS), 1)
    target_samples = max(1, int(round(target_duration * SR_TTS)))
    current = source
    for attempt in range(5):
        current_info = sf.info(str(current))
        current_samples = int(round(current_info.frames * SR_TTS / max(int(current_info.samplerate or SR_TTS), 1)))
        if abs(current_samples - target_samples) <= int(0.012 * SR_TTS):
            if current != destination:
                shutil.copy2(current, destination)
            fitted = current_samples / SR_TTS
            return destination, actual, fitted
        factor = current_samples / target_samples
        candidate = destination.with_name(f"{destination.stem}.sync-{attempt}.wav")
        run([
            "ffmpeg", "-y", "-i", str(current), "-filter:a", atempo_filter(factor),
            "-ar", str(SR_TTS), "-ac", "1", str(candidate),
        ])
        current = candidate
    final_info = sf.info(str(current))
    final_samples = int(round(final_info.frames * SR_TTS / max(int(final_info.samplerate or SR_TTS), 1)))
    if abs(final_samples - target_samples) > int(0.025 * SR_TTS):
        raise RuntimeError(f"speech sync mismatch after tempo correction: {final_samples} vs {target_samples} samples")
    os.replace(current, destination)
    return destination, actual, final_samples / SR_TTS


def match_duration_bounded(
    source: Path, destination: Path, target_duration: float, *,
    min_tempo: float = SLOW_TEMPO_FLOOR, max_tempo: float | None = None,
) -> tuple[Path, float, float]:
    """Fill the speech window in both directions, but never slow below ``min_tempo``.

    Long speech is accelerated to end with the original phrase unless
    ``max_tempo`` caps that (``max_tempo=1.0`` means: never speed up here, an
    earlier budget fit already did). Short speech is slowed only as far as
    ``min_tempo`` allows; whatever remains of the window stays silent rather
    than dragging the read. Tempo only: nothing is trimmed or sliced.
    """
    if target_duration <= 0.10:
        raise RuntimeError(f"invalid sync target: {target_duration:.3f}s")
    min_tempo = min(1.0, max(0.5, float(min_tempo)))
    info = sf.info(str(source))
    actual = info.frames / max(int(info.samplerate or SR_TTS), 1)
    slowest = actual / min_tempo
    effective_target = min(float(target_duration), slowest)
    if max_tempo is not None and actual > effective_target:
        effective_target = max(effective_target, actual / max(1.0, float(max_tempo)))
    if abs(actual - effective_target) <= 0.012:
        if Path(source) != Path(destination):
            shutil.copy2(source, destination)
        return destination, actual, actual
    return match_duration_without_cutting(source, destination, effective_target)


def plausible_tts_seconds(text: str) -> float:
    """Upper bound for a sane take of ``text``; longer takes are regenerated."""
    words = len(normalize_tokens(text or "")) or len((text or "").split()) or 1
    return TTS_LONG_TAKE_RATIO * (words / TTS_WORDS_PER_SECOND) + TTS_LONG_TAKE_SLACK_SECONDS


def boundary_fades(chunks: list[dict], index: int) -> tuple[float, float]:
    """Voice fade lengths for chunk ``index``: click guards where speech continues across the cut."""
    chunk = chunks[index]
    previous = chunks[index - 1] if index > 0 else None
    fade_in = DECLICK_FADE_SECONDS if previous and previous.get("cut_reason") in CONTINUATION_CUTS else EDGE_FADE_SECONDS
    fade_out = DECLICK_FADE_SECONDS if chunk.get("cut_reason") in CONTINUATION_CUTS else EDGE_FADE_SECONDS
    return fade_in, fade_out


def convert_analysis_audio(source: Path, speech: Path, background: Path | None = None) -> None:
    speech.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-i", str(source), "-ar", "24000", "-ac", "1", "-c:a", "flac", str(speech)])
    if background is not None:
        run(["ffmpeg", "-y", "-i", str(source), "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "128k", str(background)])


def slice_audio(source: Path, start: float, duration: float, destination: Path) -> Path:
    run([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", str(source), "-ar", str(SR_TTS), "-ac", "1", "-c:a", "pcm_s16le", str(destination),
    ])
    return destination


def expand_short_phrase_window(chunk: dict, natural_duration: float | None) -> dict:
    """Borrow preceding silence when ASR assigns an impossibly short phrase window."""
    result = dict(chunk)
    if not natural_duration or natural_duration <= 0:
        return result
    raw_start = float(result["speech_start"])
    original_start = float(result.get("speech_start_original", raw_start))
    start = max(float(result["start"]), raw_start)
    end = float(result["speech_end"])
    if start != raw_start:
        result["speech_start_original"] = original_start
        result["speech_start"] = start
        result["timing_adjustment"] = "clamped_to_chunk_start"
    current = max(0.0, end - start)
    natural = min(float(natural_duration), max(0.0, end - float(result["start"])))
    if natural > 0 and current < natural * 0.80:
        result["speech_start_original"] = original_start
        result["speech_start"] = max(float(result["start"]), end - natural)
        result["timing_adjustment"] = "expanded_into_preceding_silence_for_complete_phrase"
    if result.get("speech_start") != raw_start or result.get("speech_start_original") is not None:
        result["timing_shift_seconds"] = round(original_start - float(result["speech_start"]), 3)
    return result


def build_chunk_audio(
    chunk: dict,
    fitted_voice: Path | None,
    background_audio: Path | None,
    destination: Path,
    *,
    background_gain: float,
    voice_is_full_timeline: bool = False,
    fade_in_seconds: float = EDGE_FADE_SECONDS,
    fade_out_seconds: float = EDGE_FADE_SECONDS,
) -> Path:
    duration = float(chunk["end"]) - float(chunk["start"])
    total = max(1, int(round(duration * SR_TTS)))
    voice_track = np.zeros(total, dtype=np.float32)
    bed_track = np.zeros(total, dtype=np.float32)

    if background_audio and background_audio.exists():
        background_clip = destination.with_name("background.wav")
        slice_audio(background_audio, float(chunk["start"]), duration, background_clip)
        bed, _ = sf.read(str(background_clip), dtype="float32")
        if getattr(bed, "ndim", 1) > 1:
            bed = bed.mean(axis=1)
        count = min(total, len(bed))
        bed_track[:count] = bed[:count]

    if fitted_voice and fitted_voice.exists():
        voice, _ = sf.read(str(fitted_voice), dtype="float32")
        if getattr(voice, "ndim", 1) > 1:
            voice = voice.mean(axis=1)
        offset_seconds = 0.0 if voice_is_full_timeline else max(0.0, float(chunk["speech_start"]) - float(chunk["start"]))
        offset = min(total, int(round(offset_seconds * SR_TTS)))
        available = total - offset
        if len(voice) > available:
            raise RuntimeError(f"fitted voice would be truncated: {len(voice)} samples > {available}")
        count = len(voice)
        if count > 0:
            fade_in = min(int(max(0.0, fade_in_seconds) * SR_TTS), count // 2)
            fade_out = min(int(max(0.0, fade_out_seconds) * SR_TTS), count // 2)
            voice = voice.copy()
            if fade_in > 1:
                voice[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
            if fade_out > 1:
                voice[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
            voice_track[offset:offset + count] = voice

    # Duck the background only while translated speech is active. A smoothed
    # envelope avoids pumping at word boundaries and restores full ambience in silence.
    active = (np.abs(voice_track) > (10 ** (-44.0 / 20.0))).astype(np.float32)
    smooth_samples = max(1, int(0.12 * SR_TTS))
    if np.any(active):
        envelope = np.convolve(active, np.ones(smooth_samples, dtype=np.float32) / smooth_samples, mode="same")
        envelope = np.clip(envelope * 2.0, 0.0, 1.0)
    else:
        envelope = active
    duck_floor = 0.28
    duck_curve = 1.0 - envelope * (1.0 - duck_floor)
    mix = voice_track + bed_track * float(background_gain) * duck_curve

    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 0.94:
        mix *= 0.94 / peak
    sf.write(str(destination), mix, SR_TTS, subtype="PCM_16")
    atomic_write_json(destination.with_name("mix-report.json"), {
        "background_present": bool(background_audio and background_audio.exists()),
        "background_gain": float(background_gain),
        "duck_floor": duck_floor,
        "speech_active_ratio": round(float(np.mean(active)), 4),
        "peak_before_limit": round(peak, 6),
        "fade_in_seconds": round(float(fade_in_seconds), 4),
        "fade_out_seconds": round(float(fade_out_seconds), 4),
    })
    return destination


def render_chunk(source_video: Path, chunk: dict, audio: Path, destination: Path) -> Path:
    start = float(chunk["start"])
    duration = float(chunk["end"]) - start
    destination.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", str(source_video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", "setpts=PTS-STARTPTS,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-t", f"{duration:.3f}", "-movflags", "+faststart", str(destination),
    ])
    measured = ffprobe_duration(destination)
    if abs(measured - duration) > 0.25:
        raise RuntimeError(f"chunk duration mismatch: {measured:.3f}s vs {duration:.3f}s")
    return destination


def slice_source_chunk(source_video: Path, chunk: dict, destination: Path) -> Path:
    start = float(chunk["start"])
    duration = float(chunk["end"]) - start
    run([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", str(source_video), "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", "setpts=PTS-STARTPTS,format=yuv420p", "-af", "asetpts=PTS-STARTPTS",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-t", f"{duration:.3f}",
        "-movflags", "+faststart", str(destination),
    ])
    return destination


def apply_seed_vc_audio(reference_audio: Path, voice_audio: Path, destination: Path, space: str) -> Path:
    """Convert only spoken voice; never send timeline silence/background to Seed-VC."""
    command = [
        sys.executable, str(Path(__file__).with_name("seed_vc_enhance.py")),
        "--original", str(reference_audio), "--dubbed", str(voice_audio),
        "--output", str(destination), "--space", space, "--audio-only",
    ]
    result = run(command, check=False)
    if result.returncode != 0 or not destination.exists() or destination.stat().st_size < 1024:
        detail = ((result.stderr or "") + "\n" + (result.stdout or ""))[-1600:]
        raise RuntimeError(f"Seed-VC failed for this chunk: {detail.strip()}")
    return destination


def find_verified_word_clip(
    store: CheckpointStore, target_word: str, destination: Path, *, exclude_index: int,
) -> tuple[Path, int] | None:
    target = normalize_tokens(target_word)
    if not target:
        return None
    token = target[0]
    candidates: list[tuple[float, float, int, dict, Path]] = []
    for donor in store.data.get("chunks", []):
        index = int(donor.get("index", -1))
        if index == exclude_index or donor.get("status") != "completed":
            continue
        if not bool((donor.get("content_validation") or {}).get("ok")):
            continue
        speech_start = float(donor.get("speech_start", donor.get("start", 0.0)))
        for word in (donor.get("word_alignment") or {}).get("words", []):
            normalized = normalize_tokens(str(word.get("word", "")))
            if not normalized or normalized[0] != token:
                continue
            local_start = max(0.0, float(word["actual_start"]) - speech_start)
            local_end = max(local_start + 0.04, float(word["actual_end"]) - speech_start)
            directory = store.chunk_dir(index)
            retry_attempt = int(donor.get("content_retry_attempt") or 0)
            voice_paths = []
            if retry_attempt:
                voice_paths.append(directory / f"content-retry-{retry_attempt}.delivery.wav")
            voice_paths.extend(sorted(directory.glob("delivery*.fitted.wav")))
            voice_paths.extend(sorted(directory.glob("pre-render*.fitted.wav")))
            voice_paths.extend(sorted(directory.glob("generated*.fitted.wav")))
            voice = next((path for path in voice_paths if path.exists() and path.stat().st_size > 1024), None)
            if voice:
                # Prefer a donor word already at the beginning of its phrase.
                candidates.append((0.0 if local_start <= 0.08 else 1.0, abs(float(word.get("end_drift", 0.0))), index, {"start": local_start, "end": local_end}, voice))
            break
    if not candidates:
        return None
    _edge, _drift, index, timing, voice = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    start = max(0.0, float(timing["start"]) - 0.015)
    duration = max(0.08, float(timing["end"]) - start + 0.025)
    slice_audio(voice, start, duration, destination)
    return destination, index


def concatenate_voice_parts(parts: list[Path], destination: Path, gap_seconds: float = 0.06) -> Path:
    gap = np.zeros(int(round(gap_seconds * SR_TTS)), dtype=np.float32)
    waves: list[np.ndarray] = []
    for position, path in enumerate(parts):
        audio, rate = sf.read(str(path), dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        if int(rate) != SR_TTS:
            raise RuntimeError(f"unexpected sample rate in borrowed word: {rate}")
        waves.append(np.asarray(audio, dtype=np.float32))
        if position + 1 < len(parts):
            waves.append(gap)
    sf.write(str(destination), np.concatenate(waves), SR_TTS, subtype="PCM_16")
    return destination


def create_comparison_previews(
    source_chunk: Path, before_seed: Path, after_seed_voice: Path | None,
    final_chunk: Path, directory: Path,
) -> dict[str, str | None]:
    sources = {
        "original": source_chunk,
        "before_seed_vc": before_seed,
        "after_seed_vc": after_seed_voice,
        "final": final_chunk,
    }
    result: dict[str, str | None] = {}
    for label, source in sources.items():
        if not source or not Path(source).exists():
            result[label] = None
            continue
        destination = directory / f"preview-{label}.mp3"
        run(["ffmpeg", "-y", "-i", str(source), "-vn", "-ar", "24000", "-ac", "1",
             "-c:a", "libmp3lame", "-b:a", "64k", str(destination)])
        result[label] = destination.name
    return result


def combine_voice_batch(items: list[tuple[int, Path]], destination: Path, gap_seconds: float = 0.25) -> list[dict]:
    gap = np.zeros(int(round(gap_seconds * SR_TTS)), dtype=np.float32)
    pieces: list[np.ndarray] = []
    mapping: list[dict] = []
    cursor = 0
    for position, (index, path) in enumerate(items):
        voice, rate = sf.read(str(path), dtype="float32")
        if getattr(voice, "ndim", 1) > 1:
            voice = voice.mean(axis=1)
        if int(rate) != SR_TTS:
            normalized = destination.with_name(f"normalize-{index:04d}.wav")
            run(["ffmpeg", "-y", "-i", str(path), "-ar", str(SR_TTS), "-ac", "1", str(normalized)])
            voice, _ = sf.read(str(normalized), dtype="float32")
        start = cursor
        end = start + len(voice)
        mapping.append({"chunk": index, "start_sample": start, "end_sample": end})
        pieces.append(np.asarray(voice, dtype=np.float32))
        cursor = end
        if position + 1 < len(items):
            pieces.append(gap)
            cursor += len(gap)
    combined = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    sf.write(str(destination), combined, SR_TTS, subtype="PCM_16")
    return mapping


def split_voice_batch(source: Path, mapping: list[dict], destinations: dict[int, Path]) -> None:
    audio, rate = sf.read(str(source), dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if int(rate) != SR_TTS:
        raise RuntimeError(f"unexpected Seed-VC batch sample rate: {rate}")
    required = max((int(item["end_sample"]) for item in mapping), default=0)
    if len(audio) < required:
        missing = required - len(audio)
        if missing > int(0.03 * SR_TTS):
            raise RuntimeError(f"Seed-VC batch is too short by {missing} samples")
        # Codec rounding may remove a few milliseconds at the final boundary.
        # Add digital silence after all speech; never slice a spoken sample.
        audio = np.pad(audio, (0, missing))
    for item in mapping:
        index = int(item["chunk"])
        start = int(item["start_sample"])
        end = int(item["end_sample"])
        if end <= start:
            raise RuntimeError(f"Seed-VC batch mapping is invalid for chunk {index}")
        # Boundaries fall inside the synthetic 250 ms separators, never words.
        sf.write(str(destinations[index]), audio[start:end], SR_TTS, subtype="PCM_16")


def probe_frame_rate(path: Path) -> tuple[float, str] | None:
    """Average video frame rate of ``path`` as (float, ffmpeg fraction) or None."""
    result = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], check=False)
    text = (result.stdout or "").strip().splitlines()
    if result.returncode != 0 or not text:
        return None
    raw = text[0].strip()
    numerator, _sep, denominator = raw.partition("/")
    try:
        fps = float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        return None
    return (fps, raw) if fps > 0 else None


def probe_video_frames(path: Path) -> int | None:
    """Exact number of video frames stored in ``path`` (one packet per frame)."""
    result = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], check=False)
    text = (result.stdout or "").strip().splitlines()
    if result.returncode != 0 or not text:
        return None
    try:
        frames = int(text[0].strip())
    except ValueError:
        return None
    return frames if frames > 0 else None


def frame_aligned_frame_counts(chunks: list[dict], fps: float, held: list[int]) -> list[int]:
    """How many frames of every rendered chunk belong in the final timeline.

    Stream-copy concatenation of the 158 Napoleon chunks ran 3.08 s past the
    source: every chunk file carries codec padding (and can carry the frame
    that straddles its planned end), and the concat demuxer simply adds those
    up.  Here each chunk keeps all of its frames unless that would push the
    assembled timeline a whole frame or more past the chunk's planned end; only
    then is the trailing frame left out.  The result stays within one frame of
    the plan at every boundary, no frame is ever duplicated, and no chunk file
    is modified or re-rendered.
    """
    if not chunks:
        return []
    counts: list[int] = []
    cursor = float(math.ceil(float(chunks[0]["start"]) * fps - 1e-6))
    for chunk, frames_in_file in zip(chunks, held):
        frames = max(1, int(frames_in_file))
        ideal_end = float(chunk["end"]) * fps
        while frames > 1 and cursor + frames - ideal_end >= 1.0 - 1e-6:
            frames -= 1
        counts.append(frames)
        cursor += frames
    return counts


def concatenate_chunks_frame_accurate(
    paths: list[Path], frame_counts: list[int], fps: float, fps_text: str, destination: Path,
) -> subprocess.CompletedProcess:
    """Join chunks through the concat *filter* with exact per-chunk frame counts.

    Each video segment is cut to ``frame_counts[i]`` whole frames, its audio is
    bounded to the same span (dropping per-file AAC padding), and the timeline
    is encoded once at the source frame rate.  The stream-copy path cannot do
    this: it can only cut on packet boundaries and inherits every file's padding.
    """
    script = destination.with_suffix(".concat.filter")
    lines = []
    for index, frames in enumerate(frame_counts):
        seconds = frames / fps
        lines.append(f"[{index}:v:0]trim=end_frame={frames},setpts=PTS-STARTPTS[v{index}];")
        lines.append(f"[{index}:a:0]atrim=end={seconds:.6f},asetpts=PTS-STARTPTS[a{index}];")
    inputs = "".join(f"[v{index}][a{index}]" for index in range(len(frame_counts)))
    lines.append(f"{inputs}concat=n={len(frame_counts)}:v=1:a=1[vout][aout]")
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    duration = max(1.0, sum(frame_counts) / fps)
    audio_kbps = 192 if duration <= 600 else 128
    # The verified video is committed to the repository twice (projects/ and
    # docs/); GitHub rejects any file above 100 MB, so bound the encode to a
    # PUBLISH_SIZE_BUDGET_BYTES total.  Short clips never reach the cap.
    result = None
    for attempt in range(2):
        budget_kbps = PUBLISH_SIZE_BUDGET_BYTES * 8 / duration / 1000 * (1.0 - 0.25 * attempt)
        maxrate_kbps = int(max(300, min(6000, budget_kbps - audio_kbps)))
        command = ["ffmpeg", "-y"]
        for path in paths:
            command += ["-i", str(path)]
        command += [
            "-filter_complex_script", str(script), "-map", "[vout]", "-map", "[aout]",
            "-r", fps_text, "-fps_mode", "cfr",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-maxrate", f"{maxrate_kbps}k", "-bufsize", f"{2 * maxrate_kbps}k",
            "-c:a", "aac", "-b:a", f"{audio_kbps}k", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart", str(destination),
        ]
        result = run(command, check=False)
        if result.returncode != 0 or not destination.exists():
            return result
        if destination.stat().st_size <= PUBLISH_SIZE_HARD_LIMIT_BYTES:
            return result
        print(f"final video is {destination.stat().st_size} bytes; re-encoding with a tighter rate cap", file=sys.stderr)
    return result


def concatenate_chunks(
    paths: list[Path], destination: Path, *, chunks: list[dict] | None = None,
    frame_rate: tuple[float, str] | None = None, held_frames: list[int | None] | None = None,
) -> Path:
    if chunks is not None and frame_rate and held_frames and len(chunks) == len(paths) == len(held_frames) and all(held_frames):
        fps, fps_text = frame_rate
        counts = frame_aligned_frame_counts(chunks, fps, [int(value) for value in held_frames])
        exact = concatenate_chunks_frame_accurate(paths, counts, fps, fps_text, destination)
        if exact.returncode == 0 and destination.exists() and destination.stat().st_size > 1024:
            trimmed = sum(int(value) for value in held_frames) - sum(counts)
            print(f"Assembled {len(paths)} chunks frame-accurately: {sum(counts)} frames at {fps_text} fps ({trimmed} straddling frames left out)")
            return destination
        print("Frame-accurate assembly failed; falling back to stream-copy concatenation", file=sys.stderr)
        print((exact.stderr or "")[-1200:], file=sys.stderr)
    list_path = destination.with_suffix(".concat.txt")
    list_path.write_text("".join(f"file '{path.resolve()}'\n" for path in paths), encoding="utf-8")
    first = run([
        "ffmpeg", "-y", "-fflags", "+genpts", "-f", "concat", "-safe", "0",
        "-i", str(list_path), "-c", "copy", "-movflags", "+faststart", str(destination),
    ], check=False)
    if first.returncode != 0:
        run([
            "ffmpeg", "-y", "-fflags", "+genpts", "-f", "concat", "-safe", "0",
            "-i", str(list_path), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(destination),
        ])
    return destination


def reusable_final(
    manifest: dict, delivery_key: str, delivery_chunks: list[str], candidate: Path, *, chunks_changed: bool = False,
) -> bool:
    """Is the final video restored from the checkpoint still the right film?

    It must be the exact file the manifest recorded (sha256) and must have been
    built from the current completed chunks: either the delivery key matches,
    or, for manifests written before delivery keys existed, the project had
    already reached the completed state and no chunk was rebuilt in this run.
    """
    expected = manifest.get("final_sha256")
    if not expected or not candidate.exists() or sha256_file(candidate) != expected:
        return False
    if manifest.get("delivery_key"):
        return manifest.get("delivery_key") == delivery_key and manifest.get("delivery_chunks") == delivery_chunks
    return (
        not chunks_changed
        and manifest.get("state") == "completed_waiting_for_cleanup_approval"
        and all(delivery_chunks)
    )


def is_derived_asr_cache(path: Path) -> bool:
    """Whisper's 16 kHz working copies are rebuilt from their source on demand.

    They must never travel inside a checkpoint archive: a restored copy could be
    older than the regenerated 24 kHz take it claims to mirror, and content
    validation would then judge audio that is no longer delivered.
    """
    name = Path(path).name
    return name.endswith("_16k.wav") or ".tmp-" in name


def write_text_files(store: CheckpointStore, index: int) -> None:
    chunk = store.chunk(index)
    directory = store.chunk_dir(index)
    (directory / "source.txt").write_text(chunk.get("source_text", ""), encoding="utf-8")
    (directory / "translation.txt").write_text(chunk.get("translated_text", ""), encoding="utf-8")


def prepare_profile_references(
    profiles: dict[str, dict], chunks: list[dict], speech_audio: Path,
    global_reference: Path, analysis_dir: Path,
) -> dict[str, Path | None]:
    out: dict[str, Path | None] = {}
    root = analysis_dir / "speaker-references"
    root.mkdir(parents=True, exist_ok=True)
    for speaker, profile in profiles.items():
        mode = profile.get("reference_mode")
        destination = root / f"{safe_project_id(speaker)}-{stable_hash(profile)[:10]}.wav"
        if destination.exists() and destination.stat().st_size > 1024:
            out[speaker] = destination
            continue
        if mode == "custom":
            source = Path(profile["reference_path"])
            run(["ffmpeg", "-y", "-i", str(source), "-af",
                 "highpass=f=80,lowpass=f=9000,afftdn=nr=10,dynaudnorm=f=150:g=7",
                 "-ar", "22050", "-ac", "1", str(destination)])
        else:
            turns = [chunk for chunk in chunks if (chunk.get("speaker") or "SPEAKER_00") == speaker and chunk.get("source_text")]
            if not turns:
                shutil.copy2(global_reference, destination)
            else:
                best = max(turns, key=lambda item: float(item["speech_end"]) - float(item["speech_start"]))
                start = float(best["speech_start"])
                duration = min(12.0, max(1.0, float(best["speech_end"]) - start))
                run(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
                     "-i", str(speech_audio), "-af",
                     "highpass=f=80,lowpass=f=9000,afftdn=nr=10,dynaudnorm=f=150:g=7",
                     "-ar", "22050", "-ac", "1", str(destination)])
        if not destination.exists() or destination.stat().st_size < 1024:
            raise RuntimeError(f"invalid voice reference for {speaker}")
        out[speaker] = destination
    return out


def observed_transcript(raw: list[dict]) -> tuple[str, list[dict]]:
    text = " ".join(str(item.get("text", "")).strip() for item in raw).strip()
    words = []
    for item in raw:
        words.extend(item.get("words") or [])
    return text, words


async def synthesize(
    args, profile: dict, text: str, destination: Path, reference: Path | None,
    *, max_seconds: float | None = None,
) -> str:
    engine = profile.get("tts_engine") or args.tts_engine
    style = profile.get("style") or "natural"
    if max_seconds is None:
        max_seconds = plausible_tts_seconds(text)
    if engine == "voxcpm":
        await speak_voxcpm(
            text, destination, language=args.target_lang,
            control=f"{style}; delivery: {infer_emotion(text)}",
            reference_audio=reference,
            max_seconds=max_seconds,
        )
        return "voxcpm"
    if engine == "xtts":
        if not args.allow_xtts:
            raise RuntimeError("XTTS v2 is locked; explicit --allow-xtts approval is required")
        if not reference:
            raise RuntimeError("XTTS requires an approved source or custom voice reference")
        ok = await asyncio.to_thread(
            xtts_clone.clone_speak, text, reference, destination,
            args.target_lang, pick_device(),
        )
        if not ok:
            raise RuntimeError("XTTS failed for this chunk")
        return "xtts"
    raise RuntimeError(f"production TTS engine is not allowed: {engine}")

async def translate_with_llm(
    store: CheckpointStore, mirror: "ReleaseMirror", pending: list[dict], *, source_lang: str, target_lang: str,
) -> list[dict]:
    """Translate ``pending`` chunks with the configured LLM; returns chunks still untranslated.

    Only active when ``TRANSLATE_API_KEY`` is set. Speech is translated in
    transcript order with the surrounding, already translated lines as context
    and the planned speech window as a spoken-time budget. Progress is saved
    and mirrored after every window, so a resumed run continues where it
    stopped. Non-speech labels are kept verbatim. With TRANSLATE_FALLBACK=google
    an API outage hands the remaining chunks to the default translator; by
    default it stops the run resumably instead of mixing translation engines.
    """
    try:
        config = LLMTranslateConfig.from_env()
    except TranslationConfigError as exc:
        raise RuntimeError(f"LLM translation is misconfigured: {exc}") from exc
    if config is None:
        return pending
    remaining: list[dict] = []
    segments: list[dict] = []
    for chunk in pending:
        text = str(chunk.get("source_text") or "")
        index = int(chunk["index"])
        if is_non_speech_text(text):
            store.update_chunk(index, translated_text=text, status="translated", translation_engine="label")
            write_text_files(store, index)
            continue
        window_seconds = max(0.6, float(chunk.get("speech_end", chunk["end"])) - float(chunk.get("speech_start", chunk["start"])))
        segments.append({
            "index": index,
            "text": text,
            "seconds": window_seconds,
            "continues": chunk.get("cut_reason") in CONTINUATION_CUTS,
        })
    if not segments:
        return remaining
    first_pending = segments[0]["index"]
    prior = [
        {"index": int(chunk["index"]), "text": chunk["source_text"], "translation": chunk["translated_text"]}
        for chunk in store.data["chunks"]
        if int(chunk["index"]) < first_pending and chunk.get("source_text") and chunk.get("translated_text")
        and not is_non_speech_text(chunk.get("source_text", ""))
    ]
    store.data["translation"] = {**config.public_summary(), "status": "in_progress"}
    store.save()
    translator = LLMTranslator(config)
    translated_indices: set[int] = set()

    def checkpoint(results: list[dict]) -> None:
        for item in results:
            index = int(item["index"])
            store.update_chunk(
                index, translated_text=item["text"], status="translated",
                translation_engine=item["engine"], translation_words=item["words"],
                translation_budget_words=item["max_words"],
            )
            write_text_files(store, index)
            translated_indices.add(index)
        store.data["translation"]["usage"] = dict(translator.usage)
        store.save()
        mirror.upload_manifest(store)

    try:
        await translator.translate_segments(
            segments, source_lang=source_lang, target_lang=target_lang, prior=prior, on_window=checkpoint,
        )
        store.data["translation"]["status"] = "completed"
    except TranslationAPIError as exc:
        left = [chunk for chunk in pending if int(chunk["index"]) not in translated_indices
                and not is_non_speech_text(str(chunk.get("source_text") or ""))]
        store.data["translation"].update({"status": "failed", "error": str(exc), "untranslated_chunks": [int(c["index"]) for c in left]})
        store.save()
        mirror.upload_manifest(store)
        if config.fallback == "google":
            print(f"LLM translation unavailable ({exc}); TRANSLATE_FALLBACK=google translates the remaining {len(left)} chunks", file=sys.stderr)
            return left
        raise RuntimeError(
            f"LLM translation failed after retries ({exc}); {len(left)} chunks are untranslated. "
            "Fix the API access and re-run to resume, or set TRANSLATE_FALLBACK=google."
        ) from exc
    finally:
        store.save()
        await translator.close()
    return remaining


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    ap.add_argument("--output-dir", type=Path, default=Path("output"))
    ap.add_argument("--source-lang", default="ar")
    ap.add_argument("--target-lang", default="en")
    ap.add_argument("--tts-engine", choices=["voxcpm", "xtts"], default="voxcpm")
    ap.add_argument("--gender", choices=["male", "female"], default="male")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--target-seconds", type=float, default=8.0)
    ap.add_argument("--min-seconds", type=float, default=2.5)
    ap.add_argument("--separate-sources", action="store_true")
    ap.add_argument("--diarize", action="store_true")
    ap.add_argument("--no-vad", action="store_true")
    ap.add_argument("--preserve-background", action="store_true")
    ap.add_argument("--background-gain", type=float, default=0.50)
    ap.add_argument("--seed-vc", action="store_true")
    ap.add_argument("--seed-vc-space", default="phuoc2005/seed-vc")
    ap.add_argument("--seed-batch-size", type=int, default=8)
    ap.add_argument("--seed-quota-policy", choices=["fail", "voxcpm"], default="fail")
    ap.add_argument("--speaker-voices", type=Path)
    ap.add_argument("--allow-xtts", action="store_true", help="explicit authorization required before any XTTS v2 synthesis")
    ap.add_argument("--require-voice-approval", action="store_true")
    ap.add_argument("--validate-content", action="store_true")
    ap.add_argument("--analysis-only", action="store_true", help="checkpoint source, ASR, speakers, and chunk plan, then stop before translation")
    ap.add_argument("--content-min-recall", type=float, default=0.70)
    ap.add_argument("--content-min-sequence", type=float, default=0.58)
    ap.add_argument("--release-tag")
    return ap


async def main_async(args) -> None:
    source_value = str(args.source).strip()
    if not source_value:
        raise ValueError("source is required")
    is_url = source_value.startswith(("http://", "https://"))
    if not is_url:
        local_source = Path(source_value).expanduser().resolve()
        if not local_source.exists():
            raise FileNotFoundError(local_source)
        source_value = str(local_source)
    project_id = safe_project_id(args.project_id)
    project_root = args.checkpoint_root / project_id
    mirror = ReleaseMirror(args.release_tag)
    mirror.ensure_and_restore(project_root)
    store = CheckpointStore(project_root)

    loaded = load_source(source_value)
    source_path = loaded.video_path if is_url else Path(source_value)
    source_hash = sha256_file(source_path)
    config = {
        "pipeline": "smart-resumable-v1",
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "tts_engine": args.tts_engine,
        "allow_xtts": args.allow_xtts,
        "gender": args.gender,
        "model": args.model,
        "max_seconds": args.max_seconds,
        "target_seconds": args.target_seconds,
        "min_seconds": args.min_seconds,
        "separate_sources": args.separate_sources,
        "diarize": args.diarize,
        "no_vad": args.no_vad,
        "preserve_background": args.preserve_background,
        "background_gain": args.background_gain,
        "seed_vc": args.seed_vc,
        "seed_vc_space": args.seed_vc_space,
    }

    source_duration = float(loaded.metadata.duration)
    device = pick_device()
    analysis = store.analysis_dir
    asr_path = analysis / "asr.json"
    meta_path = analysis / "analysis.json"
    speech_audio = analysis / "speech.flac"
    background_audio = analysis / "background.m4a"
    reference = analysis / "reference.wav"

    if asr_path.exists() and meta_path.exists():
        raw = json.loads(asr_path.read_text(encoding="utf-8"))
        analysis_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        detected_language = analysis_meta.get("detected_language") or args.source_lang
        asr_coverage = float(analysis_meta.get("asr_coverage", timeline_metrics(raw, source_duration)[0]))
        asr_max_gap = float(analysis_meta.get("asr_max_gap", timeline_metrics(raw, source_duration)[1]))
        speech_coverage = analysis_meta.get("speech_coverage")
        print(f"Reusing full-video ASR ({len(raw)} timed segments)")
    else:
        working_speech = loaded.audio_path
        working_background = None
        if args.separate_sources:
            demucs_work = Path(os.environ.get("YAD_TEMP_DIR", "temp")) / f"smart-demucs-{project_id}"
            stems = separate_dialogue_background(loaded.audio_path, demucs_work)
            stem_report = validate_stems(stems, source_duration)
            if stem_report.get("valid"):
                separated_speech, separated_background = stems
                convert_analysis_audio(separated_speech, speech_audio)
                run(["ffmpeg", "-y", "-i", str(separated_background), "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "128k", str(background_audio)])
                working_speech = speech_audio
                working_background = background_audio
            else:
                print(f"Demucs rejected: {stem_report.get('reason')}; using original audio")
        forced = None if args.source_lang in ("", "auto") else args.source_lang
        use_vad = not args.no_vad
        raw, detected_language = transcribe(
            working_speech, model_name=args.model, device=device, language=forced, use_vad=use_vad,
        )
        asr_coverage, asr_max_gap = timeline_metrics(raw, source_duration)
        if use_vad and asr_max_gap > 4.0:
            print(f"ASR gap {asr_max_gap:.2f}s detected; retrying once without VAD")
            recovered, recovered_language = transcribe(
                working_speech, model_name=args.model, device=device, language=forced, use_vad=False,
            )
            recovered_coverage, recovered_gap = timeline_metrics(recovered, source_duration)
            if (recovered_gap, -recovered_coverage) < (asr_max_gap, -asr_coverage):
                raw, detected_language = recovered, recovered_language or detected_language
                asr_coverage, asr_max_gap = recovered_coverage, recovered_gap
        # Whisper can drop a whole phrase while the speaker keeps talking; the
        # dub would then fall silent there. Compare the speech-stem energy with
        # the word timeline and re-transcribe only the loud, wordless spans.
        uncovered_before = find_uncovered_speech(working_speech, raw, source_duration)
        speech_coverage = {"spans_before": uncovered_before, "recovered_words": 0, "spans_after": uncovered_before}
        if uncovered_before:
            total_uncovered = sum(float(s["seconds"]) for s in uncovered_before)
            print(f"ASR left {total_uncovered:.2f}s of loud speech without words in {len(uncovered_before)} span(s); re-transcribing them without VAD")
            raw, recovered_words = recover_uncovered_speech(
                working_speech, raw, uncovered_before, duration=source_duration,
                work_dir=analysis / "asr-recovery", model_name=args.model, device=device,
                language=forced or detected_language,
            )
            speech_coverage["recovered_words"] = recovered_words
            speech_coverage["spans_after"] = find_uncovered_speech(working_speech, raw, source_duration)
            asr_coverage, asr_max_gap = timeline_metrics(raw, source_duration)
            print(f"ASR recovery added {recovered_words} words; {sum(float(s['seconds']) for s in speech_coverage['spans_after']):.2f}s remain uncovered")
        if args.diarize:
            raw = annotate_segments(working_speech, raw)
        atomic_write_json(asr_path, raw)
        atomic_write_json(meta_path, {
            "detected_language": detected_language,
            "source_duration": source_duration,
            "source_sha256": source_hash,
            "asr_coverage": asr_coverage,
            "asr_max_gap": asr_max_gap,
            "speech_audio": str(working_speech),
            "background_audio": str(working_background) if working_background else None,
            "speech_coverage": speech_coverage,
        })
        if not speech_audio.exists():
            convert_analysis_audio(working_speech, speech_audio)
        if working_background and not background_audio.exists():
            shutil.copy2(working_background, background_audio)
        mirror.upload_tree("analysis.zip", project_root, analysis)

    if not reference.exists():
        run([
            "ffmpeg", "-y", "-i", str(speech_audio if speech_audio.exists() else loaded.audio_path),
            "-t", "45", "-af", "highpass=f=80,lowpass=f=9000,afftdn=nr=12",
            "-ar", "22050", "-ac", "1", str(reference),
        ])
        mirror.upload_tree("analysis.zip", project_root, analysis)

    plans = plan_smart_chunks(
        raw, source_duration, max_seconds=args.max_seconds,
        target_seconds=args.target_seconds, min_seconds=args.min_seconds,
    )
    speakers = sorted({str(chunk.get("speaker") or "SPEAKER_00") for chunk in plans if chunk.get("source_text")}) or ["SPEAKER_00"]
    profile_defaults = {
        "reference_mode": "source",
        "tts_engine": args.tts_engine,
        "voice_conversion": "seed-vc" if args.seed_vc else "none",
        "gender": args.gender,
        "style": "natural",
        "approved": not args.require_voice_approval,
    }
    profiles = load_voice_profiles(
        args.speaker_voices, speakers, defaults=profile_defaults,
        require_approval=bool(args.speaker_voices) or args.require_voice_approval,
    )
    xtts_speakers = [speaker for speaker, profile in profiles.items() if profile.get("tts_engine") == "xtts"]
    if (args.tts_engine == "xtts" or xtts_speakers) and not args.allow_xtts:
        requested = ", ".join(xtts_speakers) if xtts_speakers else "project default"
        raise RuntimeError(f"XTTS v2 is locked; explicit approval is required for: {requested}")
    atomic_write_json(analysis / "voice-profiles-template.json", template_for_speakers(speakers, profile_defaults))
    atomic_write_json(analysis / "voice-profiles-active.json", {"version": 1, "speakers": profiles})
    profile_references = prepare_profile_references(
        profiles, plans, speech_audio if speech_audio.exists() else loaded.audio_path,
        reference, analysis,
    )
    speaker_analysis = {
        "project_id": project_id,
        "speakers": [
            {
                "speaker": speaker,
                "profile": profiles[speaker],
                "reference": str(profile_references.get(speaker) or ""),
                "chunk_count": sum(1 for chunk in plans if (chunk.get("speaker") or "SPEAKER_00") == speaker),
                "spoken_seconds": round(sum(max(0.0, float(chunk["speech_end"]) - float(chunk["speech_start"])) for chunk in plans if (chunk.get("speaker") or "SPEAKER_00") == speaker), 3),
            }
            for speaker in speakers
        ],
    }
    atomic_write_json(analysis / "speaker-analysis.json", speaker_analysis)
    mirror.upload_tree("analysis.zip", project_root, analysis)
    store.initialize(
        source={"path": source_path.name, "sha256": source_hash, "duration": source_duration},
        config=config,
        chunks=plans,
    )
    store.data["voice_profiles"] = profiles
    store.save()
    mirror.upload_manifest(store)

    override_path = args.speaker_voices.with_name("translation-overrides.json") if args.speaker_voices else None
    if override_path and override_path.exists():
        override_doc = json.loads(override_path.read_text(encoding="utf-8"))
        if isinstance(override_doc.get("targets"), dict):
            override_chunks = override_doc["targets"].get(args.target_lang, {})
        else:
            override_chunks = override_doc.get("chunks", override_doc)
        if not isinstance(override_chunks, dict):
            raise ValueError("translation-overrides.json must contain chunks or targets.<language>")
        for raw_index, override_text in override_chunks.items():
            index = int(raw_index)
            if index < 0 or index >= len(store.data["chunks"]):
                raise ValueError(f"translation override references unknown chunk {index}")
            text = str(override_text).strip()
            if not text:
                raise ValueError(f"translation override for chunk {index} is empty")
            chunk = store.chunk(index)
            text_hash = stable_hash(text)
            if chunk.get("translated_text") != text or chunk.get("translation_override_hash") != text_hash:
                store.invalidate_from(index, "translation", "approved project translation override changed")
                store.update_chunk(
                    index,
                    translated_text=text,
                    translation_engine="project_override",
                    translation_override_hash=text_hash,
                    status="translated",
                    error=None,
                )
                write_text_files(store, index)
        store.save()
        mirror.upload_manifest(store)

    if args.analysis_only:
        store.mark_state(
            "analysis_completed_waiting_for_voice_approval",
            cleanup_authorized=False,
            detected_speakers=speakers,
            analysis_only=True,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output_dir / "progress-report.json", store.summary())
        shutil.copy2(store.manifest_path, args.output_dir / "checkpoint-manifest.json")
        shutil.copy2(analysis / "speaker-analysis.json", args.output_dir / "speaker-analysis.json")
        mirror.upload_manifest(store)
        print(json.dumps(store.summary(), ensure_ascii=False, indent=2))
        print("ANALYSIS_ONLY=completed; configure and approve every detected speaker before dubbing")
        return

    missing_translation = [chunk for chunk in store.data["chunks"] if chunk.get("source_text") and not chunk.get("translated_text")]
    effective_source = str(detected_language or args.source_lang).lower().split("-")[0]
    effective_target = str(args.target_lang).lower().split("-")[0]
    if missing_translation and effective_source == effective_target:
        for chunk in missing_translation:
            index = int(chunk["index"])
            text = str(chunk.get("source_text") or "").strip()
            store.update_chunk(
                index, translated_text=text, status="translated",
                translation_engine="same_language_passthrough", error=None,
            )
            write_text_files(store, index)
        store.save()
        mirror.upload_manifest(store)
        missing_translation = []
    if missing_translation:
        missing_translation = await translate_with_llm(
            store, mirror, missing_translation,
            source_lang=detected_language or args.source_lang, target_lang=args.target_lang,
        )
    if missing_translation:
        translator = GoogleTranslator()
        try:
            for offset in range(0, len(missing_translation), 10):
                batch = missing_translation[offset:offset + 10]
                translations = await translator.translate_batch(
                    [chunk["source_text"] for chunk in batch],
                    source=detected_language or args.source_lang,
                    target=args.target_lang,
                )
                for chunk, translation in zip(batch, translations):
                    index = int(chunk["index"])
                    translated = translation.strip()
                    if not translated:
                        raise RuntimeError(f"empty translation for chunk {index}; refusing to synthesize source-language text")
                    store.update_chunk(index, translated_text=translated, status="translated", translation_engine="google")
                    write_text_files(store, index)
                mirror.upload_manifest(store)
        finally:
            await translator.close()

    for chunk in store.data["chunks"]:
        index = int(chunk["index"])
        if not chunk.get("source_text") or is_non_speech_text(chunk.get("source_text", "")):
            store.mark_stage(index, "translation", "skipped", details={"reason": "non_speech_or_empty"})
        elif chunk.get("translated_text"):
            store.mark_stage(
                index, "translation", "success",
                input_hash=stable_hash({"source": chunk.get("source_text"), "target": args.target_lang}),
                details={"translated_text": chunk.get("translated_text")},
            )

    # Pass 1: generate and checkpoint all TTS voices before consuming any
    # Seed-VC quota. A later run resumes at the first missing voice file.
    generation_failures: list[int] = []
    for chunk in store.data["chunks"]:
        if not chunk.get("source_text"):
            continue
        index = int(chunk["index"])
        if is_non_speech_text(chunk.get("source_text", "")):
            store.update_chunk(index, status="non_speech", engine_used="silence", non_speech_label=True, error=None)
            for stage_name in ("tts", "seed_vc", "timing_fit", "content_validation"):
                store.mark_stage(index, stage_name, "skipped", details={"reason": "non_speech"})
            mirror.upload_chunk(store, index)
            continue
        speaker = str(chunk.get("speaker") or "SPEAKER_00")
        non_speech = is_non_speech_text(chunk.get("source_text", ""))
        profile = profiles[speaker]
        profile_hash = stable_hash(profile)
        variant = f".{profile_hash[:10]}" if args.speaker_voices else ""
        if chunk.get("translation_override_hash"):
            variant += f".text-{chunk['translation_override_hash'][:8]}"
        directory = store.chunk_dir(index)
        generated = directory / f"generated{variant}.wav"
        fitted = directory / f"generated{variant}.fitted.wav"
        tts_input_hash = stable_hash({"text": chunk.get("translated_text"), "profile": profile_hash})
        if store.stage_valid(index, "tts", fitted, input_hash=tts_input_hash):
            continue
        if store.stage(index, "tts").get("state") == "success":
            store.invalidate_from(index, "tts", "TTS input or output checksum changed")
        try:
            engine_used = await synthesize(
                args, profile, chunk["translated_text"], generated,
                profile_references.get(speaker),
            ) if not generated.exists() or generated.stat().st_size < 1024 else (chunk.get("engine_used") or profile.get("tts_engine") or args.tts_engine)
            trimmed = trim_generated(generated, directory / f"generated{variant}.trim.wav")
            budget = max(0.12, float(chunk["end"]) - float(chunk["speech_start"]) - 0.03)
            fitted, original_tts_duration, fitted_tts_duration = fit_without_cutting(trimmed, fitted, budget)
            store.update_chunk(
                index, status="voice_generated", engine_used=engine_used,
                original_tts_duration=round(original_tts_duration, 3),
                fitted_tts_duration=round(fitted_tts_duration, 3),
                tts_plausible_seconds=round(plausible_tts_seconds(chunk["translated_text"]), 3),
                tts_duration_warning=bool(original_tts_duration > plausible_tts_seconds(chunk["translated_text"])),
            )
            store.mark_stage(
                index, "tts", "success", output=fitted, input_hash=tts_input_hash,
                details={"engine": engine_used, "duration": round(fitted_tts_duration, 3)},
            )
            mirror.upload_chunk(store, index)
        except Exception as exc:
            generation_failures.append(index)
            store.mark_stage(index, "tts", "failed", input_hash=tts_input_hash, error=str(exc))
            store.update_chunk(index, status="failed", error=str(exc))
            store.add_error(index, str(exc))
            mirror.upload_chunk(store, index)
    if generation_failures:
        store.mark_state("failed_resumable", failed_chunks=generation_failures)
        mirror.upload_manifest(store)
        raise RuntimeError(f"TTS chunks preserved for resume: {generation_failures}")

    # Pass 2: convert compact voice-only batches. Seed-VC itself supports long
    # inputs via internal overlapping windows; batching amortizes ZeroGPU startup
    # quota while 250 ms synthetic separators protect every phrase boundary.
    seed_failures: list[int] = []
    seed_quota_fallback = bool(
        args.seed_quota_policy == "voxcpm"
        and (store.data.get("seed_quota_fallback") or {}).get("active")
    )
    if not seed_quota_fallback and args.seed_batch_size > 1 and any(profile.get("voice_conversion") == "seed-vc" for profile in profiles.values()):
        grouped: dict[tuple[str, str], list[tuple[int, Path, Path]]] = {}
        for chunk in store.data["chunks"]:
            if not chunk.get("source_text") or is_non_speech_text(chunk.get("source_text", "")):
                continue
            index = int(chunk["index"])
            speaker = str(chunk.get("speaker") or "SPEAKER_00")
            profile = profiles[speaker]
            if profile.get("voice_conversion") != "seed-vc":
                continue
            profile_hash = stable_hash(profile)
            variant = f".{profile_hash[:10]}" if args.speaker_voices else ""
            if chunk.get("translation_override_hash"):
                variant += f".text-{chunk['translation_override_hash'][:8]}"
            directory = store.chunk_dir(index)
            fitted = directory / f"generated{variant}.fitted.wav"
            seed_voice = directory / f"seedvc.voice-only{variant}.wav"
            if seed_voice.exists() and seed_voice.stat().st_size > 1024:
                continue
            grouped.setdefault((speaker, profile_hash), []).append((index, fitted, seed_voice))
        batch_root = analysis / "seed-batches"
        batch_root.mkdir(parents=True, exist_ok=True)
        stop_for_quota = False
        for (speaker, profile_hash), items in grouped.items():
            for offset in range(0, len(items), max(1, args.seed_batch_size)):
                batch = items[offset:offset + max(1, args.seed_batch_size)]
                batch_id = stable_hash({"speaker": speaker, "profile": profile_hash, "chunks": [item[0] for item in batch]})[:14]
                batch_dir = batch_root / batch_id
                batch_dir.mkdir(parents=True, exist_ok=True)
                combined = batch_dir / "voice-input.wav"
                mapping_path = batch_dir / "batch-map.json"
                converted = batch_dir / "voice-seed.wav"
                synced = batch_dir / "voice-seed.synced.wav"
                try:
                    mapping = combine_voice_batch([(item[0], item[1]) for item in batch], combined)
                    atomic_write_json(mapping_path, mapping)
                    if not converted.exists() or converted.stat().st_size < 1024:
                        converted = apply_seed_vc_audio(
                            profile_references.get(speaker) or reference,
                            combined, converted, args.seed_vc_space,
                        )
                    target_duration = sf.info(str(combined)).frames / SR_TTS
                    synced, _actual, _fitted = match_duration_without_cutting(converted, synced, target_duration)
                    destinations = {item[0]: item[2] for item in batch}
                    split_voice_batch(synced, mapping, destinations)
                    for index, _input, output in batch:
                        store.update_chunk(index, status="seed_vc_generated", seed_vc_batch=batch_id)
                        store.mark_stage(
                            index, "seed_vc", "success", output=output,
                            details={"batch": batch_id, "mode": "voice_only"},
                        )
                        mirror.upload_chunk(store, index)
                    mirror.upload_tree(f"seed-batch-{batch_id}.zip", project_root, batch_dir)
                except Exception as exc:
                    message = str(exc)
                    quota_exhausted = "quota" in message.lower() or "zerogpu" in message.lower() or "runs limit" in message.lower()
                    if quota_exhausted and args.seed_quota_policy == "voxcpm":
                        seed_quota_fallback = True
                        store.data["seed_quota_fallback"] = {
                            "active": True,
                            "mode": "voxcpm_reference_clone",
                            "reason": "Seed-VC ZeroGPU quota exhausted",
                        }
                        for index, _input, _output in batch:
                            store.update_chunk(
                                index, status="voice_generated", error=None,
                                seed_vc_skipped="quota_exhausted_explicit_voxcpm_policy",
                            )
                            store.mark_stage(
                                index, "seed_vc", "skipped",
                                details={"reason": "quota_exhausted", "fallback": "voxcpm_reference_clone"},
                            )
                            mirror.upload_chunk(store, index)
                        store.save()
                    else:
                        for index, _input, _output in batch:
                            seed_failures.append(index)
                            store.mark_stage(index, "seed_vc", "failed", error=message)
                            store.update_chunk(index, status="failed", error=message)
                            store.add_error(index, message)
                            mirror.upload_chunk(store, index)
                    if quota_exhausted:
                        stop_for_quota = True
                        break
            if stop_for_quota:
                break
    if seed_failures:
        store.mark_state("failed_resumable", failed_chunks=seed_failures)
        mirror.upload_manifest(store)
        raise RuntimeError(f"Seed-VC batches preserved for resume: {seed_failures}")

    failures: list[int] = []
    rendered_this_run = 0
    for chunk in store.data["chunks"]:
        index = int(chunk["index"])
        speaker = str(chunk.get("speaker") or "SPEAKER_00")
        non_speech = is_non_speech_text(chunk.get("source_text", ""))
        profile = profiles[speaker]
        profile_hash = stable_hash(profile)
        variant = f".{profile_hash[:10]}" if args.speaker_voices else ""
        if chunk.get("translation_override_hash"):
            variant += f".text-{chunk['translation_override_hash'][:8]}"
        seed_requested = profile.get("voice_conversion") == "seed-vc" and bool(chunk.get("source_text")) and not non_speech
        seed_required = seed_requested and not seed_quota_fallback
        profile_current = not args.speaker_voices or chunk.get("voice_profile_hash") == profile_hash
        content_current = (
            not args.validate_content
            or non_speech
            or not chunk.get("source_text")
            or bool((chunk.get("content_validation") or {}).get("ok"))
        )
        complete = store.completed_file(index)
        seed_mode_current = chunk.get("seed_vc_mode") == "voice_only_sync_v3"
        if complete and profile_current and content_current and (not seed_required or seed_mode_current):
            if chunk.get("status") != "completed":
                # Pass 1 re-labels finished non-speech chunks as "non_speech" on
                # every run; a validated render is still a completed chunk.
                store.update_chunk(index, status="completed", error=None)
            print(f"Chunk {index:04d}: restored and validated; skipping")
            continue
        rendered_this_run += 1
        if complete and seed_required and not seed_mode_current:
            print(f"Chunk {index:04d}: outdated Seed-VC timing detected; rebuilding alignment only")
        directory = store.chunk_dir(index)
        current_stage = "timing_fit"
        expected_word_count = len(normalize_tokens(chunk.get("translated_text", "")))
        natural_window = max(
            float(chunk.get("original_tts_duration") or 0.0),
            min(3.0, expected_word_count * 0.34),
        )
        adjusted_chunk = expand_short_phrase_window(chunk, natural_window)
        if adjusted_chunk.get("speech_start") != chunk.get("speech_start"):
            store.update_chunk(
                index,
                speech_start_original=adjusted_chunk.get("speech_start_original"),
                speech_start=adjusted_chunk["speech_start"],
                timing_adjustment=adjusted_chunk.get("timing_adjustment"),
                timing_shift_seconds=adjusted_chunk.get("timing_shift_seconds"),
                error=None,
            )
            chunk = store.chunk(index)
        write_text_files(store, index)
        attempts = int(chunk.get("attempts", 0)) + 1
        store.update_chunk(index, status="processing", attempts=attempts, error=None)
        try:
            fitted_voice: Path | None = None
            engine_used = "silence"
            if chunk.get("source_text") and not non_speech:
                generated = directory / f"generated{variant}.wav"
                if not generated.exists() or generated.stat().st_size < 1024:
                    engine_used = await synthesize(
                        args, profile, chunk["translated_text"], generated,
                        profile_references.get(speaker),
                    )
                else:
                    engine_used = chunk.get("engine_used") or profile.get("tts_engine") or args.tts_engine
                trimmed = trim_generated(generated, directory / f"generated{variant}.trim.wav")
                budget = max(0.12, float(chunk["end"]) - float(chunk["speech_start"]) - 0.03)
                fitted_voice, original_tts_duration, fitted_tts_duration = fit_without_cutting(
                    trimmed, directory / f"generated{variant}.fitted.wav", budget,
                )
                store.update_chunk(
                    index, status="voice_generated", engine_used=engine_used,
                    original_tts_duration=round(original_tts_duration, 3),
                    fitted_tts_duration=round(fitted_tts_duration, 3),
                    tempo_factor=round(max(original_tts_duration / max(fitted_tts_duration, 0.001), 1.0), 4),
                )
            if fitted_voice:
                pre_render_budget = max(
                    0.12,
                    float(store.chunk(index)["end"]) - max(
                        float(store.chunk(index)["start"]), float(store.chunk(index)["speech_start"])
                    ) - 0.015,
                )
                fitted_voice, _pre_original, pre_fitted_duration = fit_without_cutting(
                    fitted_voice,
                    directory / f"pre-render{variant}.fitted.wav",
                    pre_render_budget,
                )
                store.mark_stage(
                    index, "timing_fit", "success", output=fitted_voice,
                    details={"duration": round(pre_fitted_duration, 3), "budget": round(pre_render_budget, 3)},
                )
                if not seed_required:
                    # Without Seed-VC nothing used to slow short speech down, so a
                    # line that came out shorter than the original left a hole in
                    # the middle of continuous narration. Fill the planned speech
                    # window in both directions, bounded by SLOW_TEMPO_FLOOR.
                    window_target = min(
                        pre_render_budget,
                        max(0.12, float(store.chunk(index)["speech_end"]) - float(store.chunk(index)["speech_start"])),
                    )
                    fitted_voice, fill_original, fill_fitted = match_duration_bounded(
                        fitted_voice, directory / f"window-fill{variant}.wav", window_target,
                        min_tempo=SLOW_TEMPO_FLOOR, max_tempo=1.0,
                    )
                    store.update_chunk(
                        index, timing_mode="window_fill_v1",
                        window_fill_target=round(window_target, 3),
                        window_fill_duration=round(fill_fitted, 3),
                        window_fill_tempo=round(fill_original / max(fill_fitted, 0.001), 4),
                    )
                    store.mark_stage(
                        index, "timing_fit", "success", output=fitted_voice,
                        details={"duration": round(fill_fitted, 3), "window": round(window_target, 3), "mode": "window_fill_v1"},
                    )
            # Render and preserve the complete pre-Seed voice-only checkpoint.
            fade_in_seconds, fade_out_seconds = boundary_fades(store.data["chunks"], index)
            raw_voice_audio = build_chunk_audio(
                store.chunk(index), fitted_voice, None,
                directory / f"voice-before-seedvc{variant}.wav", background_gain=0.0,
                fade_in_seconds=fade_in_seconds, fade_out_seconds=fade_out_seconds,
            )
            raw_dubbed = render_chunk(
                loaded.video_path, store.chunk(index), raw_voice_audio,
                directory / f"dubbed-before-seedvc{variant}.mp4",
            )
            source_chunk = directory / "source.mp4"
            if not source_chunk.exists() or source_chunk.stat().st_size < 1024:
                source_chunk = slice_source_chunk(loaded.video_path, store.chunk(index), source_chunk)
            final_voice = fitted_voice
            full_timeline = False
            seed_applied = False
            if seed_required:
                store.update_chunk(index, status="seed_vc_processing")
                # Keep the original media slice for audit, but use the stable
                # global clean speaker reference and voice-only content for VC.
                seed_voice = directory / f"seedvc.voice-only{variant}.wav"
                if not seed_voice.exists() or seed_voice.stat().st_size < 1024:
                    if args.seed_batch_size > 1:
                        raise RuntimeError(f"missing completed Seed-VC batch output for chunk {index}")
                    seed_voice = apply_seed_vc_audio(
                        profile_references.get(speaker) or reference, fitted_voice, seed_voice, args.seed_vc_space,
                    )
                speech_target = max(0.12, float(chunk["speech_end"]) - float(chunk["speech_start"]))
                final_voice, seed_original_duration, seed_fitted_duration = match_duration_bounded(
                    seed_voice, directory / f"seedvc.voice-only{variant}.synced.wav", speech_target,
                    min_tempo=SLOW_TEMPO_FLOOR,
                )
                # The converted voice is placed at speech_start and ends with
                # the original ASR phrase window; trailing media silence remains silent.
                full_timeline = False
                seed_applied = True
                store.update_chunk(
                    index, status="seed_vc_completed", seed_vc_applied=True,
                    seed_vc_mode="voice_only_sync_v3",
                    seed_original_duration=round(seed_original_duration, 3),
                    seed_fitted_duration=round(seed_fitted_duration, 3),
                    speech_target_duration=round(speech_target, 3),
                )
            delivery_budget = max(0.12, float(chunk["end"]) - float(chunk["speech_start"]) - 0.015)
            if final_voice:
                final_voice, _delivery_original, delivery_fitted_duration = fit_without_cutting(
                    final_voice, directory / f"delivery{variant}.fitted.wav", delivery_budget,
                )
                store.update_chunk(
                    index, status="delivery_fitted",
                    delivery_fitted_duration=round(delivery_fitted_duration, 3),
                    delivery_budget=round(delivery_budget, 3),
                )
                store.mark_stage(
                    index, "timing_fit", "success", output=final_voice,
                    details={"duration": round(delivery_fitted_duration, 3), "budget": round(delivery_budget, 3)},
                )

            content_result = None
            timing_result = None
            if args.validate_content and chunk.get("source_text") and not non_speech and final_voice:
                current_stage = "content_validation"
                for content_attempt in range(3):
                    checked_raw, _checked_language = transcribe(
                        final_voice, model_name=args.model, device=device,
                        language=args.target_lang, use_vad=False,
                    )
                    spoken_text, spoken_words = observed_transcript(checked_raw)
                    content_result = validate_spoken_content(
                        chunk["translated_text"], spoken_text,
                        min_recall=args.content_min_recall,
                        min_sequence_ratio=args.content_min_sequence,
                    )
                    timing_result = word_timing_report(
                        spoken_words, float(chunk["speech_start"]), float(chunk["speech_end"]),
                    )
                    atomic_write_json(directory / f"content-validation-{content_attempt}.json", content_result)
                    atomic_write_json(directory / f"word-alignment-{content_attempt}.json", timing_result)
                    if content_result["ok"]:
                        break
                    if content_attempt >= 2:
                        raise RuntimeError(
                            f"spoken phrase incomplete after 3 retained takes: recall={content_result['recall']:.3f}, "
                            f"sequence={content_result['sequence_ratio']:.3f}"
                        )
                    retry_profile = dict(profile)
                    retry_profile["style"] = (
                        str(profile.get("style") or "natural")
                        + "; pronounce every written word distinctly; do not omit conjunctions or short words"
                    )
                    retry_text = str(chunk["translated_text"]).strip()
                    retry_parts = retry_text.split(maxsplit=1)
                    if len(retry_parts) == 2 and retry_parts[0].lower().rstrip(".,!?") in {"and", "but", "or"}:
                        retry_text = f"{retry_parts[0].rstrip('.,!?')}. {retry_parts[1]}"
                    retry_raw = directory / f"content-retry-{content_attempt + 1}.wav"
                    borrowed = None
                    expected_tokens = normalize_tokens(chunk["translated_text"])
                    missing_tokens = set(content_result.get("missing_words") or [])
                    if content_attempt == 0 and expected_tokens and expected_tokens[0] in missing_tokens and final_voice:
                        borrowed_path = directory / f"content-retry-{content_attempt + 1}.borrowed-word.wav"
                        borrowed = find_verified_word_clip(
                            store, expected_tokens[0], borrowed_path, exclude_index=index,
                        )
                    if borrowed:
                        retry_raw = concatenate_voice_parts(
                            [borrowed[0], final_voice], retry_raw, gap_seconds=0.06,
                        )
                        store.update_chunk(
                            index, content_retry_mode="borrowed_verified_word",
                            content_retry_donor_chunk=borrowed[1],
                            content_retry_synthesis_text=chunk["translated_text"],
                        )
                    else:
                        if content_attempt >= 1:
                            # Never switch engines during recovery. Short, balanced
                            # parts make the selected engine pronounce every word.
                            words = retry_text.split()
                            part_count = max(1, math.ceil(len(words) / 5))
                            part_size = max(1, math.ceil(len(words) / part_count))
                            split_texts = [
                                " ".join(words[offset:offset + part_size])
                                for offset in range(0, len(words), part_size)
                            ]
                            split_audio = []
                            for part_index, part_text in enumerate(split_texts, start=1):
                                part_raw = directory / f"content-retry-{content_attempt + 1}.part{part_index}.wav"
                                await synthesize(
                                    args, retry_profile, part_text, part_raw,
                                    profile_references.get(speaker),
                                )
                                split_audio.append(trim_generated(
                                    part_raw, directory / f"content-retry-{content_attempt + 1}.part{part_index}.trim.wav",
                                ))
                            retry_raw = concatenate_voice_parts(split_audio, retry_raw, gap_seconds=0.05)
                            store.update_chunk(
                                index, content_retry_mode="split_exact_selected_engine",
                                content_retry_parts=split_texts,
                                content_retry_synthesis_text=retry_text,
                            )
                        else:
                            await synthesize(
                                args, retry_profile, retry_text, retry_raw,
                                profile_references.get(speaker),
                            )
                            store.update_chunk(index, content_retry_synthesis_text=retry_text)
                    if borrowed:
                        # The borrowed word is intentionally at sample zero; silence
                        # trimming would remove this short, lower-energy consonant.
                        retry_trimmed = retry_raw
                    else:
                        retry_trimmed = trim_generated(
                            retry_raw, directory / f"content-retry-{content_attempt + 1}.trim.wav",
                        )
                    speech_target = max(0.12, float(chunk["speech_end"]) - float(chunk["speech_start"]))
                    retry_voice, _retry_actual, _retry_fitted = match_duration_bounded(
                        retry_trimmed,
                        directory / f"content-retry-{content_attempt + 1}.synced.wav",
                        speech_target,
                        min_tempo=SLOW_TEMPO_FLOOR,
                    )
                    if seed_required:
                        try:
                            retry_seed = apply_seed_vc_audio(
                                profile_references.get(speaker) or reference,
                                retry_voice,
                                directory / f"content-retry-{content_attempt + 1}.seed.wav",
                                args.seed_vc_space,
                            )
                            retry_voice, _seed_actual, _seed_fitted = match_duration_bounded(
                                retry_seed,
                                directory / f"content-retry-{content_attempt + 1}.seed.synced.wav",
                                speech_target,
                                min_tempo=SLOW_TEMPO_FLOOR,
                            )
                        except Exception as seed_error:
                            message = str(seed_error).lower()
                            quota_exhausted = "quota" in message or "zerogpu" in message or "runs limit" in message
                            if not (quota_exhausted and args.seed_quota_policy == "voxcpm"):
                                raise
                            # The selected profile explicitly authorizes VoxCPM
                            # fallback. Keep the newly generated complete phrase
                            # rather than failing after the primary Seed-VC pass.
                            store.update_chunk(
                                index,
                                content_retry_seed_vc_skipped="quota_exhausted_explicit_voxcpm_policy",
                                content_retry_seed_vc_error=str(seed_error),
                            )
                    final_voice, _retry_delivery_original, retry_delivery_fitted = fit_without_cutting(
                        retry_voice,
                        directory / f"content-retry-{content_attempt + 1}.delivery.wav",
                        delivery_budget,
                    )
                    store.update_chunk(
                        index, status="content_retry", content_retry_attempt=content_attempt + 1,
                        delivery_fitted_duration=round(retry_delivery_fitted, 3),
                    )
                atomic_write_json(directory / "content-validation.json", content_result)
                atomic_write_json(directory / "word-alignment.json", timing_result)
                store.update_chunk(
                    index, status="content_validated", content_validation=content_result,
                    word_alignment=timing_result,
                    content_retry_attempt=content_attempt,
                )
                store.mark_stage(
                    index, "content_validation", "success",
                    output=directory / "content-validation.json",
                    details={"recall": content_result.get("recall"), "sequence_ratio": content_result.get("sequence_ratio")},
                )

            current_stage = "audio_mix"
            chunk_audio = build_chunk_audio(
                store.chunk(index), final_voice,
                background_audio if args.preserve_background and background_audio.exists() else None,
                directory / "mixed.wav", background_gain=args.background_gain,
                voice_is_full_timeline=full_timeline,
                fade_in_seconds=fade_in_seconds, fade_out_seconds=fade_out_seconds,
            )
            store.mark_stage(index, "audio_mix", "success", output=chunk_audio)
            current_stage = "video_render"
            dubbed = render_chunk(loaded.video_path, store.chunk(index), chunk_audio, directory / "dubbed.mp4")
            store.mark_stage(index, "video_render", "success", output=dubbed)
            preview_tracks = create_comparison_previews(
                source_chunk, raw_dubbed,
                final_voice if seed_required else None,
                dubbed, directory,
            )
            comparison = {
                "speaker": speaker,
                "profile": profile,
                "tracks": preview_tracks,
                "content_validation": content_result,
                "word_alignment": timing_result,
            }
            atomic_write_json(directory / "comparison.json", comparison)
            store.update_chunk(
                index, status="completed", engine_used=engine_used,
                speaker=speaker, voice_profile_hash=profile_hash, voice_profile=profile,
                seed_vc_requested=seed_requested,
                seed_vc_required=seed_required,
                seed_vc_applied=seed_applied,
                delivery_voice_mode="seed_vc" if seed_applied else ("voxcpm_reference_clone" if seed_quota_fallback else "base_tts"),
                content_validation=content_result, word_alignment=timing_result,
                dubbed_sha256=sha256_file(dubbed), measured_duration=round(ffprobe_duration(dubbed), 3),
            )
            current_stage = "checkpoint_upload"
            mirror.upload_chunk(store, index)
            store.mark_stage(index, "checkpoint_upload", "success", details={"asset": f"chunk-{index:04d}.zip"})
            print(f"Chunk {index:04d}: completed and checkpointed")
        except Exception as exc:
            failures.append(index)
            store.mark_stage(index, current_stage, "failed", error=str(exc))
            store.update_chunk(index, status="failed", error=str(exc))
            store.add_error(index, str(exc))
            mirror.upload_chunk(store, index)
            print(f"Chunk {index:04d}: FAILED: {exc}", file=sys.stderr)
            message = str(exc).lower()
            if "quota" in message or "zerogpu" in message:
                print("External GPU quota is exhausted; stopping safely after checkpoint upload", file=sys.stderr)
                break

    if failures:
        store.mark_state("failed_resumable", failed_chunks=failures)
        mirror.upload_manifest(store)
        raise RuntimeError(f"failed chunks preserved for resume: {failures}")

    ordered = [store.completed_file(index) for index in range(len(store.data["chunks"]))]
    if not all(ordered):
        raise RuntimeError("not all completed chunk files are present")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source-path.txt").write_text(str(source_path.resolve()), encoding="utf-8")
    chunk_files = [path for path in ordered if path]
    frame_rate = probe_frame_rate(chunk_files[0]) if chunk_files else None
    held_frames = [probe_video_frames(path) for path in chunk_files]
    delivery_chunks = [str(chunk.get("dubbed_sha256") or "") for chunk in store.data["chunks"]]
    delivery_key = stable_hash({
        "assembler": "frame_accurate_v1",
        "chunks": delivery_chunks,
        "frames": (
            frame_aligned_frame_counts(store.data["chunks"], frame_rate[0], [int(v) for v in held_frames])
            if frame_rate and all(held_frames) else None
        ),
        "fps": frame_rate[1] if frame_rate else None,
        "budget": [PUBLISH_SIZE_BUDGET_BYTES, PUBLISH_SIZE_HARD_LIMIT_BYTES],
    })
    final = args.output_dir / "final-dub.mp4"
    reused_final = False
    cached_final = mirror.cached_asset("final-dub.mp4")
    if cached_final and reusable_final(store.data, delivery_key, delivery_chunks, cached_final, chunks_changed=rendered_this_run > 0):
        # The delivery stage is a checkpoint too: the same completed chunks,
        # assembled the same way, are the same film.  Do not encode it again.
        shutil.copy2(cached_final, final)
        reused_final = True
        print("Reusing the assembled final video from the checkpoint release (delivery unchanged)")
    else:
        final = concatenate_chunks(
            chunk_files, final,
            chunks=store.data["chunks"], frame_rate=frame_rate, held_frames=held_frames,
        )
    final_duration = ffprobe_duration(final)
    if abs(final_duration - source_duration) > 1.0:
        raise RuntimeError(f"final duration mismatch: {final_duration:.3f}s vs {source_duration:.3f}s")

    speech_chunks = [chunk for chunk in store.data["chunks"] if chunk.get("source_text")]
    segment_report = {
        "source_duration": source_duration,
        "transcript_source": "asr",
        "source_language": detected_language or args.source_lang,
        "target_language": args.target_lang,
        "background_preserved": bool(args.preserve_background),
        "asr_timeline": {
            "coverage": asr_coverage, "max_gap": asr_max_gap,
            "uncovered_speech": summarize_speech_coverage(speech_coverage),
        },
        "smart_chunking": {
            "max_seconds": args.max_seconds,
            "target_seconds": args.target_seconds,
            "coverage": "contiguous_exactly_once",
            "chunk_count": len(store.data["chunks"]),
        },
        "segments": [
            {
                "index": int(chunk["index"]),
                "start": float(chunk["speech_start"]),
                "end": float(chunk["speech_end"]),
                "chunk_start": float(chunk["start"]),
                "chunk_end": float(chunk["end"]),
                "source_text": chunk.get("source_text", ""),
                "translated_text": chunk.get("translated_text", ""),
                "speaker": chunk.get("speaker"),
                "confidence": float(chunk.get("confidence", 1.0)),
                "status": chunk.get("status"),
                "cut_reason": chunk.get("cut_reason"),
            }
            for chunk in speech_chunks
        ],
        "chunks": [
            {
                "index": int(chunk["index"]), "start": float(chunk["start"]),
                "end": float(chunk["end"]), "status": chunk.get("status"),
                "word_count": int(chunk.get("word_count", 0)),
                "cut_reason": chunk.get("cut_reason"),
            }
            for chunk in store.data["chunks"]
        ],
    }
    atomic_write_json(args.output_dir / "segments-report.json", segment_report)
    store.mark_state(
        "completed_waiting_for_cleanup_approval",
        cleanup_authorized=False,
        final_sha256=sha256_file(final),
        final_duration=round(final_duration, 3),
        final_path=str(final),
        final_reused=reused_final,
        delivery_key=delivery_key,
        delivery_chunks=delivery_chunks,
        checkpoint_release_tag=mirror.tag,
        delivery_voice_mode="voxcpm_reference_clone" if seed_quota_fallback else "configured",
        seed_quota_policy=args.seed_quota_policy,
    )
    atomic_write_json(args.output_dir / "progress-report.json", store.summary())
    shutil.copy2(store.manifest_path, args.output_dir / "checkpoint-manifest.json")
    shutil.copy2(analysis / "speaker-analysis.json", args.output_dir / "speaker-analysis.json")
    mirror.upload_manifest(store)
    if not reused_final:
        mirror.upload_final(final)
    print(json.dumps(store.summary(), ensure_ascii=False, indent=2))
    print(f"FINAL={final.resolve()}")


def main() -> None:
    args = parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
