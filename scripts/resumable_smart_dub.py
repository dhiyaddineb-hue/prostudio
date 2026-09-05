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
from youtube_auto_dub.voice import pick_voice, speak_edge, speak_qwen
from youtube_auto_dub import xtts_clone
from youtube_auto_dub.voxcpm_tts import speak_voxcpm
from youtube_auto_dub.youtube import load_source


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
            fade = min(int(0.015 * SR_TTS), count // 2)
            voice = voice.copy()
            if fade > 1:
                voice[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
                voice[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
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


def concatenate_chunks(paths: list[Path], destination: Path) -> Path:
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


def is_derived_asr_cache(path: Path) -> bool:
    """Whisper's 16 kHz working copies are rebuilt from their source on demand.

    They must never travel inside a checkpoint archive: a restored copy could be
    older than the regenerated 24 kHz take it claims to mirror, and content
    validation would then judge audio that is no longer delivered.
    """
    name = Path(path).name
    return name.endswith("_16k.wav") or ".tmp-" in name


def ensure_pcm_wav(path: Path) -> Path:
    """Guarantee a RIFF/PCM WAV at ``path`` (Edge-TTS writes MP3 regardless of suffix).

    The original bytes are kept next to it as ``<stem>.edge.mp3`` so nothing is
    discarded; only the working copy is transcoded to the project sample rate.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size < 12:
        return path
    with path.open("rb") as handle:
        header = handle.read(4)
    if header == b"RIFF":
        return path
    original = path.with_name(f"{path.stem}.edge.mp3")
    shutil.move(str(path), str(original))
    run([
        "ffmpeg", "-y", "-i", str(original), "-ar", str(SR_TTS), "-ac", "1",
        "-c:a", "pcm_s16le", str(path),
    ])
    if not path.exists() or path.stat().st_size < 1024:
        raise RuntimeError("Edge-TTS output could not be converted to PCM WAV")
    return path


class ReleaseMirror:
    """Incremental durable checkpoint mirror using a draft GitHub Release."""

    def __init__(self, tag: str | None):
        self.tag = safe_project_id(tag or "") if tag else ""
        self.enabled = bool(self.tag and os.environ.get("GH_TOKEN") and shutil.which("gh"))
        self.tmp = Path(tempfile.mkdtemp(prefix="dub-checkpoint-release-"))

    def _gh(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
        return run(["gh", *args], check=check)

    def ensure_and_restore(self, project_root: Path) -> None:
        if not self.enabled:
            print("Checkpoint release mirror disabled; using local state only")
            return
        view = self._gh(["release", "view", self.tag, "--json", "tagName"], check=False)
        if view.returncode != 0:
            target = os.environ.get("GITHUB_SHA", "")
            command = ["release", "create", self.tag, "--draft", "--title", f"Checkpoint {self.tag}",
                       "--notes", "Resumable smart-dub chunks. Delete only after explicit owner approval."]
            if target:
                command += ["--target", target]
            self._gh(command)
            return
        download = self.tmp / "download"
        download.mkdir(parents=True, exist_ok=True)
        result = self._gh(["release", "download", self.tag, "--dir", str(download), "--clobber"], check=False)
        if result.returncode != 0:
            return
        for archive in sorted(download.glob("*.zip")):
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(project_root)
        manifest = download / "checkpoint-manifest.json"
        if manifest.exists():
            project_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest, project_root / "manifest.json")
        print(f"Restored checkpoint release {self.tag}")

    def _upload(self, path: Path) -> None:
        if self.enabled:
            self._gh(["release", "upload", self.tag, str(path), "--clobber"])

    def upload_manifest(self, store: CheckpointStore) -> None:
        if not self.enabled:
            return
        asset = self.tmp / "checkpoint-manifest.json"
        shutil.copy2(store.manifest_path, asset)
        self._upload(asset)

    def upload_tree(self, asset_name: str, project_root: Path, tree: Path) -> None:
        if not self.enabled or not tree.exists():
            return
        asset = self.tmp / asset_name
        with zipfile.ZipFile(asset, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as handle:
            for path in sorted(tree.rglob("*")):
                if path.is_file() and not is_derived_asr_cache(path):
                    handle.write(path, path.relative_to(project_root))
        self._upload(asset)

    def upload_chunk(self, store: CheckpointStore, index: int) -> None:
        directory = store.chunk_dir(index)
        self.upload_tree(f"chunk-{index:04d}.zip", store.root, directory)
        if self.enabled:
            for preview in sorted(directory.glob("preview-*.mp3")):
                asset = self.tmp / f"chunk-{index:04d}-{preview.name}"
                shutil.copy2(preview, asset)
                self._upload(asset)
        self.upload_manifest(store)

    def upload_final(self, path: Path) -> None:
        if not self.enabled:
            return
        asset = self.tmp / "final-dub.mp4"
        shutil.copy2(path, asset)
        self._upload(asset)


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
        if mode == "synthetic":
            out[speaker] = None
            continue
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


async def synthesize(args, profile: dict, text: str, destination: Path, reference: Path | None) -> str:
    engine = profile.get("tts_engine") or args.tts_engine
    style = profile.get("style") or "natural"
    if engine == "voxcpm":
        try:
            await speak_voxcpm(
                text, destination, language=args.target_lang,
                control=f"{style}; delivery: {infer_emotion(text)}",
                reference_audio=reference,
            )
            return "voxcpm"
        except Exception as exc:
            if not args.fallback_edge:
                raise
            print(f"VoxCPM failed, using Edge-TTS fallback: {exc}")
    elif engine == "xtts":
        if not reference:
            raise RuntimeError("XTTS requires the persisted source voice reference")
        ok = await asyncio.to_thread(
            xtts_clone.clone_speak, text, reference, destination,
            args.target_lang, pick_device(),
        )
        if not ok:
            raise RuntimeError("XTTS failed for this chunk")
        return "xtts"
    elif engine == "qwen":
        await speak_qwen(
            text, destination, voice_sample=reference,
            language=args.target_lang, device=f"{pick_device()}:0",
        )
        return "qwen"
    voice = profile.get("voice") or args.voice or pick_voice(args.target_lang, profile.get("gender") or args.gender)
    await speak_edge(text, voice, destination, lang=args.target_lang, gender=profile.get("gender") or args.gender)
    return "edge"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    ap.add_argument("--output-dir", type=Path, default=Path("output"))
    ap.add_argument("--source-lang", default="ar")
    ap.add_argument("--target-lang", default="en")
    ap.add_argument("--tts-engine", choices=["voxcpm", "edge", "xtts", "qwen"], default="voxcpm")
    ap.add_argument("--voice")
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
    ap.add_argument("--fallback-edge", action="store_true", default=False)
    ap.add_argument("--seed-vc", action="store_true")
    ap.add_argument("--seed-vc-space", default="phuoc2005/seed-vc")
    ap.add_argument("--seed-batch-size", type=int, default=8)
    ap.add_argument("--seed-quota-policy", choices=["fail", "voxcpm"], default="fail")
    ap.add_argument("--no-fallback-edge", dest="fallback_edge", action="store_false")
    ap.add_argument("--speaker-voices", type=Path)
    ap.add_argument("--require-voice-approval", action="store_true")
    ap.add_argument("--validate-content", action="store_true")
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
        "voice": args.voice,
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
        "fallback_edge": args.fallback_edge,
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
        "voice": args.voice or "",
        "voice_conversion": "seed-vc" if args.seed_vc else "none",
        "gender": args.gender,
        "style": "natural",
        "approved": not args.require_voice_approval,
    }
    profiles = load_voice_profiles(
        args.speaker_voices, speakers, defaults=profile_defaults,
        require_approval=bool(args.speaker_voices) or args.require_voice_approval,
    )
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

    missing_translation = [chunk for chunk in store.data["chunks"] if chunk.get("source_text") and not chunk.get("translated_text")]
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
                    store.update_chunk(index, translated_text=translated, status="translated")
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
    for chunk in store.data["chunks"]:
        index = int(chunk["index"])
        speaker = str(chunk.get("speaker") or "SPEAKER_00")
        non_speech = is_non_speech_text(chunk.get("source_text", ""))
        profile = profiles[speaker]
        profile_hash = stable_hash(profile)
        variant = f".{profile_hash[:10]}" if args.speaker_voices else ""
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
            print(f"Chunk {index:04d}: restored and validated; skipping")
            continue
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
            # Render and preserve the complete pre-Seed voice-only checkpoint.
            raw_voice_audio = build_chunk_audio(
                store.chunk(index), fitted_voice, None,
                directory / f"voice-before-seedvc{variant}.wav", background_gain=0.0,
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
                final_voice, seed_original_duration, seed_fitted_duration = match_duration_without_cutting(
                    seed_voice, directory / f"seedvc.voice-only{variant}.synced.wav", speech_target,
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
                        if content_attempt >= 1 and len(expected_tokens) <= 5:
                            retry_voice_name = profile.get("voice") or args.voice or pick_voice(args.target_lang, profile.get("gender") or args.gender)
                            await speak_edge(
                                retry_text, retry_voice_name, retry_raw,
                                lang=args.target_lang, gender=profile.get("gender") or args.gender,
                            )
                            retry_raw = ensure_pcm_wav(retry_raw)
                            store.update_chunk(
                                index, content_retry_mode="edge_exact_short_phrase",
                                content_retry_voice=retry_voice_name,
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
                    retry_voice, _retry_actual, _retry_fitted = match_duration_without_cutting(
                        retry_trimmed,
                        directory / f"content-retry-{content_attempt + 1}.synced.wav",
                        speech_target,
                    )
                    if seed_required:
                        retry_seed = apply_seed_vc_audio(
                            profile_references.get(speaker) or reference,
                            retry_voice,
                            directory / f"content-retry-{content_attempt + 1}.seed.wav",
                            args.seed_vc_space,
                        )
                        retry_voice, _seed_actual, _seed_fitted = match_duration_without_cutting(
                            retry_seed,
                            directory / f"content-retry-{content_attempt + 1}.seed.synced.wav",
                            speech_target,
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
    final = concatenate_chunks([path for path in ordered if path], args.output_dir / "final-dub.mp4")
    final_duration = ffprobe_duration(final)
    if abs(final_duration - source_duration) > 1.0:
        raise RuntimeError(f"final duration mismatch: {final_duration:.3f}s vs {source_duration:.3f}s")

    speech_chunks = [chunk for chunk in store.data["chunks"] if chunk.get("source_text")]
    segment_report = {
        "source_duration": source_duration,
        "transcript_source": "asr",
        "source_language": detected_language or args.source_lang,
        "target_language": args.target_lang,
        "asr_timeline": {"coverage": asr_coverage, "max_gap": asr_max_gap},
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
        delivery_voice_mode="voxcpm_reference_clone" if seed_quota_fallback else "configured",
        seed_quota_policy=args.seed_quota_policy,
    )
    atomic_write_json(args.output_dir / "progress-report.json", store.summary())
    shutil.copy2(store.manifest_path, args.output_dir / "checkpoint-manifest.json")
    shutil.copy2(analysis / "speaker-analysis.json", args.output_dir / "speaker-analysis.json")
    mirror.upload_manifest(store)
    mirror.upload_final(final)
    print(json.dumps(store.summary(), ensure_ascii=False, indent=2))
    print(f"FINAL={final.resolve()}")


def main() -> None:
    args = parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
