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
    for attempt in range(3):
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
            "ffmpeg", "-y", "-i", str(destination), "-filter:a", atempo_filter(correction * 1.002),
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
    mix = np.zeros(total, dtype=np.float32)

    if background_audio and background_audio.exists():
        background_clip = destination.with_name("background.wav")
        slice_audio(background_audio, float(chunk["start"]), duration, background_clip)
        bed, _ = sf.read(str(background_clip), dtype="float32")
        if getattr(bed, "ndim", 1) > 1:
            bed = bed.mean(axis=1)
        count = min(total, len(bed))
        mix[:count] += bed[:count] * float(background_gain)

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
            # Tiny fades prevent clicks without deleting spoken content.
            fade = min(int(0.015 * SR_TTS), count // 2)
            voice = voice.copy()
            if fade > 1:
                voice[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
                voice[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            mix[offset:offset + count] += voice

    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 0.94:
        mix *= 0.94 / peak
    sf.write(str(destination), mix, SR_TTS, subtype="PCM_16")
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
                if path.is_file():
                    handle.write(path, path.relative_to(project_root))
        self._upload(asset)

    def upload_chunk(self, store: CheckpointStore, index: int) -> None:
        self.upload_tree(f"chunk-{index:04d}.zip", store.root, store.chunk_dir(index))
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


async def synthesize(args, text: str, destination: Path, reference: Path | None) -> str:
    if args.tts_engine == "voxcpm":
        try:
            await speak_voxcpm(
                text, destination, language=args.target_lang,
                control=f"A natural, clear narrator; delivery: {infer_emotion(text)}",
                reference_audio=reference,
            )
            return "voxcpm"
        except Exception as exc:
            if not args.fallback_edge:
                raise
            print(f"VoxCPM failed, using Edge-TTS fallback: {exc}")
    elif args.tts_engine == "xtts":
        if not reference:
            raise RuntimeError("XTTS requires the persisted source voice reference")
        ok = await asyncio.to_thread(
            xtts_clone.clone_speak, text, reference, destination,
            args.target_lang, pick_device(),
        )
        if not ok:
            raise RuntimeError("XTTS failed for this chunk")
        return "xtts"
    elif args.tts_engine == "qwen":
        await speak_qwen(
            text, destination, voice_sample=reference,
            language=args.target_lang, device=f"{pick_device()}:0",
        )
        return "qwen"
    voice = args.voice or pick_voice(args.target_lang, args.gender)
    await speak_edge(text, voice, destination, lang=args.target_lang, gender=args.gender)
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
    ap.add_argument("--no-fallback-edge", dest="fallback_edge", action="store_false")
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
        device = pick_device()
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
    store.initialize(
        source={"path": source_path.name, "sha256": source_hash, "duration": source_duration},
        config=config,
        chunks=plans,
    )
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

    failures: list[int] = []
    for chunk in store.data["chunks"]:
        index = int(chunk["index"])
        complete = store.completed_file(index)
        seed_mode_current = chunk.get("seed_vc_mode") == "voice_only_sync_v3"
        if complete and (not args.seed_vc or seed_mode_current):
            print(f"Chunk {index:04d}: restored and validated; skipping")
            continue
        if complete and args.seed_vc and not seed_mode_current:
            print(f"Chunk {index:04d}: outdated Seed-VC timing detected; rebuilding alignment only")
        directory = store.chunk_dir(index)
        write_text_files(store, index)
        attempts = int(chunk.get("attempts", 0)) + 1
        store.update_chunk(index, status="processing", attempts=attempts, error=None)
        try:
            fitted_voice: Path | None = None
            engine_used = "silence"
            if chunk.get("source_text"):
                generated = directory / "generated.wav"
                if not generated.exists() or generated.stat().st_size < 1024:
                    engine_used = await synthesize(args, chunk["translated_text"], generated, reference if reference.exists() else None)
                else:
                    engine_used = chunk.get("engine_used") or args.tts_engine
                trimmed = trim_generated(generated, directory / "generated.trim.wav")
                budget = max(0.12, float(chunk["end"]) - float(chunk["speech_start"]) - 0.03)
                fitted_voice, original_tts_duration, fitted_tts_duration = fit_without_cutting(
                    trimmed, directory / "generated.fitted.wav", budget,
                )
                store.update_chunk(
                    index, status="voice_generated", engine_used=engine_used,
                    original_tts_duration=round(original_tts_duration, 3),
                    fitted_tts_duration=round(fitted_tts_duration, 3),
                    tempo_factor=round(max(original_tts_duration / max(fitted_tts_duration, 0.001), 1.0), 4),
                )
            # Render and preserve the complete pre-Seed voice-only checkpoint.
            raw_voice_audio = build_chunk_audio(
                store.chunk(index), fitted_voice, None,
                directory / "voice-before-seedvc.wav", background_gain=0.0,
            )
            raw_dubbed = render_chunk(
                loaded.video_path, store.chunk(index), raw_voice_audio,
                directory / "dubbed-before-seedvc.mp4",
            )
            final_voice = fitted_voice
            full_timeline = False
            seed_applied = False
            if args.seed_vc and chunk.get("source_text"):
                store.update_chunk(index, status="seed_vc_processing")
                # Keep the original media slice for audit, but use the stable
                # global clean speaker reference and voice-only content for VC.
                slice_source_chunk(loaded.video_path, store.chunk(index), directory / "source.mp4")
                seed_voice = directory / "seedvc.voice-only.wav"
                if not seed_voice.exists() or seed_voice.stat().st_size < 1024:
                    seed_voice = apply_seed_vc_audio(
                        reference, fitted_voice, seed_voice, args.seed_vc_space,
                    )
                speech_target = max(0.12, float(chunk["speech_end"]) - float(chunk["speech_start"]))
                final_voice, seed_original_duration, seed_fitted_duration = match_duration_without_cutting(
                    seed_voice, directory / "seedvc.voice-only.synced.wav", speech_target,
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
            chunk_audio = build_chunk_audio(
                store.chunk(index), final_voice,
                background_audio if args.preserve_background and background_audio.exists() else None,
                directory / "mixed.wav", background_gain=args.background_gain,
                voice_is_full_timeline=full_timeline,
            )
            dubbed = render_chunk(loaded.video_path, store.chunk(index), chunk_audio, directory / "dubbed.mp4")
            store.update_chunk(
                index, status="completed", engine_used=engine_used,
                seed_vc_required=args.seed_vc and bool(chunk.get("source_text")),
                seed_vc_applied=seed_applied,
                dubbed_sha256=sha256_file(dubbed), measured_duration=round(ffprobe_duration(dubbed), 3),
            )
            mirror.upload_chunk(store, index)
            print(f"Chunk {index:04d}: completed and checkpointed")
        except Exception as exc:
            failures.append(index)
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
    )
    atomic_write_json(args.output_dir / "progress-report.json", store.summary())
    shutil.copy2(store.manifest_path, args.output_dir / "checkpoint-manifest.json")
    mirror.upload_manifest(store)
    mirror.upload_final(final)
    print(json.dumps(store.summary(), ensure_ascii=False, indent=2))
    print(f"FINAL={final.resolve()}")


def main() -> None:
    args = parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
