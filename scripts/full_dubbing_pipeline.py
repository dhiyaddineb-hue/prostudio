#!/usr/bin/env python3
"""End-to-end dubbing pipeline: Demucs -> pyannote -> TTS -> Seed-VC -> Wav2Lip.

This is an orchestration script: the heavyweight projects are installed separately
and their official command/API interfaces are called from one reproducible file.
It is designed for consenting speakers and requires a Hugging Face token for
pyannote model access.

Example:
  python scripts/full_dubbing_pipeline.py input.mp4 output.mp4 \
    --source-lang ar --target-lang en --hf-token "$HF_TOKEN" \
    --seed-vc-dir third_party/seed-vc \
    --wav2lip-dir third_party/Wav2Lip \
    --wav2lip-checkpoint checkpoints/wav2lip_gan.pth

Install the orchestration dependencies:
  pip install demucs pyannote.audio faster-whisper edge-tts pydub

Seed-VC and Wav2Lip are intentionally kept as explicit checkout paths because
both projects have their own model checkpoints and version-specific dependencies.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

class PipelineError(RuntimeError):
    pass


def run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=cwd, text=True, check=True)


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        text=True, capture_output=True, check=True,
    )
    return float(result.stdout.strip())


def extract_audio(video: Path, out: Path) -> None:
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "2", "-ar", "44100", str(out)])


def demucs_vocals(audio: Path, work: Path, model: str) -> Path:
    run([sys.executable, "-m", "demucs", "-n", model, "--two-stems=vocals",
         "-o", str(work / "demucs"), str(audio)])
    matches = list((work / "demucs").glob("**/vocals.wav"))
    if not matches:
        raise PipelineError("Demucs completed but vocals.wav was not found")
    return matches[0]


def diarize(vocals: Path, token: str, model: str) -> list[dict[str, Any]]:
    if not token:
        raise PipelineError("pyannote requires --hf-token or HF_TOKEN")
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(model, token=token)
    diarization = pipeline(str(vocals))
    turns: list[dict[str, Any]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if turn.end - turn.start >= 0.20:
            turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
    if not turns:
        raise PipelineError("pyannote found no speech turns")
    return turns


def speaker_at(turns: list[dict[str, Any]], start: float, end: float) -> str:
    midpoint = (start + end) / 2
    candidates = [t for t in turns if t["start"] <= midpoint <= t["end"]]
    if candidates:
        return candidates[0]["speaker"]
    return min(turns, key=lambda t: abs((t["start"] + t["end"]) / 2 - midpoint))["speaker"]


def diarized_reference(vocals: Path, turns: list[dict[str, Any]], work: Path) -> dict[str, Path]:
    from pydub import AudioSegment

    source = AudioSegment.from_file(vocals)
    refs: dict[str, Path] = {}
    for speaker in sorted({t["speaker"] for t in turns}):
        own = [t for t in turns if t["speaker"] == speaker]
        # Prefer several clean turns, capped at 20 seconds for Seed-VC.
        clip = AudioSegment.silent(duration=0, frame_rate=source.frame_rate)
        for turn in sorted(own, key=lambda x: x["end"] - x["start"], reverse=True):
            piece = source[int(turn["start"] * 1000):int(turn["end"] * 1000)]
            clip += piece
            if len(clip) >= 20000:
                break
        if len(clip) < 1000:
            continue
        path = work / f"reference_{speaker}.wav"
        clip[:20000].set_channels(1).export(path, format="wav")
        refs[speaker] = path
    if not refs:
        raise PipelineError("No usable speaker references were created")
    return refs


def transcribe(vocals: Path, model_name: str, source_lang: str) -> list[dict[str, Any]]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="auto", compute_type="int8")
    segments, _ = model.transcribe(str(vocals), language=source_lang, vad_filter=True,
                                   beam_size=5, word_timestamps=False)
    result = []
    for index, segment in enumerate(segments):
        text = segment.text.strip()
        if text:
            result.append({"i": index, "start": float(segment.start),
                           "end": float(segment.end), "text": text})
    if not result:
        raise PipelineError("Whisper produced no transcript")
    return result


async def translate_texts(texts: list[str], source: str, target: str) -> list[str]:
    from youtube_auto_dub.googlev4 import GoogleTranslator

    translator = GoogleTranslator()
    try:
        return await translator.translate_batch(texts, source=source, target=target)
    finally:
        await translator.close()


async def edge_tts(text: str, voice: str, output: Path) -> None:
    import edge_tts
    await edge_tts.Communicate(text, voice).save(str(output))


def seed_convert(seed_dir: Path, source: Path, reference: Path, out_dir: Path,
                 steps: int, length_adjust: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "inference.py", "--source", str(source), "--target", str(reference),
         "--output", str(out_dir), "--diffusion-steps", str(steps),
         "--length-adjust", str(length_adjust), "--inference-cfg-rate", "0.7",
         "--f0-condition", "False", "--auto-f0-adjust", "False", "--fp16", "True"],
        cwd=seed_dir)
    candidates = sorted(out_dir.glob("**/*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise PipelineError(f"Seed-VC produced no WAV for {source.name}")
    return candidates[0]


def assemble_voice(cues: list[dict[str, Any]], audio_paths: dict[int, Path], duration: float,
                   output: Path) -> None:
    timeline = AudioSegment.silent(duration=int(duration * 1000), frame_rate=24000).set_channels(1)
    for cue in cues:
        path = audio_paths.get(cue["i"])
        if not path or not path.exists():
            continue
        clip = AudioSegment.from_file(path).set_frame_rate(24000).set_channels(1)
        slot = max(350, int((cue["end"] - cue["start"]) * 1000))
        if len(clip) > slot:
            clip = clip[:slot]
        timeline = timeline.overlay(clip, position=max(0, int(cue["start"] * 1000)))
    timeline.export(output, format="wav")


def mux_audio(video: Path, audio: Path, output: Path) -> None:
    run(["ffmpeg", "-y", "-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output)])


def wav2lip(wav2lip_dir: Path, checkpoint: Path, face: Path, audio_video: Path,
            output: Path, pads: str) -> None:
    inference = wav2lip_dir / "inference.py"
    if not inference.exists():
        raise PipelineError(f"Wav2Lip inference.py not found in {wav2lip_dir}")
    run([sys.executable, str(inference), "--checkpoint_path", str(checkpoint),
         "--face", str(face), "--audio", str(audio_video), "--outfile", str(output),
         "--pads", pads], cwd=wav2lip_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-lang", default="ar")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--voice", default="en-US-AndrewMultilingualNeural")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""))
    parser.add_argument("--diarization-model", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--demucs-model", default="htdemucs")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--seed-vc-dir", type=Path, required=True)
    parser.add_argument("--wav2lip-dir", type=Path, required=True)
    parser.add_argument("--wav2lip-checkpoint", type=Path, required=True)
    parser.add_argument("--diffusion-steps", type=int, default=40)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--pads", default="0,10,0,0")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input video not found: {args.input}")
    work = args.output.parent / f".{args.output.stem}.pipeline-work"
    if work.exists() and not args.keep_work:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    try:
        print("[1/8] Extracting audio")
        source_audio = work / "source.wav"
        extract_audio(args.input, source_audio)
        print("[2/8] Demucs vocal separation")
        vocals = demucs_vocals(source_audio, work, args.demucs_model)
        print("[3/8] pyannote speaker diarization")
        turns = diarize(vocals, args.hf_token, args.diarization_model)
        references = diarized_reference(vocals, turns, work)
        print(f"    speakers: {', '.join(references)}")
        print("[4/8] Whisper transcription")
        cues = transcribe(vocals, args.whisper_model, args.source_lang)
        for cue in cues:
            cue["speaker"] = speaker_at(turns, cue["start"], cue["end"])
        print("[5/8] Dialogue translation")
        translations = asyncio.run(translate_texts([c["text"] for c in cues], args.source_lang, args.target_lang))
        for cue, translated in zip(cues, translations):
            cue["translated"] = translated.strip() or cue["text"]
        print("[6/8] TTS and Seed-VC voice conversion")
        converted: dict[int, Path] = {}
        for cue in cues:
            raw_tts = work / f"tts_{cue['i']:04d}.mp3"
            asyncio.run(edge_tts(cue["translated"], args.voice, raw_tts))
            reference = references.get(cue["speaker"])
            if not reference:
                print(f"    warning: no reference for {cue['speaker']}; using raw TTS")
                converted[cue["i"]] = raw_tts
                continue
            try:
                converted[cue["i"]] = seed_convert(
                    args.seed_vc_dir, raw_tts, reference, work / "seed" / str(cue["i"]),
                    args.diffusion_steps, args.length_adjust,
                )
            except Exception as exc:
                print(f"    warning: Seed-VC failed on cue {cue['i']}: {exc}; using raw TTS")
                converted[cue["i"]] = raw_tts
        voice_wav = work / "dubbed_voice.wav"
        assemble_voice(cues, converted, ffprobe_duration(args.input), voice_wav)
        print("[7/8] Muxing clean dubbed audio")
        audio_video = work / "dubbed_audio.mp4"
        mux_audio(args.input, voice_wav, audio_video)
        print("[8/8] Wav2Lip lip synchronization")
        wav2lip(args.wav2lip_dir, args.wav2lip_checkpoint, args.input, audio_video, args.output, args.pads)
        manifest = args.output.with_suffix(".json")
        manifest.write_text(json.dumps({"cues": cues, "speakers": sorted(references),
                                        "components": ["demucs", "pyannote", "seed-vc", "wav2lip"]},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Done: {args.output}")
    finally:
        if work.exists() and not args.keep_work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except PipelineError as exc:
        raise SystemExit(f"Pipeline error: {exc}") from exc
