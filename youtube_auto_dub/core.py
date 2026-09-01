"""Core pipeline — orchestrates download → transcribe → translate → speak → assemble → render."""

import asyncio
import json
import logging
import os
import re
import subprocess
from pathlib import Path

from rich.table import Table

from youtube_auto_dub.align_text import guess_language
from youtube_auto_dub.translate_fit import adapt_length
from youtube_auto_dub.audio import (
    align_segments,
    finalize_audio,
    group_segments,
    fixed_window_segments,
    overlay_dub,
    render_video,
    write_srt,
)
from youtube_auto_dub.googlev4 import GoogleTranslator
from youtube_auto_dub.models import (
    AUDIO_DEFAULT_AMBIENT_GAIN,
    DEFAULT_TTS_ENGINE,
    SR_TTS,
    TEMP_DIR,
    WHISPER_DEFAULT_MODEL,
    SubtitleSegment,
)
from youtube_auto_dub.offline_asr import available as offline_asr_available
from youtube_auto_dub.offline_asr import transcribe_offline
from youtube_auto_dub.runtime import have_whisper, pick_device
from youtube_auto_dub.speech import build_hint, transcribe
from youtube_auto_dub.subs import read_srt
from youtube_auto_dub.ui import console
from youtube_auto_dub.voice import (
    auto_clone_voice,
    pick_voice,
    resolve_persona,
    speak_edge,
    speak_qwen,
)
from youtube_auto_dub.voxcpm_tts import speak_voxcpm
from youtube_auto_dub.emotion import infer_emotion
from youtube_auto_dub.speaker_diarization import annotate_segments
from youtube_auto_dub.source_separation import separate_dialogue_background, validate_stems
from youtube_auto_dub import xtts_clone
from youtube_auto_dub.youtube import load_source

log = logging.getLogger(__name__)


def _polish_english_dialogue(text: str) -> str:
    """Make machine translation sound conversational without changing meaning."""
    replacements = (
        (r"\bI am\b", "I'm"), (r"\bI have\b", "I've"),
        (r"\bdo not\b", "don't"), (r"\bdoes not\b", "doesn't"),
        (r"\bcannot\b", "can't"), (r"\bit is\b", "it's"),
        (r"\byou are\b", "you're"), (r"\bwe are\b", "we're"),
        (r"\bthat is\b", "that's"), (r"\bthere is\b", "there's"),
    )
    polished = text.strip()
    for pattern, replacement in replacements:
        polished = re.sub(pattern, replacement, polished, flags=re.IGNORECASE)
    return polished


def _asr_timeline_metrics(raw: list[dict], duration: float) -> tuple[float, float]:
    """Return (covered ratio, longest gap) for ASR segments."""
    spans = sorted(
        (max(0.0, float(x.get("start", 0.0))), min(duration, float(x.get("end", 0.0))))
        for x in raw if float(x.get("end", 0.0)) > float(x.get("start", 0.0))
    )
    if not spans or duration <= 0:
        return 0.0, max(duration, 0.0)
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    covered = sum(end - start for start, end in merged) / duration
    gaps = [merged[0][0], max(0.0, duration - merged[-1][1])]
    gaps.extend(merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1))
    return covered, max(gaps + [0.0])


def _asr_candidate_score(raw: list[dict], duration: float) -> float:
    coverage, max_gap = _asr_timeline_metrics(raw, duration)
    confidence = sum(float(x.get("confidence", 0.0)) for x in raw) / max(len(raw), 1)
    no_speech = sum(float(x.get("no_speech_prob", 0.0)) for x in raw) / max(len(raw), 1)
    gap_penalty = min(max_gap / max(duration, 1.0), 1.0)
    return coverage + 0.65 * confidence - 0.35 * no_speech - 0.5 * gap_penalty


def _report(progress, step: str, message: str, percent: int) -> None:
    if progress:
        progress(step, message, percent)


async def run(args, progress=None) -> Path:
    base_lang = args.lang or "en"
    sub_lang = args.lang_sub or base_lang
    dub_lang = args.lang_dub or base_lang
    out_root = Path(args.output_dir) if getattr(args, "output_dir", None) else Path("output")
    device = pick_device()
    model_name = args.whisper_model or WHISPER_DEFAULT_MODEL

    tts_engine = getattr(args, "tts_engine", DEFAULT_TTS_ENGINE)
    use_tempo = not getattr(args, "no_tempo", False)
    keep_bg = getattr(args, "preserve_bg", False)
    do_clone = getattr(args, "auto_clone", False)
    persona = getattr(args, "voice_theme", None)

    # ── UI ────────────────────────────────────────────────────────────
    console.header("YouTube Auto Dub")
    console.header("Configuration", center=False)
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_row("URL", f"[#e5e7eb]{args.url}[/#e5e7eb]")
    t.add_row("Mode", f"[#e5e7eb]{args.mode.upper()}[/#e5e7eb]")
    if args.mode in ("sub", "both"):
        t.add_row("Subs", f"[#e5e7eb]{sub_lang.upper()}[/#e5e7eb]")
    if args.mode in ("dub", "both"):
        t.add_row("Dub", f"[#e5e7eb]{dub_lang.upper()}[/#e5e7eb]")
        t.add_row("Gender", f"[#e5e7eb]{args.gender.title()}[/#e5e7eb]")
    t.add_row("ASR", f"[#e5e7eb]{model_name.upper()} ({device.upper()})[/#e5e7eb]")
    t.add_row("TTS", f"[#e5e7eb]{tts_engine.upper()}[/#e5e7eb]")
    if persona:
        t.add_row("Persona", f"[#e5e7eb]{persona}[/#e5e7eb]")
    if do_clone:
        t.add_row("Clone", "[#e5e7eb]ON[/#e5e7eb]")
    if keep_bg:
        t.add_row("Ambient", "[#e5e7eb]ON[/#e5e7eb]")
    console.print(t)
    console.print()

    with console.status("Processing..."):
        # ── 1. Download ──────────────────────────────────────────────
        console.info("Downloading media")
        _report(progress, "download", "تنزيل الفيديو / Download", 8)
        project = load_source(args.url, getattr(args, "browser", None))
        _report(progress, "download", f"تم التحميل: {project.video_id}", 16)

        # Separate before ASR/diarization so music does not contaminate the
        # transcript or the voice reference. Keep the original as a safe
        # fallback when Demucs is unavailable or fails validation.
        speech_audio = project.audio_path
        background_audio = None
        if getattr(args, "separate_sources", False):
            console.info("Separating dialogue and background (Demucs)")
            source_work = TEMP_DIR / "source-separation"
            stems = separate_dialogue_background(project.audio_path, source_work)
            source_duration = float(project.metadata.duration) if project.metadata else 0.0
            stem_report = validate_stems(stems, source_duration)
            if stem_report.get("valid"):
                speech_audio, background_audio = stems
                console.step("Validated Demucs speech/background stems")
            else:
                console.warning(f"Demucs stems rejected; using original audio: {stem_report.get('reason')}")

        # ── 2. Transcribe ────────────────────────────────────────────
        console.info(f"Transcribing ({model_name})")
        _report(progress, "transcribe", f"تفريغ الصوت ({model_name})", 22)

        cached = project.load_cache("segments")
        transcript_source = "asr"
        transcript = (getattr(args, "transcript", None) or "").strip()
        if transcript:
            transcript_source = "provided"
            from youtube_auto_dub.align_text import segments_from_transcript
            console.step("Using provided transcript")
            duration = float(project.metadata.duration) if project.metadata else 0.0
            project.segments = segments_from_transcript(
                transcript, project.audio_path, duration=duration
            )
            lang_detected = getattr(args, "source_lang", None) or guess_language(
                transcript
            )
            console.step(f"Source language: {lang_detected}")
        elif not str(args.url).startswith(("http://", "https://")) and Path(args.url).with_suffix(".srt").exists():
            sidecar = Path(args.url).with_suffix(".srt")
            transcript_source = "sidecar"
            console.step(f"Using trusted sidecar transcript: {sidecar.name}")
            entries = read_srt(str(sidecar))
            project.segments = [
                SubtitleSegment(
                    start=float(item["start"]), end=float(item["end"]),
                    source_text=item["text"].strip(), confidence=1.0, index=i,
                )
                for i, item in enumerate(entries) if item.get("text", "").strip()
            ]
            if not project.segments:
                raise RuntimeError(f"Sidecar transcript is empty: {sidecar}")
            lang_detected = getattr(args, "source_lang", None) or guess_language(
                " ".join(seg.source_text for seg in project.segments)
            )
        elif cached and not getattr(args, "refresh", False):
            transcript_source = "cache"
            console.step("Using cached transcription")
            project.segments = [
                SubtitleSegment(
                    start=s["start"], end=s["end"],
                    source_text=s["source_text"],
                    speaker=s.get("speaker"),
                    confidence=float(s.get("confidence", 1.0)),
                    index=i,
                )
                for i, s in enumerate(cached)
            ]
            lang_detected = cached[0].get("lang")
        else:
            raw = lang_detected = None
            if have_whisper():
                hint = build_hint(project.metadata)
                if hint:
                    console.step("Prompting with video metadata")
                try:
                    requested_source = getattr(args, "source_lang", None)
                    forced_language = None if requested_source in (None, "", "auto") else requested_source
                    use_vad = not getattr(args, "no_vad", False)
                    raw, lang_detected = transcribe(
                        speech_audio,
                        model_name=model_name,
                        device=device,
                        language=forced_language,
                        hint=hint,
                        use_vad=use_vad,
                    )
                    source_duration = float(project.metadata.duration) if project.metadata else 0.0
                    asr_audio = speech_audio
                    if speech_audio != project.audio_path:
                        console.step("Cross-checking ASR against the original audio track")
                        original_raw, original_lang = transcribe(
                            project.audio_path,
                            model_name=model_name,
                            device=device,
                            language=forced_language,
                            hint=hint,
                            use_vad=use_vad,
                        )
                        stem_score = _asr_candidate_score(raw, source_duration)
                        original_score = _asr_candidate_score(original_raw, source_duration)
                        console.step(f"ASR candidates: separated={stem_score:.3f}, original={original_score:.3f}")
                        if original_score > stem_score:
                            raw, lang_detected, asr_audio = original_raw, original_lang or lang_detected, project.audio_path
                            console.step("Original track selected for transcription; separated stem remains the voice reference")
                    coverage, max_gap = _asr_timeline_metrics(raw, source_duration)
                    console.step(f"ASR timeline: {coverage:.1%} span coverage; longest gap {max_gap:.2f}s")
                    # VAD can mistake low-energy Arabic speech for silence.  If it
                    # creates a suspicious hole, rerun once without VAD and keep
                    # only the objectively better timeline.
                    if use_vad and source_duration > 0 and max_gap > 4.0:
                        console.warning("Large ASR gap detected; retrying transcription without VAD")
                        recovered, recovered_lang = transcribe(
                            asr_audio,
                            model_name=model_name,
                            device=device,
                            language=forced_language,
                            hint=hint,
                            use_vad=False,
                        )
                        recovered_coverage, recovered_gap = _asr_timeline_metrics(recovered, source_duration)
                        if (recovered_gap, -recovered_coverage) < (max_gap, -coverage):
                            raw, lang_detected = recovered, recovered_lang or lang_detected
                            coverage, max_gap = recovered_coverage, recovered_gap
                            console.step(f"Recovered ASR timeline: {coverage:.1%}; longest gap {max_gap:.2f}s")
                    console.success(f"Detected: {lang_detected}")
                except Exception as exc:
                    console.warning(f"Whisper failed ({exc})")
                    raw = None

            if raw is None:
                # Whisper is missing or could not fetch its weights. Rather than
                # refuse to transcribe, fall back to the fully offline
                # recogniser — clearly announced, since it is English-only and
                # less accurate.
                requested_source = getattr(args, "source_lang", None)
                if requested_source != "en":
                    raise RuntimeError(
                        "لا يمكن كشف وتفريغ لغة الصوت تلقائياً في هذه البيئة. "
                        "الصق نص الفيديو في خانة النص أو وفّر نموذج Whisper متاحاً."
                    )
                if not offline_asr_available():
                    raise RuntimeError(
                        "تعذر التفريغ الآلي. الصق نص الفيديو في خانة النص، "
                        "أو ثبّت pocketsphinx للتفريغ دون إنترنت."
                    )
                console.warning(
                    "Using offline PocketSphinx (English only, lower accuracy)"
                )
                _report(progress, "transcribe", "تفريغ محلي دون إنترنت", 28)
                raw = transcribe_offline(project.audio_path)
                if not raw:
                    raise RuntimeError(
                        "تعذّر التعرف على أي كلام. الصق نص الفيديو في خانة النص."
                    )
                lang_detected = "en"

            if getattr(args, "diarize", False):
                console.info("Identifying speakers")
                raw = annotate_segments(speech_audio, raw)
            console.info("Grouping segments")
            if os.environ.get("YAD_FIXED_WINDOWS", "false").lower() == "true":
                project.segments = fixed_window_segments(
                    raw,
                    float(project.metadata.duration) if project.metadata else 0.0,
                    float(os.environ.get("YAD_FIXED_WINDOW_SECONDS", "6.0")),
                )
            else:
                project.segments = group_segments(raw)

            cache_data = [
                {"index": i, "start": s.start, "end": s.end,
                 "source_text": s.source_text, "lang": lang_detected,
                 "speaker": s.speaker, "confidence": s.confidence}
                for i, s in enumerate(project.segments)
            ]
            project.save_cache("segments", cache_data)

        texts = [seg.source_text for seg in project.segments]
        _report(progress, "chunk", f"تقسيم إلى {len(project.segments)} مقطع", 38)

        # ── 3. Translate ─────────────────────────────────────────────
        target_sidecar_entries = None
        if not str(args.url).startswith(("http://", "https://")):
            target_sidecar = Path(args.url).with_suffix(f".{dub_lang}.srt")
            if target_sidecar.exists():
                candidate_entries = read_srt(str(target_sidecar))
                if len(candidate_entries) == len(project.segments):
                    target_sidecar_entries = candidate_entries
                    console.step(f"Using trusted target transcript: {target_sidecar.name}")
                else:
                    console.warning(f"Target sidecar cue count does not match source: {target_sidecar}")
        console.info("Translating")
        _report(progress, "translate", f"ترجمة إلى {dub_lang if args.mode != 'sub' else sub_lang}", 44)

        if args.mode in ("sub", "both"):
            if target_sidecar_entries is not None and sub_lang == dub_lang:
                sub_out = [item["text"] for item in target_sidecar_entries]
            elif lang_detected and lang_detected == sub_lang:
                console.step(f"Source == target ({sub_lang}), skipping")
                sub_out = texts
            else:
                console.step(f"Translating {len(texts)} segs -> {sub_lang.upper()}")
                xl = GoogleTranslator()
                sub_out = await xl.translate_batch(
                    texts, source=lang_detected or "auto", target=sub_lang
                )
                await xl.close()
            for i, seg in enumerate(project.segments):
                seg.translated_text_sub = sub_out[i].strip() or seg.source_text

        if args.mode in ("dub", "both"):
            if target_sidecar_entries is not None:
                dub_out = [item["text"] for item in target_sidecar_entries]
            elif args.mode == "both" and dub_lang == sub_lang:
                console.step("Reusing subtitle translation for dubbing")
                dub_out = sub_out
            elif lang_detected and lang_detected == dub_lang:
                console.step(f"Source == target ({dub_lang}), skipping")
                dub_out = texts
            else:
                console.step(f"Translating {len(texts)} segs -> {dub_lang.upper()}")
                xl = GoogleTranslator()
                dub_out = await xl.translate_batch(
                    texts, source=lang_detected or "auto", target=dub_lang
                )
                await xl.close()
            for i, seg in enumerate(project.segments):
                translated = dub_out[i].strip() or seg.source_text
                seg.translated_text_dub = (
                    _polish_english_dialogue(translated)
                    if dub_lang.lower() in ("en", "en-us", "english")
                    else translated
                )
                if getattr(args, "adapt_translation", True):
                    seg.translated_text_dub = adapt_length(
                        seg.translated_text_dub,
                        max(float(seg.end) - float(seg.start), 0.1),
                        dub_lang,
                    )

        console.success("Translation done")

        # ── 4. Speech synthesis ───────────────────────────────────────
        if args.mode in ("dub", "both"):
            console.info(f"Synthesizing ({tts_engine})")
            _report(progress, "tts", "توليد الصوت العصبي", 58)

            # Resolve voice source
            sample = ref_txt = None
            xtts_reference = None
            if tts_engine == "voxcpm":
                console.step("VoxCPM2: generating clean target-language speech")
                # VoxCPM-Demo rejects references longer than 50 seconds. Build
                # one bounded, denoised reference from the dialogue stem.
                voxcpm_reference = TEMP_DIR / "voxcpm_reference.wav"
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(speech_audio), "-t", "45",
                    "-af", "highpass=f=80,lowpass=f=9000,afftdn=nr=12",
                    "-ar", "22050", "-ac", "1", str(voxcpm_reference),
                ], check=True, capture_output=True)

                # When diarization is reliable, do not feed every speaker the
                # same voice reference. Pick each speaker's first real turn,
                # pad it slightly, and clean it independently. If a turn is
                # unavailable, retain the conservative global reference.
                speaker_references = {}
                speaker_turns = {}
                for seg in project.segments:
                    speaker = getattr(seg, "speaker", None)
                    if speaker:
                        previous = speaker_turns.get(speaker)
                        if previous is None or (seg.end - seg.start) > (previous.end - previous.start):
                            speaker_turns[speaker] = seg
                for seg in sorted(speaker_turns.values(), key=lambda item: item.start):
                    speaker = getattr(seg, "speaker", None)
                    if not speaker:
                        continue
                    ref = TEMP_DIR / f"voxcpm_reference_{re.sub(r'[^A-Za-z0-9_-]', '_', str(speaker))}.wav"
                    start = max(0.0, float(seg.start) - 0.25)
                    length = min(15.0, max(2.5, float(seg.end) - float(seg.start) + 0.5))
                    subprocess.run([
                        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(speech_audio),
                        "-t", f"{length:.3f}", "-af",
                        "highpass=f=80,lowpass=f=9000,afftdn=nr=12,dynaudnorm=f=150:g=7",
                        "-ar", "22050", "-ac", "1", str(ref),
                    ], check=True, capture_output=True)
                    if ref.exists() and ref.stat().st_size > 1024:
                        speaker_references[speaker] = ref
            else:
                voxcpm_reference = None
                speaker_references = {}

            if tts_engine == "xtts":
                # XTTS needs a few seconds of the original speaker. Use the
                # first available speech window, not the translated text, so
                # the cloned timbre is genuine and language-independent.
                import soundfile as sf

                source_audio, source_sr = sf.read(speech_audio, dtype="float32")
                if getattr(source_audio, "ndim", 1) > 1:
                    source_audio = source_audio.mean(axis=1)
                ref_end = min(len(source_audio) / source_sr, xtts_clone.REF_MAX_SEC)
                xtts_reference = xtts_clone.write_reference(
                    source_audio, source_sr, 0.0, ref_end,
                    TEMP_DIR / "xtts_reference.wav",
                )
                if not xtts_reference or not xtts_clone.available():
                    console.warning("XTTS-v2 unavailable; falling back to Edge-TTS")
                    tts_engine = "edge"
                speaker_references = {}
                if xtts_reference and project.segments:
                    speaker_turns = {}
                    for seg in project.segments:
                        speaker = getattr(seg, "speaker", None)
                        if speaker:
                            previous = speaker_turns.get(speaker)
                            if previous is None or (seg.end - seg.start) > (previous.end - previous.start):
                                speaker_turns[speaker] = seg
                    for speaker, seg in speaker_turns.items():
                        speaker_audio, speaker_sr = sf.read(speech_audio, dtype="float32")
                        if getattr(speaker_audio, "ndim", 1) > 1:
                            speaker_audio = speaker_audio.mean(axis=1)
                        start = max(0.0, float(seg.start) - 0.25)
                        end = min(len(speaker_audio) / speaker_sr, float(seg.end) + 0.5)
                        ref = xtts_clone.write_reference(
                            speaker_audio, speaker_sr, start, end,
                            TEMP_DIR / f"xtts_reference_{re.sub(r'[^A-Za-z0-9_-]', '_', str(speaker))}.wav",
                        )
                        if ref:
                            speaker_references[speaker] = ref

            if tts_engine == "qwen" and do_clone:
                srt_path = TEMP_DIR / "ref.srt"
                write_srt(project.segments, srt_path)
                entries = read_srt(str(srt_path))
                sample = auto_clone_voice(project.audio_path, entries,
                                          project.project_dir / "clone")

            if tts_engine == "qwen" and persona and not sample:
                sample, ref_txt = resolve_persona(
                    persona, dub_lang, device=f"{device}:0",
                )
                console.step(f"Persona: {persona}")

            async def synth_xtts_or_edge(seg):
                ok = await asyncio.to_thread(
                    xtts_clone.clone_speak,
                    seg.translated_text_dub,
                    speaker_references.get(getattr(seg, "speaker", None), xtts_reference),
                    seg.tts_audio_path,
                    dub_lang,
                    device,
                )
                if not ok:
                    voice = pick_voice(
                        dub_lang, args.gender,
                        voice=getattr(args, "edge_voice", None),
                    )
                    await speak_edge(
                        seg.translated_text_dub, voice, seg.tts_audio_path,
                        lang=dub_lang, gender=args.gender,
                    )

            # Generate TTS per fixed/timed window. Remote failures are handled
            # per window so one failed request cannot erase later speech.
            async def synth_vox_resilient(seg, dest):
                console.info(
                    f"WINDOW {seg.start:.3f}-{seg.end:.3f}s | engine=VoxCPM | text={seg.translated_text_dub[:90]!r}"
                )
                try:
                    await speak_voxcpm(
                        seg.translated_text_dub,
                        dest,
                        language=dub_lang,
                        control=(
                            (getattr(args, "voice_theme", None) or "A natural, clear, warm narrator")
                            + "; delivery: " + infer_emotion(seg.translated_text_dub)
                        ),
                        reference_audio=speaker_references.get(
                            getattr(seg, "speaker", None), voxcpm_reference
                        ),
                    )
                    console.success(f"WINDOW {seg.start:.3f}-{seg.end:.3f}s | used=VoxCPM | status=success")
                except Exception as exc:
                    console.warning(f"VoxCPM exhausted retries for window {seg.start:.2f}s; trying XTTS: {exc}")
                    xtts_ref = speaker_references.get(
                        getattr(seg, "speaker", None), voxcpm_reference
                    )
                    xtts_ok = False
                    if xtts_ref and xtts_clone.available():
                        xtts_ok = await asyncio.to_thread(
                            xtts_clone.clone_speak,
                            seg.translated_text_dub,
                            xtts_ref,
                            dest,
                            dub_lang,
                            device,
                        )
                    if xtts_ok:
                        console.success(f"WINDOW {seg.start:.3f}-{seg.end:.3f}s | used=XTTS | status=success")
                    if not xtts_ok:
                        console.warning(f"WINDOW {seg.start:.2f}s | XTTS failed; using Edge-TTS as final fallback")
                        voice = pick_voice(dub_lang, args.gender, voice=getattr(args, "edge_voice", None))
                        await speak_edge(seg.translated_text_dub, voice, dest, lang=dub_lang, gender=args.gender)
                        console.success(f"WINDOW {seg.start:.3f}-{seg.end:.3f}s | used=Edge-TTS | status=final-fallback")

            tasks = []
            for i, seg in enumerate(project.segments):
                seg.tts_audio_path = TEMP_DIR / f"tts_{i}.wav"
                if tts_engine == "xtts" and xtts_reference:
                    tasks.append(synth_xtts_or_edge(seg))
                elif tts_engine == "voxcpm":
                    tasks.append(synth_vox_resilient(seg, seg.tts_audio_path))
                elif tts_engine == "qwen":
                    tasks.append(speak_qwen(
                        seg.translated_text_dub, seg.tts_audio_path,
                        voice_sample=Path(sample) if sample else None,
                        ref_text=ref_txt, language=dub_lang,
                        device=f"{device}:0",
                    ))
                else:
                    voice = pick_voice(
                        dub_lang,
                        args.gender,
                        voice=getattr(args, "edge_voice", None),
                    )
                    tasks.append(speak_edge(
                        seg.translated_text_dub, voice, seg.tts_audio_path,
                        lang=dub_lang, gender=args.gender,
                    ))

            await asyncio.gather(*tasks)

            missing = [
                (i, round(seg.start, 3), seg.source_text[:60])
                for i, seg in enumerate(project.segments)
                if seg.source_text.strip() and (
                    not seg.tts_audio_path
                    or not seg.tts_audio_path.exists()
                    or seg.tts_audio_path.stat().st_size < 1024
                )
            ]
            if missing:
                raise RuntimeError(f"Audio coverage validation failed for windows: {missing}")

            # VoxCPM may return a noticeable leading pause and trailing room
            # tone. Remove only silence around each utterance before fitting it
            # into the original speaker turn; otherwise every line sounds late.
            for seg in project.segments:
                if not seg.tts_audio_path or not seg.tts_audio_path.exists():
                    continue
                trimmed = seg.tts_audio_path.with_name(seg.tts_audio_path.stem + "_trim.wav")
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(seg.tts_audio_path),
                    "-af", "silenceremove=start_periods=1:start_duration=0.08:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_duration=0.12:start_threshold=-45dB,areverse",
                    "-ar", str(SR_TTS), "-ac", "1", str(trimmed),
                ], check=True, capture_output=True)
                if trimmed.exists() and trimmed.stat().st_size > 1024:
                    seg.tts_audio_path = trimmed

            # ── 5. Assemble & finalise ────────────────────────────────
            _report(progress, "mix", "المزامنة ومزج الموسيقى", 78)
            project.dub_audio_path = TEMP_DIR / "dub_final.wav"

            if use_tempo:
                info_list = []
                for seg in project.segments:
                    tts_dur = 0.0
                    if seg.tts_audio_path and seg.tts_audio_path.exists():
                        import soundfile as sf
                        tts_dur = len(sf.read(seg.tts_audio_path, dtype="float32")[0]) / SR_TTS
                    info_list.append({
                        "start": seg.start,
                        "target_dur": max(seg.duration, 0.35),
                        "wav_path": seg.tts_audio_path,
                    })

                src_dur = float(project.metadata.duration) if project.metadata else 0.0
                raw_mix = TEMP_DIR / "dub_raw.wav"
                align_segments(info_list, src_dur, raw_mix)
                ambient_gain = getattr(
                    args, "ambient_gain", AUDIO_DEFAULT_AMBIENT_GAIN
                )
                finalize_audio(
                    raw_mix, speech_audio,
                    project.dub_audio_path,
                    match_loudness=True,
                    mix_ambient=keep_bg,
                    ambient_gain=ambient_gain if keep_bg else 0.0,
                    # The working audio is mono for Whisper; centre separation
                    # needs the original stereo mix.
                    stereo_source=project.video_path,
                    ambient_source=background_audio,
                )
            else:
                overlay_dub(project.audio_path, project.segments,
                            project.dub_audio_path)

            console.success("Dubbing complete")

        # ── 6. Subtitles ──────────────────────────────────────────────
        if args.mode in ("sub", "both"):
            console.info("Writing subtitles")
            project.subtitle_path = TEMP_DIR / "subtitles.srt"
            write_srt(project.segments, project.subtitle_path)
            console.success("Subtitles saved")

        # ── 7. Render ─────────────────────────────────────────────────
        console.info("Rendering video")
        _report(progress, "render", "تصدير الفيديو النهائي", 90)
        info = f"L-{base_lang}"
        if args.lang_sub:
            info += f"_S-{sub_lang}"
        if args.lang_dub:
            info += f"_D-{dub_lang}"
        # Keep the container honest: an audio-only source produces audio.
        from youtube_auto_dub.audio import _has_video_stream
        ext = "mp4" if _has_video_stream(project.video_path) else "mp3"
        out = out_root / f"Output_{args.mode}_{info}_{project.video_id}.{ext}"

        render_video(
            video_path=project.video_path,
            subtitle_path=None,
            dub_audio_path=project.dub_audio_path if args.mode in ("dub", "both") else None,
            output_path=out,
        )
        console.success("Video rendered")
        project.output_path = out

        # Persist the real timing/text contract next to the deliverable.  Quality
        # gates and the studio must inspect what was actually rendered instead of
        # inferring coverage from a successful process exit code.
        segment_report = out_root / "segments-report.json"
        segment_report.write_text(json.dumps({
            "source_duration": float(project.metadata.duration) if project.metadata else 0.0,
            "transcript_source": transcript_source,
            "source_language": lang_detected,
            "target_language": dub_lang,
            "segments": [
                {
                    "index": i,
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "source_text": seg.source_text,
                    "translated_text": seg.translated_text_dub,
                    "speaker": seg.speaker,
                    "confidence": float(seg.confidence),
                }
                for i, seg in enumerate(project.segments)
            ],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        _report(progress, "done", "اكتمل الدوبلاج", 100)

    console.print()
    console.print(f"[bold #38bdf8]Output: {out.resolve()}[/bold #38bdf8]")
    return out
