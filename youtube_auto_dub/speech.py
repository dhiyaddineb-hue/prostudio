"""Speech-to-text with Whisper — VAD, prompt conditioning, resegmentation."""

import math
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from youtube_auto_dub.ffmpeg_bin import ffmpeg_exe
from youtube_auto_dub.models import (
    SR_WHISPER,
    VAD_GUARD_SECONDS,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
    VAD_THRESHOLD,
    WHISPER_BATCH,
    WHISPER_BEAM,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_DEFAULT_MODEL,
    WHISPER_LOG_PROB_THRESHOLD,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_TEMPERATURES,
    VideoMetadata,
    pick_whisper_compute_type,
)
from youtube_auto_dub.runtime import empty_cuda_cache
from youtube_auto_dub.subs import refine_segments


def build_hint(meta: Optional[VideoMetadata]) -> Optional[str]:
    if not meta or not meta.title:
        return None
    parts = [meta.title]
    if meta.description:
        para = meta.description.split("\n\n")[0].strip()
        parts.append(para[:200])
    if meta.tags:
        parts.append(", ".join(meta.tags[:15]))
    prompt = ". ".join(parts)
    return prompt[:800] if len(prompt) > 800 else prompt


# ── Text cleanup ────────────────────────────────────────────────────────

_PHANTOM = re.compile(r"ترجمة\s+نانسي\s+قنقر")
_REP_CHAR = re.compile(r"(.)\1{2,}")
_REP_WORD = re.compile(r"\b(\S+)(?:\s+\1){2,}\b")


def _scrub(text: str) -> str:
    text = _PHANTOM.sub("", text)
    text = _REP_CHAR.sub(r"\1", text)
    text = _REP_WORD.sub(r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# ── Whisper model cache ─────────────────────────────────────────────────

_MODEL_CACHE = {}


def asr_cache_path(audio: Path) -> Path:
    """Derived 16 kHz mono copy that Whisper actually listens to."""
    audio = Path(audio)
    return audio.with_name(audio.stem + "_16k.wav")


def prepare_asr_audio(audio: Path) -> Path:
    """Return a 16 kHz mono PCM file that reflects the *current* content of ``audio``.

    ``<stem>_16k.wav`` is a derived cache, not a result.  It used to be reused
    whenever it already existed, which silently fed Whisper a stale copy after a
    checkpoint restore (the 24 kHz take was regenerated, the 16 kHz copy was
    not).  The copy is therefore rebuilt on every call, written atomically so a
    concurrent reader never sees a half-written file, and never deleted.
    """
    audio = Path(audio)
    if audio.suffix == ".wav" and _is_16k_mono(audio):
        return audio
    wav = asr_cache_path(audio)
    tmp = wav.with_name(f"{wav.stem}.tmp-{os.getpid()}{wav.suffix}")
    try:
        subprocess.run(
            [ffmpeg_exe(), "-y", "-i", str(audio), "-ac", "1", "-ar", str(SR_WHISPER),
             "-sample_fmt", "s16", "-c:a", "pcm_s16le", str(tmp)],
            check=True, capture_output=True,
        )
        os.replace(tmp, wav)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return wav


def transcribe(
    audio: Path,
    model_name: str = WHISPER_DEFAULT_MODEL,
    device: str = "cpu",
    language: Optional[str] = None,
    hint: Optional[str] = None,
    use_vad: bool = True,
    beam: int = None,
    batch: int = None,
):
    if beam is None:
        beam = WHISPER_BEAM
    if batch is None:
        batch = WHISPER_BATCH

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    ct = pick_whisper_compute_type(device)
    key = f"{model_name}|{device}|{ct}"

    if key not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[key] = WhisperModel(model_name, device=device, compute_type=ct)
        except ValueError:
            ct = "int8" if device == "cpu" else "int8_float16"
            key = f"{model_name}|{device}|{ct}"
            if key not in _MODEL_CACHE:
                _MODEL_CACHE[key] = WhisperModel(model_name, device=device, compute_type=ct)
    model = _MODEL_CACHE[key]

    # Normalise to 16kHz mono WAV. The derived copy is rebuilt on every call so
    # that Whisper always hears the *current* audio (see prepare_asr_audio).
    wav = prepare_asr_audio(audio)

    # VAD
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    sr = model.feature_extractor.sampling_rate
    samples = decode_audio(str(wav), sampling_rate=sr)
    total_samples = samples.shape[0]

    if use_vad:
        clips = get_speech_timestamps(samples, VadOptions(
            max_speech_duration_s=model.feature_extractor.chunk_length,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
            threshold=VAD_THRESHOLD,
        ))
        guard = int(VAD_GUARD_SECONDS * sr)
        if clips:
            if clips[0]["start"] > guard:
                clips.insert(0, {"start": 0, "end": clips[0]["start"]})
            if total_samples - clips[-1]["end"] > guard:
                clips.append({"start": clips[-1]["end"], "end": total_samples})
        else:
            clips = [{"start": 0, "end": total_samples}]
        clip_sec = [{"start": c["start"] / sr, "end": c["end"] / sr} for c in clips]
    else:
        clip_sec = [{"start": 0, "end": total_samples / sr}]

    pipe = BatchedInferencePipeline(model=model)
    kw = dict(
        batch_size=batch, language=language, beam_size=beam,
        word_timestamps=True, clip_timestamps=clip_sec,
        condition_on_previous_text=True,
        temperature=WHISPER_TEMPERATURES,
        compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
    )
    if hint:
        kw["initial_prompt"] = hint

    segs_gen, info = pipe.transcribe(samples, **kw)
    detected = info.language or None

    raw = []
    for seg in segs_gen:
        avg_logprob = float(getattr(seg, "avg_logprob", -10.0))
        d = {
            "start": seg.start, "end": seg.end, "text": _scrub(seg.text.strip()),
            "confidence": max(0.0, min(1.0, math.exp(avg_logprob))),
            "no_speech_prob": float(getattr(seg, "no_speech_prob", 0.0)),
        }
        if seg.words:
            d["words"] = [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words]
        raw.append(d)

    del pipe
    if device == "cuda":
        empty_cuda_cache()

    refined = refine_segments(raw)
    # Preserve model confidence after word-level resegmentation.
    for item in refined:
        parents = [x for x in raw if min(float(item["end"]), float(x["end"])) > max(float(item["start"]), float(x["start"]))]
        if parents:
            parent = max(parents, key=lambda x: min(float(item["end"]), float(x["end"])) - max(float(item["start"]), float(x["start"])))
            item["confidence"] = parent.get("confidence", 0.0)
            item["no_speech_prob"] = parent.get("no_speech_prob", 0.0)
    return refined, detected


def _is_16k_mono(path: Path) -> bool:
    try:
        res = subprocess.run(
            [ffmpeg_exe(), "-i", str(path)],
            capture_output=True, text=True,
        )
        text = (res.stderr or "") + (res.stdout or "")
        return f"{SR_WHISPER} Hz" in text and re.search(r"Audio:.*mono", text) is not None
    except Exception:
        return False
