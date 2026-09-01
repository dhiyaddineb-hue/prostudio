"""VoxCPM2 speech adapter with two interchangeable backends.

Backend is chosen by the env var YAD_VOXCPM_BACKEND:
  * "local" (default): official openbmb/VoxCPM2 loaded on the runner. Reliable
    reference passing, but slow on CPU.
  * "space": the hosted openbmb/VoxCPM-Demo Gradio Space (GPU on their side, much
    faster). Uses Controllable Cloning: the reference wav is uploaded with
    handle_file() so the timbre is actually applied (the old integration failed
    because the reference was never sent). Busy/timeout/queue failures are
    retried with exponential backoff.

Both backends receive the speaker reference audio for every turn and serialize
generation through a single lock.
"""
from __future__ import annotations
import asyncio
import logging
import os
import subprocess
from pathlib import Path
from youtube_auto_dub.models import SR_TTS, VOICE_MIN_FILE_SIZE

log = logging.getLogger(__name__)

_BACKEND = os.environ.get("YAD_VOXCPM_BACKEND", "local").strip().lower()
_SPACE = os.environ.get("YAD_VOXCPM_SPACE", "openbmb/VoxCPM-Demo").strip()

_MODEL = None
_CLIENT = None
_LOCK = asyncio.Lock()


# ---------------- local backend ----------------
def _load_model():
    global _MODEL
    if _MODEL is None:
        from voxcpm import VoxCPM
        _MODEL = VoxCPM.from_pretrained(
            os.environ.get("YAD_VOXCPM_MODEL", "openbmb/VoxCPM2"),
            load_denoiser=False,
        )
    return _MODEL


def _generate_local(text, dest, control, reference_audio):
    import soundfile as sf
    model = _load_model()
    prompt = text.strip()
    if control and control.strip():
        prompt = f"({control.strip()}){prompt}"
    kwargs = {"text": prompt, "cfg_value": 2.0, "inference_timesteps": 10}
    if reference_audio and Path(reference_audio).exists():
        kwargs["reference_wav_path"] = str(reference_audio)
    wav = model.generate(**kwargs)
    sf.write(str(dest), wav, int(getattr(model.tts_model, "sample_rate", SR_TTS)), subtype="PCM_16")
    if not dest.exists() or dest.stat().st_size < VOICE_MIN_FILE_SIZE:
        raise RuntimeError("VoxCPM2 (local) returned empty audio")


# ---------------- hosted Space backend ----------------
def _space_client():
    global _CLIENT
    if _CLIENT is None:
        from gradio_client import Client
        tok = os.environ.get("HF_TOKEN") or None
        try:
            _CLIENT = Client(_SPACE, hf_token=tok) if tok else Client(_SPACE)
        except TypeError:
            # older/newer gradio_client without hf_token kwarg
            _CLIENT = Client(_SPACE)
    return _CLIENT


def _generate_space(text, dest, control, reference_audio):
    from gradio_client import handle_file
    client = _space_client()
    ref = None
    if reference_audio and Path(reference_audio).exists():
        ref = handle_file(str(reference_audio))
    out = client.predict(
        text.strip(),                 # text_input
        (control or "").strip(),      # control_instruction (emotion/style)
        ref,                          # reference_wav_path_input (timbre source)
        False,                        # use_prompt_text (cross-lingual -> Controllable Cloning)
        "",                           # prompt_text_input
        2.0,                          # cfg_value_input
        True,                         # do_normalize
        True,                         # denoise reference
        api_name="/generate",
    )
    outp = Path(out)
    if not outp.exists() or outp.stat().st_size < VOICE_MIN_FILE_SIZE:
        raise RuntimeError("VoxCPM Space returned empty audio")
    # The Space may return mp3; convert to mono wav at the pipeline sample rate.
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(outp), "-ac", "1", "-ar", str(SR_TTS), str(dest)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not dest.exists() or dest.stat().st_size < VOICE_MIN_FILE_SIZE:
        raise RuntimeError(f"VoxCPM Space audio conversion failed: {proc.stderr[-300:]}")


async def speak_voxcpm(text, dest, language="en", control="", reference_audio=None):
    generate = _generate_space if _BACKEND == "space" else _generate_local
    # Hosted Spaces can transiently reject queued requests. Keep the preferred
    # engine alive through several independent attempts before falling back.
    attempts = 7 if _BACKEND == "space" else 3
    last = None
    async with _LOCK:
        for attempt in range(attempts):
            try:
                await asyncio.to_thread(generate, text, dest, control, reference_audio)
                return
            except Exception as exc:
                last = exc
                dest.unlink(missing_ok=True)
                if _BACKEND == "space" and attempt >= 1:
                    global _CLIENT
                    _CLIENT = None
                log.warning("VoxCPM[%s] attempt %d/%d failed for %s: %s",
                            _BACKEND, attempt + 1, attempts, language, exc)
                if attempt < attempts - 1:
                    await asyncio.sleep(min(6 * (2 ** attempt), 60))
    raise RuntimeError(f"VoxCPM[{_BACKEND}] failed for {language} speech") from last
