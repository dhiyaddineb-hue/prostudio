"""Official VoxCPM2 local inference adapter.

Loads openbmb/VoxCPM2 once per runner and serializes generation. Reference audio
is passed to the model for every speaker turn; no remote Gradio demo is used.
"""
from __future__ import annotations
import asyncio
import logging
import os
from pathlib import Path
from youtube_auto_dub.models import SR_TTS, VOICE_MIN_FILE_SIZE
log = logging.getLogger(__name__)
_MODEL = None
_LOCK = asyncio.Lock()
def _load_model():
    global _MODEL
    if _MODEL is None:
        from voxcpm import VoxCPM
        _MODEL = VoxCPM.from_pretrained(
            os.environ.get("YAD_VOXCPM_MODEL", "openbmb/VoxCPM2"),
            load_denoiser=False,
        )
    return _MODEL
def _generate_sync(text, dest, control, reference_audio):
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
        raise RuntimeError("VoxCPM2 returned empty audio")
async def speak_voxcpm(text, dest, language="en", control="", reference_audio=None):
    last = None
    async with _LOCK:
        for attempt in range(2):
            try:
                await asyncio.to_thread(_generate_sync, text, dest, control, reference_audio)
                return
            except Exception as exc:
                last = exc; dest.unlink(missing_ok=True)
                log.warning("VoxCPM2 attempt %d/2 failed for %s: %s", attempt + 1, language, exc)
                if attempt == 0: await asyncio.sleep(3)
    raise RuntimeError(f"VoxCPM2 failed for {language} speech") from last
