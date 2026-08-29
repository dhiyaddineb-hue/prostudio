"""Optional VoxCPM-Demo client for GitHub Actions dubbing.

The remote Space generates the target-language speech; Seed-VC can then be
used as a separate post-process to transfer the source speaker's timbre.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from youtube_auto_dub.models import SR_TTS, VOICE_MIN_FILE_SIZE

log = logging.getLogger(__name__)
SPACE = "openbmb/VoxCPM-Demo"


def _generate_sync(
    text: str,
    dest: Path,
    control: str = "",
    reference_audio: Path | None = None,
) -> None:
    from gradio_client import Client

    dest.parent.mkdir(parents=True, exist_ok=True)
    client = Client(SPACE)
    # Inputs mirror the Space's public api_name="generate":
    # target text, control instruction, reference audio, ultimate cloning,
    # reference transcript, CFG, normalize text, denoise reference.
    result = client.predict(
        text,
        control,
        str(reference_audio) if reference_audio else None,
        False,
        "",
        2.0,
        True,
        bool(reference_audio),
        api_name="/generate",
    )
    path = result.get("path") if isinstance(result, dict) else result
    if isinstance(path, (list, tuple)):
        path = path[0]
    if not path:
        raise RuntimeError(f"VoxCPM returned no audio path: {result!r}")

    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32")
    sf.write(str(dest), audio, int(sr or SR_TTS), subtype="PCM_16")
    if not dest.exists() or dest.stat().st_size < VOICE_MIN_FILE_SIZE:
        raise RuntimeError("VoxCPM returned an empty audio file")


async def speak_voxcpm(
    text: str,
    dest: Path,
    language: str = "en",
    control: str = "A natural, clear, warm English documentary narrator",
    reference_audio: Path | None = None,
) -> None:
    """Generate one line with VoxCPM-Demo, with bounded retry and clean output."""
    last: Exception | None = None
    for attempt in range(2):
        try:
            await asyncio.to_thread(
                _generate_sync, text, dest, control, reference_audio
            )
            return
        except Exception as exc:
            last = exc
            log.warning("VoxCPM attempt %d failed: %s", attempt + 1, exc)
            dest.unlink(missing_ok=True)
            if attempt == 0:
                await asyncio.sleep(3)
    raise RuntimeError(f"VoxCPM failed for {language} speech") from last
