"""Optional VoxCPM-Demo client for GitHub Actions dubbing.

The remote Space is rate-limited, so requests are deliberately serialized and
retried with bounded exponential backoff. Seed-VC remains a separate post-process.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from youtube_auto_dub.models import SR_TTS, VOICE_MIN_FILE_SIZE

log = logging.getLogger(__name__)
SPACE = "openbmb/VoxCPM-Demo"
_REQUEST_LOCK = asyncio.Lock()


def _generate_sync(
    text: str,
    dest: Path,
    control: str = "",
    reference_audio: Path | None = None,
) -> None:
    from gradio_client import Client, handle_file

    dest.parent.mkdir(parents=True, exist_ok=True)
    client = Client(SPACE)
    result = client.predict(
        text,
        control,
        handle_file(str(reference_audio)) if reference_audio else None,
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
    control: str = "A natural, clear, warm narrator",
    reference_audio: Path | None = None,
) -> None:
    """Generate one line through VoxCPM without saturating the remote queue."""
    last: Exception | None = None
    async with _REQUEST_LOCK:
        for attempt in range(4):
            try:
                await asyncio.to_thread(
                    _generate_sync, text, dest, control, reference_audio
                )
                return
            except Exception as exc:
                last = exc
                dest.unlink(missing_ok=True)
                delay = min(30, 5 * (2 ** attempt))
                log.warning(
                    "VoxCPM attempt %d/4 failed for %s: %s; retrying in %ss",
                    attempt + 1, language, exc, delay,
                )
                if attempt < 3:
                    await asyncio.sleep(delay)
    raise RuntimeError(f"VoxCPM failed for {language} speech") from last
