"""Runtime capability probing — optional heavy deps, device pick, offline checks.

The studio must boot on a plain CPU box with no ``torch`` and no network. Every
heavy dependency (torch, faster-whisper, librosa) is therefore probed lazily and
reported instead of crashing at import time.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import ssl
from functools import lru_cache
from pathlib import Path


def have_module(name: str) -> bool:
    """True when ``name`` is importable without actually importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def pick_device() -> str:
    """Return ``cuda`` when a usable GPU is present, otherwise ``cpu``.

    ``torch`` is optional: without it we simply run on CPU.
    """
    if not have_module("torch"):
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def empty_cuda_cache() -> None:
    """Free GPU memory when torch/CUDA are actually in play. Never raises."""
    if not have_module("torch"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def whisper_cached_models() -> list[str]:
    """Whisper model names whose weights are already on disk.

    faster-whisper downloads from HuggingFace on first use, so an importable
    package proves nothing: on an offline box the import succeeds and the very
    first transcription fails. Only a cached snapshot means it will work.
    """
    found: list[str] = []
    roots = [
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    seen: set[Path] = set()
    for root in roots:
        if root in seen or not root.is_dir():
            continue
        seen.add(root)
        for entry in root.glob("models--*aster-whisper*"):
            snapshots = entry / "snapshots"
            if not snapshots.is_dir():
                continue
            # A usable snapshot carries the actual weights, not just refs.
            if any(
                any(rev.glob("model*.bin")) or any(rev.glob("*.safetensors"))
                for rev in snapshots.iterdir()
                if rev.is_dir()
            ):
                found.append(entry.name.split("--")[-1])
    return sorted(set(found))


def have_whisper() -> bool:
    """True only when transcription can actually run right now.

    Requires both the package and either cached weights or a reachable hub to
    fetch them from.
    """
    if not have_module("faster_whisper"):
        return False
    if whisper_cached_models():
        return True
    return huggingface_reachable()


@lru_cache(maxsize=1)
def huggingface_reachable() -> bool:
    """Whether model weights can still be downloaded."""
    return host_reachable("huggingface.co")


def have_espeak() -> bool:
    """Offline TTS fallback availability (binary or bundled shared library)."""
    root = Path(__file__).resolve().parent.parent
    if (root / ".local" / "bin" / "espeak-ng").exists():
        return True
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        return True
    return have_module("espeakng_loader")


def studio_takes() -> int:
    """Number of approved professional voice takes available offline."""
    try:
        from youtube_auto_dub.studio_tts import load_index

        return len(load_index())
    except Exception:
        return 0


def host_reachable(host: str, port: int = 443, timeout: float = 2.0) -> bool:
    """Reachability probe that completes a real TLS handshake.

    A bare TCP connect is not enough: filtering proxies routinely accept the
    socket and then drop the TLS session, which would make an offline sandbox
    look online.
    """
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host):
                return True
    except (OSError, ssl.SSLError):
        return False


@lru_cache(maxsize=1)
def edge_tts_reachable() -> bool:
    """Edge-TTS streams from Microsoft's speech endpoint."""
    return host_reachable("speech.platform.bing.com")


def capabilities() -> dict:
    """Snapshot of what this machine can actually do, for /api/health."""
    from youtube_auto_dub.ffmpeg_bin import ffmpeg_exe

    try:
        ffmpeg = bool(ffmpeg_exe())
    except Exception:
        ffmpeg = False

    edge = edge_tts_reachable()
    espeak = have_espeak()
    whisper = have_whisper()
    studio = studio_takes()
    cached = whisper_cached_models()
    return {
        "ffmpeg": ffmpeg,
        "device": pick_device(),
        "torch": have_module("torch"),
        "whisper": whisper,
        "whisper_installed": have_module("faster_whisper"),
        "whisper_models_cached": cached,
        "model_hub": huggingface_reachable(),
        "edge_tts": edge,
        "espeak": espeak,
        "studio_voices": studio,
        # A dub needs *some* way to make speech.
        "can_dub": ffmpeg and (edge or espeak or studio > 0),
        # Without Whisper the user must paste a transcript.
        "needs_transcript": not whisper,
    }
