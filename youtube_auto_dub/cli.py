#!/usr/bin/env python3
"""CLI — global-standard flags, launch pipeline."""

import argparse
import asyncio
import shutil

from youtube_auto_dub import __version__
from youtube_auto_dub.core import run
from youtube_auto_dub.models import DEFAULT_GENDER, DEFAULT_TTS_ENGINE, TEMP_DIR
from youtube_auto_dub.ui import console


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="youtube-auto-dub",
        description="Translate, subtitle, and dub any YouTube video automatically",
    )

    # ── Positional ────────────────────────────────────────────────────
    p.add_argument("url", help="YouTube video URL")

    # ── Language ──────────────────────────────────────────────────────
    lang = p.add_argument_group("Language")
    lang.add_argument("-l", "--lang", default="ar",
                      help="Target language code (default: ar)")
    lang.add_argument("-s", "--sub-lang",
                      help="Subtitle language (overrides --lang)")
    lang.add_argument("-d", "--dub-lang",
                      help="Dubbing language (overrides --lang)")
    lang.add_argument("--source-lang", default="auto",
                      help="Source audio language (default: auto)")


    # ── Mode ──────────────────────────────────────────────────────────
    mode = p.add_argument_group("Mode")
    mode.add_argument("-m", "--mode", choices=["sub", "dub", "both"], default="both",
                      help="Processing mode (default: both)")
    mode.add_argument("-g", "--gender", choices=["male", "female"], default=DEFAULT_GENDER,
                      help=f"Voice gender (default: {DEFAULT_GENDER})")

    # ── Model ─────────────────────────────────────────────────────────
    mdl = p.add_argument_group("Model")
    mdl.add_argument("--model", "--whisper",
                     help="Whisper model size: tiny, base, small, medium, large (default: auto)")
    mdl.add_argument("-b", "--browser",
                     help="Browser for cookie auth: chrome, edge, firefox")

    # ── Engine ────────────────────────────────────────────────────────
    eng = p.add_argument_group("Engine")
    eng.add_argument("-e", "--tts-engine", choices=["edge", "qwen"], default=DEFAULT_TTS_ENGINE,
                     help=f"TTS engine (default: {DEFAULT_TTS_ENGINE})")
    eng.add_argument("--voice",
                     help="Edge-TTS voice (e.g. ar-SA-HamedNeural) or Qwen persona")
    eng.add_argument("--voice-clone", action="store_true",
                     help="Auto-clone voice from source audio (Qwen3-TTS only)")

    # ── Processing ────────────────────────────────────────────────────
    proc = p.add_argument_group("Processing")
    proc.add_argument("--no-tempo", action="store_true",
                      help="Disable tempo alignment (paste at original speed)")
    proc.add_argument("--no-vad", action="store_true",
                      help="Disable voice-activity detection")
    proc.add_argument("--bg-music", action="store_true",
                      help="Mix original background audio into dub")
    proc.add_argument("-o", "--output-dir",
                      help="Output directory (default: ./output)")

    # ── Info ──────────────────────────────────────────────────────────
    p.add_argument("--version", action="version",
                   version=f"youtube-auto-dub v{__version__}")

    return p


def main():
    p = _parser()
    args = p.parse_args()

    # Normalise internal attribute names for core.py
    args.lang_sub = args.sub_lang
    args.lang_dub = args.dub_lang
    # --voice is an Edge-TTS ShortName unless the Qwen engine is selected
    if args.tts_engine == "qwen":
        args.voice_theme = args.voice
        args.edge_voice = None
    else:
        args.voice_theme = None
        args.edge_voice = args.voice
    args.auto_clone = args.voice_clone
    args.preserve_bg = args.bg_music
    args.whisper_model = args.model

    # Clean temp
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise SystemExit(1)
