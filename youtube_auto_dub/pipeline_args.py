"""Helpers for building pipeline arguments from the CLI or the web studio."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Optional

from youtube_auto_dub.models import DEFAULT_GENDER, DEFAULT_TTS_ENGINE, WHISPER_DEFAULT_MODEL


def build_args(
    url: str,
    *,
    lang: str = "ar",
    mode: str = "both",
    gender: str = DEFAULT_GENDER,
    sub_lang: Optional[str] = None,
    dub_lang: Optional[str] = None,
    model: Optional[str] = None,
    browser: Optional[str] = None,
    tts_engine: str = DEFAULT_TTS_ENGINE,
    voice: Optional[str] = None,
    voice_clone: bool = False,
    no_tempo: bool = False,
    no_vad: bool = False,
    bg_music: bool = True,
    output_dir: Optional[str] = None,
) -> Namespace:
    args = Namespace(
        url=url,
        lang=lang,
        sub_lang=sub_lang,
        dub_lang=dub_lang,
        mode=mode,
        gender=gender,
        model=model or WHISPER_DEFAULT_MODEL,
        browser=browser,
        tts_engine=tts_engine,
        voice=voice,
        voice_clone=voice_clone,
        no_tempo=no_tempo,
        no_vad=no_vad,
        bg_music=bg_music,
        output_dir=output_dir,
    )
    args.lang_sub = sub_lang
    args.lang_dub = dub_lang
    if tts_engine == "qwen":
        args.voice_theme = voice
        args.edge_voice = None
    else:
        args.voice_theme = None
        args.edge_voice = voice
    args.auto_clone = voice_clone
    args.preserve_bg = bg_music
    args.whisper_model = args.model
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    return args
