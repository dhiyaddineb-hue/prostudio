from youtube_auto_dub.pipeline_args import build_args
from youtube_auto_dub.voice import pick_voice


def test_build_args_arabic_defaults():
    args = build_args("https://youtube.com/watch?v=abc")
    assert args.lang == "ar"
    assert args.mode == "both"
    assert args.preserve_bg is True
    assert args.edge_voice is None
    assert args.whisper_model


def test_build_args_edge_voice():
    args = build_args("https://youtube.com/watch?v=abc", voice="ar-SA-HamedNeural")
    assert args.edge_voice == "ar-SA-HamedNeural"
    assert args.voice_theme is None


def test_build_args_arabic_source_english_dub():
    args = build_args(
        "video.mp4",
        lang="en",
        source_lang="ar",
        mode="dub",
    )
    assert args.lang == "en"
    assert args.lang_dub == "en"
    assert args.source_lang == "ar"


def test_arabic_preferred_voices():
    assert pick_voice("ar", "male") == "ar-SA-HamedNeural"
    assert pick_voice("ar", "female") == "ar-EG-SalmaNeural"


def test_explicit_voice_override():
    assert pick_voice("ar", "male", voice="ar-EG-ShakirNeural") == "ar-EG-ShakirNeural"
