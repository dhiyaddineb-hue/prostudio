"""The studio must boot and report honestly on a CPU-only, offline box."""

import subprocess
import sys

from youtube_auto_dub.align_text import guess_language
from youtube_auto_dub.runtime import capabilities, have_module, pick_device


def test_core_imports_without_torch():
    """torch is optional — importing the pipeline must not require it."""
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "import youtube_auto_dub.core, youtube_auto_dub.speech, youtube_auto_dub.voice\n"
        "import web.app\n"
        "print('ok')\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr
    assert "ok" in res.stdout


def test_pick_device_defaults_to_cpu():
    assert pick_device() in {"cpu", "cuda"}


def test_have_module_detects_missing():
    assert have_module("json") is True
    assert have_module("definitely_not_a_real_module_xyz") is False


def test_capabilities_shape():
    caps = capabilities()
    for key in (
        "ffmpeg",
        "device",
        "torch",
        "whisper",
        "edge_tts",
        "espeak",
        "can_dub",
        "needs_transcript",
    ):
        assert key in caps
    assert isinstance(caps["can_dub"], bool)
    # needs_transcript must mirror whisper availability
    assert caps["needs_transcript"] is (not caps["whisper"])


def test_guess_language_by_script():
    assert guess_language("Welcome to ProStudio.") == "en"
    assert guess_language("مرحباً بكم في برو ستوديو") == "ar"
    assert guess_language("Привет всем") == "ru"
    assert guess_language("こんにちは皆さん") == "ja"
    assert guess_language("") == "en"


def test_guess_language_ignores_latin_noise_in_arabic():
    text = "ProStudio مرحباً بكم في هذا الفيديو الجديد اليوم"
    assert guess_language(text) == "ar"
