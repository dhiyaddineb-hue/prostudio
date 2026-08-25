"""The studio voice bank must match lines robustly and fail soft."""

from pathlib import Path

import pytest

from youtube_auto_dub import studio_tts


def test_normalize_ignores_diacritics_and_orthography():
    a = studio_tts.normalize("مرحباً بكم في برو ستوديو.")
    b = studio_tts.normalize("مرحبا بكم فى برو ستوديو")
    assert a == b


def test_normalize_collapses_whitespace_and_punctuation():
    assert studio_tts.normalize("  أولاً،   نُفرّغ   الكلام!  ") == studio_tts.normalize(
        "اولا نفرغ الكلام"
    )


def test_key_is_stable_and_short():
    key = studio_tts.key_for("ثم نترجم المعنى إلى العربية.")
    assert key == studio_tts.key_for("ثم نترجم المعنى الى العربيه")
    assert len(key) == 16


def test_bundled_demo_lines_are_covered():
    """The five demo lines ship with approved takes."""
    lines = [
        "مرحباً بكم في برو ستوديو.",
        "يعرض هذا الفيلم القصير الدبلجة الآلية للفيديو.",
        "أولاً نُفرّغ الكلام.",
        "ثم نترجم المعنى إلى العربية.",
        "وأخيراً نولّد صوتاً جديداً ونزامنه مع الصورة.",
    ]
    have, total = studio_tts.coverage(lines)
    assert have == total == 5


def test_unknown_line_has_no_take(tmp_path):
    assert studio_tts.lookup("جملة غير مسجلة إطلاقاً في البنك") is None
    assert studio_tts.speak_studio(
        "جملة غير مسجلة إطلاقاً في البنك", tmp_path / "out.wav"
    ) is False


def test_missing_file_is_not_returned(monkeypatch):
    """An index entry pointing at a deleted file must not be used."""
    monkeypatch.setattr(
        studio_tts,
        "load_index",
        lambda: {studio_tts.key_for("مفقود"): {"text": "مفقود", "file": "nope.mp3"}},
    )
    assert studio_tts.lookup("مفقود") is None


def test_speak_studio_renders_known_line(tmp_path):
    dest = tmp_path / "line.wav"
    assert studio_tts.speak_studio("مرحباً بكم في برو ستوديو.", dest) is True
    assert dest.exists() and dest.stat().st_size > 1000
