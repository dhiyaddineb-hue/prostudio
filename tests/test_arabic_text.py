"""Arabic diacritisation must help pronunciation without ever breaking a dub."""

import pytest

from youtube_auto_dub import arabic_text
from youtube_auto_dub.arabic_text import (
    add_diacritics,
    diacritic_ratio,
    is_arabic,
    prepare,
)


def test_detects_arabic():
    assert is_arabic("ليه مش متجوز؟") is True
    assert is_arabic("Hello world") is False
    assert is_arabic("") is False


def test_diacritic_ratio():
    assert diacritic_ratio("راجل") == 0.0
    assert diacritic_ratio("رَاجِل") > 0.3
    assert diacritic_ratio("hello") == 0.0


def test_non_arabic_is_untouched():
    assert prepare("Hello world") == "Hello world"
    assert prepare("") == ""


def test_already_diacritised_text_is_left_alone():
    marked = "أَنَا رَاجِلٌ"
    assert prepare(marked) == marked


def test_missing_model_returns_the_input(monkeypatch):
    """A dub must never fail because the diacritiser is absent."""
    monkeypatch.setattr(arabic_text, "_diacritiser", lambda: None)
    text = "ليه مش متجوز؟"
    assert add_diacritics(text) == text


def test_model_failure_returns_the_input(monkeypatch):
    class Broken:
        def diacritize(self, _text):
            raise RuntimeError("boom")

    monkeypatch.setattr(arabic_text, "_diacritiser", lambda: Broken())
    text = "ليه مش متجوز؟"
    assert add_diacritics(text) == text


def test_altered_letters_are_rejected(monkeypatch):
    """If the model rewrites words rather than vowel-marking them, don't use it."""
    class Rewriter:
        def diacritize(self, _text):
            return "كلام مختلف تماما"

    monkeypatch.setattr(arabic_text, "_diacritiser", lambda: Rewriter())
    text = "ليه مش متجوز؟"
    assert add_diacritics(text) == text


@pytest.mark.skipif(not arabic_text.available(), reason="piper-tts not installed")
def test_real_diacritisation_adds_marks_and_keeps_letters():
    text = "أنا راجل اتخلقت للعزوبية"
    out = prepare(text)
    assert out != text
    assert diacritic_ratio(out) > 0.3
    # The consonants must survive: only vowel marks are added.
    import re
    strip = re.compile(r"[\u064b-\u0652\u0670]")
    assert strip.sub("", out).replace(" ", "") == text.replace(" ", "")


@pytest.mark.skipif(not arabic_text.available(), reason="piper-tts not installed")
def test_studio_takes_still_match_after_diacritisation():
    """The voice bank keys on normalised text, which ignores vowel marks."""
    from youtube_auto_dub.studio_tts import normalize

    text = "مرحباً بكم في برو ستوديو."
    assert normalize(text) == normalize(prepare(text))
