from youtube_auto_dub.align_text import split_sentences
from youtube_auto_dub.local_translate import translate_offline


def test_demo_phrase_translation():
    text = "Welcome to ProStudio."
    out = translate_offline(text, source="en", target="ar")
    assert "برو ستوديو" in out


def test_same_language_skips():
    assert translate_offline("hello", source="ar", target="ar") == "hello"


def test_sentence_split():
    parts = split_sentences("Hello. Welcome to ProStudio. Let's begin!")
    assert len(parts) == 3
