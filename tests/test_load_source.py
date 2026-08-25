"""An explicit source must never be hijacked by a stray inbox/ file."""

import pytest

from youtube_auto_dub import youtube


@pytest.fixture
def fake_inbox(monkeypatch, tmp_path):
    """Pretend inbox/ always holds a leftover upload."""
    stale = tmp_path / "stale.mp4"
    stale.write_bytes(b"x" * 2048)
    monkeypatch.setattr(youtube, "_inbox_video", lambda: stale)
    return stale


@pytest.fixture
def spies(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        youtube, "import_local_video", lambda p, *a, **k: calls.setdefault("local", p)
    )
    monkeypatch.setattr(
        youtube, "download_project", lambda u, *a, **k: calls.setdefault("url", u)
    )
    return calls


def test_explicit_path_beats_inbox(fake_inbox, spies):
    youtube.load_source("samples/prostudio_en.mp4")
    assert spies.get("local") == "samples/prostudio_en.mp4"
    assert "url" not in spies


def test_youtube_url_beats_inbox(fake_inbox, spies):
    youtube.load_source("https://www.youtube.com/watch?v=abc123")
    assert spies.get("url") == "https://www.youtube.com/watch?v=abc123"
    assert "local" not in spies


def test_inbox_used_only_when_source_blank(fake_inbox, spies):
    youtube.load_source("   ")
    assert spies.get("local") == fake_inbox


def test_blank_source_without_inbox_raises(monkeypatch, spies):
    monkeypatch.setattr(youtube, "_inbox_video", lambda: None)
    with pytest.raises(ValueError):
        youtube.load_source("")
