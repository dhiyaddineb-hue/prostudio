from youtube_auto_dub.youtube import _URL_RE, _extract_metadata


def test_url_detection():
    assert _URL_RE.match("https://www.youtube.com/watch?v=abc")
    assert _URL_RE.match("http://youtu.be/abc")
    assert not _URL_RE.match("/tmp/video.mp4")
    assert not _URL_RE.match("video.mp4")


def test_extract_metadata_tags_string():
    meta = _extract_metadata({
        "title": "Hello",
        "description": "World",
        "tags": "one, two",
        "duration": 12.5,
        "channel": "Studio",
    })
    assert meta.title == "Hello"
    assert meta.tags == ["one", "two"]
    assert meta.duration == 12.5
