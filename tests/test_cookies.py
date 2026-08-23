from youtube_auto_dub.youtube import _normalize_cookie_text


def test_header_cookies_become_netscape():
    raw = "SID=abc; HSID=def; APISID=ghi"
    out = _normalize_cookie_text(raw)
    assert "youtube.com" in out
    assert "\tSID\t" in out
    assert "Netscape" in out


def test_escaped_newlines():
    raw = "# Netscape HTTP Cookie File\\n.youtube.com\\tTRUE\\t/\\tTRUE\\t0\\tSID\\tx"
    out = _normalize_cookie_text(raw)
    assert "\n.youtube.com\tTRUE" in out
