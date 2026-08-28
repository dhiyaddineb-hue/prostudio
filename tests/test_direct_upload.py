"""Tests for uploading a file straight to the studio in parts.

Splitting a file, saving every piece by hand and dragging them back into
GitHub's form is work the machine should do. These endpoints take the parts
directly, and the properties worth pinning are the ones whose failure is
invisible until much later: the reassembled file must be byte-identical, a part
damaged in transit must be refused rather than written, and a filename from the
network must never escape the inbox.
"""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from web.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    import web.app as web_app

    inbox = tmp_path / "inbox"
    temp = tmp_path / "temp"
    inbox.mkdir()
    temp.mkdir()
    monkeypatch.setattr(web_app, "INBOX_DIR", inbox)
    monkeypatch.setattr(web_app, "TEMP_DIR", temp)
    with TestClient(app) as c:
        c.inbox = inbox
        yield c


def send(client, name, index, total, payload, *, sha=True):
    data = {"name": name, "index": str(index), "total": str(total)}
    if sha:
        data["sha256"] = hashlib.sha256(payload).hexdigest()
    return client.post(
        "/api/upload/part",
        files={"chunk": (str(index), payload)},
        data=data,
    )


def test_parts_reassemble_byte_identically(client):
    payload = bytes(range(256)) * 700
    parts = [payload[i:i + 5000] for i in range(0, len(payload), 5000)]

    for i, part in enumerate(parts, 1):
        res = send(client, "clip.mp4", i, len(parts), part)
        assert res.status_code == 200

    assert res.json()["complete"] is True
    assert (client.inbox / "clip.mp4").read_bytes() == payload


def test_parts_may_arrive_out_of_order(client):
    payload = b"".join(bytes([i]) * 100 for i in range(20))
    parts = [payload[i:i + 500] for i in range(0, len(payload), 500)]

    for i in reversed(range(1, len(parts) + 1)):
        res = send(client, "clip.mp4", i, len(parts), parts[i - 1])
        assert res.status_code == 200

    assert (client.inbox / "clip.mp4").read_bytes() == payload


def test_nothing_is_written_until_every_part_arrives(client):
    res = send(client, "clip.mp4", 1, 3, b"first")
    assert res.json()["complete"] is False
    assert not (client.inbox / "clip.mp4").exists()


def test_a_part_damaged_in_transit_is_refused(client):
    res = client.post(
        "/api/upload/part",
        files={"chunk": ("1", b"actual bytes")},
        data={
            "name": "clip.mp4", "index": "1", "total": "1",
            "sha256": hashlib.sha256(b"what was promised").hexdigest(),
        },
    )
    assert res.status_code == 422
    assert not (client.inbox / "clip.mp4").exists()


def test_a_refused_part_can_be_resent(client):
    payload = b"real content"
    bad = client.post(
        "/api/upload/part",
        files={"chunk": ("1", payload)},
        data={"name": "clip.mp4", "index": "1", "total": "1",
              "sha256": hashlib.sha256(b"wrong").hexdigest()},
    )
    assert bad.status_code == 422

    good = send(client, "clip.mp4", 1, 1, payload)
    assert good.status_code == 200
    assert (client.inbox / "clip.mp4").read_bytes() == payload


def test_status_reports_what_has_arrived_so_that_a_resume_can_skip_it(client):
    send(client, "clip.mp4", 1, 3, b"one")
    send(client, "clip.mp4", 3, 3, b"three")

    body = client.get("/api/upload/status/clip.mp4").json()
    assert body["have"] == [1, 3]
    assert body["complete"] is False


def test_status_reports_a_finished_upload(client):
    send(client, "clip.mp4", 1, 1, b"whole file")
    assert client.get("/api/upload/status/clip.mp4").json()["complete"] is True


@pytest.mark.parametrize("hostile", [
    "../escape.bin",
    "../../etc/passwd",
    "/etc/passwd",
    "..\\windows.bin",          # backslash is an ordinary character on POSIX
    "sub/dir/clip.mp4",
])
def test_a_hostile_filename_cannot_escape_the_inbox(client, hostile):
    res = send(client, hostile, 1, 1, b"payload")

    if res.status_code == 200:
        written = client.inbox / res.json()["path"].split("/", 1)[1]
        assert written.parent == client.inbox
        assert ".." not in written.name
    else:
        assert res.status_code == 400

    assert not (client.inbox.parent / "escape.bin").exists()
    assert not (client.inbox.parent / "windows.bin").exists()


@pytest.mark.parametrize("bad", ["", ".", "..", ".hidden"])
def test_empty_and_dot_filenames_are_refused(client, bad):
    assert send(client, bad, 1, 1, b"x").status_code in (400, 422)


def test_an_index_outside_the_declared_range_is_refused(client):
    assert send(client, "clip.mp4", 5, 3, b"x").status_code == 400
    assert send(client, "clip.mp4", 0, 3, b"x").status_code == 400


def test_the_hash_is_optional(client):
    # An older client that does not send one should still work.
    res = send(client, "clip.mp4", 1, 1, b"no hash", sha=False)
    assert res.status_code == 200
    assert (client.inbox / "clip.mp4").read_bytes() == b"no hash"
