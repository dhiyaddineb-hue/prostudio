"""Tests for rejoining browser-split uploads.

Splitting exists so a large video can clear GitHub's 25 MB form limit without
losing the picture. That promise only holds if the join is exact, so the join
verifies rather than trusts: a rebuilt file is kept only when every part
matches its recorded hash and the total size is right.

These pin the three refusals. Delivering a silently corrupt video that fails
three stages later is far worse than stopping here with a clear reason.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json

import pytest

SPEC = importlib.util.spec_from_file_location(
    "join_parts",
    __import__("pathlib").Path(__file__).resolve().parent.parent
    / "scripts" / "join_parts.py",
)
join_parts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(join_parts)


@pytest.fixture
def inbox(tmp_path, monkeypatch):
    folder = tmp_path / "inbox"
    folder.mkdir()
    monkeypatch.setattr(join_parts, "INBOX", folder)
    return folder


def write_split(folder, name, payload, part_size, *, manifest=True):
    chunks = [payload[i:i + part_size] for i in range(0, len(payload), part_size)]
    total = len(chunks)
    for index, chunk in enumerate(chunks, 1):
        (folder / f"{name}.part{index:02d}of{total:02d}").write_bytes(chunk)
    if manifest:
        (folder / f"{name}.parts.json").write_text(json.dumps({
            "original": name,
            "size": len(payload),
            "parts": total,
            "part_bytes": part_size,
            "sha256": [hashlib.sha256(c).hexdigest() for c in chunks],
        }), encoding="utf-8")
    return total


def test_rejoined_file_is_byte_identical(inbox):
    payload = bytes(range(256)) * 400
    write_split(inbox, "clip.mp4", payload, 10_000)

    groups = join_parts.find_groups(inbox)
    out = join_parts.join("clip.mp4", groups["clip.mp4"], keep=False)

    assert out is not None
    assert out.read_bytes() == payload


def test_parts_are_removed_once_joined(inbox):
    write_split(inbox, "clip.mp4", b"x" * 5000, 1000)
    join_parts.join("clip.mp4", join_parts.find_groups(inbox)["clip.mp4"], keep=False)
    assert not list(inbox.glob("*.part??of??"))
    assert not (inbox / "clip.mp4.parts.json").exists()


def test_keep_leaves_the_parts_alone(inbox):
    write_split(inbox, "clip.mp4", b"x" * 5000, 1000)
    join_parts.join("clip.mp4", join_parts.find_groups(inbox)["clip.mp4"], keep=True)
    # "*.part*" would also catch clip.mp4.parts.json; match the real pattern.
    assert len(list(inbox.glob("*.part??of??"))) == 5
    assert (inbox / "clip.mp4.parts.json").exists()


def test_a_single_flipped_byte_is_refused(inbox):
    payload = bytes(range(256)) * 100
    write_split(inbox, "clip.mp4", payload, 5000)

    victim = inbox / "clip.mp4.part02of06"
    data = bytearray(victim.read_bytes())
    data[10] ^= 0xFF
    victim.write_bytes(data)

    out = join_parts.join("clip.mp4", join_parts.find_groups(inbox)["clip.mp4"], keep=True)
    assert out is None
    assert not (inbox / "clip.mp4").exists()


def test_a_missing_part_is_refused(inbox):
    write_split(inbox, "clip.mp4", b"y" * 5000, 1000)
    (inbox / "clip.mp4.part03of05").unlink()

    out = join_parts.join("clip.mp4", join_parts.find_groups(inbox)["clip.mp4"], keep=True)
    assert out is None
    assert not (inbox / "clip.mp4").exists()


def test_size_mismatch_is_refused_and_cleaned_up(inbox):
    # Hashes are consistent but the manifest claims a different total, which is
    # what a truncated final part looks like.
    write_split(inbox, "clip.mp4", b"z" * 3000, 1000)
    path = inbox / "clip.mp4.parts.json"
    data = json.loads(path.read_text())
    data["size"] = 999999
    path.write_text(json.dumps(data), encoding="utf-8")

    out = join_parts.join("clip.mp4", join_parts.find_groups(inbox)["clip.mp4"], keep=True)
    assert out is None
    # A half-written file left behind would be picked up as a real upload.
    assert not (inbox / "clip.mp4").exists()


def test_joining_works_without_a_manifest(inbox):
    payload = b"abc" * 2000
    write_split(inbox, "clip.mp4", payload, 1000, manifest=False)

    out = join_parts.join("clip.mp4", join_parts.find_groups(inbox)["clip.mp4"], keep=False)
    assert out is not None and out.read_bytes() == payload


def test_ordinary_files_are_not_mistaken_for_parts(inbox):
    (inbox / "holiday.mp4").write_bytes(b"whole file")
    (inbox / "notes.part.txt").write_bytes(b"not a part")
    assert join_parts.find_groups(inbox) == {}


def test_ten_or_more_parts_join_in_numeric_order(inbox):
    # Zero padding is what keeps part 10 after part 9 in a sorted listing.
    payload = bytes(range(256)) * 50
    total = write_split(inbox, "clip.mp4", payload, 500)
    assert total >= 10

    out = join_parts.join("clip.mp4", join_parts.find_groups(inbox)["clip.mp4"], keep=False)
    assert out is not None and out.read_bytes() == payload
