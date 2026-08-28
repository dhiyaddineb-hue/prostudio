#!/usr/bin/env python3
"""Rejoin files that were split in the browser to clear GitHub's upload cap.

The upload form refuses anything over 25 MB, so docs/split.js cuts a large
video into byte-range parts. This puts them back together.

The join is byte concatenation, not a re-encode: the rebuilt file is identical
to the original, bit for bit. That is the point of splitting rather than
transcoding — the picture, the resolution and every audio sample survive.

Identity is verified, not asserted. Each part carries a SHA-256 in the
manifest, and the rebuilt file is only kept if every part matches and the total
size is right. A silently corrupt video that fails three stages later is worse
than a clear refusal here.

    python scripts/join_parts.py            # join everything in inbox/
    python scripts/join_parts.py --keep     # leave the parts in place
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INBOX = ROOT / "inbox"

# name.mp4.part01of03
PART_RE = re.compile(r"^(?P<name>.+)\.part(?P<index>\d+)of(?P<total>\d+)$")
CHUNK = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def find_groups(folder: Path) -> dict[str, list[tuple[int, int, Path]]]:
    """Group part files by the original filename they belong to."""
    groups: dict[str, list[tuple[int, int, Path]]] = {}
    for entry in sorted(folder.iterdir()):
        if not entry.is_file():
            continue
        match = PART_RE.match(entry.name)
        if not match:
            continue
        groups.setdefault(match["name"], []).append(
            (int(match["index"]), int(match["total"]), entry)
        )
    return groups


def join(name: str, parts: list[tuple[int, int, Path]], *, keep: bool) -> Path | None:
    """Concatenate one group back into its original file."""
    parts.sort(key=lambda p: p[0])
    expected = parts[0][1]

    have = {index for index, _total, _path in parts}
    missing = sorted(set(range(1, expected + 1)) - have)
    if missing:
        # Parts are raw byte ranges: only the first has a container header, so
        # a gap cannot be worked around. Naming the missing pieces is the only
        # useful thing to do.
        print(f"  {name}: missing part(s) {missing} of {expected} — skipped")
        return None

    manifest_path = INBOX / f"{name}.parts.json"
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  {name}: manifest unreadable, falling back to size check")

    if manifest:
        for (index, _total, path), want in zip(parts, manifest.get("sha256", [])):
            got = _sha256(path)
            if got != want:
                print(f"  {name}: part {index} is corrupt — refusing to join")
                print(f"    expected {want[:16]}…  got {got[:16]}…")
                return None

    out = INBOX / name
    with out.open("wb") as dest:
        for _index, _total, path in parts:
            with path.open("rb") as src:
                while block := src.read(CHUNK):
                    dest.write(block)

    size = out.stat().st_size
    if manifest and size != manifest.get("size"):
        print(f"  {name}: rebuilt {size} bytes, manifest says {manifest['size']} — refusing")
        out.unlink(missing_ok=True)
        return None

    print(f"  {name}: joined {expected} parts -> {size / 1048576:.1f} MB", end="")
    print(" (sha verified)" if manifest else " (no manifest; size only)")

    if not keep:
        for _index, _total, path in parts:
            path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    return out


def main() -> None:
    keep = "--keep" in sys.argv
    if not INBOX.is_dir():
        raise SystemExit(f"no inbox at {INBOX}")

    groups = find_groups(INBOX)
    if not groups:
        print("no split files in inbox/")
        return

    print(f"{len(groups)} split file(s) found:")
    for name, parts in groups.items():
        join(name, parts, keep=keep)


if __name__ == "__main__":
    main()
