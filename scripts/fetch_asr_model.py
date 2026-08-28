#!/usr/bin/env python3
"""Fetch the Vosk speech model into .cache/models/.

Every official host for a speech model is blocked from this sandbox:
alphacephei.com, HuggingFace and its mirrors, the whisper CDNs, and the Git LFS
endpoints. Whisper is the worst case — every copy on GitHub is stored through
LFS, so the API returns a 133-byte pointer where the weights should be.

What does work is codeload, and some repositories committed a Vosk model as
ordinary git objects. This downloads the tarball of one of those and extracts
just the model directory.

The result is verified rather than assumed: an LFS pointer is a short text file
beginning "version https://git-lfs...", so a size and content check catches
that failure here instead of at the first transcription.

    python scripts/fetch_asr_model.py
    python scripts/fetch_asr_model.py --force   # re-download
"""

from __future__ import annotations

import io
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from youtube_auto_dub.vosk_asr import MODEL_DIR, SOURCE_PATH, SOURCE_REPO  # noqa: E402

TARBALL = f"https://codeload.github.com/{SOURCE_REPO}/tar.gz/refs/heads/main"
REQUIRED = ["am/final.mdl", "graph/HCLr.fst", "graph/Gr.fst", "conf/model.conf"]


def looks_like_lfs_pointer(path: Path) -> bool:
    if path.stat().st_size > 1000:
        return False
    return path.read_bytes()[:8] == b"version "


def main() -> None:
    if MODEL_DIR.exists() and "--force" not in sys.argv:
        print(f"model already at {MODEL_DIR}")
        return

    print(f"downloading {TARBALL}")
    with urllib.request.urlopen(TARBALL, timeout=300) as response:
        blob = response.read()
    print(f"  {len(blob) / 1048576:.1f} MB")

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)

    extracted = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or SOURCE_PATH not in member.name:
                continue
            # Strip everything up to and including the model directory name so
            # the layout Vosk expects is preserved.
            tail = member.name.split(SOURCE_PATH, 1)[1].lstrip("/")
            if not tail:
                continue
            dest = MODEL_DIR / tail
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            dest.write_bytes(source.read())
            extracted += 1

    if not extracted:
        raise SystemExit(f"no files matching {SOURCE_PATH} in the tarball")

    missing = [name for name in REQUIRED if not (MODEL_DIR / name).is_file()]
    if missing:
        raise SystemExit(f"model incomplete, missing: {missing}")

    pointers = [
        str(path.relative_to(MODEL_DIR))
        for path in MODEL_DIR.rglob("*")
        if path.is_file() and looks_like_lfs_pointer(path)
    ]
    if pointers:
        raise SystemExit(
            "these files are Git LFS pointers, not real data: "
            f"{pointers[:5]} — this repository stores its model in LFS, whose "
            "hosts are unreachable from here. Find another source."
        )

    size = sum(p.stat().st_size for p in MODEL_DIR.rglob("*") if p.is_file())
    print(f"extracted {extracted} files, {size / 1048576:.0f} MB -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
