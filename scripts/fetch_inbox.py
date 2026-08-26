#!/usr/bin/env python3
"""Pull files uploaded to inbox/ on GitHub into the local checkout.

Attachments on issues are stored on S3, which is unreachable from this
sandbox — a file dropped into an issue arrives as a name and a size and
nothing else. Files committed to the repository come down through the Git
blobs API instead, which does work here.

So the send page points at GitHub's own upload form for ``inbox/``, and this
script fetches whatever landed there:

    python scripts/fetch_inbox.py               # list what is waiting
    python scripts/fetch_inbox.py --pull        # download new files
    python scripts/fetch_inbox.py --pull --project  # …and start a project
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPO = "dhiyaddineb-hue/prostudio"
BRANCH = "arena/01a03969-prostudio"
MEDIA = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi",
         ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


def _gh(path: str) -> str:
    res = subprocess.run(
        ["gh", "api", path], capture_output=True, text=True, timeout=180
    )
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip()[:200] or "gh api failed")
    return res.stdout


def listing() -> list[dict]:
    """Media files currently sitting in the repository's inbox/."""
    raw = _gh(f"repos/{REPO}/contents/inbox?ref={BRANCH}")
    return [
        item for item in json.loads(raw)
        if item.get("type") == "file"
        and Path(item["name"]).suffix.lower() in MEDIA
    ]


def fetch(item: dict, dest_dir: Path) -> Path:
    """Download one file by blob sha.

    The contents endpoint returns an empty body above ~1 MB; the blobs endpoint
    carries the bytes whatever the size.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / item["name"]
    raw = _gh(f"repos/{REPO}/git/blobs/{item['sha']}")
    payload = json.loads(raw)
    data = base64.b64decode(payload["content"].replace("\n", ""))
    if not data:
        raise RuntimeError(f"{item['name']}: empty blob")
    dest.write_bytes(data)
    if dest.stat().st_size != item["size"]:
        raise RuntimeError(
            f"{item['name']}: got {dest.stat().st_size} bytes, expected {item['size']}"
        )
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch uploads from inbox/ on GitHub")
    ap.add_argument("--pull", action="store_true", help="download new files")
    ap.add_argument("--project", action="store_true",
                    help="create a dub project for each new file")
    args = ap.parse_args()

    try:
        items = listing()
    except Exception as exc:
        raise SystemExit(f"Could not read inbox/: {exc}")

    if not items:
        print("inbox/ is empty — nothing waiting.")
        return

    inbox = ROOT / "inbox"
    print(f"{len(items)} file(s) in inbox/:\n")
    for item in items:
        local = inbox / item["name"]
        have = local.exists() and local.stat().st_size == item["size"]
        mb = item["size"] / 1048576
        print(f"  {'✓' if have else '↓'} {item['name']}  ({mb:.1f} MB)")

        if not args.pull or have:
            continue
        try:
            path = fetch(item, inbox)
            print(f"      downloaded to {path.relative_to(ROOT)}")
        except Exception as exc:
            print(f"      FAILED: {exc}")
            continue

        if args.project:
            from youtube_auto_dub.project_dirs import create

            stem = Path(item["name"]).stem[:40]
            project = create(stem, title=stem, lang="ar", dialect="eg").ensure_dirs()
            target = project.source_dir / item["name"]
            target.write_bytes(path.read_bytes())
            project.source_name = item["name"]
            project.save()
            print(f"      project: projects/{project.slug}/")

    if not args.pull:
        print("\nRun with --pull to download, or --pull --project to start a dub.")


if __name__ == "__main__":
    main()
