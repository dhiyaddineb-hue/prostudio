#!/usr/bin/env python3
"""Transcribe a clip and save the result where it will survive a reset.

The first full transcription of the Bob Proctor talk — 132 segments, 4619
words, two minutes of compute — was written to /tmp and lost when the sandbox
was wiped. Anything worth two minutes of compute is worth committing, so
transcripts land in ``transcripts/`` as JSON and are tracked in git.

The JSON keeps per-word timings, which is the part that matters for dubbing: a
line can be cut where a phrase actually ends rather than at a guess from
character count.

    python scripts/transcribe.py inbox/clip.mp4
    python scripts/transcribe.py inbox/clip.mp4 --force   # redo an existing one
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from youtube_auto_dub.vosk_asr import available, transcribe  # noqa: E402

OUT_DIR = ROOT / "transcripts"


def slug(name: str) -> str:
    text = unicodedata.normalize("NFKC", name)
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"[\s_]+", "-", text).strip("-")[:60] or "clip"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit("usage: transcribe.py <media file> [--force]")

    source = Path(args[0])
    if not source.is_file():
        raise SystemExit(f"no such file: {source}")

    if not available():
        raise SystemExit(
            "Vosk model missing — run: python scripts/fetch_asr_model.py"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / f"{slug(source.stem)}.json"
    if dest.exists() and "--force" not in sys.argv:
        existing = json.loads(dest.read_text(encoding="utf-8"))
        print(f"already transcribed: {dest} ({len(existing['segments'])} segments)")
        print("pass --force to redo it")
        return

    print(f"transcribing {source.name}…")
    started = time.time()
    last = [0.0]

    def progress(fraction: float) -> None:
        if fraction - last[0] >= 0.1 or fraction >= 1.0:
            print(f"  {fraction * 100:.0f}%", flush=True)
            last[0] = fraction

    segments = transcribe(source, words=True, progress=progress)
    elapsed = time.time() - started
    words = sum(len(s["words"]) for s in segments)

    dest.write_text(json.dumps({
        "source": source.name,
        "engine": "vosk-model-small-en-us-0.15",
        "segments": segments,
        "word_count": words,
        "seconds_to_transcribe": round(elapsed, 1),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{len(segments)} segments, {words} words in {elapsed:.0f}s")
    print(f"saved {dest}")


if __name__ == "__main__":
    main()
