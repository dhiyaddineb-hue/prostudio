#!/usr/bin/env python3
"""Put the workspace back after the environment is reset.

The sandbox is wiped periodically. Everything under ``.gitignore`` disappears
with it: the virtualenv, the 68 MB speech model, and any uploaded video that
was joined from its parts. During one session this happened five times, and
each recovery was a different sequence of half-remembered commands.

This does the whole recovery, and each step is skipped when it is already done,
so running it twice is harmless and running it after a partial wipe fixes only
what is missing.

    python3 scripts/restore.py            # everything
    python3 scripts/restore.py --check    # report, change nothing
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
PY = VENV / "bin" / "python"
MODEL = ROOT / ".cache" / "models" / "vosk-model-small-en-us-0.15"
INBOX = ROOT / "inbox"
TRANSCRIPTS = ROOT / "transcripts"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, **kw)


def have_venv() -> bool:
    if not PY.is_file():
        return False
    probe = run(
        [str(PY), "-c", "import numpy, soundfile, parselmouth, vosk"],
        capture_output=True,
    )
    return probe.returncode == 0


def restore_venv(check: bool) -> bool:
    if have_venv():
        print("  venv        ok")
        return True
    if check:
        print("  venv        MISSING")
        return False
    print("  venv        rebuilding…")
    run([sys.executable, "-m", "venv", str(VENV)], capture_output=True)
    run(
        [str(PY), "-m", "pip", "install", "-q", "-r", "requirements.txt",
         "pytest", "pytest-asyncio", "vosk"],
        capture_output=True,
    )
    ok = have_venv()
    print(f"  venv        {'rebuilt' if ok else 'FAILED'}")
    return ok


def restore_model(check: bool) -> bool:
    mdl = MODEL / "am" / "final.mdl"
    if mdl.is_file() and mdl.stat().st_size > 1_000_000:
        print("  asr model   ok")
        return True
    if check:
        print("  asr model   MISSING")
        return False
    print("  asr model   downloading…")
    result = run([str(PY), "scripts/fetch_asr_model.py"], capture_output=True, text=True)
    ok = mdl.is_file()
    print(f"  asr model   {'restored' if ok else 'FAILED: ' + result.stderr[-200:]}")
    return ok


def restore_uploads(check: bool) -> bool:
    """Rejoin any split uploads that are on the branch but not on disk."""
    parts = sorted(INBOX.glob("*.part*of*"))
    if not parts:
        print("  uploads     nothing split to join")
        return True
    if check:
        print(f"  uploads     {len(parts)} part(s) waiting to join")
        return False
    result = run([str(PY), "scripts/join_parts.py", "--keep"],
                 capture_output=True, text=True)
    print("  uploads     " + (result.stdout.strip().split("\n")[-1] if result.stdout else "joined"))
    return True


def report_transcripts() -> None:
    if not TRANSCRIPTS.is_dir():
        print("  transcripts none saved")
        return
    files = sorted(TRANSCRIPTS.glob("*.json"))
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            segments = data if isinstance(data, list) else data.get("segments", [])
            words = sum(len(s.get("words", [])) for s in segments)
            print(f"  transcript  {path.name}: {len(segments)} segments, {words} words")
        except (json.JSONDecodeError, OSError):
            print(f"  transcript  {path.name}: unreadable")
    if not files:
        print("  transcripts none saved")


def main() -> None:
    check = "--check" in sys.argv
    print("restoring workspace" if not check else "checking workspace")

    ok = restore_venv(check)
    if ok or check:
        restore_model(check)
        restore_uploads(check)
    report_transcripts()

    if check:
        print("\n(--check only; run without it to repair)")


if __name__ == "__main__":
    main()
