#!/usr/bin/env python3
"""Snap a project's cue windows onto the dialogue actually present in the clip.

Cue windows are typed off the burned-in subtitles, and a subtitle lingers well
past the last syllable. Dubbing into that window leaves the dubbed voice running
over a closed mouth. Measured on the Vikings clip the captions claim 1.50 s for
a line delivered in 1.22 s, and the rendered dub talked 1.30 s past the actor.

This replaces each cue's start/end with the speech window it overlaps, and
reports how much text each window can carry at a natural Arabic rate, so lines
that no longer fit can be shortened rather than rushed.

    python scripts/retime_from_audio.py            # report only
    python scripts/retime_from_audio.py --write    # save into project.json
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from youtube_auto_dub.ffmpeg_bin import ensure_ffmpeg_on_path  # noqa: E402
from youtube_auto_dub.project_dirs import load  # noqa: E402
from youtube_auto_dub.speech_windows import find_windows, match_cues  # noqa: E402
from youtube_auto_dub.stem_split import decode_stereo, split_center  # noqa: E402

SR = 16000
# Comfortable Arabic delivery. Above this a line is being rushed to fit.
COMFORTABLE_RATE = 14.0
# A line may outlast the actor's mouth by this much before it reads as late.
MAX_OVERRUN = 1.25


def source_of(project) -> Path:
    local = sorted(project.source_dir.glob("*.mp4"))
    if local:
        return local[0]
    inbox = sorted((ROOT / "inbox").glob("*.mp4"))
    if inbox:
        return inbox[0]
    raise SystemExit("no source clip for this project")


def main() -> None:
    slug = os.environ.get("PROJECT", "Vikings-Ragnar-Floki")
    project = load(slug)
    ensure_ffmpeg_on_path()

    left, right = decode_stereo(source_of(project), SR)
    centre, _ = split_center(left, right, SR)
    windows = find_windows(centre, SR)

    print(f"{len(windows)} speech windows detected:")
    for w in windows:
        print(f"  {w.start:6.2f} - {w.end:6.2f}  ({w.dur:.2f}s)")

    fixed = match_cues(project.cues, windows)
    print("\ncue  window          was            chars  fits at "
          f"{COMFORTABLE_RATE:.0f} ch/s")
    for cue, old in zip(fixed, sorted(project.cues, key=lambda c: c["i"])):
        chars = len([c for c in cue["text"] if not c.isspace()])
        budget = (cue["end"] - cue["start"]) * MAX_OVERRUN
        room = int(budget * COMFORTABLE_RATE)
        flag = "ok" if chars <= room else f"CUT {chars - room}"
        mark = " " if cue["detected"] else "?"
        print(
            f"{cue['i']:3d}{mark} {cue['start']:6.2f}-{cue['end']:6.2f}  "
            f"{old['start']:6.2f}-{old['end']:6.2f}  {chars:4d}   "
            f"{room:3d}  {flag}"
        )

    if "--write" in sys.argv:
        for cue in fixed:
            cue.pop("detected", None)
        project.cues = fixed
        project.save()
        print(f"\nwritten to {project.manifest_path}")
    else:
        print("\n(report only — pass --write to save)")


if __name__ == "__main__":
    main()
