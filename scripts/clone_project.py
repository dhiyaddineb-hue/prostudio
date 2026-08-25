#!/usr/bin/env python3
"""Re-record a project's lines in the original actors' voices.

Run where HuggingFace is reachable (GitHub Actions, or any normal machine):

    pip install f5-tts
    PROJECT=Phantom-Thread python scripts/clone_project.py

For each speaker the cleanest stretch of that actor's separated voice is used as
the cloning reference, and every one of their lines is then generated in that
voice. Existing takes are backed up rather than overwritten, so a failed or
disappointing clone can be rolled back.

Nothing here is destructive if the weights are missing: the script says so and
exits, leaving the project exactly as it was.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from youtube_auto_dub.clone_tts import (  # noqa: E402
    REF_MAX_SEC,
    REF_MIN_SEC,
    CloneRef,
    available,
    clone_speak,
)
from youtube_auto_dub.project_dirs import load  # noqa: E402
from youtube_auto_dub.stem_split import decode_stereo, split_center  # noqa: E402

SR = 44100


def pick_reference_window(
    stem: np.ndarray,
    windows: list[tuple[float, float]],
    want: float,
) -> tuple[float, float] | None:
    """Loudest ``want``-second stretch inside this speaker's own lines.

    On a separated stem the quiet parts are mostly leftover score, so the
    loudest run is the most reliably clean sample of the actor.
    """
    best: tuple[float, float] | None = None
    best_energy = -1.0
    step = 0.25
    for start, end in windows:
        if end - start < REF_MIN_SEC:
            continue
        span = min(end - start, want)
        t = start
        while t + span <= end:
            seg = stem[int(t * SR): int((t + span) * SR)]
            if seg.size:
                energy = float(np.mean(seg ** 2))
                if energy > best_energy:
                    best_energy, best = energy, (t, t + span)
            t += step
    return best


def main() -> None:
    slug = os.environ.get("PROJECT", "Phantom-Thread")
    want = float(os.environ.get("REF_SECONDS", "8"))
    want = max(REF_MIN_SEC, min(want, REF_MAX_SEC))

    project = load(slug)
    if not project.cues:
        raise SystemExit(f"{slug}: no cues in project.json — nothing to clone")

    if not available():
        raise SystemExit(
            "F5-TTS weights are not available here.\n"
            "Install f5-tts on a machine that can reach HuggingFace, or run "
            "the 'Clone voices and dub' workflow on GitHub Actions."
        )

    sources = sorted(project.source_dir.glob("*.mp4")) or sorted(
        (ROOT / "inbox").glob("*.mp4")
    )
    if not sources:
        raise SystemExit(f"{slug}: no source clip in source/")

    print(f"separating {sources[0].name}…")
    left, right = decode_stereo(sources[0], SR)
    actor_voice, _ = split_center(left, right, SR)

    # Group each speaker's own cues; a reference must come from that actor.
    by_speaker: dict[str, list[tuple[float, float]]] = {}
    for cue in project.cues:
        by_speaker.setdefault(cue["speaker"], []).append(
            (float(cue["start"]), float(cue["end"]))
        )

    refs: dict[str, CloneRef] = {}
    for speaker, windows in by_speaker.items():
        window = pick_reference_window(actor_voice, windows, want)
        if window is None:
            print(f"  [{speaker}] no window long enough to clone from — skipping")
            continue
        start, end = window
        # The reference transcript must be what is actually said in that window.
        spoken = " ".join(
            c["text"] for c in project.cues
            if c["speaker"] == speaker and float(c["start"]) < end
            and float(c["end"]) > start
        ).strip()
        if not spoken:
            print(f"  [{speaker}] no transcript for the reference window — skipping")
            continue

        import soundfile as sf

        clip = actor_voice[int(start * SR): int(end * SR)]
        peak = float(np.max(np.abs(clip)) or 0.0)
        if peak < 1e-4:
            print(f"  [{speaker}] reference is silent — skipping")
            continue
        path = project.work_dir / f"ref_{speaker}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, (clip / peak * 0.89).astype(np.float32), SR)
        refs[speaker] = CloneRef(audio_path=path, text=spoken)
        print(f"  [{speaker}] reference {start:.2f}-{end:.2f}s: {spoken[:48]}…")

    if not refs:
        raise SystemExit("no usable reference audio — nothing cloned")

    backup = project.voices_dir.parent / "voices_before_clone"
    if project.voices_dir.exists() and not backup.exists():
        shutil.copytree(project.voices_dir, backup)
        print(f"backed up existing takes to {backup.name}/")

    done = failed = 0
    for cue in project.cues:
        speaker = cue["speaker"]
        ref = refs.get(speaker)
        if ref is None:
            continue
        dest = project.cue_take(int(cue["i"]), speaker)
        if clone_speak(cue["text"], ref, dest):
            done += 1
            print(f"  cue {cue['i']:2d} [{speaker}] cloned")
        else:
            failed += 1
            print(f"  cue {cue['i']:2d} [{speaker}] FAILED — keeping previous take")

    print(f"\ncloned {done} lines, {failed} failed")
    if failed and done == 0:
        raise SystemExit("every clone failed; project left unchanged")


if __name__ == "__main__":
    main()
