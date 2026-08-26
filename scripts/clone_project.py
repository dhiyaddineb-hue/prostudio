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


def merge_runs(
    windows: list[tuple[float, float]],
    max_gap: float = 1.5,
) -> list[tuple[float, float]]:
    """Join a speaker's consecutive cues into continuous stretches.

    Individual captions are short — the longest here is 2.8 s, under the 3 s a
    clone needs — but consecutive lines from the same actor are one unbroken
    piece of speech in the source. Merging across the small gaps between them
    yields references of 10 s instead of failing outright.
    """
    if not windows:
        return []
    ordered = sorted(windows)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


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
    usable = [w for w in windows if w[1] - w[0] >= REF_MIN_SEC]
    if not usable:
        # Fall back to the longest stretch available, even if it is short:
        # a 2.5 s reference still clones better than refusing to try.
        longest = max(windows, key=lambda w: w[1] - w[0], default=None)
        if longest is None or longest[1] - longest[0] < 1.5:
            return None
        usable = [longest]

    for start, end in usable:
        span = min(end - start, want)
        t = start
        while t + span <= end + 1e-6:
            seg = stem[int(t * SR): int((t + span) * SR)]
            if seg.size:
                energy = float(np.mean(seg ** 2))
                if energy > best_energy:
                    best_energy, best = energy, (t, t + span)
            t += step
    return best


def preflight(project) -> list[str]:
    """Refuse clones that cannot work, instead of burning an hour to find out.

    Two failures produced the garbled, non-Arabic audio in run #17:

    1. The F5-TTS vocabulary is English/Chinese. It carries 25 Arabic glyphs and
       is missing ``ا`` — the commonest letter in the language — so roughly a
       fifth of the script is silently dropped before synthesis.

    2. The reference clip is the actor speaking *English*, while ref_text is the
       *Arabic* translation. F5 expects ref_text to be a literal transcript of
       ref_audio; when they disagree the alignment collapses and the output
       slurs into itself.
    """
    problems: list[str] = []

    vocab: set[str] = set()
    try:
        import f5_tts

        # f5_tts is a namespace package on some installs, so __file__ can be
        # None; fall back to its search paths.
        roots = [Path(f5_tts.__file__).parent] if f5_tts.__file__ else [
            Path(p) for p in getattr(f5_tts, "__path__", [])
        ]
        for root in roots:
            candidate = root / "infer" / "examples" / "vocab.txt"
            if candidate.exists():
                vocab = {
                    line.rstrip("\n")
                    for line in candidate.read_text(encoding="utf-8").splitlines()
                }
                break
    except Exception:
        vocab = set()

    if vocab:
        letters = [
            ch
            for cue in project.cues
            for ch in cue["text"]
            if "\u0600" <= ch <= "\u06ff"
        ]
        missing = [ch for ch in letters if ch not in vocab]
        if letters and len(missing) / len(letters) > 0.05:
            unique = "".join(sorted(set(missing)))
            problems.append(
                f"  ✗ vocabulary: {len(missing)}/{len(letters)} Arabic characters "
                f"({len(missing) / len(letters) * 100:.0f}%) are not in the F5-TTS "
                f"vocab and will be dropped. Missing: {unique}"
            )

    # The reference audio is the original performance, in its own language.
    problems.append(
        "  ✗ reference mismatch: ref_audio is the actor speaking the source "
        "language, but ref_text is the Arabic translation. F5-TTS needs them to "
        "match, so the clone will not track the words."
    )

    if problems:
        problems.insert(0, "\nPreflight found blocking problems:")
    return problems


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

    # Two hard checks before spending an hour of synthesis on a bad result.
    problems = preflight(project)
    if problems and os.environ.get("FORCE_CLONE") != "1":
        print("\n".join(problems))
        raise SystemExit(
            "Refusing to run: the clone would produce unusable audio.\n"
            "Set FORCE_CLONE=1 to run anyway."
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
        # Consecutive lines are continuous speech in the source; merging them
        # is what makes a reference long enough to clone from.
        runs = merge_runs(windows)
        window = pick_reference_window(actor_voice, runs, want)
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

    # A preview run clones only a few lines, so the voices can be judged in
    # minutes rather than after an hour of synthesis.
    max_cues = int(os.environ.get("MAX_CUES", "0") or 0)
    todo = [c for c in project.cues if c["speaker"] in refs]
    if max_cues > 0:
        picked: list = []
        # Take them alternately per speaker, so a preview covers both voices.
        for speaker in refs:
            picked += [c for c in todo if c["speaker"] == speaker][
                : max(1, max_cues // max(len(refs), 1))
            ]
        todo = sorted(picked, key=lambda c: c["i"])[:max_cues]
        print(f"preview mode: cloning {len(todo)} of {len(project.cues)} lines")

    done = failed = 0
    for cue in todo:
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
