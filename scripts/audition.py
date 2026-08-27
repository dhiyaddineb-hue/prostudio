#!/usr/bin/env python3
"""Score candidate voices against the actor they have to replace.

Casting by ear wastes takes: three rounds of auditions were rejected before
anyone measured what the on-screen actors actually sound like. This scores each
candidate recording on the two things a viewer notices immediately.

**Pitch.** How far the candidate's median F0 sits from the actor's, in
semitones — the unit the ear works in. A 12-semitone error is an octave and
reads as the wrong person entirely; under 2 semitones is a close match.

**Pace.** Whether the line fits the actor's mouth. A take that needs heavy
speed-up to fit will sound rushed no matter how good the timbre is.

Run it after generating candidates into work/cand/ as ``vNN_cMM.wav``.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from youtube_auto_dub.ffmpeg_bin import ensure_ffmpeg_on_path  # noqa: E402
from youtube_auto_dub.project_dirs import load  # noqa: E402
from youtube_auto_dub.speech_windows import find_windows  # noqa: E402
from youtube_auto_dub.stem_split import decode_stereo, split_center  # noqa: E402

SR = 44100
TAKE = re.compile(r"^v(\d+)_c(\d+)$")


def median_f0(audio: np.ndarray, sr: int) -> float:
    if audio.size < sr // 20:
        return 0.0
    pitch = parselmouth.Sound(audio.astype(np.float64), sr).to_pitch(
        pitch_floor=60, pitch_ceiling=400
    )
    voiced = pitch.selected_array["frequency"]
    voiced = voiced[voiced > 0]
    return float(np.median(voiced)) if voiced.size else 0.0


def trim_db(audio: np.ndarray, sr: int, floor: float = 40.0) -> np.ndarray:
    hop = max(int(sr * 0.01), 1)
    count = audio.size // hop
    if count < 2:
        return audio
    frames = audio[: count * hop].reshape(count, hop)
    db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-12)
    keep = np.where(db > db.max() - floor)[0]
    return audio[keep[0] * hop : (keep[-1] + 1) * hop] if keep.size else audio


def semitones(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return float("inf")
    return abs(12 * np.log2(a / b))


def main() -> None:
    slug = os.environ.get("PROJECT", "Vikings-Ragnar-Floki")
    project = load(slug)
    ensure_ffmpeg_on_path()

    src = sorted(project.source_dir.glob("*.mp4"))
    if not src:
        raise SystemExit("no source clip; cannot measure the actors")
    left, right = decode_stereo(src[0], SR)
    centre, _ = split_center(left, right, SR)

    cues = sorted(project.cues, key=lambda c: c["i"])
    actor: dict[int, float] = {}
    for cue in cues:
        seg = centre[int(float(cue["start"]) * SR) : int(float(cue["end"]) * SR)]
        actor[cue["i"]] = median_f0(seg, SR)

    print("actor pitch per cue:")
    for cue in cues:
        print(f"  cue {cue['i']}  [{cue['speaker']}]  {actor[cue['i']]:5.0f} Hz")

    cand_dir = project.work_dir / "cand"
    if not cand_dir.is_dir():
        raise SystemExit(f"no candidates in {cand_dir}")

    rows: dict[str, list] = {}
    for wav in sorted(cand_dir.glob("*.wav")):
        m = TAKE.match(wav.stem)
        if not m:
            continue
        voice, idx = f"voice-{m.group(1)}", int(m.group(2))
        cue = next((c for c in cues if c["i"] == idx), None)
        if cue is None:
            continue
        audio, sr = sf.read(wav, dtype="float64")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = trim_db(audio, sr)
        f0 = median_f0(audio, sr)
        window = float(cue["end"]) - float(cue["start"])
        dur = audio.size / sr
        rows.setdefault(voice, []).append(
            {
                "cue": idx,
                "speaker": cue["speaker"],
                "f0": f0,
                "err": semitones(f0, actor[idx]),
                "squeeze": dur / window if window > 0 else float("inf"),
            }
        )

    print("\nvoice     cues  median F0  pitch error  speed-up needed")
    scored = []
    for voice, takes in sorted(rows.items()):
        f0s = [t["f0"] for t in takes if t["f0"] > 0]
        errs = [t["err"] for t in takes if np.isfinite(t["err"])]
        sq = [t["squeeze"] for t in takes]
        err = float(np.mean(errs)) if errs else float("inf")
        print(
            f"{voice}   {len(takes):3d}   {np.median(f0s):6.0f} Hz   "
            f"{err:5.1f} st     x{np.mean(sq):.2f}"
        )
        scored.append((err, np.mean(sq), voice, {t['cue'] for t in takes}))

    print("\nclosest per cue group:")
    for _err, _sq, voice, cue_set in sorted(scored):
        print(f"  {voice}: cues {sorted(cue_set)}  {_err:.1f} st, x{_sq:.2f}")


if __name__ == "__main__":
    main()
