"""Find where the original actors actually speak.

Cue windows used to be typed by hand off the burned-in subtitles. A subtitle
stays on screen well past the last syllable, so those windows are consistently
wider than the mouth movement they are supposed to describe — measured on the
Vikings clip, the captions claim 1.50 s for a line the actor delivers in 1.14 s.
Dubbing into that window puts the dubbed voice over a closed mouth, which is
exactly the "out of sync" a viewer notices.

This measures the dialogue instead. Centre-channel extraction pulls the voice
forward of the score, and speech is taken to be the frames that rise clearly
above the bed. The result is a list of (start, end) windows in seconds that a
dub can be fitted to.

Threshold is chosen relative to the clip's own noise floor rather than fixed:
an action scene and a quiet interior have very different beds, and a constant
-30 dB gate that works on one silently swallows the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Window:
    start: float
    end: float

    @property
    def dur(self) -> float:
        return self.end - self.start

    def as_tuple(self) -> tuple[float, float]:
        return (round(self.start, 2), round(self.end, 2))


def frame_db(audio: np.ndarray, sr: int, hop: float = 0.02) -> np.ndarray:
    """Per-frame RMS in dBFS."""
    n = int(sr * hop)
    if n <= 0 or audio.size < n:
        return np.zeros(0, dtype=np.float32)
    count = audio.size // n
    frames = audio[: count * n].reshape(count, n)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    return (20 * np.log10(rms + 1e-12)).astype(np.float32)


def pick_threshold(db: np.ndarray, margin_db: float = 8.0) -> float:
    """A gate that sits ``margin_db`` above this clip's own quiet frames.

    The 25th percentile stands in for "bed only": speech is sparse enough in a
    dialogue clip that a quarter of frames are reliably between lines. Clamped
    to the loud end so a wall-to-wall-music clip cannot drive the gate below
    the score and mark the whole reel as speech.
    """
    if db.size == 0:
        return -30.0
    floor = float(np.percentile(db, 25))
    ceiling = float(np.percentile(db, 95))
    return min(floor + margin_db, ceiling - 3.0)


def find_windows(
    audio: np.ndarray,
    sr: int,
    *,
    hop: float = 0.02,
    close_gap: float = 0.28,
    min_dur: float = 0.20,
    margin_db: float = 8.0,
    pad: float = 0.04,
) -> list[Window]:
    """Speech windows in ``audio``, longest-first merging of loud frames.

    ``close_gap`` bridges the pauses inside a sentence — a comma is shorter
    than the silence between two speakers, so 0.28 s keeps a line whole without
    welding two lines together. ``pad`` restores the quiet onset of a word,
    which an energy gate always clips.
    """
    db = frame_db(audio, sr, hop)
    if db.size == 0:
        return []
    gate = pick_threshold(db, margin_db)
    loud = db > gate

    runs: list[list[int]] = []
    start = None
    for i, on in enumerate(loud):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(loud)])

    merged: list[list[int]] = []
    for a, b in runs:
        if merged and (a - merged[-1][1]) * hop < close_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    out = []
    for a, b in merged:
        s, e = a * hop - pad, b * hop + pad
        if e - s >= min_dur:
            out.append(Window(max(s, 0.0), min(e, len(audio) / sr)))
    return out


def match_cues(
    cues: list[dict], windows: list[Window], tolerance: float = 1.2
) -> list[dict]:
    """Re-time each cue onto the speech window it overlaps.

    A cue keeps its text and speaker; only ``start``/``end`` move. Cues whose
    hand-typed window matches no detected speech are left untouched rather than
    dragged onto a neighbour's line — a wrong window is better than a wrong
    speaker.
    """
    out = []
    for cue in sorted(cues, key=lambda c: c["i"]):
        s, e = float(cue["start"]), float(cue["end"])
        best, score = None, 0.0
        for w in windows:
            overlap = min(e, w.end) - max(s, w.start)
            if overlap > score:
                best, score = w, overlap
        fixed = dict(cue)
        if best is not None and score > 0 and abs(best.start - s) <= tolerance:
            fixed["start"] = round(best.start, 2)
            fixed["end"] = round(best.end, 2)
            fixed["detected"] = True
        else:
            fixed["detected"] = False
        out.append(fixed)
    return out
