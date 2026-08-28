"""Work out who is speaking, from a few spans you already know.

Blind clustering does not work on this material. Running k-means over MFCC
features of the Bob Proctor interview scored a silhouette of 0.087 — barely
structure at all — and put Proctor's own line "so I started to study myself"
in the interviewer's cluster. Pitch is no better: both men sit at 87-153 Hz,
and Proctor alone swings 89-128 Hz inside a single sentence, so the ranges
overlap completely.

What does work is starting from spans a human has confirmed. Reading the
transcript makes some lines unambiguous — an interviewer asks "wasn't that
your motivation?", and only Proctor says "I'm not Bob Proctor, those are two
words my parents gave me". A handful of those, on each side, trains a
classifier that labels the rest: 98% cross-validated on the seeds, and 10 out
of 10 on held-out anchors chosen afterwards, against 4 of 5 for the blind
attempt.

This is deliberately not automatic. The seeds are the human judgement in the
loop, and without them the answer would be a guess wearing a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# A window long enough to carry timbre but short enough to sit inside one
# speaker's turn. Half-overlapped so a turn boundary is never straddled twice.
WINDOW = 1.2
HOP_RATIO = 0.5
# A speaker holds the floor for seconds; isolated flips are classifier noise.
SMOOTH_WORDS = 9


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    text: str

    @property
    def dur(self) -> float:
        return self.end - self.start


def _fingerprint(audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """MFCC mean, spread and slope — timbre rather than pitch.

    Pitch is useless here because the two speakers overlap in range. What
    separates them is the shape of the vocal tract, which is what the cepstral
    coefficients describe.
    """
    import librosa

    if audio.size < sr // 3:
        return None
    mfcc = librosa.feature.mfcc(y=audio.astype(float), sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    return np.concatenate([mfcc.mean(1), mfcc.std(1), delta.mean(1)])


def _windows(audio: np.ndarray, sr: int, spans: Sequence[tuple[float, float]]) -> list:
    out = []
    step = WINDOW * HOP_RATIO
    for start, end in spans:
        at = start
        while at + WINDOW <= end:
            fp = _fingerprint(audio[int(at * sr):int((at + WINDOW) * sr)], sr)
            if fp is not None:
                out.append(fp)
            at += step
    return out


def train(
    audio: np.ndarray,
    sr: int,
    seeds_a: Sequence[tuple[float, float]],
    seeds_b: Sequence[tuple[float, float]],
):
    """Fit a two-speaker classifier from confirmed spans.

    Returns ``(model, mean, std, accuracy)``. The accuracy is cross-validated
    on the seeds, so it says how separable the two voices are — not how right
    the seeds were. Bad seeds produce a confident, wrong model, which is why
    they must come from reading the transcript.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import cross_val_score

    a = np.array(_windows(audio, sr, seeds_a))
    b = np.array(_windows(audio, sr, seeds_b))
    if len(a) < 3 or len(b) < 3:
        raise ValueError(
            f"need at least 3 windows per speaker, got {len(a)} and {len(b)}; "
            "give longer or more seed spans"
        )

    both = np.vstack([a, b])
    mean, std = both.mean(0), both.std(0) + 1e-9
    features = (both - mean) / std
    labels = np.r_[np.zeros(len(a)), np.ones(len(b))]

    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    folds = min(5, len(a), len(b))
    accuracy = float(cross_val_score(model, features, labels, cv=folds).mean())
    model.fit(features, labels)
    return model, mean, std, accuracy


def label_words(
    audio: np.ndarray,
    sr: int,
    words: Sequence[dict],
    model,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Label every word 0 or 1, smoothed over its neighbours."""
    probs = np.full(len(words), 0.5)
    half = WINDOW / 2
    for i, word in enumerate(words):
        centre = (word["start"] + word["end"]) / 2
        chunk = audio[int(max(0.0, centre - half) * sr):int((centre + half) * sr)]
        fp = _fingerprint(chunk, sr)
        if fp is not None:
            probs[i] = model.predict_proba(((fp - mean) / std)[None])[0][1]

    pad = SMOOTH_WORDS // 2
    smooth = np.array([
        np.median(probs[max(0, i - pad):i + pad + 1]) for i in range(len(probs))
    ])
    return (smooth > 0.5).astype(int)


def to_turns(words: Sequence[dict], labels: np.ndarray, names: tuple[str, str]) -> list[Turn]:
    """Group consecutive same-speaker words into turns."""
    import itertools

    turns: list[Turn] = []
    index = 0
    for key, group in itertools.groupby(labels):
        count = len(list(group))
        chunk = words[index:index + count]
        turns.append(Turn(
            speaker=names[int(key)],
            start=chunk[0]["start"],
            end=chunk[-1]["end"],
            text=" ".join(w["w"] for w in chunk),
        ))
        index += count
    return turns


def check_anchors(turns: Sequence[Turn], anchors: Sequence[tuple[float, str]]) -> tuple[int, int]:
    """Score the labelling against moments confirmed separately.

    Anchors must be chosen *after* training and must not overlap the seeds,
    or this measures memorisation rather than accuracy.
    """
    hits = 0
    for when, expected in anchors:
        for turn in turns:
            if turn.start <= when <= turn.end:
                hits += turn.speaker == expected
                break
    return hits, len(anchors)
