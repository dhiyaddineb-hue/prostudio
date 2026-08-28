"""Tests for seeded speaker separation.

Blind clustering failed on this material -- k-means over MFCC features scored
a silhouette of 0.087 and put Proctor's own "so I started to study myself" in
the interviewer's cluster. Seeding from spans confirmed by reading the
transcript took held-out anchor accuracy from 4/5 to 10/10.

These pin the parts whose failure would be silent: too few seeds must raise
rather than fit a meaningless model, smoothing must not erase real turns, and
anchor scoring must actually detect a wrong labelling.
"""

from __future__ import annotations

import numpy as np
import pytest

from youtube_auto_dub.diarize import (
    SMOOTH_WORDS, Turn, check_anchors, to_turns, train,
)


def _two_voices(sr=16000, seconds=6.0):
    """Two synthetic 'speakers' with clearly different spectral shape."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    rng = np.random.default_rng(0)
    a = (np.sin(2 * np.pi * 110 * t) + 0.5 * np.sin(2 * np.pi * 220 * t)
         + 0.05 * rng.standard_normal(t.size))
    b = (np.sin(2 * np.pi * 190 * t) + 0.5 * np.sin(2 * np.pi * 1900 * t)
         + 0.05 * rng.standard_normal(t.size))
    return np.concatenate([a, b]).astype(np.float32), sr, seconds


def test_training_separates_two_distinct_voices():
    audio, sr, span = _two_voices()
    _model, _mean, _std, accuracy = train(
        audio, sr, [(0.2, span - 0.2)], [(span + 0.2, 2 * span - 0.2)]
    )
    assert accuracy > 0.9, f"only {accuracy:.2f} on obviously different voices"


def test_too_few_seeds_is_refused():
    # A model fitted on two windows would report a confident number and mean
    # nothing; better to say the seeds are too short.
    audio, sr, _ = _two_voices(seconds=1.0)
    with pytest.raises(ValueError, match="at least 3 windows"):
        train(audio, sr, [(0.0, 1.3)], [(1.0, 2.3)])


def test_turns_group_consecutive_words():
    words = [{"w": w, "start": i, "end": i + 0.5} for i, w in enumerate("a b c d e".split())]
    turns = to_turns(words, np.array([0, 0, 1, 1, 1]), ("I", "P"))
    assert [t.speaker for t in turns] == ["I", "P"]
    assert turns[0].text == "a b" and turns[1].text == "c d e"
    assert turns[0].start == 0.0 and turns[1].end == 4.5


def test_a_single_word_turn_survives():
    # Interjections are real: "exactly" between two long turns is a turn.
    words = [{"w": w, "start": i, "end": i + 0.5} for i, w in enumerate("a b c".split())]
    turns = to_turns(words, np.array([0, 1, 0]), ("I", "P"))
    assert [t.speaker for t in turns] == ["I", "P", "I"]
    assert turns[1].text == "b"


def test_anchor_check_counts_hits_and_catches_errors():
    turns = [Turn("I", 0.0, 10.0, "question"), Turn("P", 10.0, 60.0, "answer")]
    assert check_anchors(turns, [(5.0, "I"), (30.0, "P")]) == (2, 2)
    assert check_anchors(turns, [(5.0, "P"), (30.0, "P")]) == (1, 2)


def test_anchors_outside_every_turn_count_as_misses():
    # Silence between turns should not be scored as correct by accident.
    turns = [Turn("P", 10.0, 20.0, "x")]
    assert check_anchors(turns, [(99.0, "P")]) == (0, 1)


def test_smoothing_window_is_odd_so_the_median_has_a_middle():
    assert SMOOTH_WORDS % 2 == 1


def test_turn_duration_is_reported():
    assert Turn("P", 2.5, 9.0, "x").dur == pytest.approx(6.5)
