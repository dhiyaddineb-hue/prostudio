"""The dub must not stretch a take far enough to sound synthetic.

Time-stretching costs harmonics, measured on this project's own takes:

    x1.05  ->  -0.07 dB HNR        x1.20  ->  -1.10 dB
    x1.10  ->  -0.41 dB            x1.29  ->  -1.29 dB

No stretcher available offline does better; rubberband measured -2.54 dB at
x1.29 against atempo's -1.29. So the only real lever is to stretch less, and
these tests keep the budget from creeping back up.

The regression they guard against is specific: MAX_NATURALISE was 1.6, so the
pass meant to stop a line dragging could stretch it 1.51x on its own -- before
the fitting pass added more -- and quietly cost over a decibel in the name of
improving the read.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    """Import build_dub with a project that needs no source clip on disk."""
    import os

    os.environ.setdefault("PROJECT", "Vikings-Ragnar-Floki")
    spec = importlib.util.spec_from_file_location(
        "build_dub_under_test", ROOT / "scripts" / "build_dub.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_dub_under_test"] = module
    spec.loader.exec_module(module)
    return module


def test_no_single_pass_may_stretch_past_the_audible_threshold():
    m = _load()
    # 1.10 costs -0.41 dB; 1.20 already costs -1.10 dB.
    assert m.MAX_TEMPO <= 1.10
    assert m.MAX_NATURALISE <= 1.10


def test_naturalise_shares_the_same_budget_as_fitting():
    # Two independent caps multiply: 1.6 * 1.35 would permit x2.16 overall.
    m = _load()
    assert m.MAX_NATURALISE <= m.MAX_TEMPO


def test_warning_threshold_sits_below_the_cap():
    # A warning that only fires at the cap can never fire early enough to be
    # useful, so it has to trip before the limit is reached.
    m = _load()
    assert m.STRETCH_WARN < m.MAX_TEMPO
