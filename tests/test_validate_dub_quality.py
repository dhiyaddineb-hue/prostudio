import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location("quality",Path("scripts/validate_dub_quality.py"))
quality=importlib.util.module_from_spec(spec); spec.loader.exec_module(quality)

def test_timeline_metrics_detects_missing_middle():
    m=quality.timeline_metrics([{"start":0,"end":4},{"start":9,"end":12}],12)
    assert round(m["coverage"],3)==0.583
    assert m["max_gap"]==5

def test_timeline_metrics_merges_overlaps():
    m=quality.timeline_metrics([{"start":0,"end":5},{"start":4,"end":10}],10)
    assert m["coverage"]==1
    assert m["max_gap"]==0
