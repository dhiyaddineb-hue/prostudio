import json
from pathlib import Path


def test_publish_script_declares_seed_vc_and_report_contract():
    text = Path("scripts/publish_dub_run.py").read_text(encoding="utf-8")
    assert "seed_vc" in text
    assert "pipeline_report" in text
    assert "project.json" in text


def test_dashboard_dispatch_sends_seed_vc_and_disables_lipsync():
    text = Path("docs/dashboard.html").read_text(encoding="utf-8")
    assert "seed_vc:$('clone').checked?'true':'false'" in text
    assert "lip_sync:'false'" in text
