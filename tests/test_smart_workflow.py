from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_seed_vc_is_enabled_and_applied_per_chunk():
    text = (ROOT / ".github/workflows/dub.yml").read_text(encoding="utf-8")
    assert 'description: "apply Seed-VC independently to every spoken smart chunk"' in text
    assert "extra+=(--seed-vc)" in text
    seed_section = text.split("      seed_vc:", 1)[1].split("      lip_sync:", 1)[0]
    assert "default: true" in seed_section
    script = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    assert "status=\"seed_vc_processing\"" in script
    assert "status=\"seed_vc_completed\"" in script
    assert "dubbed-before-seedvc.mp4" in script


def test_no_automatic_cleanup_and_no_unapproved_fallback():
    dub = (ROOT / ".github/workflows/dub.yml").read_text(encoding="utf-8")
    cleanup = (ROOT / ".github/workflows/cleanup-dub-checkpoints.yml").read_text(encoding="utf-8")
    script = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    assert "gh release delete" not in dub
    assert "schedule:" not in cleanup
    assert 'expected="DELETE $PROJECT_ID"' in cleanup
    assert 'default=False' in script


def test_workflow_yamls_parse():
    for name in ("dub.yml", "cleanup-dub-checkpoints.yml"):
        assert yaml.safe_load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))


def test_seed_vc_uses_huggingface_token_when_available():
    text = (ROOT / "scripts/seed_vc_enhance.py").read_text(encoding="utf-8")
    assert 'os.environ.get("HF_TOKEN")' in text
    assert "Client(args.space, hf_token=token)" in text


def test_script_adds_repository_root_before_local_imports():
    text = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    root_insert = text.index("sys.path.insert(0, str(Path(__file__).resolve().parents[1]))")
    local_import = text.index("from youtube_auto_dub.emotion import infer_emotion")
    assert root_insert < local_import
