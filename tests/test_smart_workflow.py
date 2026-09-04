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


def test_seed_audio_is_tempo_fitted_without_sample_slicing():
    text = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    assert "seedvc.voice-only.fitted.wav" in text
    assert "budget_samples" in text
    fit_block = text.split("def fit_without_cutting", 1)[1].split("def convert_analysis_audio", 1)[0]
    assert "atempo_filter" in fit_block
    assert "audio[:" not in fit_block
    assert ".unlink(" not in fit_block


def test_seed_vc_never_receives_timeline_silence():
    smart = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    seed = (ROOT / "scripts/seed_vc_enhance.py").read_text(encoding="utf-8")
    assert "apply_seed_vc_audio" in smart
    assert "seedvc.voice-only.wav" in smart
    assert 'seed_vc_mode="voice_only_v2"' in smart
    assert 'parser.add_argument("--audio-only"' in seed
    audio_only = seed.split("if args.audio_only:", 1)[1].split("# Keep video untouched", 1)[0]
    assert '"-t"' not in audio_only
    assert '"atrim=' not in audio_only


def test_old_full_timeline_seed_checkpoints_are_invalidated():
    text = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    assert 'seed_mode_current = chunk.get("seed_vc_mode") == "voice_only_v2"' in text
    assert "old full-timeline Seed-VC detected" in text
