from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_has_smart_projects_characters_and_comparisons():
    text = (ROOT / "docs/dashboard.html").read_text(encoding="utf-8")
    for marker in (
        'id="refreshSmart"', 'id="smartProjects"', 'id="characterEditor"',
        'إعداد الشخصيات', 'reference_mode', 'tts_engine', 'voice_conversion',
        'XTTS', 'VoxCPM', 'Qwen', 'Edge-TTS',
        'preview-(original|before_seed_vc|after_seed_vc|final)', 'before_seed_vc', 'after_seed_vc',
    ):
        assert marker in text


def test_dashboard_never_triggers_cleanup():
    text = (ROOT / "docs/dashboard.html").read_text(encoding="utf-8")
    assert "gh release delete" not in text
    assert "cleanup-dub-checkpoints" not in text


def test_dashboard_requires_character_approval_before_saving():
    text = (ROOT / "docs/dashboard.html").read_text(encoding="utf-8")
    assert "يجب اعتماد إعداد كل شخصية" in text
    assert "config/voices/" in text


def test_dashboard_renders_per_chunk_stage_checklists():
    text = (ROOT / "docs/dashboard.html").read_text(encoding="utf-8")
    assert "قائمة المراحل" in text
    for stage in ("analysis", "translation", "tts", "seed_vc", "timing_fit", "content_validation", "audio_mix", "video_render", "checkpoint_upload"):
        assert stage in text
    assert "المراحل الناجحة لن تُعاد" in text
