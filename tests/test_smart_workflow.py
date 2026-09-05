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
    assert 'dubbed-before-seedvc{variant}.mp4' in script


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
    assert 'seedvc.voice-only{variant}.synced.wav' in text
    assert "budget_samples" in text
    fit_block = text.split("def fit_without_cutting", 1)[1].split("def convert_analysis_audio", 1)[0]
    assert "atempo_filter" in fit_block
    assert "audio[:" not in fit_block
    assert ".unlink(" not in fit_block


def test_seed_vc_never_receives_timeline_silence():
    smart = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    seed = (ROOT / "scripts/seed_vc_enhance.py").read_text(encoding="utf-8")
    assert "apply_seed_vc_audio" in smart
    assert 'seedvc.voice-only{variant}.wav' in smart
    assert 'seed_vc_mode="voice_only_sync_v3"' in smart
    assert 'parser.add_argument("--audio-only"' in seed
    audio_only = seed.split("if args.audio_only:", 1)[1].split("# Keep video untouched", 1)[0]
    assert '"-t"' not in audio_only
    assert '"atrim=' not in audio_only


def test_old_full_timeline_seed_checkpoints_are_invalidated():
    text = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    assert 'seed_mode_current = chunk.get("seed_vc_mode") == "voice_only_sync_v3"' in text
    assert "outdated Seed-VC timing detected" in text


def test_post_seed_sync_uses_bidirectional_tempo_without_trimming():
    text = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    block = text.split("def match_duration_without_cutting", 1)[1].split("def convert_analysis_audio", 1)[0]
    assert "atempo_filter(factor)" in block
    assert "max(factor, 1.0)" not in block
    assert '"atrim=' not in block
    assert "audio[:" not in block
    assert 'seed_vc_mode="voice_only_sync_v3"' in text


def test_selected_character_quality_features_are_wired():
    text = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    for marker in (
        "load_voice_profiles", "prepare_profile_references", "voice_profile_hash",
        "validate_spoken_content", "word_timing_report", "content-validation.json",
        "word-alignment.json", "comparison.json", "mix-report.json",
    ):
        assert marker in text
    assert "duck_floor = 0.28" in text
    assert "profile.get(\"tts_engine\")" in text


def test_workflow_dispatch_stays_within_github_input_limit():
    import re
    text = (ROOT / ".github/workflows/dub.yml").read_text(encoding="utf-8")
    section = text.split("    inputs:", 1)[1].split("\njobs:", 1)[0]
    names = re.findall(r"^      ([A-Za-z_][A-Za-z0-9_]*):\s*$", section, re.M)
    assert len(names) <= 25
    assert "speaker_voices_path" in names
    assert "validate_content" in names


def test_profile_file_forces_character_approval():
    text = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    assert "require_approval=bool(args.speaker_voices) or args.require_voice_approval" in text


def test_seed_vc_is_batched_to_reduce_quota_usage():
    text = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    for marker in ("--seed-batch-size", "combine_voice_batch", "split_voice_batch", "seed-batches", "seed_vc_batch"):
        assert marker in text
    assert "np.pad(audio, (0, missing))" in text


def test_seed_quota_exhaustion_fails_fast():
    text = (ROOT / "scripts/seed_vc_enhance.py").read_text(encoding="utf-8")
    assert '"quota" in message' in text
    assert "checkpoint and resume later" in text


def test_explicit_quota_policy_uses_one_consistent_voxcpm_delivery():
    script = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/dub.yml").read_text(encoding="utf-8")
    assert 'choices=["fail", "voxcpm"]' in script
    assert 'args.seed_quota_policy == "voxcpm"' in script
    assert 'seed_required = seed_requested and not seed_quota_fallback' in script
    assert 'delivery_voice_mode="voxcpm_reference_clone"' in script
    assert "seed_quota_voxcpm" in workflow
    assert "--seed-quota-policy voxcpm" in workflow


def test_render_loop_recomputes_non_speech_per_chunk():
    text = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    render = text.split("failures: list[int] = []", 1)[1]
    assert 'non_speech = is_non_speech_text(chunk.get("source_text", ""))' in render


def test_final_delivery_fit_prevents_last_sample_overrun():
    text = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    assert 'delivery_budget = max(0.12' in text
    assert 'directory / f"delivery{variant}.fitted.wav"' in text
    assert 'retry_text = f"{retry_parts[0].rstrip' in text
    assert 'content_retry_synthesis_text=retry_text' in text


def test_impossibly_short_phrase_borrows_preceding_silence():
    import ast
    source = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "expand_short_phrase_window")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "smart", "exec"), namespace)
    chunk = {"start": 30.0, "end": 39.94, "speech_start": 39.6, "speech_end": 39.94}
    fixed = namespace["expand_short_phrase_window"](chunk, 0.617)
    assert round(fixed["speech_start"], 3) == 39.323
    assert round(fixed["speech_end"] - fixed["speech_start"], 3) == 0.617
    assert fixed["timing_shift_seconds"] == 0.277
    assert chunk["speech_start"] == 39.6


def test_normal_phrase_window_is_not_shifted():
    import ast
    source = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "expand_short_phrase_window")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "smart", "exec"), namespace)
    chunk = {"start": 0.0, "end": 8.0, "speech_start": 0.2, "speech_end": 7.8}
    fixed = namespace["expand_short_phrase_window"](chunk, 6.0)
    assert fixed["speech_start"] == 0.2
    assert "timing_adjustment" not in fixed


def test_resume_reuses_persisted_seed_quota_decision():
    text = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    assert '(store.data.get("seed_quota_fallback") or {}).get("active")' in text
    assert "if not seed_quota_fallback and args.seed_batch_size > 1" in text
    assert 'directory / f"pre-render{variant}.fitted.wav"' in text


def test_short_three_word_phrase_receives_intelligible_window():
    import ast
    source = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "expand_short_phrase_window")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "smart", "exec"), namespace)
    chunk = {"start": 30.0, "end": 39.94, "speech_start": 39.323, "speech_start_original": 39.6, "speech_end": 39.94}
    fixed = namespace["expand_short_phrase_window"](chunk, 1.02)
    assert round(fixed["speech_start"], 2) == 38.92
    assert round(fixed["speech_end"] - fixed["speech_start"], 2) == 1.02
    assert fixed["timing_shift_seconds"] == 0.68


def test_speech_start_before_chunk_is_clamped_before_fit():
    import ast
    source = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "expand_short_phrase_window")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "smart", "exec"), namespace)
    chunk = {"start": 226.11, "end": 234.0, "speech_start": 225.82, "speech_end": 234.0}
    fixed = namespace["expand_short_phrase_window"](chunk, 7.0)
    assert fixed["speech_start"] == 226.11
    assert fixed["timing_adjustment"] == "clamped_to_chunk_start"


def test_missing_leading_word_uses_verified_same_voice_donor():
    text = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    assert "find_verified_word_clip" in text
    assert "concatenate_voice_parts" in text
    assert 'content_retry_mode="borrowed_verified_word"' in text
    retry = text.split("borrowed = None", 1)[1].split("retry_trimmed =", 1)[0]
    assert retry.index("if borrowed:") < retry.index("await synthesize(")


def test_borrowed_word_is_never_removed_by_silence_trimming():
    text = (ROOT / "scripts/resumable_smart_dub.py").read_text(encoding="utf-8")
    section = text.split("if borrowed:", 2)[2].split("speech_target =", 1)[0]
    assert "retry_trimmed = retry_raw" in section
    assert section.index("retry_trimmed = retry_raw") < section.index("trim_generated(")
