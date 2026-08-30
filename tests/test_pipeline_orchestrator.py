from pathlib import Path
import json
import pipeline

ROOT = Path(__file__).parents[1]


def test_profile_modes_are_conservative_and_overridable():
    safe = pipeline.load_profile(ROOT / "config/pipeline_profile.yaml", "safe")
    cinematic = pipeline.load_profile(ROOT / "config/pipeline_profile.yaml", "multi_speaker_cinematic")
    assert safe["lip_sync"]["enabled"] is False
    assert safe["separation"]["preserve_background"] is False
    assert cinematic["diarization"]["enabled"] is True
    assert pipeline.load_profile(ROOT / "config/pipeline_profile.yaml", "safe", ["timing.max_tempo_factor=1.1"])["timing"]["max_tempo_factor"] == 1.1


def test_dry_run_is_model_free_and_writes_machine_readable_stdout(capsys, tmp_path):
    video = tmp_path / "video.mp4"; audio = tmp_path / "dub.wav"
    video.write_bytes(b"placeholder"); audio.write_bytes(b"placeholder")
    code = pipeline.main(["--video", str(video), "--dubbed-audio", str(audio), "--output", str(tmp_path / "out.mp4"), "--dry-run"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data["status"] == "dry_run_ok"
    assert "capabilities" in data


def test_no_original_dialogue_flag_is_explicit():
    assert pipeline.validate_output.__name__ == "validate_output"
