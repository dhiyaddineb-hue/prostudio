"""Each dub gets its own self-contained folder."""

import json

import pytest

from youtube_auto_dub import project_dirs as pd


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(pd, "PROJECTS_DIR", tmp_path / "projects")
    return tmp_path


def test_creates_the_full_layout():
    p = pd.create("My Film").ensure_dirs()
    for d in (p.source_dir, p.voices_dir, p.output_dir, p.work_dir):
        assert d.is_dir()
    assert p.manifest_path.exists()


def test_slug_keeps_arabic_and_drops_path_breakers():
    assert pd.slugify("فيلم: الخيط/الخفي*") == "فيلم-الخيطالخفي"
    assert pd.slugify("  Phantom   Thread  ") == "Phantom-Thread"
    assert pd.slugify("!!!") == "dub"


def test_second_project_with_the_same_name_does_not_collide():
    a = pd.create("Same")
    b = pd.create("Same")
    assert a.slug != b.slug
    assert a.root != b.root
    assert b.root.exists()


def test_outputs_are_named_after_the_project():
    p = pd.create("Night Train")
    assert p.video_path.name == f"{p.slug}.mp4"
    assert p.srt_path.name == f"{p.slug}.srt"
    assert p.video_path.parent == p.output_dir


def test_cue_take_paths_are_zero_padded():
    p = pd.create("X")
    assert p.cue_take(7, "f").name == "c07_f.wav"
    assert p.cue_take(23, "m").name == "c23_m.wav"
    assert p.cue_take(7, "f").parent == p.voices_dir


def test_manifest_round_trips_arabic():
    p = pd.create("Reload", title="فيلم تجريبي", dialect="eg")
    p.cues = [{"i": 1, "text": "ليه مش متجوز؟"}]
    p.save()

    again = pd.load(p.slug)
    assert again.title == "فيلم تجريبي"
    assert again.dialect == "eg"
    assert again.cues[0]["text"] == "ليه مش متجوز؟"
    assert again.root == p.root

    raw = json.loads(p.manifest_path.read_text(encoding="utf-8"))
    assert "root" not in raw, "Path objects must not leak into the manifest"


def test_load_or_create_reuses_an_existing_project():
    first = pd.load_or_create("Repeat")
    second = pd.load_or_create("Repeat")
    assert first.slug == second.slug


def test_clean_work_keeps_the_deliverables():
    p = pd.create("Keep").ensure_dirs()
    (p.work_dir / "scratch.wav").write_bytes(b"tmp")
    p.output_dir.mkdir(parents=True, exist_ok=True)
    p.video_path.write_bytes(b"video")

    p.clean_work()
    assert not (p.work_dir / "scratch.wav").exists()
    assert p.work_dir.is_dir()
    assert p.video_path.exists()


def test_listing_reports_progress():
    p = pd.create("Listed").ensure_dirs()
    p.cues = [{"i": 1}, {"i": 2}]
    p.save()
    (p.voices_dir / "c01_m.wav").write_bytes(b"x")

    entry = next(e for e in pd.list_projects() if e["slug"] == p.slug)
    assert entry["cues"] == 2
    assert entry["takes"] == 1
    assert entry["rendered"] is False


def test_missing_project_raises():
    with pytest.raises(FileNotFoundError):
        pd.load("does-not-exist")
