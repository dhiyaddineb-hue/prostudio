"""The studio page and the one-folder-per-video library it manages.

The user asked for a web studio that controls the repository: a gallery of
uploaded and dubbed videos (each in its own folder), preview, deletion with an
explicit confirmation, and live progress of running dubs. The page is static
(GitHub Pages) and talks to GitHub's API directly, so the contract between the
page and the workflow lives in file layouts and workflow inputs. These tests
pin that contract.
"""
from __future__ import annotations

import importlib.util
import json
import wave
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_publisher():
    spec = importlib.util.spec_from_file_location("publish_to_library_under_test", ROOT / "scripts/publish_to_library.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_video(path: Path, seconds: float = 1.0) -> None:
    # A WAV is enough for the publisher: it only copies, hashes and (optionally) probes.
    pcm = (0.2 * np.sin(np.linspace(0, 200, int(24000 * seconds))) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(pcm.tobytes())


def test_slug_comes_from_the_library_folder_or_the_file_name():
    m = _load_publisher()
    assert m.slug_for_source("library/my-video/source.mp4") == "my-video"
    assert m.slug_for_source("library/my-video/source.mp4.part01of03") == "my-video"
    assert m.slug_for_source("/runner/work/prostudio/library/ep-01/source.mkv") == "ep-01"
    assert m.slug_for_source("samples/shorts-test.mp4") == "shorts-test"
    assert m.slug_for_source("samples/napoleon.part0") == "napoleon"
    assert m.slug_for_source("inbox/clip name (1).mp4") == "clip-name-1"
    assert m.slug_for_source("") == "video"


def test_publish_creates_one_folder_per_video_with_versions(tmp_path, monkeypatch):
    m = _load_publisher()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "LIBRARY", tmp_path / "library")
    monkeypatch.setattr(m, "DUBS", tmp_path / "dubs")
    monkeypatch.setattr(m, "probe_duration", lambda path: 33.1)
    (tmp_path / "library/ep-01").mkdir(parents=True)
    (tmp_path / "library/ep-01/meta.json").write_text(json.dumps({"slug": "ep-01", "title": "Episode 1", "status": "uploaded"}), encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    _fake_video(output / "final-dub.mp4")
    (output / "quality-report.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (output / "language-integrity.json").write_text(json.dumps({"valid": True, "detected": "en", "confidence": 0.99}), encoding="utf-8")
    (output / "segments-report.json").write_text(json.dumps({"segments": [{}, {}, {}], "translation": {"engine": "llm"}}), encoding="utf-8")

    result = m.publish(output / "final-dub.mp4", source="library/ep-01/source.mp4", run_id="101", output_dir=output,
                       settings={"target_lang": "en", "tts_engine": "voxcpm"})
    assert result == {"slug": "ep-01", "folder": "dubs/ep-01", "file": "final-dub-101.mp4", "versions": 1}
    folder = tmp_path / "dubs/ep-01"
    assert sorted(p.name for p in folder.iterdir()) == [
        "final-dub-101.language.json", "final-dub-101.mp4", "final-dub-101.quality.json", "final-dub-101.segments.json", "meta.json",
    ]
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    version = meta["versions"][0]
    assert version["run_id"] == "101" and version["quality_ok"] is True and version["language"]["valid"] is True
    assert version["segments"] == 3 and version["translation_engine"] == "llm" and version["duration"] == 33.1
    assert version["settings"] == {"target_lang": "en", "tts_engine": "voxcpm"}
    assert version["reports"] == {"quality": "final-dub-101.quality.json", "segments": "final-dub-101.segments.json", "language": "final-dub-101.language.json"}
    lib_meta = json.loads((tmp_path / "library/ep-01/meta.json").read_text(encoding="utf-8"))
    assert lib_meta["status"] == "dubbed" and lib_meta["last_dub_run"] == "101" and lib_meta["dubs"] == 1 and lib_meta["title"] == "Episode 1"

    # A second run adds a version (newest first) and never removes the first.
    _fake_video(output / "final-dub.mp4", seconds=1.5)
    result = m.publish(output / "final-dub.mp4", source="library/ep-01/source.mp4", run_id="102", output_dir=output, settings={})
    assert result["versions"] == 2
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert [v["run_id"] for v in meta["versions"]] == ["102", "101"]
    assert (folder / "final-dub-101.mp4").exists() and (folder / "final-dub-102.mp4").exists()
    assert json.loads((tmp_path / "library/ep-01/meta.json").read_text(encoding="utf-8"))["dubs"] == 2

    # Re-publishing the same run replaces its own entry instead of duplicating it.
    m.publish(output / "final-dub.mp4", source="library/ep-01/source.mp4", run_id="102", output_dir=output, settings={})
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert [v["run_id"] for v in meta["versions"]] == ["102", "101"]

    # Sources outside the library still get their own dubs folder; no library meta is invented.
    result = m.publish(output / "final-dub.mp4", source="samples/shorts-test.mp4", run_id="103", output_dir=output, settings={})
    assert result["folder"] == "dubs/shorts-test" and not (tmp_path / "library/shorts-test").exists()


def test_workflow_publishes_into_the_library_layout_and_names_runs_after_their_source():
    text = (ROOT / ".github/workflows/dub.yml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert "inputs.source_path" in data["run-name"] and "run ${{ github.run_number }}" in data["run-name"]
    publish = text.split("- name: Publish project, report, and Pages index", 1)[1].split("- name: Conditional lip synchronization", 1)[0]
    assert "scripts/publish_to_library.py" in publish and "git add -f library dubs" in publish
    assert "publish_docs.py" not in publish and "publish_dub_run.py" not in publish
    assert '--setting "target_lang=${{ inputs.target_lang }}"' in publish
    project = text.split("- name: Prepare resumable smart-chunk project", 1)[1].split("- name: Run resumable speech-aware smart chunks", 1)[0]
    assert 'if [[ "$source" == library/*/* ]]; then' in project and "cut -d/ -f2" in project
    reassemble = text.split("- name: Reassemble split source parts", 1)[1].split("- name: Prepare resumable smart-chunk project", 1)[0]
    assert "if: inputs.source_path != ''" in reassemble
    assert '"${sp}".part*of*' in reassemble and 'cat $parts > "$sp"' in reassemble
    assert 'cat "${base}.part0" "${base}.part1" > "$full"' in reassemble  # legacy two-part sources still work


def test_cleanup_workflow_accepts_library_dub_paths():
    text = (ROOT / ".github/workflows/cleanup-dub-checkpoints.yml").read_text(encoding="utf-8")
    assert yaml.safe_load(text)
    assert "dubs/*/*.mp4|docs/*.mp4) ;;" in text
    assert 'expected="DELETE $PROJECT_ID"' in text  # explicit approval stays mandatory


def test_gitignore_keeps_library_and_dubs_tracked():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for rule in ("!library/", "!library/**", "!dubs/", "!dubs/**", "checkpoints/"):
        assert rule in text
    assert (ROOT / "library/.gitkeep").exists() and (ROOT / "dubs/.gitkeep").exists()


def test_studio_page_controls_the_repository_with_explicit_confirmations():
    html = (ROOT / "docs/index.html").read_text(encoding="utf-8")
    js = (ROOT / "docs/studio.js").read_text(encoding="utf-8")
    assert 'src="studio.js"' in html and 'href="studio.css"' in html and (ROOT / "docs/studio.css").exists()
    for tab in ("tab-library", "tab-dubs", "tab-runs", "tab-settings"):
        assert f'id="{tab}"' in html
    # gallery + live runs read the branch directly and keep polling
    assert "git/trees/${encodeURIComponent(BRANCH)}?recursive=1" in js and "If-None-Match" in js
    assert "groupFolders('library/')" in js and "groupFolders('dubs/')" in js
    assert "actions/runs?branch=" in js and "/jobs`" in js and "checkpoint-manifest.json" in js
    assert "setTimeout(async () => { await fullRefresh(false); schedule(); }" in js
    # uploads: one folder per video, split into parts under the measured blob ceiling
    assert "PART_BYTES = 18 * 1024 * 1024" in js and "library/${slug}/${name}" in js and "meta.json" in js
    # dubbing dispatches the official workflow with the library source path
    assert "actions/workflows/${WORKFLOW}/dispatches" in js and "source_path: item.sourcePath" in js
    # deletions: one commit, only after the user types the folder name
    assert "sha: null" in js and "function confirmTyped" in js
    assert "go.disabled = input.value.trim() !== expect" in js
    for fn in ("confirmDeleteFolder", "confirmDeleteVersion", "confirmCancel"):
        assert f"function {fn}" in js
    # the legacy per-project pages are still reachable from the studio
    assert 'href="dashboard.html"' in html and (ROOT / "docs/dashboard.html").exists()


def test_studio_voice_map_matches_the_pipeline_profile_contract():
    """The voices dialog writes library/<slug>/voices.json; the pipeline must read it unchanged."""
    html = (ROOT / "docs/index.html").read_text(encoding="utf-8")
    js = (ROOT / "docs/studio.js").read_text(encoding="utf-8")
    assert 'id="tab-voices"' in html and "function renderVoiceBank" in js and "function uploadVoiceSample" in js
    # per-video map next to the source; samples live under library/<slug>/voices/ and bank voices under voices/
    assert "library/${item.slug}/voices.json" in js and "voices/${card.dataset.speaker}${ext}" in js
    assert "../../voices/${esc(b.name)}" in js and "SPEAKER_00" in js
    # dubbing passes the map to the official workflow input only when the user opts in
    assert "speaker_voices_path: item.voicesJson && $('xVoices')?.checked ? `library/${item.slug}/voices.json` : ''" in js
    # every field the page writes is a field voice_profiles.py understands
    spec = importlib.util.spec_from_file_location("voice_profiles_under_test", ROOT / "youtube_auto_dub/voice_profiles.py")
    vp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vp)
    known = set(vp.default_profile("SPEAKER_00"))
    for key in ("reference_mode", "reference_path", "tts_engine", "voice_conversion", "style", "gender", "approved"):
        assert f'data-key="{key}"' in js and key in known
    for mode in vp.REFERENCE_MODES:
        assert f'value="{mode}"' in js
    # the page refuses to save an unapproved speaker because load_voice_profiles(require_approval=True) would reject it
    assert "يجب اعتماد كل متحدث قبل الحفظ" in js
    # relative reference paths resolve against the voices.json folder, so both layouts the page writes are valid
    with __import__("tempfile").TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "library/ep-01/voices").mkdir(parents=True)
        (base / "voices").mkdir()
        (base / "library/ep-01/voices/SPEAKER_00.wav").write_bytes(b"RIFF")
        (base / "voices/narrator.wav").write_bytes(b"RIFF")
        doc = {"version": 1, "speakers": {
            "SPEAKER_00": {"speaker": "SPEAKER_00", "reference_mode": "custom", "reference_path": "voices/SPEAKER_00.wav", "approved": True},
            "SPEAKER_01": {"speaker": "SPEAKER_01", "reference_mode": "custom", "reference_path": "../../voices/narrator.wav", "approved": True},
        }}
        (base / "library/ep-01/voices.json").write_text(json.dumps(doc), encoding="utf-8")
        profiles = vp.load_voice_profiles(base / "library/ep-01/voices.json", ["SPEAKER_00", "SPEAKER_01"], require_approval=True)
        assert profiles["SPEAKER_00"]["reference_path"].endswith("library/ep-01/voices/SPEAKER_00.wav")
        assert profiles["SPEAKER_01"]["reference_path"].endswith("voices/narrator.wav")


def test_studio_checkpoint_dialog_reads_the_release_manifest_and_deletes_only_with_confirmation():
    js = (ROOT / "docs/studio.js").read_text(encoding="utf-8")
    assert "function openCheckpointDialog" in js and "checkpoint-manifest.json" in js
    assert "tag_name.startsWith('checkpoint-')" in js
    # per-chunk listening happens from release assets, never by re-running anything
    assert "function renderAudioCompare" in js
    # deleting a checkpoint release goes through the typed confirmation like every other deletion
    assert "confirmTyped" in js.split("function openCheckpointDialog", 1)[1]
