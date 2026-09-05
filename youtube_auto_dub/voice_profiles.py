"""Per-character voice configuration shared by CLI, workflow, and dashboard."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ENGINES = ("xtts", "voxcpm", "qwen", "edge")
REFERENCE_MODES = ("source", "custom", "synthetic")
VOICE_CONVERSIONS = ("seed-vc", "none")
_SPEAKER_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def default_profile(speaker: str, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    values = dict(defaults or {})
    return {
        "speaker": speaker,
        "label": values.get("label") or speaker,
        "reference_mode": values.get("reference_mode", "source"),
        "reference_path": values.get("reference_path", ""),
        "tts_engine": values.get("tts_engine", "voxcpm"),
        "voice": values.get("voice", ""),
        "voice_conversion": values.get("voice_conversion", "seed-vc"),
        "style": values.get("style", "natural"),
        "gender": values.get("gender", "male"),
        "preview_text": values.get("preview_text", ""),
        "approved": bool(values.get("approved", False)),
    }


def validate_profile(profile: dict[str, Any], *, base_dir: Path | None = None, require_approval: bool = False) -> dict[str, Any]:
    value = default_profile(str(profile.get("speaker", "")), profile)
    speaker = value["speaker"]
    if not speaker or not _SPEAKER_ID.fullmatch(speaker):
        raise ValueError(f"invalid speaker id: {speaker!r}")
    if value["reference_mode"] not in REFERENCE_MODES:
        raise ValueError(f"{speaker}: unsupported reference_mode {value['reference_mode']!r}")
    if value["tts_engine"] not in ENGINES:
        raise ValueError(f"{speaker}: unsupported tts_engine {value['tts_engine']!r}")
    if value["voice_conversion"] not in VOICE_CONVERSIONS:
        raise ValueError(f"{speaker}: unsupported voice_conversion {value['voice_conversion']!r}")
    if value["reference_mode"] == "custom":
        raw = str(value.get("reference_path") or "").strip()
        if not raw:
            raise ValueError(f"{speaker}: custom reference requires reference_path")
        path = Path(raw)
        if base_dir and not path.is_absolute():
            path = (base_dir / path).resolve()
        if not path.exists() or not path.is_file():
            raise ValueError(f"{speaker}: custom reference is missing: {path}")
        value["reference_path"] = str(path)
    if value["reference_mode"] == "synthetic" and value["tts_engine"] == "edge" and not str(value.get("voice") or "").strip():
        raise ValueError(f"{speaker}: Edge-TTS synthetic voice requires voice")
    if require_approval and not value["approved"]:
        raise ValueError(f"{speaker}: voice assignment is not approved")
    return value


def load_voice_profiles(
    path: Path | None,
    speakers: list[str],
    *,
    defaults: dict[str, Any] | None = None,
    require_approval: bool = False,
) -> dict[str, dict[str, Any]]:
    raw: dict[str, Any] = {}
    base = Path.cwd()
    if path:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        document = json.loads(path.read_text(encoding="utf-8"))
        raw = document.get("speakers", document)
        if not isinstance(raw, dict):
            raise ValueError("voice profile document must contain a speakers object")
        base = path.resolve().parent
    result: dict[str, dict[str, Any]] = {}
    for speaker in speakers:
        selected = dict(defaults or {})
        selected.update(raw.get(speaker, {}))
        selected["speaker"] = speaker
        result[speaker] = validate_profile(selected, base_dir=base, require_approval=require_approval)
    unknown = sorted(set(raw) - set(speakers))
    if unknown:
        raise ValueError("voice profiles reference unknown speakers: " + ", ".join(unknown))
    return result


def template_for_speakers(speakers: list[str], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "version": 1,
        "speakers": {speaker: default_profile(speaker, defaults) for speaker in speakers},
    }
