"""Smart <=10s speech-aware chunk planning and resumable checkpoint state."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2
STAGE_ORDER = (
    "analysis", "translation", "tts", "seed_vc", "timing_fit",
    "content_validation", "audio_mix", "video_render", "checkpoint_upload",
)
STAGE_STATES = {"pending", "success", "failed", "skipped"}
_SENTENCE_END = re.compile(r"[.!?؟。！？]+[\"'»”)]*$")
_CLAUSE_END = re.compile(r"[,،;؛:]+[\"'»”)]*$")
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")
_NON_SPEECH = {
    "music", "silence", "noise", "applause", "laughter", "intro", "outro",
    "موسيقى", "صمت", "ضوضاء", "تصفيق", "ضحك",
}


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_project_id(value: str, max_len: int = 80) -> str:
    cleaned = _SAFE_ID.sub("-", value.strip()).strip("-.")
    return (cleaned[:max_len] or "dub-project")


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _fallback_words(segment: dict) -> list[dict]:
    text = str(segment.get("text", "")).strip()
    tokens = [part for part in text.split() if part]
    start = max(0.0, float(segment.get("start", 0.0)))
    end = max(start, float(segment.get("end", start)))
    if not tokens or end <= start:
        return []
    step = (end - start) / len(tokens)
    return [
        {
            "word": token,
            "start": start + i * step,
            "end": start + (i + 1) * step,
            "confidence": float(segment.get("confidence", 1.0)),
            "speaker": segment.get("speaker"),
        }
        for i, token in enumerate(tokens)
    ]


def flatten_words(raw_segments: Iterable[dict], source_duration: float) -> list[dict]:
    """Return ordered, de-duplicated words with bounded timestamps.

    Whisper word timestamps are preferred. Timed proportional words are used only
    when a transcript source lacks word-level data, so a boundary is still never
    placed in the middle of a token.
    """
    words: list[dict] = []
    seen: set[tuple] = set()
    duration = max(float(source_duration), 0.0)
    for segment in sorted(raw_segments, key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0)))):
        candidates = segment.get("words") or _fallback_words(segment)
        for item in candidates:
            token = str(item.get("word", "")).strip()
            if not token:
                continue
            try:
                start = max(0.0, min(duration, float(item.get("start"))))
                end = max(start, min(duration, float(item.get("end"))))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            normalized = re.sub(r"[^\w\u0600-\u06FF]+", "", token, flags=re.UNICODE).lower()
            # Whisper sometimes emits a single MUSIC/NOISE label covering a long
            # interval. It is not spoken lexical content and must not force a
            # mid-token hard cut or a synthetic spoken word.
            confidence = float(item.get("confidence", segment.get("confidence", 1.0)))
            if end - start > 2.5 and normalized in _NON_SPEECH:
                continue
            # A single low-confidence "word" lasting several seconds is an ASR
            # placeholder/hallucination, not a credible lexical timestamp. Keep
            # the media interval as silence rather than pretending to cut a word.
            if end - start > 3.0 and confidence < 0.50:
                continue
            key = (round(start, 3), round(end, 3), token)
            if key in seen:
                continue
            seen.add(key)
            words.append({
                "id": len(words),
                "word": token,
                "start": start,
                "end": end,
                "confidence": confidence,
                "speaker": item.get("speaker", segment.get("speaker")),
            })
    words.sort(key=lambda item: (item["start"], item["end"], item["id"]))
    for index, item in enumerate(words):
        item["id"] = index
    return words


def _boundary_score(words: list[dict], index: int, cursor: float, target_seconds: float) -> tuple[float, str]:
    word = words[index]
    token = word["word"]
    boundary = float(word["end"])
    next_start = float(words[index + 1]["start"]) if index + 1 < len(words) else boundary
    gap = max(0.0, next_start - boundary)
    reason = "word_boundary"
    score = max(0.0, 28.0 - abs((boundary - cursor) - target_seconds) * 4.0)
    current_speaker = word.get("speaker")
    next_speaker = words[index + 1].get("speaker") if index + 1 < len(words) else current_speaker
    if current_speaker and next_speaker and current_speaker != next_speaker:
        score += 180.0
        reason = "speaker_change"
    elif _SENTENCE_END.search(token):
        score += 130.0
        reason = "sentence_end"
    elif gap >= 0.45:
        score += 105.0
        reason = "natural_pause"
    elif _CLAUSE_END.search(token):
        score += 75.0
        reason = "clause_end"
    elif gap >= 0.20:
        score += 48.0
        reason = "short_pause"
    # Slightly prefer later safe boundaries when scores otherwise tie.
    score += min(max(boundary - cursor, 0.0), target_seconds) * 0.05
    return score, reason


def plan_smart_chunks(
    raw_segments: Iterable[dict],
    source_duration: float,
    *,
    max_seconds: float = 10.0,
    target_seconds: float = 8.0,
    min_seconds: float = 2.5,
) -> list[dict]:
    """Build contiguous video intervals, cutting only at word/silence boundaries.

    Every interval covers source media exactly once. Speech-aware intervals end
    after a complete word; long silent spans are divided at the hard maximum.
    """
    duration = float(source_duration)
    if duration <= 0:
        raise ValueError("source_duration must be positive")
    if not (0 < min_seconds <= target_seconds <= max_seconds):
        raise ValueError("expected 0 < min_seconds <= target_seconds <= max_seconds")

    words = flatten_words(raw_segments, duration)
    chunks: list[dict] = []
    cursor = 0.0
    eps = 1e-6
    while cursor < duration - eps:
        hard = min(duration, cursor + max_seconds)
        if hard >= duration - eps:
            speaker_changes = [
                index for index, word in enumerate(words[:-1])
                if word["end"] > cursor + eps
                and word["end"] <= hard + eps
                and word.get("speaker")
                and words[index + 1].get("speaker")
                and word.get("speaker") != words[index + 1].get("speaker")
            ]
            if speaker_changes:
                boundary = float(words[speaker_changes[0]]["end"])
                reason = "speaker_change"
            else:
                boundary, reason = duration, "source_end"
        else:
            eligible = [
                index for index, word in enumerate(words)
                if word["end"] > cursor + eps and word["end"] <= hard + eps
            ]
            preferred = [
                index for index in eligible
                if words[index]["end"] >= cursor + min_seconds - eps
                or (
                    index + 1 < len(words)
                    and words[index].get("speaker")
                    and words[index + 1].get("speaker")
                    and words[index].get("speaker") != words[index + 1].get("speaker")
                )
            ]
            pool = preferred or eligible
            if pool:
                # If the only completed speech is shorter than the minimum and
                # the remainder of the window is silence, keep that silence in
                # the same media chunk instead of creating a tiny orphan chunk.
                last_index = pool[-1]
                next_word = words[last_index + 1] if last_index + 1 < len(words) else None
                quiet_to_hard = next_word is None or float(next_word["start"]) >= hard - eps
                if not preferred and quiet_to_hard and float(words[last_index]["end"]) < hard - 0.15:
                    boundary = hard
                    reason = "silence_after_speech"
                else:
                    scored = [(_boundary_score(words, index, cursor, target_seconds), index) for index in pool]
                    (_, reason), best_index = max(scored, key=lambda item: (item[0][0], words[item[1]]["end"]))
                    boundary = float(words[best_index]["end"])
            else:
                # No complete word ends in this interval. If speech begins later,
                # cut in the preceding silence; otherwise this is an abnormally
                # long token and the hard media boundary is the only safe guard.
                future = next((word for word in words if word["start"] > cursor + eps), None)
                if future and future["start"] <= hard and future["start"] > cursor + eps:
                    boundary = float(future["start"])
                    reason = "silence_before_speech"
                else:
                    boundary = hard
                    reason = "silent_window" if not any(word["start"] < hard and word["end"] > cursor for word in words) else "hard_limit_guard"

        boundary = min(duration, max(boundary, cursor + min(0.05, duration - cursor)))
        if boundary - cursor > max_seconds + eps:
            boundary = min(duration, cursor + max_seconds)
            reason = "hard_limit_guard"

        members = [word for word in words if word["end"] > cursor + eps and word["end"] <= boundary + eps]
        text = " ".join(word["word"] for word in members).strip()
        chunk = {
            "index": len(chunks),
            "start": round(cursor, 3),
            "end": round(boundary, 3),
            "duration": round(boundary - cursor, 3),
            "speech_start": round(min((word["start"] for word in members), default=cursor), 3),
            "speech_end": round(max((word["end"] for word in members), default=cursor), 3),
            "source_text": text,
            "source_words": [
                {"id": word["id"], "word": word["word"], "start": round(float(word["start"]), 3),
                 "end": round(float(word["end"]), 3), "speaker": word.get("speaker")}
                for word in members
            ],
            "word_ids": [word["id"] for word in members],
            "word_count": len(members),
            "confidence": min((word["confidence"] for word in members), default=1.0),
            "speaker": next((word.get("speaker") for word in members if word.get("speaker")), None),
            "cut_reason": reason,
            "status": "pending",
            "attempts": 0,
        }
        chunks.append(chunk)
        cursor = boundary

    validate_plan(chunks, duration, max_seconds=max_seconds, expected_word_ids=[word["id"] for word in words])
    return chunks


def validate_plan(chunks: list[dict], source_duration: float, *, max_seconds: float, expected_word_ids: list[int] | None = None) -> None:
    if not chunks:
        raise ValueError("chunk plan is empty")
    tolerance = 0.002
    cursor = 0.0
    observed: list[int] = []
    for expected_index, chunk in enumerate(chunks):
        start, end = float(chunk["start"]), float(chunk["end"])
        if int(chunk["index"]) != expected_index:
            raise ValueError("chunk indexes are not contiguous")
        if abs(start - cursor) > tolerance:
            raise ValueError(f"timeline gap/overlap before chunk {expected_index}: {cursor} -> {start}")
        if end <= start:
            raise ValueError(f"chunk {expected_index} has non-positive duration")
        if end - start > max_seconds + tolerance:
            raise ValueError(f"chunk {expected_index} exceeds hard maximum")
        observed.extend(int(value) for value in chunk.get("word_ids", []))
        cursor = end
    if abs(cursor - float(source_duration)) > tolerance:
        raise ValueError(f"chunk plan ends at {cursor}, expected {source_duration}")
    if expected_word_ids is not None and observed != expected_word_ids:
        raise ValueError("word coverage is not exactly-once and ordered")


class CheckpointStore:
    """Atomic per-project and per-chunk state; never deletes user data."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.analysis_dir = self.root / "analysis"
        self.chunks_dir = self.root / "chunks"
        self.analysis_dir.mkdir(exist_ok=True)
        self.chunks_dir.mkdir(exist_ok=True)
        self.data: dict = {}
        if self.manifest_path.exists():
            self.data = json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def initialize(self, *, source: dict, config: dict, chunks: list[dict]) -> None:
        plan_hash = stable_hash([{key: chunk.get(key) for key in ("index", "start", "end", "source_text", "word_ids")} for chunk in chunks])
        config_digest = stable_hash(config)
        if self.data:
            if self.data.get("source", {}).get("sha256") != source.get("sha256"):
                raise RuntimeError("checkpoint source hash does not match the current video")
            if self.data.get("config_hash") != config_digest:
                raise RuntimeError("checkpoint configuration changed; use a new project id")
            if self.data.get("plan_hash") != plan_hash:
                raise RuntimeError("smart chunk plan changed; use a new project id")
            previous = {int(item["index"]): item for item in self.data.get("chunks", [])}
            merged = []
            for chunk in chunks:
                saved = previous.get(int(chunk["index"]), {})
                merged.append({**chunk, **{key: value for key, value in saved.items() if key not in {"start", "end", "source_text", "word_ids", "word_count"}}})
            self.data["chunks"] = merged
        else:
            self.data = {
                "schema_version": SCHEMA_VERSION,
                "state": "in_progress",
                "cleanup_authorized": False,
                "source": source,
                "config": config,
                "config_hash": config_digest,
                "plan_hash": plan_hash,
                "chunks": chunks,
                "errors": [],
            }
        self.data["schema_version"] = SCHEMA_VERSION
        self.save()
        self.ensure_checklists()
        for chunk in self.data["chunks"]:
            self._write_chunk_status(int(chunk["index"]))

    def _legacy_checklist(self, index: int, chunk: dict) -> dict[str, dict]:
        directory = self.chunk_dir(index)
        spoken = bool(chunk.get("source_text")) and not bool(chunk.get("non_speech_label"))
        completed = chunk.get("status") == "completed"
        content_ok = bool((chunk.get("content_validation") or {}).get("ok"))
        generated = next(iter(sorted(directory.glob("generated*.fitted.wav"))), None)
        seed_voice = next(iter(sorted(directory.glob("seedvc.voice-only*.wav"))), None)
        mixed = directory / "mixed.wav"
        dubbed = directory / "dubbed.mp4"
        stages = {name: {"state": "pending", "attempts": 0, "error": None} for name in STAGE_ORDER}
        stages["analysis"]["state"] = "success"
        stages["translation"]["state"] = "success" if chunk.get("translated_text") else ("skipped" if not spoken else "pending")
        stages["tts"]["state"] = "success" if generated else ("skipped" if not spoken else "pending")
        if not spoken or not chunk.get("seed_vc_required"):
            stages["seed_vc"]["state"] = "skipped"
        elif seed_voice:
            stages["seed_vc"]["state"] = "success"
        stages["timing_fit"]["state"] = "success" if chunk.get("delivery_fitted_duration") else ("skipped" if not spoken else "pending")
        stages["content_validation"]["state"] = "success" if content_ok else ("skipped" if not spoken else "pending")
        stages["audio_mix"]["state"] = "success" if mixed.exists() and mixed.stat().st_size > 1024 and completed else "pending"
        stages["video_render"]["state"] = "success" if dubbed.exists() and dubbed.stat().st_size > 1024 and completed else "pending"
        stages["checkpoint_upload"]["state"] = "success" if completed else "pending"
        error = chunk.get("error")
        if error and not completed:
            lowered = str(error).lower()
            failed_stage = "content_validation" if "spoken phrase" in lowered else ("timing_fit" if "truncated" in lowered or "fit" in lowered else "video_render")
            stages[failed_stage]["state"] = "failed"
            stages[failed_stage]["error"] = str(error)
        return stages

    def ensure_checklists(self) -> None:
        changed = False
        for index, chunk in enumerate(self.data.get("chunks", [])):
            checklist = chunk.get("checklist")
            if not isinstance(checklist, dict):
                checklist = self._legacy_checklist(index, chunk)
                chunk["checklist"] = checklist
                changed = True
            for name in STAGE_ORDER:
                if name not in checklist:
                    checklist[name] = {"state": "pending", "attempts": 0, "error": None}
                    changed = True
        if changed:
            self.data["schema_version"] = SCHEMA_VERSION
            self.save()

    def stage(self, index: int, name: str) -> dict:
        if name not in STAGE_ORDER:
            raise KeyError(name)
        self.ensure_checklists()
        return self.chunk(index)["checklist"][name]

    def stage_valid(self, index: int, name: str, output: Path | None = None, input_hash: str | None = None) -> bool:
        stage = self.stage(index, name)
        if stage.get("state") not in {"success", "skipped"}:
            return False
        if input_hash is not None and stage.get("input_hash") not in {None, input_hash}:
            return False
        if output is not None:
            output = Path(output)
            if not output.exists() or output.stat().st_size < 1:
                return False
            expected = stage.get("output_sha256")
            if expected and sha256_file(output) != expected:
                return False
        return True

    def mark_stage(
        self, index: int, name: str, state: str, *, output: Path | None = None,
        input_hash: str | None = None, error: str | None = None, details: dict | None = None,
    ) -> dict:
        if name not in STAGE_ORDER or state not in STAGE_STATES:
            raise ValueError(f"invalid stage transition: {name}={state}")
        stage = self.stage(index, name)
        if state == "failed" or (state == "pending" and stage.get("state") != "pending"):
            stage["attempts"] = int(stage.get("attempts", 0)) + 1
        stage.update({"state": state, "error": error})
        if input_hash is not None:
            stage["input_hash"] = input_hash
        if output is not None:
            output = Path(output)
            stage["output"] = str(output.relative_to(self.root)) if output.is_relative_to(self.root) else str(output)
            if output.exists() and output.is_file():
                stage["output_sha256"] = sha256_file(output)
                stage["bytes"] = output.stat().st_size
        if details:
            stage.setdefault("details", {}).update(details)
        self.save()
        self._write_chunk_status(index)
        return stage

    def invalidate_from(self, index: int, name: str, reason: str) -> None:
        start = STAGE_ORDER.index(name)
        self.ensure_checklists()
        checklist = self.chunk(index)["checklist"]
        for stage_name in STAGE_ORDER[start:]:
            previous = checklist[stage_name].get("state")
            checklist[stage_name]["state"] = "pending"
            checklist[stage_name]["error"] = None
            checklist[stage_name]["invalidated_from"] = previous
            checklist[stage_name]["invalidation_reason"] = reason
        self.save()
        self._write_chunk_status(index)

    def save(self) -> None:
        atomic_write_json(self.manifest_path, self.data)

    def chunk_dir(self, index: int) -> Path:
        path = self.chunks_dir / f"{index:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def chunk(self, index: int) -> dict:
        return self.data["chunks"][index]

    def update_chunk(self, index: int, **changes: Any) -> dict:
        chunk = self.chunk(index)
        chunk.update(changes)
        if changes.get("status") == "completed" and isinstance(chunk.get("checklist"), dict):
            rendered = self.chunk_dir(index) / "dubbed.mp4"
            if rendered.exists() and rendered.stat().st_size > 1024:
                stage = chunk["checklist"].setdefault(
                    "video_render", {"state": "pending", "attempts": 0, "error": None},
                )
                stage.update({
                    "state": "success", "error": None,
                    "output": str(rendered.relative_to(self.root)),
                    "output_sha256": sha256_file(rendered),
                    "bytes": rendered.stat().st_size,
                })
        self.save()
        self._write_chunk_status(index)
        return chunk

    def _write_chunk_status(self, index: int) -> None:
        atomic_write_json(self.chunk_dir(index) / "status.json", self.chunk(index))

    def completed_file(self, index: int) -> Path | None:
        chunk = self.chunk(index)
        self.ensure_checklists()
        if not self.stage_valid(index, "video_render"):
            return None
        path = self.chunk_dir(index) / "dubbed.mp4"
        if not path.exists() or path.stat().st_size < 1024:
            return None
        expected = chunk.get("dubbed_sha256")
        if expected and sha256_file(path) != expected:
            return None
        return path

    def add_error(self, index: int | None, message: str) -> None:
        self.data.setdefault("errors", []).append({"chunk": index, "message": str(message)})
        self.save()

    def mark_state(self, state: str, **values: Any) -> None:
        self.data["state"] = state
        self.data.update(values)
        self.save()

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for chunk in self.data.get("chunks", []):
            status = str(chunk.get("status", "pending"))
            counts[status] = counts.get(status, 0) + 1
        return {"state": self.data.get("state"), "total": len(self.data.get("chunks", [])), "by_status": counts, "cleanup_authorized": False}
