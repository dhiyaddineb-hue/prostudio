"""Optional automatic speaker diarization with a conservative fallback."""
from __future__ import annotations

import logging
import os
from youtube_auto_dub.diarization_refine import refine_turns, resolve_overlaps, assign_segment, stats
from pathlib import Path

log = logging.getLogger(__name__)
MODEL = os.environ.get("YAD_DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")


def annotate_segments(audio: Path, segments: list[dict], token: str | None = None, model: str | None = None, min_overlap: float = .55) -> list[dict]:
    """Attach labels only when real pyannote turns cover a segment conservatively.

    Empty input remains empty: this function never fabricates a whole-video turn.
    """
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        log.warning("Speaker diarization skipped: no HF_TOKEN configured"); return segments
    try:
        import torch
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(model or MODEL, token=token)
        pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        diarization = pipeline(str(audio))
    except Exception as exc:
        log.warning("Speaker diarization unavailable; keeping conservative mode: %s", exc); return segments
    raw_turns = []
    annotation = getattr(diarization, "speaker_diarization", diarization)
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        raw_turns.append((float(turn.start), float(turn.end), str(speaker)))
    turns = resolve_overlaps(refine_turns(raw_turns))
    log.info("Diarization refined: %d raw turns -> %d turns; speakers=%s", len(raw_turns), len(turns), stats(turns))
    out = []
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        item = dict(seg)
        assigned, confidence = assign_segment(start, end, turns, min_overlap=min(.45, min_overlap))
        if assigned:
            item.update(speaker=assigned, speaker_confidence=confidence)
        else:
            item.pop("speaker", None)
            item["speaker_confidence"] = confidence
        out.append(item)
    return out
