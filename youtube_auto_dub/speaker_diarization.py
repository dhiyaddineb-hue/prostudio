"""Optional automatic speaker diarization with a conservative fallback."""
from __future__ import annotations

import logging
import os
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
    turns = []
    annotation = getattr(diarization, "speaker_diarization", diarization)
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(speaker)))
    out = []
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"]); scores: dict[str, float] = {}
        for a, b, speaker in turns:
            overlap = max(0.0, min(end, b) - max(start, a))
            if overlap: scores[speaker] = scores.get(speaker, 0.0) + overlap
        item = dict(seg)
        if scores:
            speaker, overlap = max(scores.items(), key=lambda pair: pair[1])
            coverage = overlap / max(end-start, .001)
            if coverage >= min_overlap: item.update(speaker=speaker, speaker_confidence=round(min(1., coverage), 3))
        out.append(item)
    return out
