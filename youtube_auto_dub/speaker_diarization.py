"""Optional automatic speaker diarization with a conservative fallback."""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)
MODEL = os.environ.get("YAD_DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")


def annotate_segments(audio: Path, segments: list[dict], token: str | None = None) -> list[dict]:
    """Attach speaker labels only when pyannote returns reliable overlap.

    The original ASR timestamps are preserved. When diarization is unavailable,
    the input list is returned untouched, so the pipeline never guesses.
    """
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        log.warning("Speaker diarization skipped: no HF_TOKEN configured")
        return segments
    try:
        import torch
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(MODEL, token=token)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        diarization = pipeline(str(audio))
    except Exception as exc:
        log.warning("Speaker diarization unavailable; keeping conservative mode: %s", exc)
        return segments

    turns = []
    annotation = getattr(diarization, "speaker_diarization", diarization)
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(speaker)))
    if not turns:
        return segments

    out = []
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        scores: dict[str, float] = {}
        for a, b, speaker in turns:
            overlap = max(0.0, min(end, b) - max(start, a))
            if overlap > 0:
                scores[speaker] = scores.get(speaker, 0.0) + overlap
        item = dict(seg)
        if scores:
            speaker, overlap = max(scores.items(), key=lambda p: p[1])
            coverage = overlap / max(end - start, 0.001)
            if coverage >= 0.55:
                item["speaker"] = speaker
                item["confidence"] = round(min(1.0, coverage), 3)
        out.append(item)
    return out
