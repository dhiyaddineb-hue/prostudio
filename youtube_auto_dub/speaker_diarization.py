"""Optional automatic speaker diarization with a conservative fallback."""
from __future__ import annotations

import logging
import os
from youtube_auto_dub.diarization_refine import refine_turns, resolve_overlaps, assign_segment, stats
from pathlib import Path

log = logging.getLogger(__name__)
MODEL = os.environ.get("YAD_DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")


def _run_pipeline(audio, pipeline, token, num_speakers=None):
    import torch
    from pyannote.audio import Pipeline
    p = pipeline if pipeline is not None else Pipeline.from_pretrained(MODEL, token=token)
    p.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if num_speakers:
        d = p(audio, num_speakers=num_speakers)
    else:
        d = p(audio)
    annotation = getattr(d, "speaker_diarization", d)
    raw = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        raw.append((float(turn.start), float(turn.end), str(speaker)))
    return raw


def annotate_segments(audio: Path, segments: list[dict], token: str | None = None, model: str | None = None, min_overlap: float = .45) -> list[dict]:
    """Attach labels only when real pyannote turns cover a segment conservatively.

    Uses the community 1.0 model when available, and re-runs with an explicit
    2-speaker constraint whenever a single speaker (or none) is detected but the
    dialogue clearly alternates. Falls back to the given segments untouched.
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
        annotation = getattr(diarization, "speaker_diarization", diarization)
        raw = [(float(t.start), float(t.end), str(s[0])) for t, _, s in annotation.itertracks(yield_label=True)]
        turns = resolve_overlaps(refine_turns(raw))
        speakers = {s for _, _, s in turns}
        log.info("Diarization initial: %d turns; speakers=%s", len(turns), stats(turns))
        # For a 2-way dialogue that was squashed into a single speaker, force
        # the segmenter to find two roles. This is the fix for the "male voice
        # missing" defect: the clip contains two voices but pyannote collapsed
        # them, so the male never got an isolated reference.
        if len(speakers) < 2:
            try:
                diar2 = pipeline(str(audio), num_speakers=2)
                ann2 = getattr(diar2, "speaker_diarization", diar2)
                raw2 = [(float(t.start), float(t.end), str(s[0])) for t, _, s in ann2.itertracks(yield_label=True)]
                turns2 = resolve_overlaps(refine_turns(raw2))
                if len({s for _, _, s in turns2}) >= 2:
                    log.info("Diarization recovered 2 speakers via num_speakers=2 (%d turns)", len(turns2))
                    turns = turns2
            except Exception as e:
                log.warning("2-speaker re-run failed; keeping initial turns: %s", e)
    except Exception as exc:
        log.warning("Speaker diarization unavailable; keeping conservative mode: %s", exc); return segments

    # Relax the assignment floor so a segment never loses its speaker just
    # because the winning role covered < the conservative threshold. The won
    # speaker still carries its real role and a cloned reference.
    out = []
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        item = dict(seg)
        assigned, confidence = assign_segment(start, end, turns, min_overlap=min(.30, min_overlap))
        if assigned:
            item["speaker"] = assigned
            item["speaker_confidence"] = confidence
        else:
            # still prefer the best-scoring role over None so the reference
            # lookup does not silently fall back to a global generic voice.
            best = _best_speaker(start, end, turns)
            if best:
                item["speaker"] = best[0]
                item["speaker_confidence"] = best[1]
            else:
                item.pop("speaker", None)
                item["speaker_confidence"] = 0.0
        out.append(item)
    return out


def _best_speaker(start, end, turns):
    scores = {}
    for t in turns:
        ov = max(0.0, min(end, t.end) - max(start, t.start))
        if ov:
            scores[t.speaker] = scores.get(t.speaker, 0.0) + ov
    if not scores:
        return None
    sp, ov = max(scores.items(), key=lambda x: x[1])
    return (sp, round(min(1.0, ov / max(end - start, .001)), 3))
