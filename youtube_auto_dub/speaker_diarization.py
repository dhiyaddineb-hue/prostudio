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



def _detect_gender(audio: Path, start: float, end: float) -> str:
    """Classify a speaker's gender from a short excerpt by spectral balance.

    Male voices concentrate energy at lower frequencies than female voices.
    We compute the energy ratio between low (<~150Hz) and high (>~250Hz)
    bands on a mono excerpt and pick male/female from the ratio. The model is
    a strong heuristic and is only used to tag the cloned role, not to pick it.
    """
    import subprocess, tempfile, os, shutil
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, "g.wav")
        subprocess.run(
            ["ffmpeg","-y","-ss",f"{start:.3f}","-t",f"{max(0.05,end-start):.3f}",
             "-i",str(audio),"-ac","1","-ar","16000","-vn",str(out)],
            check=True, capture_output=True)
        import numpy as np
        from scipy.io import wavfile
        try:
            sr, x = wavfile.read(out)
        except Exception:
            return "unknown"
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x.astype(np.float32)
        if x.size < 1600:
            return "unknown"
        n = len(x)
        X = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        low = np.sum(X[(freqs >= 60) & (freqs <= 150)])
        high = np.sum(X[(freqs >= 250) & (freqs <= 4000)])
        if low + high <= 0:
            return "unknown"
        ratio = low / high
        # male: more low-freq energy; female: flatter/brighter spectrum
        if ratio >= 0.9:
            return "male"
        if ratio <= 0.45:
            return "female"
        return "unknown"
    except Exception:
        return "unknown"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def speaker_genders(audio: Path, segments: list[dict]) -> dict:
    """Map speaker -> {gender, start, end} using the longest turn per speaker."""
    by_speaker = {}
    for seg in segments:
        sp = seg.get("speaker")
        if not sp:
            continue
        st, en = float(seg["start"]), float(seg["end"])
        prev = by_speaker.get(sp)
        if prev is None or (en - st) > (prev["end"] - prev["start"]):
            by_speaker[sp] = {"start": st, "end": en}
    result = {}
    for sp, span in by_speaker.items():
        result[sp] = {
            "gender": _detect_gender(audio, span["start"], span["end"]),
            "start": span["start"],
            "end": span["end"],
        }
    return result


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
        raw = [(float(t.start), float(t.end), str(s)) for t, _, s in annotation.itertracks(yield_label=True)]
        turns = resolve_overlaps(refine_turns(raw))
        speakers = {t.speaker for t in turns}
        log.info("Diarization initial: %d turns; speakers=%s", len(turns), stats(turns))
        # For a 2-way dialogue that was squashed into a single speaker, force
        # the segmenter to find two roles. Use min/max_speakers=2 (more
        # reliable than bare num_speakers) so the two roles surface even when
        # their voices are close, and keep turns if they split cleanly.
        if len(speakers) < 2:
            for attempt in ("minmax", "num"):
                try:
                    if attempt == "minmax":
                        diar2 = pipeline(str(audio), min_speakers=2, max_speakers=2)
                    else:
                        diar2 = pipeline(str(audio), num_speakers=2)
                    ann2 = getattr(diar2, "speaker_diarization", diar2)
                    raw2 = [(float(t.start), float(t.end), str(s)) for t, _, s in ann2.itertracks(yield_label=True)]
                    turns2 = resolve_overlaps(refine_turns(raw2))
                    if len({t.speaker for t in turns2}) >= 2:
                        log.info("Diarization recovered 2 speakers (%s) with %d turns", attempt, len(turns2))
                        turns = turns2
                        break
                except Exception as e:
                    log.warning("2-speaker re-run (%s) failed; keeping initial turns: %s", attempt, e)
    except Exception as exc:
        log.warning("Speaker diarization unavailable; keeping conservative mode: %s", exc); return segments

    # Assign with a low floor so a tightly-interleaved dialog does not lose a
    # role merely because no single segment overwhelmingly overlaps one speaker.
    # Every segment still gets its real dominant role + a clone reference.
    out = []
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        item = dict(seg)
        assigned, confidence = assign_segment(start, end, turns, min_overlap=min(.12, min_overlap))
        if assigned:
            item["speaker"] = assigned
            item["speaker_confidence"] = confidence
        else:
            best = _best_speaker(start, end, turns)
            if best:
                item["speaker"] = best[0]
                item["speaker_confidence"] = best[1]
            else:
                item.pop("speaker", None)
                item["speaker_confidence"] = 0.0
        out.append(item)

    # Guarantee BOTH roles surface. If only one distinct speaker ended up
    # assigned (e.g. the second role was consumed by the conservative floor or
    # an early voice is thinner), re-open the assignment so the longest genuine
    # turn of each detected role is represented. This is what makes the
    # per-speaker cloning / speaker-map actually carry two roles for a dialog.
    assigned_spk = {x.get("speaker") for x in out if x.get("speaker")}
    det = {t.speaker for t in turns}
    if len(assigned_spk) < 2 and len(det) >= 2:
        # Tally per-segment best speaker without floor, then re-assign the
        # currently-under-represented role to its most dominant segment.
        for role in sorted(det):
            if role in assigned_spk:
                continue
            # longest turn for this role = stable clone reference
            role_turns = [t for t in turns if t.speaker == role]
            if not role_turns:
                continue
            longest = max(role_turns, key=lambda t: t.duration)
            best_seg = None
            best_ov = 0.0
            for x in out:
                if x.get("speaker") or x.get("speaker_confidence", 0) > 0:
                    continue
                ov = min(longest.end, float(x["end"])) - max(longest.start, float(x["start"]))
                if ov > best_ov:
                    best_ov = ov; best_seg = x
            if best_seg is None and out:
                # fall back to the segment overlapping longest anywhere
                best_seg = max(out, key=lambda x: min(longest.end, float(x["end"])) - max(longest.start, float(x["start"])))
            if best_seg is not None:
                best_seg["speaker"] = role
                best_seg["speaker_confidence"] = 1.0
                assigned_spk.add(role)
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
