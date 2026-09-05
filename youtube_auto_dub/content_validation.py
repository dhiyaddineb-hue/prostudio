"""Phrase completeness and word-timing quality checks for generated speech."""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

_TOKEN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)?", re.UNICODE)


def normalize_tokens(text: str) -> list[str]:
    return [token.lower().replace("’", "'") for token in _TOKEN.findall(text or "")]


def validate_spoken_content(
    expected: str,
    observed: str,
    *,
    min_recall: float = 0.70,
    min_sequence_ratio: float = 0.58,
) -> dict[str, Any]:
    expected_tokens = normalize_tokens(expected)
    observed_tokens = normalize_tokens(observed)
    matcher = SequenceMatcher(a=expected_tokens, b=observed_tokens, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    recall = matched / max(len(expected_tokens), 1)
    sequence_ratio = matcher.ratio()
    length_ratio = len(observed_tokens) / max(len(expected_tokens), 1)
    missing = []
    for tag, a0, a1, _b0, _b1 in matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            missing.extend(expected_tokens[a0:a1])
    ok = bool(expected_tokens) and recall >= min_recall and sequence_ratio >= min_sequence_ratio and length_ratio >= 0.65
    return {
        "ok": ok,
        "expected_words": len(expected_tokens),
        "observed_words": len(observed_tokens),
        "matched_words": matched,
        "recall": round(recall, 4),
        "sequence_ratio": round(sequence_ratio, 4),
        "length_ratio": round(length_ratio, 4),
        "missing_words": missing,
        "expected_text": expected,
        "observed_text": observed,
    }


def word_timing_report(observed_words: list[dict], speech_start: float, speech_end: float) -> dict[str, Any]:
    valid = [word for word in observed_words if word.get("start") is not None and word.get("end") is not None]
    if not valid or speech_end <= speech_start:
        return {"ok": False, "word_count": 0, "reason": "missing_word_timestamps"}
    local_start = float(valid[0]["start"])
    local_end = float(valid[-1]["end"])
    actual_span = max(local_end - local_start, 1e-6)
    target_span = speech_end - speech_start
    rows = []
    drifts = []
    for word in valid:
        rel_start = (float(word["start"]) - local_start) / actual_span
        rel_end = (float(word["end"]) - local_start) / actual_span
        ideal_start = speech_start + rel_start * target_span
        ideal_end = speech_start + rel_end * target_span
        actual_global_start = speech_start + float(word["start"])
        actual_global_end = speech_start + float(word["end"])
        drift = actual_global_end - ideal_end
        drifts.append(abs(drift))
        rows.append({
            "word": str(word.get("word", "")).strip(),
            "actual_start": round(actual_global_start, 3),
            "actual_end": round(actual_global_end, 3),
            "ideal_start": round(ideal_start, 3),
            "ideal_end": round(ideal_end, 3),
            "end_drift": round(drift, 3),
        })
    return {
        "ok": True,
        "word_count": len(rows),
        "mean_abs_drift": round(sum(drifts) / len(drifts), 4),
        "max_abs_drift": round(max(drifts), 4),
        "words": rows,
    }
