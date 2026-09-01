#!/usr/bin/env python3
"""Verify that a rendered dub is spoken in the requested target language."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--expected", required=True)
    p.add_argument("--model", default="base")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(args.video), beam_size=3, vad_filter=True)
    text = " ".join(s.text.strip() for s in segments).strip()
    detected = (info.language or "").lower()
    probability = float(info.language_probability or 0.0)
    expected = args.expected.lower().split("-")[0]
    report = {"expected": expected, "detected": detected, "probability": round(probability, 4), "text_chars": len(text), "valid": bool(text) and detected == expected}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["valid"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
