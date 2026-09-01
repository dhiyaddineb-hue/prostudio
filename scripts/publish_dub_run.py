#!/usr/bin/env python3
"""Persist a GitHub Actions dub output as a self-contained project."""
from __future__ import annotations
import argparse, json, re, shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def slugify(value: str) -> str:
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    value = re.sub(r"[\s_]+", "-", value).strip("-")
    return value[:56] or "dub"

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--source", type=Path, required=False)
    p.add_argument("--source-lang", default="ar")
    p.add_argument("--target-lang", default="en")
    p.add_argument("--engine", default="voxcpm")
    p.add_argument("--seed-vc", default="true")
    p.add_argument("--bg-music", default="false")
    p.add_argument("--diarize", default="false")
    p.add_argument("--run-id", required=True)
    args = p.parse_args()
    if not args.video.exists(): raise SystemExit(f"missing final video: {args.video}")
    base = slugify(args.video.stem.replace("seedvc-enhanced", "dub")) + "-" + args.run_id
    root = ROOT / "projects" / base
    (root / "source").mkdir(parents=True, exist_ok=True)
    (root / "output").mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.video, root / "output" / f"{base}.mp4")
    if args.source and args.source.exists(): shutil.copy2(args.source, root / "source" / args.source.name)
    for candidate in sorted(args.video.parent.glob("*.srt")):
        shutil.copy2(candidate, root / "output" / f"{base}.srt")
        break
    reports = sorted(args.video.parent.glob("*.json"))
    report = {"run_id": args.run_id, "engine": args.engine, "source_lang": args.source_lang, "target_lang": args.target_lang, "seed_vc": args.seed_vc == "true", "bg_music": args.bg_music == "true", "diarize": args.diarize == "true", "video": str(args.video), "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for candidate in reports:
        if candidate.name not in {"lip-sync-report.json"}:
            try: report["pipeline_report"] = json.loads(candidate.read_text(encoding="utf-8")); break
            except Exception: pass
    (root / "output" / f"{base}.pipeline.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"slug": base, "title": f"{args.source_lang} → {args.target_lang} · {args.video.stem}", "source_name": args.source.name if args.source else "", "lang": args.source_lang, "dialect": "", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cues": [], "voices": {"engine": args.engine, "seed_vc": args.seed_vc == "true"}, "render": {"settings": {"target_lang": args.target_lang, "bg_music": args.bg_music == "true", "diarize": args.diarize == "true"}, "measured": {}}}
    (root / "project.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(root)

if __name__ == "__main__": main()
