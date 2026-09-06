#!/usr/bin/env python3
"""Publish a finished dub into the library layout the studio page reads.

Layout (one folder per video, sources and dubs kept apart):

    library/<slug>/source.mp4            uploaded by the studio page (or parts:
    library/<slug>/source.mp4.part01of03 rejoined by the workflow at run time)
    library/<slug>/meta.json             title, size, sha256, upload time, status

    dubs/<slug>/final-dub-<run>.mp4      the verified deliverable of one run
    dubs/<slug>/final-dub-<run>.quality.json / .segments.json / .language.json
    dubs/<slug>/meta.json                every run of this video, newest first

The slug is the library folder when the source came from library/; any other
source (samples/, inbox/, a YouTube download) gets a slug from its file name.
Nothing is deleted here: a new run adds a version, it never replaces one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "library"
DUBS = ROOT / "dubs"


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    value = re.sub(r"-{2,}", "-", value)
    return value[:60] or "video"


def slug_for_source(source: str) -> str:
    """library/<slug>/source.* -> <slug>; otherwise the file stem."""
    path = Path(str(source).replace("\\", "/"))
    parts = path.parts
    if "library" in parts:
        index = parts.index("library")
        if index + 1 < len(parts) - 1:
            return slugify(parts[index + 1])
    stem = path.name
    for suffix in (".mp4", ".mkv", ".webm", ".mov", ".part0", ".part1"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
    stem = re.sub(r"\.part\d+of\d+$", "", stem)
    return slugify(stem)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_duration(path: Path) -> float | None:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        return round(float(proc.stdout.strip()), 3)
    except Exception:  # noqa: BLE001 - metadata only
        return None


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except ValueError:
        return {}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def publish(video: Path, *, source: str, run_id: str, output_dir: Path, settings: dict) -> dict:
    if not video.exists() or video.stat().st_size < 1024:
        raise SystemExit(f"missing final video: {video}")
    slug = slug_for_source(source)
    folder = DUBS / slug
    folder.mkdir(parents=True, exist_ok=True)
    base = f"final-dub-{run_id}"
    target = folder / f"{base}.mp4"
    shutil.copy2(video, target)
    reports = {}
    for name, suffix in (("quality-report.json", "quality"), ("segments-report.json", "segments"), ("language-integrity.json", "language")):
        candidate = output_dir / name
        if candidate.exists():
            shutil.copy2(candidate, folder / f"{base}.{suffix}.json")
            reports[suffix] = f"{base}.{suffix}.json"
    quality = read_json(output_dir / "quality-report.json")
    language = read_json(output_dir / "language-integrity.json")
    segments = read_json(output_dir / "segments-report.json")
    entry = {
        "run_id": str(run_id),
        "file": target.name,
        "size": target.stat().st_size,
        "sha256": sha256_file(target),
        "duration": probe_duration(target),
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(source),
        "settings": settings,
        "quality_ok": bool(quality.get("ok")) if quality else None,
        "language": {
            "valid": language.get("valid"),
            "detected": language.get("detected") or language.get("language"),
            "confidence": language.get("confidence") or language.get("probability"),
        } if language else None,
        "segments": len(segments.get("segments", [])) if segments else None,
        "translation_engine": (segments.get("translation") or {}).get("engine") if segments else None,
        "reports": reports,
        "run_url": f"https://github.com/dhiyaddineb-hue/prostudio/actions/runs/{run_id}",
    }
    meta_path = folder / "meta.json"
    meta = read_json(meta_path)
    versions = [v for v in meta.get("versions", []) if str(v.get("run_id")) != str(run_id)]
    versions.insert(0, entry)
    meta.update({"slug": slug, "source": str(source), "updated_at": entry["published_at"], "versions": versions})
    write_json(meta_path, meta)

    # Mark the library item as dubbed (when the source lives in the library).
    lib_meta_path = LIBRARY / slug / "meta.json"
    if lib_meta_path.parent.exists():
        lib_meta = read_json(lib_meta_path)
        lib_meta.update({
            "slug": lib_meta.get("slug") or slug,
            "status": "dubbed",
            "last_dub_run": str(run_id),
            "last_dub_at": entry["published_at"],
            "dubs": len(versions),
        })
        write_json(lib_meta_path, lib_meta)
    return {"slug": slug, "folder": str(folder.relative_to(ROOT)), "file": target.name, "versions": len(versions)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source", required=True, help="the source path the workflow dubbed")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--setting", action="append", default=[], help="key=value pairs recorded with the version")
    args = parser.parse_args()
    settings = {}
    for item in args.setting:
        if "=" in item:
            key, value = item.split("=", 1)
            settings[key.strip()] = value.strip()
    result = publish(args.video, source=args.source, run_id=args.run_id, output_dir=args.output_dir, settings=settings)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
