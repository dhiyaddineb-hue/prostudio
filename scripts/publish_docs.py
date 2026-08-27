#!/usr/bin/env python3
"""Publish every finished dub to the GitHub Pages folder.

Until now ``docs/`` was hand-filled with one clip, so the second dub was built,
committed and then invisible — there was no link to hand over. This walks
``projects/``, copies each rendered deliverable next to the page that serves it,
converts the SRT to WebVTT (browsers refuse SRT in a ``<track>``), and writes
``docs/projects.json`` so the page discovers dubs instead of hard-coding them.

Every number in that manifest is measured off the rendered file, not copied
from the build log: cue count, speaking rate, tightest gap between lines, true
peak. A claim on the page is therefore always a claim about the file you can
download from the same page.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from youtube_auto_dub.ffmpeg_bin import ensure_ffmpeg_on_path, ffmpeg_exe  # noqa: E402
from youtube_auto_dub.project_dirs import list_projects, load  # noqa: E402

DOCS = ROOT / "docs"


def srt_to_vtt(srt: str) -> str:
    """WebVTT is SRT with a header and dots instead of commas in timestamps."""
    body = srt.replace("\r\n", "\n").strip()
    out = []
    for line in body.split("\n"):
        if "-->" in line:
            line = line.replace(",", ".")
        out.append(line)
    return "WEBVTT\n\n" + "\n".join(out) + "\n"


def probe(video: Path) -> dict:
    """Duration and true peak, read back from the delivered file."""
    ensure_ffmpeg_on_path()
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        subprocess.run(
            [ffmpeg_exe(), "-y", "-v", "error", "-i", str(video),
             "-vn", "-ac", "1", "-ar", "44100", str(wav)],
            check=True, capture_output=True,
        )
        audio, sr = sf.read(wav, dtype="float32")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return {
        "duration": round(len(audio) / sr, 2) if audio.size else 0.0,
        "peak_dbfs": round(20 * np.log10(peak), 2) if peak > 0 else None,
        "clipped": bool(peak >= 0.999),
    }


def metrics(project, video: Path) -> dict:
    """What the page is allowed to claim about this dub.

    Rate, gaps and overlaps come from ``render.measured`` — written by the
    build from where each take actually landed. Falling back to the caption
    windows would describe the script instead of the audio: on the Vikings dub
    the captions imply 17.5 characters/second while the rendered lines run at
    a very different speed, because each take was naturalised and refitted.
    """
    cues = sorted(project.cues, key=lambda c: c["i"])
    measured = (project.render or {}).get("measured") or {}
    m = {
        "cues": len(cues),
        "placed": measured.get("placed"),
        "rate": measured.get("rate"),
        "min_gap": measured.get("min_gap"),
        "overlaps": measured.get("overlaps"),
        "measured": bool(measured),
        "speakers": sorted({c["speaker"] for c in cues}),
    }
    m.update(probe(video))
    return m


def publish() -> list[dict]:
    DOCS.mkdir(parents=True, exist_ok=True)
    published = []

    for summary in list_projects():
        project = load(summary["slug"])
        video, srt = project.video_path, project.srt_path
        if not video.exists():
            print(f"  skip {project.slug}: not rendered yet")
            continue

        shutil.copy2(video, DOCS / video.name)
        if srt.exists():
            shutil.copy2(srt, DOCS / srt.name)
            (DOCS / f"{project.slug}.vtt").write_text(
                srt_to_vtt(srt.read_text(encoding="utf-8")), encoding="utf-8"
            )

        entry = {
            "slug": project.slug,
            "title": project.title or project.slug,
            "dialect": project.dialect or "",
            "created_at": project.created_at,
            "video": video.name,
            "srt": srt.name if srt.exists() else None,
            "vtt": f"{project.slug}.vtt" if srt.exists() else None,
            "size": video.stat().st_size,
            "cast": project.voices or {},
            "metrics": metrics(project, video),
        }
        published.append(entry)
        mt = entry["metrics"]
        print(
            f"  {project.slug}: {mt['cues']} cues, {mt['duration']}s, "
            f"{mt['rate']} ch/s, peak {mt['peak_dbfs']} dBFS"
        )

    published.sort(key=lambda e: e["created_at"] or "", reverse=True)
    (DOCS / "projects.json").write_text(
        json.dumps({"projects": published}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return published


if __name__ == "__main__":
    items = publish()
    print(f"\npublished {len(items)} dub(s) to {DOCS}/projects.json")
