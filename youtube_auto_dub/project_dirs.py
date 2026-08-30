"""Per-dub project folders.

Every dub gets its own directory under ``projects/`` instead of everything
landing in a shared ``samples/`` and ``output/``. A project is self-contained,
so it can be zipped, moved or deleted without touching anything else:

    projects/<slug>/
        source/     the original clip
        voices/     one recording per subtitle cue
        output/     the finished mp4 + srt
        work/       scratch: stems, retimed clips (safe to delete)
        project.json  script, timings, casting, render settings

The layout is deliberately flat and predictable so a human can open the folder
and immediately see what a dub is made of.
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = Path(
    __import__("os").environ.get("PROSTUDIO_PROJECTS", ROOT / "projects")
)

_SLUG_STRIP = re.compile(r"[^\w\s-]", re.UNICODE)
_SLUG_SPACE = re.compile(r"[\s_]+")


def slugify(name: str, max_len: int = 48) -> str:
    """Filesystem-safe folder name that still reads like the title.

    Arabic and other non-Latin titles are kept as-is rather than transliterated
    into noise; only characters that break paths are removed.
    """
    text = unicodedata.normalize("NFKC", (name or "").strip())
    text = _SLUG_STRIP.sub("", text)
    text = _SLUG_SPACE.sub("-", text).strip("-")
    text = text[:max_len].strip("-")
    return text or "dub"


def unique_slug(base: str, parent: Optional[Path] = None) -> str:
    """``base``, or ``base-2``, ``base-3``… if that folder already exists."""
    parent = parent or PROJECTS_DIR
    slug = slugify(base)
    if not (parent / slug).exists():
        return slug
    for n in range(2, 1000):
        candidate = f"{slug}-{n}"
        if not (parent / candidate).exists():
            return candidate
    return f"{slug}-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"


@dataclass
class DubProject:
    """One dub, on disk."""

    slug: str
    title: str = ""
    source_name: str = ""
    lang: str = "ar"
    dialect: str = ""
    created_at: str = ""
    cues: list = field(default_factory=list)
    voices: dict = field(default_factory=dict)
    render: dict = field(default_factory=dict)
    root: Path = field(default=None, repr=False)  # type: ignore[assignment]

    # ── layout ─────────────────────────────────────────────────────────
    @property
    def source_dir(self) -> Path:
        return self.root / "source"

    @property
    def voices_dir(self) -> Path:
        return self.root / "voices"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def manifest_path(self) -> Path:
        return self.root / "project.json"

    @property
    def video_path(self) -> Path:
        return self.output_dir / f"{self.slug}.mp4"

    @property
    def srt_path(self) -> Path:
        return self.output_dir / f"{self.slug}.srt"

    def cue_take(self, index: int, speaker: str) -> Path:
        """Recording for cue ``index`` (1-based), e.g. voices/c07_f.wav."""
        return self.voices_dir / f"c{index:02d}_{speaker}.wav"

    # ── lifecycle ──────────────────────────────────────────────────────
    def ensure_dirs(self) -> "DubProject":
        for d in (self.source_dir, self.voices_dir, self.output_dir, self.work_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def save(self) -> Path:
        self.ensure_dirs()
        data = {k: v for k, v in asdict(self).items() if k != "root"}
        self.manifest_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.manifest_path

    def clean_work(self) -> None:
        """Drop scratch files; the project stays reproducible without them."""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def summary(self) -> dict[str, Any]:
        takes = len(list(self.voices_dir.glob("*.wav"))) if self.voices_dir.exists() else 0
        return {
            "slug": self.slug,
            "title": self.title or self.slug,
            "lang": self.lang,
            "dialect": self.dialect,
            "cues": len(self.cues),
            "takes": takes,
            "rendered": self.video_path.exists(),
            "created_at": self.created_at,
        }


def create(
    name: str,
    *,
    title: str = "",
    lang: str = "ar",
    dialect: str = "",
    **extra,
) -> DubProject:
    """Make a fresh project folder for a new dub.

    ``name`` drives the folder slug; ``title`` is the human label and defaults
    to ``name``.
    """
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = unique_slug(name)
    project = DubProject(
        slug=slug,
        title=title or name or slug,
        lang=lang,
        dialect=dialect,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        root=PROJECTS_DIR / slug,
        **extra,
    )
    project.save()
    return project


def load(slug: str) -> DubProject:
    root = PROJECTS_DIR / slug
    manifest = root / "project.json"
    if not manifest.exists():
        raise FileNotFoundError(f"No project at {root}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data.pop("root", None)
    return DubProject(root=root, **data)


def load_or_create(name: str, **kwargs) -> DubProject:
    """Reuse the project whose slug matches ``name``, else start one."""
    slug = slugify(name)
    if (PROJECTS_DIR / slug / "project.json").exists():
        return load(slug)
    return create(name, **kwargs)


def list_projects() -> list[dict[str, Any]]:
    """Every project on disk, newest first."""
    if not PROJECTS_DIR.is_dir():
        return []
    out = []
    for entry in PROJECTS_DIR.iterdir():
        if not (entry / "project.json").exists():
            continue
        try:
            out.append(load(entry.name).summary())
        except Exception:
            continue
    return sorted(out, key=lambda p: p.get("created_at") or "", reverse=True)
