"""yt-dlp wrapper for downloading YouTube videos with metadata."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

import yt_dlp

from youtube_auto_dub.ffmpeg_bin import ensure_ffmpeg_on_path, ffmpeg_exe
from youtube_auto_dub.models import CACHE_DIR, YT_AUDIO_EXPORT_SR, YT_FORMAT, YT_MIN_FILE_SIZE, ProjectContext, VideoMetadata
from youtube_auto_dub.ui import console

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _extract_metadata(info: dict) -> VideoMetadata:
    tags = info.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    return VideoMetadata(
        title=info.get("title", ""),
        description=info.get("description", ""),
        tags=tags,
        upload_date=info.get("upload_date"),
        duration=info.get("duration", 0.0),
        channel=info.get("channel") or info.get("uploader", ""),
        view_count=info.get("view_count", 0),
        like_count=info.get("like_count", 0),
    )


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    ensure_ffmpeg_on_path()
    subprocess.run(
        [
            ffmpeg_exe(),
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(YT_AUDIO_EXPORT_SR),
            "-ac",
            "1",
            str(audio_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _probe_duration(path: Path) -> float:
    try:
        res = subprocess.run(
            [ffmpeg_exe(), "-i", str(path)],
            capture_output=True,
            text=True,
        )
        match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", res.stderr or "")
        if not match:
            return 0.0
        h, m, s = match.groups()
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return 0.0


def import_local_video(path: Union[str, Path], title: Optional[str] = None) -> ProjectContext:
    """Import a local video file into the same project layout as a YouTube download."""
    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Video file not found: {src}")

    digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:10]
    video_id = f"local-{src.stem[:24]}-{digest}"
    video_path = CACHE_DIR / f"{video_id}.mp4"
    audio_path = CACHE_DIR / f"{video_id}.wav"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() == ".mp4":
        if not video_path.exists():
            shutil.copy2(src, video_path)
    else:
        ensure_ffmpeg_on_path()
        subprocess.run(
            [ffmpeg_exe(), "-y", "-i", str(src), "-c:v", "libx264", "-c:a", "aac", str(video_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    if not audio_path.exists() or audio_path.stat().st_size < YT_MIN_FILE_SIZE:
        console.step("Extracting audio from uploaded file...")
        _extract_audio(video_path, audio_path)

    metadata = VideoMetadata(
        title=title or src.stem,
        duration=_probe_duration(video_path),
        channel="local",
    )
    console.step(f"Imported local source ({video_id})")
    project = ProjectContext(
        video_id=video_id,
        video_path=video_path,
        audio_path=audio_path,
        metadata=metadata,
    )
    project.save_cache("metadata", {
        "title": metadata.title,
        "description": metadata.description,
        "tags": metadata.tags,
        "upload_date": metadata.upload_date,
        "duration": metadata.duration,
        "channel": metadata.channel,
    })
    return project


def _inbox_video() -> Optional[Path]:
    roots = [Path("inbox"), Path.cwd() / "inbox"]
    for root in roots:
        if not root.is_dir():
            continue
        videos = []
        for ext in (".mp4", ".mkv", ".webm", ".mov", ".m4v"):
            videos.extend(root.glob(f"*{ext}"))
        videos = [p for p in videos if p.stat().st_size > 1024]
        if videos:
            return sorted(videos, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return None


def load_source(url_or_path: str, browser: Optional[str] = None) -> ProjectContext:
    """Download a YouTube URL, import a local path, or use an inbox/ upload.

    An explicit source always wins. The ``inbox/`` directory is only a
    convenience for "drop a file in and run" with no arguments — it must never
    hijack a URL or a path the caller actually asked for.
    """
    source = (url_or_path or "").strip()

    if source:
        if _URL_RE.match(source):
            return download_project(source, browser)
        return import_local_video(source)

    inbox = _inbox_video()
    if inbox:
        console.step(f"Using uploaded inbox file: {inbox}")
        return import_local_video(inbox)

    raise ValueError("No video source given: pass a YouTube URL, a file path, or drop a file in inbox/.")


def _normalize_cookie_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    if "youtube.com" not in text and "SID=" in text and ";" in text:
        pairs = [p.strip() for p in re.split(r";\s*", text) if "=" in p]
        lines = ["# Netscape HTTP Cookie File"]
        for pair in pairs:
            name, value = pair.split("=", 1)
            lines.append(f".youtube.com\tTRUE\t/\tTRUE\t0\t{name.strip()}\t{value.strip()}")
        text = "\n".join(lines) + "\n"
    return text


def _cookie_file() -> Optional[str]:
    import os

    raw = os.environ.get("YAD_COOKIES") or os.environ.get("YT_COOKIES")
    path = None
    if raw:
        path = CACHE_DIR / "cookies.txt"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(_normalize_cookie_text(raw), encoding="utf-8")
        return str(path.resolve())
    for candidate in (
        os.environ.get("YAD_COOKIES_FILE"),
        os.environ.get("YT_COOKIES_FILE"),
        "cookies.txt",
        str(Path.cwd() / "cookies.txt"),
        str(CACHE_DIR / "cookies.txt"),
    ):
        if candidate and Path(candidate).exists() and Path(candidate).stat().st_size > 20:
            src = Path(candidate)
            text = _normalize_cookie_text(src.read_text(encoding="utf-8", errors="replace"))
            dest = CACHE_DIR / "cookies.normalized.txt"
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
            return str(dest.resolve())
    return None


def _validate_cookies(path: str) -> None:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    yt_lines = [ln for ln in lines if "youtube.com" in ln]
    tabbed = sum(1 for ln in yt_lines if "\t" in ln)
    names = []
    for ln in yt_lines:
        parts = ln.split("\t")
        if len(parts) >= 6:
            names.append(parts[5])
    console.step(
        f"Cookies {path}: {Path(path).stat().st_size}B lines={len(lines)} "
        f"youtube={len(yt_lines)} netscape={tabbed} names={sorted(set(names))[:12]}"
    )
    if not yt_lines:
        raise RuntimeError("cookies.txt has no youtube.com rows")
    if tabbed == 0:
        raise RuntimeError(
            "cookies.txt is not Netscape format (no tabs). "
            "Use the 'Get cookies.txt LOCALLY' export, not DevTools copy."
        )


def _start_pot_provider() -> None:
    """Start bgutil POT HTTP server so yt-dlp can pass YouTube bot checks."""
    import sys
    import time
    import urllib.request

    console.step("Installing POT provider plugin")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "bgutil-ytdlp-pot-provider"],
        check=False, capture_output=True,
    )
    root = Path.home() / "bgutil-ytdlp-pot-provider"
    if not (root / "server").exists():
        console.step("Cloning POT provider server")
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--branch", "1.3.2",
                "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git",
                str(root),
            ],
            check=False, capture_output=True,
        )
    server = root / "server"
    main_js = server / "build" / "main.js"
    if server.exists() and not main_js.exists():
        console.step("Building POT provider")
        subprocess.run(["npm", "ci", "--ignore-scripts"], cwd=server, check=False, capture_output=True)
        subprocess.run(["npx", "--yes", "tsc"], cwd=server, check=False, capture_output=True)
    if not main_js.exists():
        console.warning("POT provider build missing")
        return
    subprocess.Popen(
        ["node", str(main_js)],
        cwd=str(server),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        try:
            urllib.request.urlopen("http://127.0.0.1:4416", timeout=1)
            console.step("POT provider ready on :4416")
            return
        except Exception:
            time.sleep(0.4)
    console.warning("POT provider did not become ready")


def _ytdlp_js_opts() -> dict:
    opts: dict = {"remote_components": ["ejs:github"]}
    deno = shutil.which("deno")
    node = shutil.which("node")
    if deno:
        opts["js_runtimes"] = {"deno": {"path": deno}}
        console.step(f"JS runtime=deno ({deno})")
    elif node:
        opts["js_runtimes"] = {"node": {"path": node}}
        console.step(f"JS runtime=node ({node})")
    else:
        console.warning("No JS runtime (node/deno) found; YouTube may reject the download")
    return opts


def download_project(url: str, browser: Optional[str] = None) -> ProjectContext:
    ensure_ffmpeg_on_path()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cookie_file = _cookie_file()
    if cookie_file:
        _validate_cookies(cookie_file)
        attempts = [
            (["web"], "bestvideo+bestaudio/best"),
            (["web"], "best"),
            (["tv"], "bv*+ba/b"),
            (["web_safari"], "best"),
            (["android"], "best"),
        ]
    else:
        console.warning("No YouTube cookies file found")
        attempts = [
            (["android", "ios"], "bv*+ba/b"),
            (["ios"], "best"),
            (["tv"], "bv*+ba/b"),
            (["web"], YT_FORMAT),
            (["web"], "bestvideo+bestaudio/best"),
            (["tv_embedded", "android"], "best"),
            (["web"], "best"),
        ]
    last_error = None
    info = None
    for player_clients, fmt in attempts:
        opts = {
            "format": fmt,
            "outtmpl": str(CACHE_DIR / "%(id)s.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "retries": 5,
            "fragment_retries": 5,
            "extractor_args": {"youtube": {"player_client": player_clients}},
        }
        opts.update(_ytdlp_js_opts())
        if browser:
            opts["cookiesfrombrowser"] = (browser.lower(),)
        elif cookie_file:
            opts["cookiefile"] = cookie_file
            console.step(f"Using cookiefile={cookie_file}")
        try:
            console.step(f"YouTube client={','.join(player_clients)} format={fmt}")
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            break
        except Exception as exc:
            last_error = exc
            console.warning(f"Download failed ({player_clients}/{fmt}): {exc}")
            continue
    if info is None:
        raise RuntimeError(f"YouTube download failed: {last_error}") from last_error

    video_id = info["id"]
    video_path = CACHE_DIR / f"{video_id}.mp4"
    audio_path = CACHE_DIR / f"{video_id}.wav"
    metadata = _extract_metadata(info)

    if not video_path.exists():
        # yt-dlp may have written a slightly different extension
        candidates = sorted(CACHE_DIR.glob(f"{video_id}.*"))
        videos = [p for p in candidates if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
        if videos:
            video_path = videos[0]

    if not audio_path.exists() or audio_path.stat().st_size < YT_MIN_FILE_SIZE:
        console.step("Extracting audio format...")
        _extract_audio(video_path, audio_path)

    console.step(f"Downloaded source ({video_id})")
    project = ProjectContext(
        video_id=video_id,
        video_path=video_path,
        audio_path=audio_path,
        metadata=metadata,
    )
    # Cache metadata for future runs
    project.save_cache("metadata", {
        "title": metadata.title,
        "description": metadata.description,
        "tags": metadata.tags,
        "upload_date": metadata.upload_date,
        "duration": metadata.duration,
        "channel": metadata.channel,
    })
    return project
