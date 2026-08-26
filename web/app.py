"""ProStudio — ready-to-use YouTube Auto Dub studio."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from youtube_auto_dub.core import run as run_pipeline
from youtube_auto_dub.ffmpeg_bin import ensure_ffmpeg_on_path
from youtube_auto_dub.models import LANG_MAP_PATH, OUTPUT_DIR, TEMP_DIR
from youtube_auto_dub.pipeline_args import build_args
from youtube_auto_dub.project_dirs import PROJECTS_DIR, create as create_project, list_projects
from youtube_auto_dub.runtime import capabilities, have_offline_asr, have_whisper
from youtube_auto_dub.voice import list_voices, load_lang_map, pick_voice

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
UPLOADS = ROOT.parent / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ensure_ffmpeg_on_path()

log = logging.getLogger("prostudio")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

STAGES = [
    {"id": "download", "ar": "التنزيل", "en": "Download"},
    {"id": "transcribe", "ar": "التفريغ", "en": "Transcribe"},
    {"id": "chunk", "ar": "التقسيم", "en": "Chunk"},
    {"id": "translate", "ar": "الترجمة", "en": "Translate"},
    {"id": "tts", "ar": "توليد الصوت", "en": "Neural TTS"},
    {"id": "mix", "ar": "المزامنة", "en": "Tempo Mix"},
    {"id": "render", "ar": "التصدير", "en": "Render"},
]

LANG_LABELS = {
    "ar": "العربية",
    "en": "English",
    "fr": "Français",
    "es": "Español",
    "de": "Deutsch",
    "tr": "Türkçe",
    "it": "Italiano",
    "pt": "Português",
    "ru": "Русский",
    "ja": "日本語",
    "ko": "한국어",
    "zh": "中文",
    "hi": "हिन्दी",
    "vi": "Tiếng Việt",
    "id": "Bahasa Indonesia",
    "ur": "اردو",
    "fa": "فارسی",
    "he": "עברית",
    "nl": "Nederlands",
    "pl": "Polski",
    "uk": "Українська",
    "th": "ไทย",
    "sw": "Kiswahili",
}

VOICE_LABELS = {
    "ar-SA-HamedNeural": "حامد — السعودية",
    "ar-EG-SalmaNeural": "سلمى — مصر",
    "ar-EG-ShakirNeural": "شاكر — مصر",
    "ar-AE-FatimaNeural": "فاطمة — الإمارات",
    "ar-AE-HamdanNeural": "حمدان — الإمارات",
    "ar-MA-JamalNeural": "جمال — المغرب",
    "ar-MA-MounaNeural": "منى — المغرب",
    "ar-IQ-BasselNeural": "باسل — العراق",
    "ar-IQ-RanaNeural": "رنا — العراق",
    "ar-LB-RamiNeural": "رامي — لبنان",
    "ar-LB-LaylaNeural": "ليلى — لبنان",
    "ar-JO-TaimNeural": "تيم — الأردن",
    "ar-JO-SanaNeural": "سناء — الأردن",
    "ar-DZ-IsmaelNeural": "إسماعيل — الجزائر",
    "ar-DZ-AminaNeural": "أمينة — الجزائر",
    "ar-TN-HediNeural": "هادي — تونس",
    "ar-TN-ReemNeural": "ريم — تونس",
}

jobs: Dict[str, Dict[str, Any]] = {}
job_queues: Dict[str, asyncio.Queue] = {}
lock = asyncio.Lock()
worker_busy = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _push(job_id: str, event: Dict[str, Any]) -> None:
    job = jobs[job_id]
    event.setdefault("ts", _now())
    job["events"].append(event)
    job["updated_at"] = event["ts"]
    if event.get("stage"):
        job["stage"] = event["stage"]
    if event.get("percent") is not None:
        job["percent"] = event["percent"]
    if event.get("message"):
        job["message"] = event["message"]
    queue = job_queues.get(job_id)
    if queue:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass


def public_job(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": job["id"],
        "status": job["status"],
        "stage": job["stage"],
        "percent": job["percent"],
        "message": job["message"],
        "url": job.get("url"),
        "lang": job.get("lang"),
        "gender": job.get("gender"),
        "mode": job.get("mode"),
        "voice": job.get("voice"),
        "transcript": job.get("transcript") or "",
        "title": job.get("title"),
        "output_name": job.get("output_name"),
        "project": job.get("project"),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "events": job.get("events", [])[-40:],
    }


app = FastAPI(title="ProStudio", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# Serve finished renders straight from samples/ so a dub can be watched or
# shared by URL without going through a job id.
SAMPLES_DIR = ROOT.parent / "samples"
if SAMPLES_DIR.is_dir():
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


if PROJECTS_DIR.is_dir():
    app.mount("/projects", StaticFiles(directory=PROJECTS_DIR), name="projects")


@app.get("/api/projects")
async def api_projects() -> List[Dict[str, Any]]:
    """Every dub project on disk, newest first."""
    items = list_projects()
    for item in items:
        if item.get("rendered"):
            item["watch"] = f"/watch/p/{item['slug']}"
            item["video"] = f"/projects/{item['slug']}/output/{item['slug']}.mp4"
    return items


@app.get("/watch/p/{slug}", response_class=HTMLResponse)
async def watch_project(slug: str) -> HTMLResponse:
    """Player for a project's finished render."""
    safe = Path(slug).name
    video = PROJECTS_DIR / safe / "output" / f"{safe}.mp4"
    if not video.exists():
        raise HTTPException(404, "Not rendered yet")
    subs = video.with_suffix(".srt")
    track = (
        f'<track kind="subtitles" srclang="ar" label="العربية" default '
        f'src="/projects/{safe}/output/{subs.name}">' if subs.exists() else ""
    )
    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe}</title>
<style>
 body{{margin:0;background:#07080c;color:#f4efe4;font-family:system-ui,sans-serif;
      display:grid;place-items:center;min-height:100vh;gap:14px}}
 video{{max-width:min(94vw,460px);max-height:78vh;border-radius:14px;
      box-shadow:0 24px 60px rgba(0,0,0,.5)}}
 a{{color:#e8b86d}}
</style></head><body>
<video controls autoplay playsinline
       src="/projects/{safe}/output/{video.name}">{track}</video>
<p><a href="/projects/{safe}/output/{video.name}" download>تحميل</a></p>
</body></html>""")


@app.get("/watch/{name}", response_class=HTMLResponse)
async def watch(name: str) -> HTMLResponse:
    """Minimal player page for a rendered clip in samples/."""
    safe = Path(name).name  # never escape the samples directory
    video = SAMPLES_DIR / safe
    if not video.exists() or video.suffix.lower() != ".mp4":
        raise HTTPException(404, "Clip not found")
    subs = video.with_suffix(".srt")
    track = (
        f'<track kind="subtitles" srclang="ar" label="العربية" default '
        f'src="/samples/{subs.name}">' if subs.exists() else ""
    )
    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe}</title>
<style>
 body{{margin:0;background:#07080c;color:#f4efe4;font-family:system-ui,sans-serif;
      display:grid;place-items:center;min-height:100vh;gap:16px}}
 video{{max-width:min(94vw,460px);max-height:80vh;border-radius:14px;
      box-shadow:0 24px 60px rgba(0,0,0,.5)}}
 a{{color:#e8b86d}}
</style></head><body>
<video controls autoplay playsinline src="/samples/{safe}">{track}</video>
<p><a href="/samples/{safe}" download>تحميل {safe}</a></p>
</body></html>""")

DEMO_ID = "demo-ready"
DEMO_VIDEO = next(
    (p for p in (
        ROOT.parent / "samples" / "ProStudio_Arabic_Pro.mp4",
        OUTPUT_DIR / "ProStudio_Arabic_Demo.mp4",
        ROOT.parent / "samples" / "ProStudio_Arabic_Demo.mp4",
    ) if p.exists()),
    OUTPUT_DIR / "ProStudio_Arabic_Demo.mp4",
)
DEMO_SRT = DEMO_VIDEO.with_suffix(".srt")
DEMO_SOURCE = ROOT.parent / "samples" / "prostudio_en.mp4"
DEMO_SCRIPT = (
    "Welcome to ProStudio. This short film shows automatic video dubbing. "
    "First we transcribe the speech. Then we translate the meaning into Arabic. "
    "Finally we generate a new voice and sync it with the picture."
)


def job_output_dir(job: Dict[str, Any]) -> Path:
    """Where a job writes its results: its own project, else the shared dir."""
    slug = job.get("project")
    if slug:
        folder = PROJECTS_DIR / slug / "output"
        folder.mkdir(parents=True, exist_ok=True)
        return folder
    return OUTPUT_DIR


def inspect_transcript(
    transcript: str, source: str, whisper: Optional[bool] = None
) -> str:
    """Resolve the transcript to dub with.

    A pasted transcript always wins. The bundled demo narration stands in only
    for the bundled demo clip — never for a video the user supplied. When there
    is no transcript and no Whisper, the job cannot proceed.
    """
    text = (transcript or "").strip()
    if text:
        return text
    if source and source == str(DEMO_SOURCE):
        return DEMO_SCRIPT
    if whisper is None:
        # Either recogniser can produce a transcript; the pipeline picks.
        whisper = have_whisper() or have_offline_asr()
    if not whisper:
        raise RuntimeError(
            "لا يتوفر التفريغ الآلي على الخادم. "
            "الصق نص الفيديو في خانة النص ثم أعد المحاولة."
        )
    return ""


def seed_demo_job() -> None:
    if not DEMO_VIDEO.exists() or DEMO_ID in jobs:
        return
    jobs[DEMO_ID] = {
        "id": DEMO_ID,
        "status": "done",
        "stage": "done",
        "percent": 100,
        "message": "تجربة جاهزة — دبلجة عربية بصوت استوديو احترافي",
        "url": "demo",
        "source": str(DEMO_SOURCE),
        "lang": "ar",
        "gender": "male",
        "mode": "both",
        "model": "offline",
        "voice": "studio-ar-male",
        "bg_music": True,
        "transcript": "",
        "title": "تجربة ProStudio الجاهزة",
        "output_path": str(DEMO_VIDEO),
        "output_name": DEMO_VIDEO.name,
        "srt_path": str(DEMO_SRT) if DEMO_SRT.exists() else None,
        "error": None,
        "events": [{"type": "done", "stage": "done", "percent": 100, "message": "جاهز"}],
        "created_at": _now(),
        "updated_at": _now(),
    }


@app.on_event("startup")
async def _startup() -> None:
    seed_demo_job()


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/send", response_class=HTMLResponse)
async def send_page() -> FileResponse:
    """Real upload page: drop a file, watch it dub, download the result."""
    return FileResponse(STATIC / "send.html")


@app.get("/projects", response_class=HTMLResponse)
async def projects_page() -> FileResponse:
    return FileResponse(STATIC / "projects.html")


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    try:
        ensure_ffmpeg_on_path()
    except Exception as exc:
        return {"ok": False, "ffmpeg": False, "error": str(exc)}
    caps = capabilities()
    return {"ok": caps["ffmpeg"], "jobs": len(jobs), **caps}


@app.get("/api/meta")
async def meta() -> Dict[str, Any]:
    lang_map = load_lang_map()
    languages = []
    preferred = list(LANG_LABELS.keys())
    for code in preferred:
        if code in lang_map:
            languages.append({
                "code": code,
                "label": LANG_LABELS[code],
                "locale": lang_map[code].get("name", code),
            })
    for code, info in sorted(lang_map.items()):
        if code in LANG_LABELS:
            continue
        languages.append({
            "code": code,
            "label": info.get("name", code),
            "locale": info.get("name", code),
        })

    voices: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    for code, info in lang_map.items():
        voices[code] = {"male": [], "female": []}
        for gender in ("male", "female"):
            for name in info.get("voices", {}).get(gender, []):
                voices[code][gender].append({
                    "id": name,
                    "label": VOICE_LABELS.get(name, name.replace("Neural", "").replace("-", " ")),
                })

    return {
        "stages": STAGES,
        "languages": languages,
        "voices": voices,
        "defaults": {
            "lang": "ar",
            "gender": "male",
            "mode": "both",
            "model": "tiny",
            "bg_music": True,
            "voice": pick_voice("ar", "male"),
        },
    }


@app.get("/api/voices/{lang}")
async def voices_for_lang(lang: str, gender: str = "male") -> Dict[str, Any]:
    items = list_voices(lang, gender)
    return {
        "lang": lang,
        "gender": gender,
        "default": pick_voice(lang, gender) if items else None,
        "voices": [
            {"id": name, "label": VOICE_LABELS.get(name, name)}
            for name in items
        ],
    }


@app.get("/api/jobs")
async def list_jobs() -> List[Dict[str, Any]]:
    ordered = sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True)
    return [public_job(j) for j in ordered[:30]]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return public_job(job)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    queue = job_queues.setdefault(job_id, asyncio.Queue(maxsize=200))

    async def gen():
        snapshot = public_job(job)
        yield f"data: {json.dumps({'type': 'snapshot', **snapshot}, ensure_ascii=False)}\n\n"
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20)
            except asyncio.TimeoutError:
                yield "event: ping\ndata: {}\n\n"
                continue
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") in {"done", "error"}:
                break

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/jobs/{job_id}/download")
async def download_job(job_id: str) -> FileResponse:
    job = jobs.get(job_id)
    if not job or not job.get("output_path"):
        raise HTTPException(404, "Output not ready")
    path = Path(job["output_path"])
    if not path.exists():
        raise HTTPException(404, "File missing")
    return FileResponse(path, filename=path.name, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/preview")
async def preview_job(job_id: str) -> FileResponse:
    job = jobs.get(job_id)
    if not job or not job.get("output_path"):
        raise HTTPException(404, "Output not ready")
    path = Path(job["output_path"])
    if not path.exists():
        raise HTTPException(404, "File missing")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/srt")
async def download_srt(job_id: str) -> FileResponse:
    job = jobs.get(job_id)
    if not job or not job.get("srt_path"):
        raise HTTPException(404, "Subtitles not ready")
    path = Path(job["srt_path"])
    if not path.exists():
        raise HTTPException(404, "File missing")
    return FileResponse(path, filename=path.name, media_type="application/x-subrip")


@app.post("/api/jobs")
async def create_job(
    url: str = Form(""),
    lang: str = Form("ar"),
    gender: str = Form("male"),
    mode: str = Form("both"),
    model: str = Form("tiny"),
    voice: str = Form(""),
    bg_music: str = Form("true"),
    transcript: str = Form(""),
    source_lang: str = Form(""),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    source = (url or "").strip()
    saved_upload: Optional[Path] = None

    project = None
    if file and file.filename:
        # Every upload gets its own project folder instead of a shared
        # uploads/ bucket, so a job's source, voices and output stay together
        # and can be zipped or deleted as one unit.
        stem = Path(file.filename).stem or "dub"
        suffix = Path(file.filename).suffix.lower() or ".mp4"
        project = create_project(stem, title=stem, lang=lang).ensure_dirs()
        saved_upload = project.source_dir / f"source{suffix}"
        with saved_upload.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        source = str(saved_upload)
    elif not source:
        if DEMO_SOURCE.exists():
            source = str(DEMO_SOURCE)
        else:
            raise HTTPException(400, "أدخل رابط يوتيوب أو ارفع ملف فيديو")

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "queued",
        "stage": "queued",
        "percent": 0,
        "message": "في الانتظار",
        "url": url.strip() if url else file.filename if file else source,
        "source": source,
        "lang": lang,
        "gender": gender,
        "mode": mode,
        "model": model,
        "voice": voice or None,
        "bg_music": bg_music.lower() in {"1", "true", "yes", "on"},
        "transcript": (transcript or "").strip(),
        "source_lang": (source_lang or "").strip() or None,
        "title": Path(source).name if saved_upload else url,
        "project": project.slug if project else None,
        "output_path": None,
        "output_name": None,
        "srt_path": None,
        "error": None,
        "events": [],
        "created_at": _now(),
        "updated_at": _now(),
    }
    jobs[job_id] = job
    job_queues[job_id] = asyncio.Queue(maxsize=200)
    _push(job_id, {"type": "status", "stage": "queued", "percent": 0, "message": "تمت إضافة المهمة"})
    asyncio.create_task(_run_job(job_id))
    return public_job(job)


async def _run_job(job_id: str) -> None:
    global worker_busy
    job = jobs[job_id]

    async with lock:
        worker_busy = True
        job["status"] = "running"
        _push(job_id, {"type": "status", "stage": "download", "percent": 3, "message": "بدء المعالجة"})
        loop = asyncio.get_running_loop()

        def on_progress(stage: str, message: str, percent: int) -> None:
            loop.call_soon_threadsafe(
                _push,
                job_id,
                {"type": "progress", "stage": stage, "percent": percent, "message": message},
            )

        try:
            transcript = inspect_transcript(
                job.get("transcript") or "", job.get("source") or ""
            )
            job["transcript"] = transcript
            args = build_args(
                job["source"],
                lang=job["lang"],
                mode=job["mode"],
                gender=job["gender"],
                model=job["model"],
                voice=job["voice"],
                bg_music=job["bg_music"],
                output_dir=str(job_output_dir(job)),
                transcript=transcript,
                source_lang=job.get("source_lang") or None,
            )
            if TEMP_DIR.exists():
                shutil.rmtree(TEMP_DIR, ignore_errors=True)
            TEMP_DIR.mkdir(parents=True, exist_ok=True)

            output = await run_pipeline(args, progress=on_progress)
            job["status"] = "done"
            job["output_path"] = str(output)
            job["output_name"] = Path(output).name
            srt = TEMP_DIR / "subtitles.srt"
            if srt.exists():
                dest = job_output_dir(job) / f"{Path(output).stem}.srt"
                shutil.copy2(srt, dest)
                job["srt_path"] = str(dest)
            _push(job_id, {
                "type": "done",
                "stage": "done",
                "percent": 100,
                "message": "الفيديو جاهز للتحميل",
                "output_name": job["output_name"],
            })
        except Exception as exc:
            log.exception("Job %s failed", job_id)
            job["status"] = "error"
            job["error"] = str(exc)
            _push(job_id, {
                "type": "error",
                "stage": job.get("stage") or "error",
                "percent": job.get("percent") or 0,
                "message": str(exc),
                "trace": traceback.format_exc()[-1500:],
            })
        finally:
            worker_busy = False
