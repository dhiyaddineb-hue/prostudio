#!/usr/bin/env python3
"""Assemble the professional Arabic demo from studio voice clips.

The neural takes in ``samples/voices`` are laid out at their natural pace with
breathing room between lines, rather than being tempo-crushed into the old
sub-second subtitle slots. The title card is then rendered to the exact length
of the finished narration and a matching SRT is written alongside it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from youtube_auto_dub.ffmpeg_bin import ensure_ffmpeg_on_path, ffmpeg_exe

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
VOICES = SAMPLES / "voices"

SR = 24000
LEAD_IN = 0.7          # silence before the first word
GAP = 0.38             # pause between lines
TAIL = 0.9             # silence after the last word

LINES = [
    "مرحباً بكم في برو ستوديو.",
    "يعرض هذا الفيلم القصير الدبلجة الآلية للفيديو.",
    "أولاً نُفرّغ الكلام.",
    "ثم نترجم المعنى إلى العربية.",
    "وأخيراً نولّد صوتاً جديداً ونزامنه مع الصورة.",
]


def _decode(path: Path) -> np.ndarray:
    """Decode any audio file to mono float32 at SR."""
    res = subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(path), "-ar", str(SR), "-ac", "1",
         "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


def _trim_silence(audio: np.ndarray, floor_db: float = -45.0) -> np.ndarray:
    """Drop leading/trailing silence so our own gaps control the pacing."""
    if audio.size == 0:
        return audio
    frame = int(SR * 0.01)
    n = len(audio) // frame
    if n == 0:
        return audio
    energy = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    loud = np.where(energy > 10 ** (floor_db / 20.0))[0]
    if loud.size == 0:
        return audio
    start = max(int(loud[0]) - 2, 0) * frame
    end = min(int(loud[-1]) + 3, n) * frame
    return audio[start:end]


def _fade(audio: np.ndarray, ms: int = 12) -> np.ndarray:
    n = min(int(SR * ms / 1000), len(audio) // 2)
    if n < 2:
        return audio
    out = audio.copy()
    ramp = np.linspace(0.0, np.pi, n)
    out[:n] *= (0.5 * (1 - np.cos(ramp))).astype(np.float32)
    out[-n:] *= (0.5 * (1 + np.cos(ramp))).astype(np.float32)
    return out


def _stamp(sec: float) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((sec % 1) * 1000):03d}"


def main() -> None:
    ensure_ffmpeg_on_path()
    clips = []
    for i in range(1, len(LINES) + 1):
        src = VOICES / f"line{i}.mp3"
        if not src.exists():
            raise SystemExit(f"missing studio take: {src}")
        clips.append(_fade(_trim_silence(_decode(src))))

    # Lay the takes out sequentially at their natural pace.
    spans: list[tuple[float, float]] = []
    cursor = LEAD_IN
    for clip in clips:
        dur = len(clip) / SR
        spans.append((cursor, cursor + dur))
        cursor += dur + GAP
    total = cursor - GAP + TAIL

    timeline = np.zeros(int(total * SR) + SR, dtype=np.float32)
    for clip, (start, _) in zip(clips, spans):
        at = int(start * SR)
        timeline[at:at + len(clip)] += clip
    timeline = timeline[: int(total * SR)]

    peak = float(np.max(np.abs(timeline)))
    if peak > 0:
        timeline *= min(0.89 / peak, 4.0)

    raw = SAMPLES / "pro_narration_raw.wav"
    narration = SAMPLES / "pro_narration.wav"
    sf.write(raw, timeline, SR)

    # Broadcast-style loudness so it sits at a consistent level.
    subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(raw),
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
         "-ar", str(SR), "-ac", "1", str(narration)],
        check=True, capture_output=True,
    )
    raw.unlink(missing_ok=True)

    video = SAMPLES / "ProStudio_Arabic_Pro.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-y",
         "-loop", "1", "-i", str(SAMPLES / "title_card.png"),
         "-i", str(narration),
         "-shortest",
         "-vf", "scale=1280:720,format=yuv420p",
         "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-r", "25",
         str(video)],
        check=True, capture_output=True,
    )

    srt = video.with_suffix(".srt")
    with srt.open("w", encoding="utf-8") as fh:
        for i, (text, (start, end)) in enumerate(zip(LINES, spans), 1):
            fh.write(f"{i}\n{_stamp(start)} --> {_stamp(end)}\n{text}\n\n")

    print(f"narration {narration} {total:.2f}s")
    print(f"video     {video} {video.stat().st_size} bytes")
    print(f"subtitles {srt}")


if __name__ == "__main__":
    main()
