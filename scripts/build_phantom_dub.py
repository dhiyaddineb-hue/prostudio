#!/usr/bin/env python3
"""Dub the Phantom Thread clip with the approved studio voices.

The source is a subtitled montage: the script was recovered from the burned-in
Arabic subtitles, and each line's timing from when that subtitle is on screen.
Lines are grouped per speaker so each take is delivered as one continuous
thought instead of chopped per caption.

Fitting strategy, in order of preference:
  1. Keep the take at its natural pace.
  2. Let it run into the silent gap after its slot (nothing is spoken there).
  3. Only then apply a mild atempo, capped so the voice never sounds rushed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from youtube_auto_dub.ffmpeg_bin import ensure_ffmpeg_on_path, ffmpeg_exe

ROOT = Path(__file__).resolve().parent.parent
CLIP = ROOT / "samples" / "phantom"
SRC = ROOT / "inbox" / (
    "Phantom Threadفنانٌ في الحياكة يريد امرأةً تلهمه من دون أن تربك نظامه،"
    " وهي ترفض أن تبقى مجرد مُ.mp4"
)
OUT_VIDEO = ROOT / "samples" / "Phantom_Thread_Pro_DUB.mp4"
OUT_SRT = OUT_VIDEO.with_suffix(".srt")

SR = 24000
MAX_TEMPO = 1.16      # beyond this the read starts to sound hurried
DUCK_DB = -15.0       # how far the original bed drops under the dialogue
DIALOG_LEAD_DB = 10.0  # how far dialogue must sit above the ducked score

# (take, slot start, hard limit before the next voice enters)
GROUPS = [
    ("g1_f", 0.40, 2.30),
    ("g2_m", 2.40, 13.30),
    ("g3_f", 13.40, 17.20),
    ("g4_m", 17.30, 19.10),
    ("g5_f", 19.20, 22.10),
    ("g6_m", 22.10, 29.30),
    ("g7_f", 29.40, 46.30),
    ("g8_m", 46.40, 58.60),
]

# Per-caption text for the sidecar SRT.
CUES = [
    (0.40, 1.90, "لماذا لستَ متزوّجًا؟"),
    (2.40, 4.70, "أنا على يقين بأن الزواج لم يُكتب لي أبدًا"),
    (5.00, 7.20, "أنا رجلٌ خُلق للعزوبة"),
    (7.70, 9.00, "ولا شفاء لي من ذلك"),
    (9.70, 11.30, "الزواج سيجعلني مخادعًا"),
    (11.30, 12.90, "وأنا لا أريد ذلك أبدًا"),
    (13.40, 15.80, "أظنّ أنك تتظاهر بالقوة فقط"),
    (17.30, 18.90, "لا، أنا قوي"),
    (19.20, 20.30, "قوي أمام مَن؟"),
    (20.50, 22.10, "أتمنى ألا تكون كذلك معي"),
    (22.10, 24.80, "أظنّ أن توقّعات الآخرين"),
    (24.80, 26.40, "وافتراضاتهم"),
    (27.40, 29.00, "هي ما يورث القلب وجعَه"),
    (29.40, 30.50, "أريدك..."),
    (31.10, 32.70, "مستلقيًا على ظهرك"),
    (33.00, 34.00, "عاجزًا"),
    (34.70, 35.60, "رقيقًا"),
    (36.50, 37.60, "مكشوفًا"),
    (38.10, 39.80, "وليس لك سواي لأعتني بك"),
    (41.10, 43.50, "ثم أريدك أن تستعيد قوتك من جديد"),
    (46.40, 49.20, "أسمع صوتك يناديني باسمي في أحلامي"),
    (49.20, 50.90, "وحين أستيقظ"),
    (50.90, 53.40, "أجد الدموع تنهمر على وجهي"),
    (54.80, 55.40, "أشتاقُ إليكِ."),
]


def decode(path: Path, sr: int = SR) -> np.ndarray:
    res = subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(path), "-ar", str(sr), "-ac", "1",
         "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


def trim(audio: np.ndarray, floor_db: float = -45.0) -> np.ndarray:
    if audio.size == 0:
        return audio
    fr = int(SR * 0.01)
    n = len(audio) // fr
    if n == 0:
        return audio
    e = np.sqrt(np.mean(audio[: n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)
    loud = np.where(e > 10 ** (floor_db / 20.0))[0]
    if loud.size == 0:
        return audio
    a = max(int(loud[0]) - 2, 0) * fr
    b = min(int(loud[-1]) + 3, n) * fr
    return audio[a:b]


def tempo(path: Path, factor: float, dst: Path) -> np.ndarray:
    chain, r = [], factor
    while r > 2.0:
        chain.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        chain.append("atempo=0.5")
        r /= 0.5
    chain.append(f"atempo={r:.6f}")
    subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(path), "-filter:a", ",".join(chain),
         "-ar", str(SR), "-ac", "1", str(dst)],
        capture_output=True, check=True,
    )
    return sf.read(str(dst), dtype="float32")[0]


def fade(a: np.ndarray, ms: int = 14) -> np.ndarray:
    n = min(int(SR * ms / 1000), len(a) // 2)
    if n < 2:
        return a
    out = a.copy()
    ramp = np.linspace(0.0, np.pi, n)
    out[:n] *= (0.5 * (1 - np.cos(ramp))).astype(np.float32)
    out[-n:] *= (0.5 * (1 + np.cos(ramp))).astype(np.float32)
    return out


def stamp(sec: float) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((sec % 1) * 1000)):03d}"


def main() -> None:
    ensure_ffmpeg_on_path()
    if not SRC.exists():
        raise SystemExit(f"source clip missing: {SRC}")

    bed = decode(SRC)
    total = len(bed)
    voice = np.zeros(total + SR, dtype=np.float32)
    gate = np.zeros(total + SR, dtype=np.float32)

    for name, start, limit in GROUPS:
        take = CLIP / f"{name}.mp3"
        if not take.exists():
            raise SystemExit(f"missing take: {take}")
        audio = trim(decode(take))
        budget = limit - start
        dur = len(audio) / SR

        if dur > budget:
            factor = min(dur / budget, MAX_TEMPO)
            audio = trim(tempo(take, factor, CLIP / f"{name}_fit.wav"))
            dur = len(audio) / SR
            note = f"atempo x{factor:.3f}"
            if dur > budget:  # still long: allow the tail to breathe past the limit
                note += f" (+{dur - budget:.2f}s tail)"
        else:
            note = "natural"

        audio = fade(audio)
        at = int(start * SR)
        end = min(at + len(audio), len(voice))
        voice[at:end] += audio[: end - at]
        gate[at:end] = 1.0
        print(f"{name}: {dur:5.2f}s into {budget:5.2f}s  {note}")

    voice = voice[:total]
    gate = gate[:total]

    # Smooth the duck so the bed dips around speech rather than stepping.
    win = int(SR * 0.25)
    kernel = np.ones(win, dtype=np.float32) / win
    smooth = np.convolve(gate, kernel, mode="same")
    smooth = np.clip(smooth, 0.0, 1.0)
    duck = 10 ** (DUCK_DB / 20.0)
    ducked = bed * (1.0 - (1.0 - duck) * smooth)

    # Set the dialogue level against the *ducked* score it will actually play
    # over, rather than hoping a fixed gain lands in the right place.
    speaking = gate > 0
    voice_rms = float(np.sqrt(np.mean(voice[speaking] ** 2) + 1e-12))
    bed_rms = float(np.sqrt(np.mean(ducked[speaking] ** 2) + 1e-12))
    if voice_rms > 0:
        target = bed_rms * (10 ** (DIALOG_LEAD_DB / 20.0))
        voice *= min(target / voice_rms, 24.0)
        print(f"dialogue lead: {20 * np.log10(target / bed_rms):+.1f} dB over score")

    mixed = ducked + voice

    peak = float(np.max(np.abs(mixed)))
    if peak > 0.99:
        mixed *= 0.99 / peak

    raw = CLIP / "mix_raw.wav"
    final = CLIP / "mix_final.wav"
    sf.write(raw, mixed, SR)
    subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(raw),
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-ac", "2",
         str(final)],
        check=True, capture_output=True,
    )
    raw.unlink(missing_ok=True)

    subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(SRC), "-i", str(final),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "192k", "-shortest", str(OUT_VIDEO)],
        check=True, capture_output=True,
    )
    final.unlink(missing_ok=True)

    with OUT_SRT.open("w", encoding="utf-8") as fh:
        for i, (s, e, t) in enumerate(CUES, 1):
            fh.write(f"{i}\n{stamp(s)} --> {stamp(e)}\n{t}\n\n")

    print(f"\nvideo     {OUT_VIDEO} ({OUT_VIDEO.stat().st_size} bytes)")
    print(f"subtitles {OUT_SRT}")


if __name__ == "__main__":
    main()
