#!/usr/bin/env python3
"""Dub the Phantom Thread clip over a separated music bed.

Two things make this a real dub rather than a voice-over:

1. **Stem separation.** The original English dialogue is pulled out of the
   centre channel (``youtube_auto_dub.stem_split``) so the Arabic takes play
   over score and effects alone, instead of on top of the English performance.

2. **Measured casting.** Each take is pitch-shifted toward the median F0 of the
   *original* actor in that window, so the dub tracks the performance instead
   of sounding like a narrator. The shift is capped, skipped on lines too short
   to measure, and reverted if it does not actually move toward the actor.

Speaker assignment was confirmed against the picture, not pitch alone: the
centre stem still carries score, and an early pass mis-cast the man at 22-29s
as a woman on a bad reading.

Timing comes from the burned-in subtitles: the script was read off the frames
and each line's slot is when its caption is on screen.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from youtube_auto_dub.ffmpeg_bin import ensure_ffmpeg_on_path, ffmpeg_exe
from youtube_auto_dub.stem_split import decode_stereo, split_center
from youtube_auto_dub.voice_profile import convert, measure

ROOT = Path(__file__).resolve().parent.parent
CLIP = ROOT / "samples" / "phantom"
SRC = ROOT / "inbox" / (
    "Phantom Threadفنانٌ في الحياكة يريد امرأةً تلهمه من دون أن تربك نظامه،"
    " وهي ترفض أن تبقى مجرد مُ.mp4"
)
OUT_VIDEO = ROOT / "samples" / "Phantom_Thread_Pro_DUB.mp4"
OUT_SRT = OUT_VIDEO.with_suffix(".srt")

SR = 44100
MAX_TEMPO = 1.16        # past this the read sounds hurried
DIALOG_LEAD_DB = 11.0   # dialogue level above the residual score
RESIDUAL_DUCK_DB = -9.0  # gentle duck of the music under speech
MAX_SEMITONES = 2.0     # cap on pitch matching, to protect voice quality
MIN_PITCH_MATCH_SEC = 2.5  # shorter lines give an unreliable F0 median

# take, slot start, hard limit before the next voice enters
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
    return audio[max(int(loud[0]) - 2, 0) * fr: min(int(loud[-1]) + 3, n) * fr]


def _filter(src: Path, chain: str, dst: Path) -> np.ndarray:
    subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(src), "-filter:a", chain,
         "-ar", str(SR), "-ac", "1", str(dst)],
        capture_output=True, check=True,
    )
    return sf.read(str(dst), dtype="float32")[0]


def probe_rate(path: Path) -> int:
    """Sample rate of a media file, read from ffmpeg's own report."""
    import re

    res = subprocess.run([ffmpeg_exe(), "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"(\d+) Hz", res.stderr or "")
    return int(m.group(1)) if m else SR


def _atempo_chain(factor: float) -> str:
    parts, r = [], factor
    while r > 2.0:
        parts.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        parts.append("atempo=0.5")
        r /= 0.5
    parts.append(f"atempo={r:.6f}")
    return ",".join(parts)


def shift_pitch(src: Path, semitones: float, dst: Path) -> np.ndarray:
    """Resample-and-retime: changes pitch while preserving duration.

    ``asetrate`` reinterprets the stream's *own* sample rate, so the multiplier
    must be based on the file's rate — using a fixed constant would also
    resample the clip and wreck its duration. It shortens audio by ``ratio``,
    so tempo is divided by the same ratio to restore the original length.
    """
    ratio = 2 ** (semitones / 12.0)
    src_sr = probe_rate(src)
    chain = (
        f"asetrate={int(round(src_sr * ratio))},aresample={SR},"
        + _atempo_chain(1.0 / ratio)
    )
    return _filter(src, chain, dst)


def fade(a: np.ndarray, ms: int = 14) -> np.ndarray:
    n = min(int(SR * ms / 1000), len(a) // 2)
    if n < 2:
        return a
    out = a.copy()
    ramp = np.linspace(0.0, np.pi, n)
    out[:n] *= (0.5 * (1 - np.cos(ramp))).astype(np.float32)
    out[-n:] *= (0.5 * (1 + np.cos(ramp))).astype(np.float32)
    return out


def median_f0(seg: np.ndarray, sr: int = SR) -> float | None:
    """Median fundamental of a window.

    Uses a longer analysis frame plus cumulative mean normalisation (the YIN
    idea) and picks the *first* strong dip rather than the global best. Plain
    autocorrelation happily locks onto a harmonic, which on short lines made
    a downward shift look like an upward one.
    """
    w, h = int(sr * 0.06), int(sr * 0.02)
    lo, hi = int(sr / 300), int(sr / 70)
    vals = []
    for i in range(max((len(seg) - w) // h, 0)):
        s = seg[i * h: i * h + w]
        if np.sqrt(np.mean(s ** 2) + 1e-12) < 1e-3:
            continue
        s = s - s.mean()
        n = len(s)
        ac = np.correlate(s, s, "full")[n - 1:]
        energy = np.concatenate(([np.dot(s, s)], np.cumsum(s[::-1] ** 2)[::-1][1:]))
        # difference function, then cumulative mean normalisation
        d = energy[0] + energy - 2 * ac[: len(energy)]
        d = d[: hi + 1]
        if len(d) <= lo + 2:
            continue
        cm = np.ones_like(d)
        run = np.cumsum(d[1:])
        idx = np.arange(1, len(d))
        cm[1:] = d[1:] * idx / np.maximum(run, 1e-12)
        window = cm[lo:hi]
        if window.size == 0:
            continue
        below = np.where(window < 0.25)[0]
        peak = lo + int(below[0] if below.size else np.argmin(window))
        if cm[peak] >= 0.6:
            continue
        # parabolic refinement around the chosen dip
        refined = float(peak)
        if 0 < peak < len(d) - 1:
            a, b, c = d[peak - 1], d[peak], d[peak + 1]
            denom = a - 2 * b + c
            if abs(denom) > 1e-12:
                refined = peak + 0.5 * float(a - c) / float(denom)
        if refined > 0:
            vals.append(sr / refined)
    return float(np.median(vals)) if len(vals) >= 6 else None


def voiced_f0(seg: np.ndarray, sr: int = SR, keep: float = 0.35) -> float | None:
    """F0 of the loudest, most speech-like part of a separated stem.

    The centre-channel stem still carries some score, and quiet frames are
    mostly that residue. Restricting the estimate to the strongest frames keeps
    the music from dragging the median off the actor's real pitch.
    """
    frame = int(sr * 0.06)
    hop = int(sr * 0.03)
    count = max((len(seg) - frame) // hop, 0)
    if count < 4:
        return None
    energies = np.array([
        np.sqrt(np.mean(seg[i * hop: i * hop + frame] ** 2) + 1e-12)
        for i in range(count)
    ])
    cutoff = np.quantile(energies, 1.0 - keep)
    loud = [i for i in range(count) if energies[i] >= cutoff]
    if not loud:
        return None
    # Concatenate the loud frames into one signal: the estimator needs a run of
    # audio, not isolated 60 ms slices.
    voiced = np.concatenate([seg[i * hop: i * hop + frame] for i in loud])
    return median_f0(voiced, sr)


def stamp(sec: float) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((sec % 1) * 1000)):03d}"


def main() -> None:
    ensure_ffmpeg_on_path()
    if not SRC.exists():
        raise SystemExit(f"source clip missing: {SRC}")

    print("separating stems…")
    left, right = decode_stereo(SRC, SR)
    original_voice, music = split_center(left, right, SR)
    total = len(music)

    voice = np.zeros(total + SR, dtype=np.float32)
    gate = np.zeros(total + SR, dtype=np.float32)

    for name, start, limit in GROUPS:
        take = CLIP / f"{name}.mp3"
        if not take.exists():
            raise SystemExit(f"missing take: {take}")

        # Voice conversion: measure the real actor here and reshape the take
        # toward that voice (pitch + vocal-tract size), not just its pitch.
        work = take
        audio = trim(decode(take))
        notes = []

        actor = measure(original_voice[int(start * SR): int(limit * SR)], SR)
        speaker = measure(audio, SR, focus=False)

        if actor and speaker and actor.reliable and speaker.f0_median > 0:
            converted, report = convert(audio, SR, speaker, actor)
            check = measure(converted, SR, focus=False)
            moved_closer = (
                check
                and abs(check.f0_median - actor.f0_median)
                < abs(speaker.f0_median - actor.f0_median)
            )
            if moved_closer and np.isfinite(converted).all():
                audio = trim(converted)
                work = CLIP / f"{name}_voice.wav"
                sf.write(work, audio, SR)
                notes.append(
                    f"voice {speaker.f0_median:.0f}->{check.f0_median:.0f}Hz"
                    f" (target {actor.f0_median:.0f}), formants x{report['formant_ratio']:.3f}"
                )
            else:
                notes.append("conversion reverted (no improvement)")
        elif actor and not actor.reliable:
            notes.append(f"actor unreliable (F0 IQR {100 * actor.f0_iqr / max(actor.f0_median,1):.0f}%)")

        budget = limit - start
        dur = len(audio) / SR
        if dur > budget:
            factor = min(dur / budget, MAX_TEMPO)
            audio = trim(_filter(work, _atempo_chain(factor), CLIP / f"{name}_fit.wav"))
            dur = len(audio) / SR
            notes.append(f"atempo x{factor:.3f}")

        audio = fade(audio)
        at = int(start * SR)
        end = min(at + len(audio), len(voice))
        voice[at:end] += audio[: end - at]
        gate[at:end] = 1.0
        print(f"  {name}: {dur:5.2f}s / {budget:5.2f}s  {'; '.join(notes) or 'natural'}")

    voice = voice[:total]
    gate = gate[:total]

    # Duck what little score sits under the dialogue, then set the voice
    # against that ducked bed so the lead is exact.
    win = int(SR * 0.25)
    smooth = np.clip(
        np.convolve(gate, np.ones(win, dtype=np.float32) / win, mode="same"), 0.0, 1.0
    )
    bed = music * (1.0 - (1.0 - 10 ** (RESIDUAL_DUCK_DB / 20.0)) * smooth)

    speaking = gate > 0
    v_rms = float(np.sqrt(np.mean(voice[speaking] ** 2) + 1e-12))
    b_rms = float(np.sqrt(np.mean(bed[speaking] ** 2) + 1e-12))
    if v_rms > 0:
        voice *= min(b_rms * (10 ** (DIALOG_LEAD_DB / 20.0)) / v_rms, 40.0)

    mixed = bed + voice
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
    for pattern in ("*_pitch.wav", "*_voice.wav"):
        for leftover in CLIP.glob(pattern):
            leftover.unlink(missing_ok=True)
    for leftover in CLIP.glob("*_fit.wav"):
        leftover.unlink(missing_ok=True)

    with OUT_SRT.open("w", encoding="utf-8") as fh:
        for i, (s, e, t) in enumerate(CUES, 1):
            fh.write(f"{i}\n{stamp(s)} --> {stamp(e)}\n{t}\n\n")

    print(f"\nvideo     {OUT_VIDEO} ({OUT_VIDEO.stat().st_size} bytes)")
    print(f"subtitles {OUT_SRT}")


if __name__ == "__main__":
    main()
