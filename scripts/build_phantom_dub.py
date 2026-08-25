#!/usr/bin/env python3
"""Dub the Phantom Thread clip, one subtitle cue at a time.

Two problems with the earlier pass drove this rewrite.

**Drift.** Speaker turns were recorded as a single continuous take and dropped
at the turn's start time. Every line after the first then landed wherever the
narration happened to reach — measured drift ran to 1.9 s by the end of a long
turn. Now each cue is anchored to its own timestamp, so an error in one line
cannot propagate into the next.

**Quality.** Pitch/formant conversion measurably degraded the recordings
(harmonics-to-noise ratio dropped by up to 1.4 dB) and tempo compression made
them sound hurried. Casting is still decided by the actor's measured pitch, but
the audio itself is left alone unless a line genuinely overruns its slot.

Timing and script both come from the burned-in subtitles.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from youtube_auto_dub.ffmpeg_bin import ensure_ffmpeg_on_path, ffmpeg_exe
from youtube_auto_dub.stem_split import decode_stereo, split_center

ROOT = Path(__file__).resolve().parent.parent
CLIP = ROOT / "samples" / "phantom"
SRC = ROOT / "inbox" / (
    "Phantom Threadفنانٌ في الحياكة يريد امرأةً تلهمه من دون أن تربك نظامه،"
    " وهي ترفض أن تبقى مجرد مُ.mp4"
)
OUT_VIDEO = ROOT / "samples" / "Phantom_Thread_Pro_DUB.mp4"
OUT_SRT = OUT_VIDEO.with_suffix(".srt")

SR = 44100
MAX_TEMPO = 1.12        # gentler than before; past this the read sounds rushed
TAIL_ALLOWANCE = 0.45   # a line may run this far past the next cue's start
DIALOG_LEAD_DB = 11.0
RESIDUAL_DUCK_DB = -15.0

# (start, end, speaker, text). ``end`` is the caption's own end; the audio may
# breathe into the gap that follows if nothing else is speaking.
CUES = [
    (0.40, 1.90, "f", "لماذا لستَ متزوّجًا؟"),
    (2.40, 4.70, "m", "أنا على يقين بأن الزواج لم يُكتب لي أبدًا"),
    (5.00, 7.20, "m", "أنا رجلٌ خُلق للعزوبة"),
    (7.70, 9.00, "m", "ولا شفاء لي من ذلك"),
    (9.70, 11.30, "m", "الزواج سيجعلني مخادعًا"),
    (11.30, 12.90, "m", "وأنا لا أريد ذلك أبدًا"),
    (13.40, 15.80, "f", "أظنّ أنك تتظاهر بالقوة فقط"),
    (17.30, 18.90, "m", "لا، أنا قوي"),
    (19.20, 20.30, "f", "قوي أمام مَن؟"),
    (20.50, 22.10, "f", "أتمنى ألا تكون كذلك معي"),
    (22.10, 24.80, "m", "أظنّ أن توقّعات الآخرين"),
    (24.80, 26.40, "m", "وافتراضاتهم"),
    (27.40, 29.00, "m", "هي ما يورث القلب وجعَه"),
    (29.40, 30.50, "f", "أريدك..."),
    (31.10, 32.70, "f", "مستلقيًا على ظهرك"),
    (33.00, 34.00, "f", "عاجزًا"),
    (34.70, 35.60, "f", "رقيقًا"),
    (36.50, 37.60, "f", "مكشوفًا"),
    (38.10, 39.80, "f", "وليس لك سواي لأعتني بك"),
    (41.10, 43.50, "f", "ثم أريدك أن تستعيد قوتك من جديد"),
    (46.40, 49.20, "m", "أسمع صوتك يناديني باسمي في أحلامي"),
    (49.20, 50.90, "m", "وحين أستيقظ"),
    (50.90, 53.40, "m", "أجد الدموع تنهمر على وجهي"),
    (54.80, 55.40, "m", "أشتاقُ إليكِ."),
]

# Cues still served by slicing a multi-line group take, until each has its own
# recording. cue index (0-based) -> (group file, position, total pieces).
GROUP_SLICES = {
    0: ("g1_f", 0, 1),
    6: ("g3_f", 0, 1),
    7: ("g4_m", 0, 1),
    8: ("g5_f", 0, 2),
    13: ("g7_f", 0, 7),
    15: ("g7_f", 2, 7),
    16: ("g7_f", 3, 7),
    17: ("g7_f", 4, 7),
    18: ("g7_f", 5, 7),
    19: ("g7_f", 6, 7),
    20: ("g8_m", 0, 4),
    21: ("g8_m", 1, 4),
    22: ("g8_m", 2, 4),
    23: ("g8_m", 3, 4),
}


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


def retime(audio: np.ndarray, factor: float) -> np.ndarray:
    """Speed a clip up by ``factor`` without changing its pitch."""
    tmp_in = CLIP / "_retime_in.wav"
    tmp_out = CLIP / "_retime_out.wav"
    sf.write(tmp_in, audio, SR)
    subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(tmp_in), "-filter:a", _atempo_chain(factor),
         "-ar", str(SR), "-ac", "1", str(tmp_out)],
        capture_output=True, check=True,
    )
    out = sf.read(str(tmp_out), dtype="float32")[0]
    tmp_in.unlink(missing_ok=True)
    tmp_out.unlink(missing_ok=True)
    return out


def fade(a: np.ndarray, ms: int = 15) -> np.ndarray:
    n = min(int(SR * ms / 1000), len(a) // 2)
    if n < 2:
        return a
    out = a.copy()
    ramp = np.linspace(0.0, np.pi, n)
    out[:n] *= (0.5 * (1 - np.cos(ramp))).astype(np.float32)
    out[-n:] *= (0.5 * (1 + np.cos(ramp))).astype(np.float32)
    return out


def split_on_pauses(audio: np.ndarray, pieces: int) -> list[np.ndarray]:
    """Cut a multi-line take into ``pieces`` at its longest internal pauses."""
    if pieces <= 1:
        return [audio]
    fr = int(SR * 0.02)
    n = len(audio) // fr
    if n < pieces * 2:
        return [audio] * pieces
    energy = np.sqrt(np.mean(audio[: n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)
    quiet = energy < max(float(np.median(energy)) * 0.30, 1e-5)

    runs, start = [], None
    for i, is_quiet in enumerate(quiet):
        if is_quiet and start is None:
            start = i
        elif not is_quiet and start is not None:
            if i - start >= 5:  # >=100 ms of silence
                runs.append((start, i, i - start))
            start = None
    runs.sort(key=lambda r: -r[2])
    cuts = sorted(int((a + b) / 2) for a, b, _ in runs[: pieces - 1])

    bounds = [0] + cuts + [n]
    out = [audio[bounds[k] * fr: bounds[k + 1] * fr] for k in range(len(bounds) - 1)]
    while len(out) < pieces:
        out.append(np.zeros(fr, dtype=np.float32))
    return out[:pieces]


def cue_audio(index: int, speaker: str, cache: dict) -> np.ndarray | None:
    """Audio for one cue: its own recording if present, else a group slice.

    Lossless WAV wins over MP3. The takes were originally rendered as 32 kb/s
    MP3, which caps the voice near 8 kHz and is audibly muffled; a WAV of the
    same line carries ~10.7 kHz and far more high-frequency detail.
    """
    for suffix in (".wav", ".mp3"):
        own = CLIP / f"c{index + 1:02d}_{speaker}{suffix}"
        if own.exists():
            return trim(decode(own))

    spec = GROUP_SLICES.get(index)
    if not spec:
        return None
    name, position, total = spec
    src = CLIP / f"{name}.mp3"
    if not src.exists():
        return None
    if name not in cache:
        cache[name] = split_on_pauses(trim(decode(src)), total)
    parts = cache[name]
    return trim(parts[position]) if position < len(parts) else None


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
    _, music = split_center(left, right, SR)
    total = len(music)

    voice = np.zeros(total + SR, dtype=np.float32)
    gate = np.zeros(total + SR, dtype=np.float32)
    cache: dict = {}
    placed = 0

    for i, (start, end, speaker, text) in enumerate(CUES):
        audio = cue_audio(i, speaker, cache)
        if audio is None or audio.size == 0:
            print(f"  cue {i + 1:2d}: MISSING")
            continue

        # A line may use its own slot plus the silence before the next cue.
        next_start = CUES[i + 1][0] if i + 1 < len(CUES) else start + 8.0
        budget = max(next_start - start - 0.08, end - start)
        dur = len(audio) / SR

        note = "natural"
        if dur > budget + TAIL_ALLOWANCE:
            factor = min(dur / (budget + TAIL_ALLOWANCE), MAX_TEMPO)
            audio = trim(retime(audio, factor))
            dur = len(audio) / SR
            note = f"atempo x{factor:.3f}"
            if dur > budget + TAIL_ALLOWANCE:
                note += f" (+{dur - budget - TAIL_ALLOWANCE:.2f}s over)"

        audio = fade(audio)
        at = int(start * SR)
        stop = min(at + len(audio), len(voice))
        voice[at:stop] += audio[: stop - at]
        gate[at:stop] = 1.0
        placed += 1
        print(f"  cue {i + 1:2d}: {start:6.2f}s  {dur:5.2f}s / {budget:5.2f}s  {note}")

    print(f"placed {placed}/{len(CUES)} cues")
    voice = voice[:total]
    gate = gate[:total]

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

    with OUT_SRT.open("w", encoding="utf-8") as fh:
        for i, (s, e, _speaker, text) in enumerate(CUES, 1):
            fh.write(f"{i}\n{stamp(s)} --> {stamp(e)}\n{text}\n\n")

    print(f"\nvideo     {OUT_VIDEO} ({OUT_VIDEO.stat().st_size} bytes)")
    print(f"subtitles {OUT_SRT}")


if __name__ == "__main__":
    main()
