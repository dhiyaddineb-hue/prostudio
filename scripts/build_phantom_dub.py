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
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
# Run correctly as `python scripts/build_phantom_dub.py` from the repo root,
# without needing PYTHONPATH set. Setting it locally is what hid this from me.
sys.path.insert(0, str(ROOT))

from youtube_auto_dub.ffmpeg_bin import ensure_ffmpeg_on_path, ffmpeg_exe  # noqa: E402
from youtube_auto_dub.project_dirs import load_or_create  # noqa: E402
from youtube_auto_dub.stem_split import decode_stereo, split_center  # noqa: E402

# Everything for this dub lives in one project folder.
PROJECT = load_or_create(
    __import__("os").environ.get("PROJECT", "Phantom-Thread"),
    title="Phantom Thread — دبلجة مصرية", lang="ar", dialect="eg",
).ensure_dirs()

CLIP = PROJECT.voices_dir
WORK = PROJECT.work_dir
OUT_VIDEO = PROJECT.video_path
OUT_SRT = PROJECT.srt_path


def _source() -> Path:
    """The clip to dub: the project's own copy, else whatever is in inbox/."""
    local = sorted(PROJECT.source_dir.glob("*.mp4"))
    if local:
        return local[0]
    inbox = sorted((ROOT / "inbox").glob("*.mp4"))
    if inbox:
        return inbox[0]
    raise SystemExit("no source clip: put one in the project's source/ folder")


SRC = _source()

SR = 44100
MAX_TEMPO = 1.12        # gentler than before; past this the read sounds rushed
# Natural Arabic speech runs 11-15 characters/second. The synthesised takes come
# out around 7.4, which reads as sluggish rather than deliberate.
TARGET_RATE_MIN = 11.0
MAX_NATURALISE = 1.6    # cap, so a very short line is not chipmunked
TAIL_ALLOWANCE = 0.45   # a line may run this far past the next cue's start
DIALOG_LEAD_DB = 11.0
RESIDUAL_DUCK_DB = -15.0

# (start, end, speaker, text). ``end`` is the caption's own end; the audio may
# breathe into the gap that follows if nothing else is speaking.
CUES = [
    (0.40, 1.90, "f", "ليه مش متجوز؟"),
    (2.40, 4.70, "m", "أنا متأكد إن الجواز مش مكتوب لي أبدًا"),
    (5.00, 7.20, "m", "أنا راجل اتخلقت للعزوبية"),
    (7.70, 9.00, "m", "ومفيش علاج لده"),
    (9.70, 11.30, "m", "الجواز هيخليني كدّاب"),
    (11.30, 12.90, "m", "وأنا عمري ما هعوز كده"),
    (13.40, 15.80, "f", "أظن إنك بتتظاهر بالقوة وبس"),
    (17.30, 18.90, "m", "لأ، أنا قوي"),
    (19.20, 20.30, "f", "قوي قدام مين؟"),
    (20.50, 22.10, "f", "أتمنى ما تكونش كده معايا"),
    (22.10, 24.80, "m", "أظن إن توقعات الناس"),
    (24.80, 26.40, "m", "وافتراضاتهم"),
    (27.40, 29.00, "m", "هي اللي بتوجع القلب"),
    (29.40, 30.50, "f", "عايزاك..."),
    (31.10, 32.70, "f", "مستلقي على ضهرك"),
    (33.00, 34.00, "f", "عاجز"),
    (34.70, 35.60, "f", "رقيق"),
    (36.50, 37.60, "f", "مكشوف"),
    (38.10, 39.80, "f", "ومفيش قدامك غيري يهتم بيك"),
    (41.10, 43.50, "f", "وبعدين عايزاك تستعيد قوتك من تاني"),
    (46.40, 49.20, "m", "بسمع صوتك بينده على اسمي في أحلامي"),
    (49.20, 50.90, "m", "ولما بصحى"),
    (50.90, 53.40, "m", "بلاقي الدموع على وشي"),
    (54.80, 55.40, "m", "وحشتيني."),
]

# Cues still served by slicing a multi-line group take, until each has its own
# recording. cue index (0-based) -> (group file, position, total pieces).
# Continuous takes: one recording per speaker turn, cut at its own pauses.
# Generating a whole turn at once keeps the intonation contour intact — the
# model builds a real arc across the sentence instead of resetting on every
# caption. Measured: pitch variation 0.44 continuous vs 0.35 line-by-line.
SEGMENTS = {
    "seg_A_m": [2, 3, 4, 5, 6],
    "seg_E_f": [7, 9, 10],
    "seg_B_m": [11, 12, 13],
    "seg_D_f": [15, 16, 17, 18, 19, 20],
    "seg_C_m": [21, 22, 23, 24],
}

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


def trim(audio: np.ndarray, floor_db: float = -40.0) -> np.ndarray:
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
    # Keep only a hair of room: 20 ms in, 60 ms out.
    return audio[max(int(loud[0]) - 2, 0) * fr: min(int(loud[-1]) + 6, n) * fr]


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
    tmp_in = WORK / "_retime_in.wav"
    tmp_out = WORK / "_retime_out.wav"
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
    # A continuous turn recording wins: it carries the intonation of the whole
    # thought, which a per-caption take cannot.
    for name, members in SEGMENTS.items():
        if index + 1 in members:
            src = CLIP / f"{name}.wav"
            if src.exists():
                if name not in cache:
                    cache[name] = split_on_pauses(trim(decode(src)), len(members))
                pos = members.index(index + 1)
                part = cache[name][pos] if pos < len(cache[name]) else None
                if part is not None and part.size:
                    return trim(part)

    own = CLIP / f"c{index + 1:02d}_{speaker}.wav"
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
    _, separated = split_center(left, right, SR)
    original = ((left + right) / 2.0).astype(np.float32)
    total = len(separated)

    # Centre removal is only needed where the original dialogue plays. Measured
    # on this clip it costs -8.7 dB in the low-mids and -10.6 dB in the presence
    # band, which guts the body of the score. So the separated stem is used only
    # under our own dialogue, and the untouched original is restored everywhere
    # else — the music keeps its full weight between lines.
    music = separated  # replaced below, once we know where speech lands

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

        # The takes are uniformly slow — measured at ~7.4 characters/second
        # against a natural Arabic rate of 11-15 — which is what makes the
        # delivery drag. Speed each line toward a natural rate for its own text
        # before worrying about whether it fits the slot.
        chars = len([c for c in text if not c.isspace()])
        if chars >= 3 and dur > 0.2:
            rate = chars / dur
            if rate < TARGET_RATE_MIN:
                factor = min(TARGET_RATE_MIN / rate, MAX_NATURALISE)
                audio = trim(retime(audio, factor))
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

    # Crossfade between the two stems: the separated one only where our voice
    # plays (so the original English underneath is gone), the untouched mix
    # everywhere else (so the score keeps its full body).
    blend = np.clip(
        np.convolve(gate, np.ones(int(SR * 0.4), dtype=np.float32) / int(SR * 0.4),
                    mode="same") * 2.5,
        0.0, 1.0,
    )
    music = separated * blend + original[:total] * (1.0 - blend)

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

    raw = WORK / "mix_raw.wav"
    final = WORK / "mix_final.wav"
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
