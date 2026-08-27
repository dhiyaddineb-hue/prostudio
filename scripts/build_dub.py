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
    __import__("os").environ.get("PROJECT", "Phantom-Thread")
).ensure_dirs()

CLIP = PROJECT.voices_dir
WORK = PROJECT.work_dir
OUT_VIDEO = PROJECT.video_path
OUT_SRT = PROJECT.srt_path


def _source() -> Path:
    """The clip to dub: the project's own copy, else whatever is in inbox/.

    Resolved when rendering starts rather than at import, so the measurement
    pass below can load this module for a project whose source clip has been
    archived away. Resolving it eagerly used to hand such a project whatever
    happened to be sitting in inbox/ — a different film entirely.
    """
    local = sorted(PROJECT.source_dir.glob("*.mp4"))
    if local:
        return local[0]
    inbox = sorted((ROOT / "inbox").glob("*.mp4"))
    if inbox:
        return inbox[0]
    raise SystemExit("no source clip: put one in the project's source/ folder")

SR = 44100
MAX_TEMPO = 1.35        # past this the read sounds rushed
# Natural Arabic speech runs 11-15 characters/second. The synthesised takes come
# out around 7.4, which reads as sluggish rather than deliberate.
TARGET_RATE_MIN = 11.0
MAX_NATURALISE = 1.6    # cap, so a very short line is not chipmunked
# How far a line may outlast the actor's own mouth movement, as a multiple of
# the caption window. Arabic needs more syllables than English for the same
# thought, so some overshoot is unavoidable; 1.25x stays under the ~0.2 s that
# viewers notice on a close-up. It used to be "anything up to the next cue",
# which let one line run 2.23x and talk over a silent face for 1.30 s.
MAX_OVERRUN = 1.25
TAIL_ALLOWANCE = 0.10   # slack before a line counts as overrunning
CUE_GUARD_SEC = 0.08    # minimum silence before the next line starts
DIALOG_LEAD_DB = 11.0
RESIDUAL_DUCK_DB = -15.0

# (start, end, speaker, text). ``end`` is the caption's own end; the audio may
# breathe into the gap that follows if nothing else is speaking.

# Cues and speaker turns come from project.json, so one engine serves every dub.
CUES = [
    (float(c["start"]), float(c["end"]), c["speaker"], c["text"])
    for c in sorted(PROJECT.cues, key=lambda c: c["i"])
]

def _segments() -> dict:
    """Which cues each seg_*.wav recording covers.

    A stored map in ``render.segments`` wins. Guessing by zipping speaker runs
    against the sorted filenames only holds when every turn was recorded in
    order and nothing was re-cut: on the Phantom dub, where three takes carry a
    trailing filler sentence and the files were recorded out of order, the
    guess assigns cues to the wrong recordings entirely.
    """
    stored = (PROJECT.render or {}).get("segments")
    if stored:
        return {
            name: [None if m in (None, "None") else int(m) for m in members]
            for name, members in stored.items()
        }

    runs, cur = [], None
    for c in sorted(PROJECT.cues, key=lambda c: c["i"]):
        if cur and cur[0] == c["speaker"]:
            cur[1].append(c["i"])
        else:
            cur = (c["speaker"], [c["i"]])
            runs.append(cur)
    takes = sorted(PROJECT.voices_dir.glob("seg_*.wav"))
    out = {}
    for (speaker, members), take in zip(runs, takes):
        out[take.stem] = members
    return out

SEGMENTS = _segments()
# Takes that end on a throwaway sentence: the model needs somewhere to carry
# the phrase to, so the kept line does not die on a flat final syllable.
SEGMENTS_WITH_FILLER: set = set((PROJECT.render or {}).get("segments_with_filler") or [])
GROUP_SLICES: dict = {
    int(k): tuple(v) for k, v in ((PROJECT.render or {}).get("group_slices") or {}).items()
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


def split_by_text(
    audio: np.ndarray,
    weights: list[float],
    min_piece: float = 0.28,
) -> list[np.ndarray]:
    """Cut a continuous take into pieces proportional to their text length.

    Splitting purely on the loudest pauses mis-assigns lines: in seg_D_f it gave
    "مكشوف" (5 characters) 2.17 s while a 21-character line got 0.67 s, so the
    wrong words landed on the wrong cues. Pause positions are still preferred,
    but only the pause nearest each text-proportional boundary is used.
    """
    total = float(sum(weights)) or 1.0
    n = len(audio)
    if len(weights) <= 1 or n == 0:
        return [audio]

    fr = int(SR * 0.02)
    frames = n // fr
    quiet = np.zeros(max(frames, 1), dtype=bool)
    if frames:
        energy = np.sqrt(
            np.mean(audio[: frames * fr].reshape(frames, fr) ** 2, axis=1) + 1e-12
        )
        quiet = energy < max(float(np.median(energy)) * 0.35, 1e-5)

    cuts = []
    running = 0.0
    for w in weights[:-1]:
        running += w
        ideal = int(running / total * n)
        # Snap to the nearest silent frame within 0.35 s, else cut where the
        # text says to.
        window = int(0.35 * SR) // fr
        centre = min(max(ideal // fr, 0), max(frames - 1, 0))
        best = None
        for off in range(window + 1):
            for cand in (centre - off, centre + off):
                if 0 <= cand < frames and quiet[cand]:
                    best = cand
                    break
            if best is not None:
                break
        cut = (best * fr) if best is not None else ideal
        cuts.append(max(cut, int(min_piece * SR) * len(cuts)))

    bounds = [0] + sorted(cuts) + [n]
    out = []
    for k in range(len(bounds) - 1):
        piece = audio[bounds[k]: bounds[k + 1]]
        out.append(piece if piece.size else np.zeros(int(min_piece * SR), dtype=np.float32))
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


_ACTOR_CACHE: dict = {}


def _match_actor(audio: np.ndarray, speaker: str) -> np.ndarray:
    """Nudge a take's pitch and vocal-tract size toward the on-screen actor.

    Measured on this clip the synthetic woman sits at F0 199 Hz with a formant
    mean of 1961, against the actress at 190 Hz and 1477 — she reads younger and
    thinner than the performance. The shift is clamped inside voice_profile, and
    reverted here if it does not actually move closer.
    """
    try:
        from youtube_auto_dub.voice_profile import convert, measure
    except Exception:
        return audio

    if speaker not in _ACTOR_CACHE:
        windows = [
            (c[0], c[1]) for c in CUES if c[2] == speaker
        ]
        if not windows:
            _ACTOR_CACHE[speaker] = None
        else:
            left, right = decode_stereo(_source(), SR)
            stem, _ = split_center(left, right, SR)
            span = (min(w[0] for w in windows), max(w[1] for w in windows))
            _ACTOR_CACHE[speaker] = measure(
                stem[int(span[0] * SR): int(span[1] * SR)], SR
            )

    target = _ACTOR_CACHE.get(speaker)
    if target is None or not target.reliable:
        return audio

    source = measure(audio, SR, focus=False)
    if source is None or source.f0_median <= 0:
        return audio

    converted, _ = convert(audio, SR, source, target)
    check = measure(converted, SR, focus=False)
    if check is None or not np.isfinite(converted).all():
        return audio
    # Keep it only if it genuinely moved toward the actor.
    if abs(check.f0_median - target.f0_median) < abs(
        source.f0_median - target.f0_median
    ):
        return converted.astype(np.float32)
    return audio


def cue_audio(index: int, speaker: str, cache: dict) -> np.ndarray | None:
    """Audio for one cue: its own recording if present, else a group slice.

    Lossless WAV wins over MP3. The takes were originally rendered as 32 kb/s
    MP3, which caps the voice near 8 kHz and is audibly muffled; a WAV of the
    same line carries ~10.7 kHz and far more high-frequency detail.
    """
    # A continuous turn recording wins: it carries the intonation of the whole
    # thought, which a per-caption take cannot.
    # Timbre is nudged toward the on-screen actor where the measurement of them
    # is trustworthy — see _match_actor.
    for name, members in SEGMENTS.items():
        if index + 1 in members:
            src = CLIP / f"{name}.wav"
            if src.exists():
                if name not in cache:
                    # Actor matching is deliberately not applied: the reference
                    # measurement comes from a stem that still carries score, so
                    # its targets are wrong. Applying it pushed the man to 94 Hz
                    # against his own 111 and the actor's 116 — further away.
                    matched = trim(decode(src))
                    weights = [
                        max(len([c for c in CUES[m - 1][3] if not c.isspace()]), 1)
                        if m
                        # Trailing filler: roughly as long as the line it
                        # follows, so the kept piece gets its fair share.
                        else max(
                            len([c for c in CUES[members[0] - 1][3] if not c.isspace()]),
                            1,
                        )
                        for m in members
                    ]
                    cache[name] = split_by_text(matched, weights)
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


def place_cues(total: int) -> tuple[np.ndarray, np.ndarray, int, list[dict]]:
    """Lay every take onto a silent timeline of ``total`` samples.

    Split out of the render so the same arithmetic can be replayed without the
    source clip. A project whose source has been archived can then still report
    honest numbers, instead of the page falling back to caption windows.
    """
    voice = np.zeros(total + SR, dtype=np.float32)
    gate = np.zeros(total + SR, dtype=np.float32)
    cache: dict = {}
    placed = 0
    spoken: list[dict] = []

    for i, (start, end, speaker, text) in enumerate(CUES):
        audio = cue_audio(i, speaker, cache)
        if audio is None or audio.size == 0:
            print(f"  cue {i + 1:2d}: MISSING")
            continue

        # How long the line may run. The caption's own window is the honest
        # answer: it is roughly how long the actor's mouth moves. Letting a
        # line spill into the silence before the next cue is what broke lip
        # sync — measured on the Vikings clip, cue 1 ran 2.23x the actor's
        # 1.14 s of speech and was still talking 1.30 s after he closed his
        # mouth. A small overshoot is inaudible; a doubled one is not.
        next_start = CUES[i + 1][0] if i + 1 < len(CUES) else start + 8.0
        room_to_next = next_start - start - CUE_GUARD_SEC
        budget = min(max(end - start, 0.35) * MAX_OVERRUN, max(room_to_next, 0.35))
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

        # Hard guard: a line must never run into the next speaker's slot, nor
        # keep talking long after this speaker's mouth has stopped.
        room = min(budget, room_to_next) if i + 1 < len(CUES) else budget
        room = max(room, 0.35)
        if len(audio) / SR > room:
            # Prefer speeding the line up over cutting a word off its end;
            # only trim if it is still too long after a modest squeeze.
            factor = min((len(audio) / SR) / room, 1.25)
            audio = trim(retime(audio, factor))
            note += f"; fitted x{factor:.2f}"
            if len(audio) / SR > room:
                audio = audio[: int(room * SR)]
                note += "; trimmed"

        audio = fade(audio)
        at = int(start * SR)
        stop = min(at + len(audio), len(voice))
        voice[at:stop] += audio[: stop - at]
        gate[at:stop] = 1.0
        placed += 1
        # Record where the line actually landed. The subtitle window is a
        # request, not a result: after naturalising and fitting, a cue can end
        # well before its caption does, and a rate computed from the caption
        # then describes a file that does not exist.
        spoken.append({
            "i": i + 1,
            "start": round(start, 3),
            "end": round(start + len(audio) / SR, 3),
            "chars": chars,
        })
        print(f"  cue {i + 1:2d}: {start:6.2f}s  {dur:5.2f}s / {budget:5.2f}s  {note}")

    print(f"placed {placed}/{len(CUES)} cues")
    return voice[:total], gate[:total], placed, spoken


def measurements(placed: int, spoken: list[dict]) -> dict:
    """Rate, tightest gap and overlaps, from where the takes actually landed."""
    gaps = [b["start"] - a["end"] for a, b in zip(spoken, spoken[1:])]
    talk = sum(s["end"] - s["start"] for s in spoken)
    chars = sum(s["chars"] for s in spoken)
    return {
        "placed": placed,
        "cues": len(CUES),
        "rate": round(chars / talk, 1) if talk > 0 else None,
        "min_gap": round(min(gaps), 3) if gaps else None,
        "overlaps": sum(1 for g in gaps if g < 0),
    }


def main() -> None:
    ensure_ffmpeg_on_path()
    src = _source()
    if not src.exists():
        raise SystemExit(f"source clip missing: {src}")

    print("separating stems…")
    left, right = decode_stereo(src, SR)
    _, separated = split_center(left, right, SR)
    original = ((left + right) / 2.0).astype(np.float32)
    total = len(separated)

    # Centre removal is only needed where the original dialogue plays. Measured
    # on this clip it costs -8.7 dB in the low-mids and -10.6 dB in the presence
    # band, which guts the body of the score. So the separated stem is used only
    # under our own dialogue, and the untouched original is restored everywhere
    # else — the music keeps its full weight between lines.
    voice, gate, placed, spoken = place_cues(total)

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

    # Dialogue polish, measured on this mix rather than guessed:
    #   highpass 70 Hz  — 7.4% of the voice energy sat below 120 Hz as rumble
    #   -2 dB @ 8 kHz   — 26% sat above 5 kHz, which reads as harsh sibilance
    #   +1.5 dB @ 2.5k  — lifts consonants so words cut through the score
    #   compressor      — gentle only. A 2.5:1 pass squashed the range from
    #                     10 dB to 7.4, which is flatter than where it started.
    chain = (
        "highpass=f=70,"
        "equalizer=f=2500:t=q:w=1.2:g=1.5,"
        "equalizer=f=8000:t=q:w=1.0:g=-2,"
        "acompressor=threshold=-16dB:ratio=1.6:attack=12:release=250:makeup=1,"
        "loudnorm=I=-16:TP=-1.5:LRA=11"
    )
    subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(raw),
         "-af", chain, "-ar", "48000", "-ac", "2", str(final)],
        check=True, capture_output=True,
    )
    raw.unlink(missing_ok=True)

    subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(src), "-i", str(final),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "192k", "-shortest", str(OUT_VIDEO)],
        check=True, capture_output=True,
    )
    final.unlink(missing_ok=True)

    with OUT_SRT.open("w", encoding="utf-8") as fh:
        for i, (s, e, _speaker, text) in enumerate(CUES, 1):
            fh.write(f"{i}\n{stamp(s)} --> {stamp(e)}\n{text}\n\n")

    # Hand the measurements to whatever publishes this dub. Without them the
    # web page can only compute a rate from the captions, which describes the
    # script rather than the rendered audio.
    PROJECT.render = dict(
        PROJECT.render or {}, spoken=spoken, measured=measurements(placed, spoken)
    )
    PROJECT.save()

    print(f"\nvideo     {OUT_VIDEO} ({OUT_VIDEO.stat().st_size} bytes)")
    print(f"subtitles {OUT_SRT}")


if __name__ == "__main__":
    if "--measure-only" in sys.argv:
        # Re-derive the numbers for an already-rendered dub whose source clip
        # is no longer on disk. No audio is written.
        ensure_ffmpeg_on_path()
        span = int((max(c[1] for c in CUES) + 8.0) * SR)
        _v, _g, _placed, _spoken = place_cues(span)
        PROJECT.render = dict(
            PROJECT.render or {},
            spoken=_spoken,
            measured=measurements(_placed, _spoken),
        )
        PROJECT.save()
        print(PROJECT.render["measured"])
    else:
        main()
