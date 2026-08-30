"""Design a voice from a description, a mixer, or a reference clip.

Ten fixed voices is not a voice bank, it is a menu. This turns each of them
into a range: a recording is reshaped by six controls, so the same base take
can read as a heavy older man or a bright young one.

Three ways in, one representation out:

* ``parse_description`` maps Arabic words ("رجل عميق غاضب") onto the controls.
* the controls themselves, moved by hand in the browser mixer.
* ``clone_from_reference`` measures a reference clip and returns the controls
  that move a base voice toward it.

What "cloning" honestly means here
----------------------------------
Neural cloning needs model weights this container cannot reach — HuggingFace,
the F5/XTTS checkpoints, every mirror. What is reachable is measurement, so
cloning here matches the *measurable* qualities of a reference: median pitch,
brightness and speaking rate. That reproduces build and delivery, not identity.
It is stated plainly rather than dressed up, and the numbers are reported so
the result can be checked instead of believed.

Pitch and vocal tract move together
-----------------------------------
Shifting pitch by resampling drags the formants with it, which is why a naive
octave shift sounds like a chipmunk rather than a different person. There is no
formant-independent shifter available offline, so instead of pretending, the
tract is modelled separately with shelving and peaking EQ: ``body`` for chest
resonance, ``clarity`` for presence, ``air`` for breath. Those are real filters
with audible effect, and they are named for what they do rather than claimed to
be something they are not.
"""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# ffmpeg's atempo only accepts 0.5-2.0 per instance; wider ranges are chained.
_ATEMPO_MIN, _ATEMPO_MAX = 0.5, 2.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class VoiceSpec:
    """Six controls. Neutral is all-zero and must leave audio untouched."""

    pitch: float = 0.0     # semitones, -12..+12
    rate: float = 1.0      # speaking rate multiplier, 0.6..1.6
    body: float = 0.0      # dB at 250 Hz  — chest, size of the speaker
    warmth: float = 0.0    # dB at 700 Hz  — roundness vs hollowness
    clarity: float = 0.0   # dB at 3 kHz   — consonants, presence
    air: float = 0.0       # dB at 9 kHz   — breath, closeness to the mic

    def clamped(self) -> "VoiceSpec":
        return VoiceSpec(
            pitch=_clamp(self.pitch, -12.0, 12.0),
            rate=_clamp(self.rate, 0.6, 1.6),
            body=_clamp(self.body, -8.0, 8.0),
            warmth=_clamp(self.warmth, -8.0, 8.0),
            clarity=_clamp(self.clarity, -8.0, 8.0),
            air=_clamp(self.air, -8.0, 8.0),
        )

    @property
    def is_neutral(self) -> bool:
        s = self.clamped()
        return (
            abs(s.pitch) < 0.01
            and abs(s.rate - 1.0) < 0.005
            and all(abs(v) < 0.05 for v in (s.body, s.warmth, s.clarity, s.air))
        )

    def to_dict(self) -> dict:
        return {k: round(v, 3) for k, v in asdict(self.clamped()).items()}

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceSpec":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**{k: float(v) for k, v in known.items()}).clamped()


# Words that describe a voice, and what they do to the controls. Deltas rather
# than absolutes, so "رجل عجوز عميق" stacks into one coherent instruction.
_WORDS: list[tuple[tuple[str, ...], dict]] = [
    (("عميق", "غليظ", "جهوري", "أجش", "خشن"), {"pitch": -3.5, "body": 3.5, "air": -1.0}),
    (("رفيع", "حاد", "نحيل"), {"pitch": 3.5, "body": -2.5, "clarity": 1.5}),
    (("رجل", "رجالي", "ذكر", "ذكوري"), {"pitch": -2.0, "body": 2.0}),
    (("امرأة", "نسائي", "أنثى", "أنثوي", "سيدة"), {"pitch": 4.0, "body": -2.0, "air": 1.0}),
    (("طفل", "طفلة", "صبي"), {"pitch": 7.0, "body": -4.0, "rate": 0.12, "clarity": 1.5}),
    (("شاب", "شابة", "صغير"), {"pitch": 1.5, "rate": 0.08, "clarity": 1.0}),
    (("عجوز", "مسن", "كبير", "شيخ", "حكيم"), {"pitch": -2.0, "rate": -0.15, "air": 1.5, "clarity": -1.0}),
    (("غاضب", "صارم", "حازم", "عنيف", "قوي"), {"clarity": 2.5, "rate": 0.1, "body": 1.5}),
    (("هادئ", "رقيق", "حنون", "لطيف", "ناعم"), {"rate": -0.1, "warmth": 2.5, "clarity": -1.0}),
    (("همس", "هامس", "خافت", "سري"), {"air": 4.0, "clarity": -2.0, "body": -2.0, "rate": -0.08}),
    (("دافئ", "دافي"), {"warmth": 3.0, "air": -0.5}),
    (("واضح", "لامع", "ساطع", "مشرق"), {"clarity": 3.0, "air": 1.5}),
    (("مهيب", "ضخم", "عريض", "ملحمي", "فخم"), {"pitch": -2.5, "body": 4.0, "warmth": 2.0}),
    (("سريع", "متسرع", "متحمس", "حماسي"), {"rate": 0.2, "clarity": 1.0}),
    (("بطيء", "متمهل", "متأنٍ", "متأني", "رزين"), {"rate": -0.2}),
    (("مذيع", "إخباري", "أخبار", "رسمي"), {"clarity": 2.0, "body": 1.0, "rate": 0.05}),
    (("راوي", "وثائقي", "سرد", "قصة"), {"warmth": 2.0, "body": 1.5, "rate": -0.08}),
    (("شرير", "مخيف", "مرعب", "مظلم"), {"pitch": -4.0, "body": 3.0, "clarity": -1.0, "rate": -0.1}),
    (("مرح", "بشوش", "ودود", "مبتهج"), {"pitch": 1.0, "clarity": 1.5, "rate": 0.1, "warmth": 1.5}),
    (("متعب", "حزين", "مكسور", "يائس"), {"pitch": -1.5, "rate": -0.18, "clarity": -1.5, "air": 1.0}),
]

# Intensity modifiers. "جداً" after a word should mean more of it, not the same.
_STRONG = ("جدا", "جداً", "للغاية", "قوي جدا", "كثيرا", "كثيراً", "أكثر")
_WEAK = ("قليلا", "قليلاً", "خفيف", "بعض", "شوية", "نوعا ما")

_NORMALISE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u0640]")


def _normalise(text: str) -> str:
    text = _NORMALISE.sub("", text or "")
    for src, dst in (("أإآٱ", "ا"), ("ى", "ي"), ("ة", "ه")):
        text = text.translate({ord(c): dst for c in src})
    return text.lower()


def parse_description(text: str) -> VoiceSpec:
    """Turn a plain Arabic description into mixer settings.

    Unknown words are ignored rather than guessed at: a description that
    matches nothing returns neutral, which leaves the base voice alone. That is
    the honest failure — silently inventing a transform would make the mixer
    untrustworthy.
    """
    norm = _normalise(text)
    scale = 1.0
    if any(_normalise(w) in norm for w in _STRONG):
        scale = 1.5
    elif any(_normalise(w) in norm for w in _WEAK):
        scale = 0.5

    spec = VoiceSpec()
    for words, delta in _WORDS:
        if not any(_normalise(w) in norm for w in words):
            continue
        for field, amount in delta.items():
            if field == "rate":
                spec.rate += amount * scale
            else:
                setattr(spec, field, getattr(spec, field) + amount * scale)
    return spec.clamped()


def clone_from_reference(
    ref_f0: float,
    ref_brightness: float,
    base_f0: float,
    base_brightness: float,
    ref_rate: Optional[float] = None,
    base_rate: Optional[float] = None,
) -> VoiceSpec:
    """Controls that move a base voice toward a measured reference.

    Pitch is matched exactly in semitones — that part is a true match, not an
    approximation. Brightness is matched with presence EQ, which is a proxy for
    vocal tract shape rather than a reproduction of it. Rate is matched only
    when both sides were measured.

    Raises ValueError on a non-positive pitch: a silent or unvoiced reference
    yields 0 Hz from every detector, and a ratio against it is meaningless.
    Refusing beats returning a confident-looking wrong number.
    """
    if ref_f0 <= 0 or base_f0 <= 0:
        raise ValueError("pitch measurement failed; reference may be silent")

    spec = VoiceSpec(pitch=12.0 * math.log2(ref_f0 / base_f0))
    if ref_brightness > 0 and base_brightness > 0:
        tilt = 20.0 * math.log10(ref_brightness / base_brightness)
        spec.clarity = _clamp(tilt, -6.0, 6.0)
        spec.air = _clamp(tilt * 0.5, -4.0, 4.0)
    if ref_rate and base_rate and base_rate > 0:
        spec.rate = _clamp(ref_rate / base_rate, 0.6, 1.6)
    return spec.clamped()


def _atempo_chain(factor: float) -> list[str]:
    """Split a tempo change into legal atempo steps."""
    steps: list[str] = []
    remaining = factor
    while remaining > _ATEMPO_MAX:
        steps.append(f"atempo={_ATEMPO_MAX}")
        remaining /= _ATEMPO_MAX
    while remaining < _ATEMPO_MIN:
        steps.append(f"atempo={_ATEMPO_MIN}")
        remaining /= _ATEMPO_MIN
    if abs(remaining - 1.0) > 1e-4:
        steps.append(f"atempo={remaining:.6f}")
    return steps


def ffmpeg_chain(spec: VoiceSpec, sample_rate: int) -> str:
    """The -af chain implementing ``spec`` for audio at ``sample_rate``.

    ``sample_rate`` is required, not assumed. Hard-coding 44100 here once cut
    every 24 kHz recording in half, because asetrate reinterprets the file at
    whatever rate it is handed; the multiplier only means "shift pitch" when it
    is applied to the file's own rate.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    spec = spec.clamped()
    parts: list[str] = []

    if abs(spec.pitch) >= 0.01:
        ratio = 2.0 ** (spec.pitch / 12.0)
        parts.append(f"asetrate={int(round(sample_rate * ratio))}")
        parts.append(f"aresample={sample_rate}")
        # asetrate changed speed as a side effect; undo it so pitch moves alone.
        parts.extend(_atempo_chain(1.0 / ratio))

    if abs(spec.rate - 1.0) >= 0.005:
        parts.extend(_atempo_chain(spec.rate))

    if abs(spec.body) >= 0.05:
        parts.append(f"equalizer=f=250:t=q:w=1.0:g={spec.body:.2f}")
    if abs(spec.warmth) >= 0.05:
        parts.append(f"equalizer=f=700:t=q:w=1.2:g={spec.warmth:.2f}")
    if abs(spec.clarity) >= 0.05:
        parts.append(f"equalizer=f=3000:t=q:w=1.2:g={spec.clarity:.2f}")
    if abs(spec.air) >= 0.05:
        parts.append(f"highshelf=f=9000:g={spec.air:.2f}")

    return ",".join(parts) if parts else "anull"


def apply_spec(src: Path, dst: Path, spec: VoiceSpec, sample_rate: int) -> Path:
    """Render ``src`` through ``spec`` into ``dst``."""
    from youtube_auto_dub.ffmpeg_bin import ensure_ffmpeg_on_path, ffmpeg_exe

    ensure_ffmpeg_on_path()
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg_exe(), "-y", "-v", "error", "-i", str(src),
         "-af", ffmpeg_chain(spec, sample_rate), str(dst)],
        check=True, capture_output=True,
    )
    return dst


# Ready-made starting points, so the mixer is useful before anyone learns it.
PRESETS: dict[str, VoiceSpec] = {
    "مهيب عميق": VoiceSpec(pitch=-3.5, body=4.5, warmth=2.0, clarity=1.0),
    "شاب حماسي": VoiceSpec(pitch=2.0, rate=1.12, clarity=2.5, air=1.0),
    "عجوز حكيم": VoiceSpec(pitch=-2.0, rate=0.85, air=2.0, warmth=2.0, clarity=-1.0),
    "همس قريب": VoiceSpec(pitch=-0.5, rate=0.92, air=5.0, body=-2.0, clarity=-1.5),
    "مذيع أخبار": VoiceSpec(pitch=-1.0, rate=1.05, clarity=3.0, body=1.5),
    "راوي وثائقي": VoiceSpec(pitch=-1.5, rate=0.92, warmth=3.0, body=2.0),
    "شرير مظلم": VoiceSpec(pitch=-5.0, rate=0.9, body=3.5, clarity=-1.0),
    "طفل": VoiceSpec(pitch=7.0, rate=1.1, body=-4.0, clarity=2.0),
}
