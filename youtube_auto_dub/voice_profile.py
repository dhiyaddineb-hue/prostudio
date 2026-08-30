"""Measure a speaker's vocal identity and transfer it onto a synthetic take.

This is voice conversion by explicit acoustic matching, not a neural clone.
A pretrained cloning model cannot be used here — every weight host (HuggingFace
and its mirrors, dl.fbaipublicfiles.com, the GitHub release CDN) is unreachable
from this sandbox — so instead we measure what actually distinguishes a voice
and impose those properties on the studio take:

* **F0** — pitch register, the strongest cue to speaker identity.
* **Formants** — resonances set by vocal tract length; scaling them is what
  makes a voice read as a bigger or smaller person rather than a chipmunk.

Both are measured with Praat (via parselmouth) on the speech-dominant frames
only, because a separated centre stem still carries some score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VoiceProfile:
    """Acoustic fingerprint of a speaker."""

    f0_median: float
    f0_iqr: float
    f1: float
    f2: float
    f3: float
    voiced_ratio: float
    frames: int

    @property
    def reliable(self) -> bool:
        """Whether this measurement is stable enough to drive a conversion.

        A wide F0 spread means the estimator was tracking music as much as
        speech, and acting on it would move the voice the wrong way.
        """
        return (
            self.frames >= 25
            and self.f0_median > 0
            and self.f0_iqr / max(self.f0_median, 1e-6) < 0.45
        )

    @property
    def formant_mean(self) -> float:
        vals = [v for v in (self.f1, self.f2, self.f3) if v > 0]
        return float(np.mean(vals)) if vals else 0.0


def _speech_frames(audio: np.ndarray, sr: int, keep: float = 0.2) -> np.ndarray:
    """Keep the loudest fraction of frames, where speech dominates residue."""
    frame = int(sr * 0.03)
    hop = frame
    count = len(audio) // hop
    if count < 4:
        return audio
    energies = np.array([
        np.sqrt(np.mean(audio[i * hop:(i + 1) * hop] ** 2) + 1e-12)
        for i in range(count)
    ])
    cutoff = np.quantile(energies, 1.0 - keep)
    picks = [audio[i * hop:(i + 1) * hop] for i in range(count) if energies[i] >= cutoff]
    return np.concatenate(picks) if picks else audio


def measure(
    audio: np.ndarray,
    sr: int,
    floor: float = 60.0,
    ceiling: float = 350.0,
    focus: bool = True,
    keep: float = 0.2,
) -> VoiceProfile | None:
    """Measure the vocal profile of ``audio``. Returns None if unusable.

    ``keep`` is the loudest fraction of frames to analyse when ``focus`` is on.
    A separated stem still carries score in its quiet frames, and including
    them widens the F0 spread until the reading is worthless.
    """
    try:
        import parselmouth
    except ImportError:
        return None

    signal = _speech_frames(audio, sr, keep=keep) if focus else audio
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size < sr * 0.2:
        return None

    sound = parselmouth.Sound(signal, sampling_frequency=sr)
    pitch = sound.to_pitch(pitch_floor=floor, pitch_ceiling=ceiling)
    track = pitch.selected_array["frequency"]
    voiced = track[track > 0]
    if voiced.size < 10:
        return None

    formants = sound.to_formant_burg(max_number_of_formants=5, maximum_formant=5000)
    picks: list[list[float]] = [[], [], []]
    for i in range(1, formants.get_number_of_frames() + 1):
        t = formants.get_time_from_frame_number(i)
        for k in range(3):
            value = formants.get_value_at_time(k + 1, t)
            if value == value and 150.0 < value < 4500.0:
                picks[k].append(value)

    return VoiceProfile(
        f0_median=float(np.median(voiced)),
        f0_iqr=float(np.percentile(voiced, 75) - np.percentile(voiced, 25)),
        f1=float(np.median(picks[0])) if picks[0] else 0.0,
        f2=float(np.median(picks[1])) if picks[1] else 0.0,
        f3=float(np.median(picks[2])) if picks[2] else 0.0,
        voiced_ratio=float(voiced.size / max(track.size, 1)),
        frames=int(voiced.size),
    )


def convert(
    audio: np.ndarray,
    sr: int,
    source: VoiceProfile,
    target: VoiceProfile,
    max_semitones: float = 4.0,
    max_formant_ratio: float = 0.18,
) -> tuple[np.ndarray, dict]:
    """Reshape ``audio`` from the ``source`` voice toward the ``target``.

    Pitch and formants are moved independently, so the take can take on a
    larger or smaller apparent vocal tract without changing speed. Both
    corrections are clamped: past a point the artefacts cost more than the
    resemblance gains.
    """
    import parselmouth
    from parselmouth.praat import call

    report: dict = {"pitch_semitones": 0.0, "formant_ratio": 1.0}
    if not (source and target and source.f0_median > 0 and target.f0_median > 0):
        return audio, report

    semitones = 12.0 * np.log2(target.f0_median / source.f0_median)
    semitones = float(np.clip(semitones, -max_semitones, max_semitones))

    ratio = 1.0
    if source.formant_mean > 0 and target.formant_mean > 0:
        ratio = target.formant_mean / source.formant_mean
        ratio = float(np.clip(ratio, 1.0 - max_formant_ratio, 1.0 + max_formant_ratio))

    sound = parselmouth.Sound(np.asarray(audio, dtype=np.float64), sampling_frequency=sr)
    pitch_factor = 2 ** (semitones / 12.0)

    # "Change gender" is Praat's formant+pitch shifter; driving formant_shift
    # and new_pitch_median separately keeps duration intact.
    converted = call(
        sound,
        "Change gender",
        75,            # pitch floor for analysis
        600,           # pitch ceiling
        ratio,         # formant shift ratio
        source.f0_median * pitch_factor,  # new median pitch
        1.0,           # keep pitch range
        1.0,           # keep duration
    )
    out = np.asarray(converted.values[0], dtype=np.float32)

    report["pitch_semitones"] = semitones
    report["formant_ratio"] = ratio
    return out, report
