"""Model-free vocal/music separation for stereo film audio.

Cinema mixes put dialogue in the phantom centre and spread score and effects
across the sides. That lets us separate them with signal processing alone, no
downloaded model — which matters on locked-down machines where Demucs cannot
fetch its weights.

Two cues are combined per time/frequency bin:

* **Panning** — a bin whose left and right content matches is centre (dialogue);
  a bin that differs is wide (music). Measured as normalised similarity between
  the L and R spectra.
* **Voice band** — speech energy concentrates roughly between 200 Hz and 4 kHz,
  so bins outside that range are biased towards the music stem.

The resulting soft mask is smoothed over time so the stems do not flutter.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from youtube_auto_dub.ffmpeg_bin import ffmpeg_exe

N_FFT = 4096
HOP = 1024
VOICE_LOW_HZ = 200.0
VOICE_HIGH_HZ = 4000.0


def _stft(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    pad = n_fft // 2
    x = np.pad(x, (pad, pad + n_fft), mode="constant")
    frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    return np.fft.rfft(x[idx] * window, axis=1).T


def _istft(spec: np.ndarray, length: int, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    frames = spec.shape[1]
    out = np.zeros((frames - 1) * hop + n_fft, dtype=np.float32)
    norm = np.zeros_like(out)
    blocks = np.fft.irfft(spec, n=n_fft, axis=0).T.astype(np.float32)
    for i in range(frames):
        at = i * hop
        out[at:at + n_fft] += blocks[i] * window
        norm[at:at + n_fft] += window ** 2
    out /= np.maximum(norm, 1e-8)
    pad = n_fft // 2
    return out[pad:pad + length]


def _smooth(mask: np.ndarray, frames: int = 5) -> np.ndarray:
    if frames < 2:
        return mask
    kernel = np.ones(frames, dtype=np.float32) / frames
    padded = np.pad(mask, ((0, 0), (frames // 2, frames // 2)), mode="edge")
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), 1, padded
    )[:, : mask.shape[1]]


def decode_stereo(path: Path, sr: int = 44100) -> tuple[np.ndarray, np.ndarray]:
    res = subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(path), "-ar", str(sr), "-ac", "2",
         "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    data = np.frombuffer(res.stdout, dtype=np.float32).reshape(-1, 2)
    return data[:, 0].copy(), data[:, 1].copy()


def split_center(
    left: np.ndarray,
    right: np.ndarray,
    sr: int = 44100,
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split stereo into (dialogue, music) using panning + voice-band cues.

    ``strength`` scales how aggressively centred content is pulled out.
    """
    length = len(left)
    spec_l = _stft(left)
    spec_r = _stft(right)

    mag_l, mag_r = np.abs(spec_l), np.abs(spec_r)
    # 1.0 when a bin is equally present in both channels, 0.0 when hard-panned.
    similarity = (2.0 * mag_l * mag_r) / (mag_l ** 2 + mag_r ** 2 + 1e-9)

    # Coherent centre content also has matching phase.
    phase = np.cos(np.angle(spec_l) - np.angle(spec_r))
    centre = similarity * np.clip(phase, 0.0, 1.0)

    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    band = np.ones_like(freqs, dtype=np.float32)
    band[freqs < VOICE_LOW_HZ] = 0.25
    band[freqs > VOICE_HIGH_HZ] = 0.35
    # Taper rather than cliff-edge at the band limits.
    lo = (freqs >= VOICE_LOW_HZ) & (freqs < VOICE_LOW_HZ * 2)
    band[lo] = np.linspace(0.25, 1.0, int(lo.sum()))
    hi = (freqs > VOICE_HIGH_HZ / 2) & (freqs <= VOICE_HIGH_HZ)
    band[hi] = np.linspace(1.0, 0.35, int(hi.sum()))

    mask = np.clip(centre * band[:, None] * strength, 0.0, 1.0)
    mask = _smooth(mask ** 1.5)

    mid = (spec_l + spec_r) / 2.0
    voice_spec = mid * mask
    voice = _istft(voice_spec, length)

    mono = (left + right) / 2.0
    music = mono - voice
    return voice, music


def separate_file(src: Path, out_dir: Path, sr: int = 44100) -> tuple[Path, Path]:
    """Write ``vocals.wav`` and ``music.wav`` next to each other."""
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    left, right = decode_stereo(src, sr)
    voice, music = split_center(left, right, sr)
    vpath = out_dir / "vocals.wav"
    mpath = out_dir / "music.wav"
    sf.write(vpath, voice, sr)
    sf.write(mpath, music, sr)
    return vpath, mpath
