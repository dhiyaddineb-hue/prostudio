"""Place a known transcript onto the audio timeline without Whisper."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf

from youtube_auto_dub.models import SubtitleSegment


_SENT_SPLIT = re.compile(r"(?<=[.!?؟。])\s+|\n+")


def split_sentences(text: str) -> List[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text.strip()) if p.strip()]
    return parts or [text.strip()]


def _speech_windows(audio: np.ndarray, sr: int) -> List[tuple[float, float]]:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    frame = max(int(sr * 0.02), 1)
    hop = frame
    n = len(audio) // hop
    if n == 0:
        return [(0.0, max(len(audio) / sr, 0.5))]
    frames = audio[: n * hop].reshape(n, hop)
    energy = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    thresh = max(float(np.median(energy) * 1.6), float(np.max(energy) * 0.08))
    speech = energy > thresh
    windows: List[tuple[float, float]] = []
    start = None
    for i, flag in enumerate(speech):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= 8:
                windows.append((start * hop / sr, i * hop / sr))
            start = None
    if start is not None:
        windows.append((start * hop / sr, len(audio) / sr))
    if not windows:
        return [(0.2, max(len(audio) / sr - 0.2, 0.8))]
    # merge tiny gaps
    merged = [windows[0]]
    for s, e in windows[1:]:
        if s - merged[-1][1] < 0.35:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def segments_from_transcript(text: str, audio_path: Path, duration: float | None = None) -> List[SubtitleSegment]:
    sentences = split_sentences(text)
    try:
        audio, sr = sf.read(str(audio_path), dtype="float32")
        total = len(audio) / float(sr)
        windows = _speech_windows(audio, sr)
    except Exception:
        total = duration or max(len(sentences) * 3.0, 4.0)
        windows = [(0.2, total)]
    if duration:
        total = duration

    if len(windows) < len(sentences):
        # split the longest windows until we have enough slots
        while len(windows) < len(sentences):
            idx = max(range(len(windows)), key=lambda i: windows[i][1] - windows[i][0])
            s, e = windows.pop(idx)
            mid = (s + e) / 2
            windows[idx:idx] = [(s, mid), (mid, e)]
    elif len(windows) > len(sentences):
        # merge adjacent windows
        while len(windows) > len(sentences):
            gaps = [windows[i + 1][0] - windows[i][1] for i in range(len(windows) - 1)]
            i = int(np.argmin(gaps))
            windows[i] = (windows[i][0], windows[i + 1][1])
            windows.pop(i + 1)

    segs = []
    for i, sentence in enumerate(sentences):
        start, end = windows[i]
        start = max(0.0, start)
        end = min(total, max(start + 0.6, end))
        segs.append(SubtitleSegment(start=start, end=end, source_text=sentence, index=i))
    return segs
