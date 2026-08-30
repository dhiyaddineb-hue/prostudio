"""Transcription with Vosk — the offline recogniser that actually works here.

PocketSphinx ships a model inside its wheel, which made it the only recogniser
available in this sandbox, and its output is close to unusable: on a minute of
this material it produced "trolls many types of art which are reacting to what
the removing of awful thing" where the speaker said "ninety five percent of the
population are reacting to life". It rendered "Bob Proctor" as "sambo proctor".
Fine for detecting *that* someone is speaking, useless for knowing *what*.

Vosk is far better but ships no model, and every model host is blocked from
here — alphacephei.com, HuggingFace and its mirrors, the whisper CDNs. Whisper
itself is worse off: every copy on GitHub is stored through Git LFS, whose
hosts are also blocked, so the API hands back a 133-byte pointer instead of
weights.

The way through was a repository that committed a Vosk model as ordinary git
objects rather than LFS, which codeload will serve in full. The model lives
under ``.cache/models/`` and is not committed here; ``ensure_model`` fetches it
on demand and says plainly what to do if it cannot.

Measured on the same minute of audio: 7 s versus PocketSphinx's 21 s, and a
transcript that reads like English.
"""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / ".cache" / "models" / "vosk-model-small-en-us-0.15"

# A repository that committed the model without Git LFS. Checked at fetch time
# rather than trusted: an LFS pointer is 133 bytes of text, so a size check
# catches the failure immediately instead of at first use.
SOURCE_REPO = "syxanash/maxheadbox"
SOURCE_PATH = "backend/assets/vosk-model-small-en-us-0.15"
MIN_MODEL_BYTES = 30 * 1024 * 1024


def available() -> bool:
    """True when both the library and a real model are present."""
    try:
        import vosk  # noqa: F401
    except ImportError:
        return False
    return model_present()


def model_present() -> bool:
    mdl = MODEL_DIR / "am" / "final.mdl"
    return mdl.is_file() and mdl.stat().st_size > 1_000_000


def _to_wav(src: Path, sample_rate: int = 16000) -> Path:
    """Vosk wants 16-bit mono PCM at a fixed rate; give it exactly that."""
    from youtube_auto_dub.ffmpeg_bin import ensure_ffmpeg_on_path, ffmpeg_exe

    ensure_ffmpeg_on_path()
    dst = src.with_suffix(f".vosk{sample_rate}.wav")
    subprocess.run(
        [ffmpeg_exe(), "-y", "-v", "error", "-i", str(src),
         "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)],
        check=True, capture_output=True,
    )
    return dst


def transcribe(
    audio: Path,
    *,
    words: bool = True,
    progress=None,
) -> list[dict]:
    """Transcribe ``audio`` into segments with start/end times.

    Each segment is ``{"start", "end", "text", "words"}``. Word timings are
    what make this useful for dubbing: a line can be split at the exact instant
    a phrase ends rather than guessed from text length.
    """
    if not model_present():
        raise RuntimeError(
            f"no Vosk model at {MODEL_DIR} — run scripts/fetch_asr_model.py"
        )

    from vosk import KaldiRecognizer, Model, SetLogLevel

    SetLogLevel(-1)
    model = Model(str(MODEL_DIR))

    wav = audio if audio.suffix == ".wav" else _to_wav(audio)
    try:
        with wave.open(str(wav), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                handle.close()
                wav = _to_wav(audio)
                handle = wave.open(str(wav), "rb")

            rate = handle.getframerate()
            total = handle.getnframes()
            rec = KaldiRecognizer(model, rate)
            rec.SetWords(words)

            out: list[dict] = []
            done = 0
            while True:
                data = handle.readframes(4000)
                if not data:
                    break
                done += 4000
                if rec.AcceptWaveform(data):
                    _collect(out, rec.Result())
                elif progress and total:
                    progress(min(0.99, done / total))
            _collect(out, rec.FinalResult())
    finally:
        if wav != audio:
            wav.unlink(missing_ok=True)

    if progress:
        progress(1.0)
    return out


def _collect(out: list[dict], raw: str) -> None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return
    text = (parsed.get("text") or "").strip()
    if not text:
        return
    word_list = parsed.get("result") or []
    out.append({
        "start": round(word_list[0]["start"], 2) if word_list else None,
        "end": round(word_list[-1]["end"], 2) if word_list else None,
        "text": text,
        "words": [
            {"w": w["word"], "start": round(w["start"], 2), "end": round(w["end"], 2)}
            for w in word_list
        ],
    })


def transcript_text(audio: Path) -> Optional[str]:
    """The whole transcript as one string, or None when unavailable."""
    if not available():
        return None
    return " ".join(seg["text"] for seg in transcribe(audio, words=False))
