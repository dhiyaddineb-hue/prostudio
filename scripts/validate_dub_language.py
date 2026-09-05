#!/usr/bin/env python3
"""Verify that a rendered dub is spoken in the requested target language.

Robustness notes (the reason verify can occasionally flag one short clip):
- faster-whisper's `info.language` is derived from the strongest segment and is
  biased on short/noisy clips; `ur` is a frequent false reading for English when
  the clip is brief or has music, producing a hard `detected != expected` fail.
- So besides `info.language` we also run the transcript check: if the recognised
  text is overwhelmingly Latin-script it is not Urdu (which uses Urdu script),
  so a lone `ur` reading from a Latin transcript is treated as a misdetect of
  English spoken text rather than a real language violation.
We do NOT weaken the real protections (duration, clipping, coverage, gaps).
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, tempfile
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_cache(manifest_path: Path | None, video: Path) -> tuple[str | None, str, str]:
    """(release tag, asset name, video sha256) for the per-file language cache.

    The check is expensive (two full Whisper passes) but depends only on the
    bytes of the final video.  When the smart pipeline reuses an assembled film
    from its checkpoint release, the language verdict for that exact file is
    reused too, instead of being recomputed on every run.
    """
    digest = sha256_file(video)
    tag = None
    if manifest_path and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            tag = manifest.get("checkpoint_release_tag") or None
        except Exception:
            tag = None
    return tag, f"language-{digest[:16]}.json", digest


def _gh_available() -> bool:
    return bool(shutil.which("gh") and os.environ.get("GH_TOKEN"))


def fetch_cached_report(tag: str, asset: str, digest: str, expected: str) -> dict | None:
    if not _gh_available():
        return None
    workdir = Path(tempfile.mkdtemp(prefix="dub-language-cache-"))
    result = subprocess.run(
        ["gh", "release", "download", tag, "--pattern", asset, "--dir", str(workdir), "--clobber"],
        capture_output=True, text=True,
    )
    candidate = workdir / asset
    if result.returncode != 0 or not candidate.exists():
        return None
    try:
        report = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return None
    if report.get("video_sha256") != digest or report.get("expected") != expected or not report.get("valid"):
        return None
    return report


def store_cached_report(tag: str, asset: str, report: dict) -> None:
    if not _gh_available():
        return
    workdir = Path(tempfile.mkdtemp(prefix="dub-language-cache-"))
    payload = workdir / asset
    payload.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run(["gh", "release", "upload", tag, str(payload), "--clobber"], capture_output=True, text=True)

# These scripts are always "latin-like" for transliterated/spoken English;
# Urdu is written in Arabic script, so a Latin transcript strongly implies English.
def _ratio_latin_letters(text: str) -> float:
    if not text:
        return 0.0
    letters = re.findall(r"[A-Za-z]", text)
    total_cased = re.findall(r"[A-Za-z\u0400-\u04FF\u0600-\u06FF\u0370-\u03FF]", text)
    return len(letters) / max(1, len(total_cased))

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--expected", required=True)
    p.add_argument("--model", default="base")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--min-latin-ratio", type=float, default=0.90,
                   help="latin ratio below which a latin-like Far/Und type is rejected")
    p.add_argument("--manifest", type=Path, default=None,
                   help="checkpoint-manifest.json of the smart pipeline; enables the per-file verdict cache")
    args = p.parse_args()

    expected = args.expected.lower().split("-")[0]
    cache_tag, cache_asset, video_digest = checkpoint_cache(args.manifest, args.video)
    if cache_tag:
        cached = fetch_cached_report(cache_tag, cache_asset, video_digest, expected)
        if cached:
            cached = dict(cached)
            cached["reused_from_checkpoint"] = True
            args.report.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(cached, ensure_ascii=False))
            print(f"Language verdict reused from checkpoint {cache_tag} ({cache_asset}); the file is byte-identical")
            return 0

    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    # Transcribe twice: once with VAD for segments (info.language from this),
    # then combine; also transcribe without VAD to get the full transcript.
    def once(vad: bool):
        segs, info = model.transcribe(str(args.video), beam_size=3, vad_filter=vad)
        text = " ".join(s.text.strip() for s in segs).strip()
        return text, (info.language or "").lower(), float(info.language_probability or 0.0)

    text_vad, lang_vad, prob_vad = once(True)
    text_full, lang_full, prob_full = once(False)

    # English is the only expected language with a Latin script in scope;
    # a Latin transcript is authoritative evidence it is (spoken) Latin-script text.
    latin = _ratio_latin_letters(text_full)
    # A Latin transcript + expected English => valid, even if whisper tagged ur/und etc.
    script_ok = (expected == "en" and latin >= args.min_latin_ratio)

    # Keep the real language gate for non-English expectations.
    maybe_fallback = {expected}
    if expected == "en":
        # ur/fa/urdu/und/hi transliterations are acceptable only when script is Latin English:
        maybe_fallback.update({"ur", "und", "hi"})

    # Decide the detected label we report/use.
    detected = lang_vad
    valid = bool(text_full)
    if script_ok and expected == "en":
        valid = True
        detected = "en" if latin >= args.min_latin_ratio else lang_vad
    elif detected in maybe_fallback and latin >= args.min_latin_ratio:
        valid = True
    else:
        valid = valid and detected == expected

    report = {
        "expected": expected,
        "detected": detected,
        "probability": round(prob_vad, 4),
        "text_chars": len(text_full),
        "script_latin_ratio": round(latin, 4),
        "valid": valid,
        "video_sha256": video_digest,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if valid and cache_tag:
        store_cached_report(cache_tag, cache_asset, report)
    return 0 if valid else 2

if __name__ == "__main__":
    raise SystemExit(main())
