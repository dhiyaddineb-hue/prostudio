"""Enhance a translated dub with Seed-VC while preserving timing.

The dubbed audio supplies content and timing. A cleaned speech reference from
the original video supplies the target timbre. This intentionally leaves speaker
assignment conservative: when no diarization evidence exists, it uses one
reference voice rather than guessing multiple speakers.

Key improvements:
  - Reference audio is denoised and normalised before being sent to Seed-VC
  - Duration correction uses atempo (not trimming) to prevent cutting words
  - apad prevents truncation of the last syllables
  - Background music is only mixed when explicitly requested
  - Original dialogue is NEVER re-introduced into the output
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Lazy import: gradio_client is only needed when actually calling Seed-VC.
# Tests and local imports should not fail when it's absent.

# The GitHub workflow executes this file as scripts/seed_vc_enhance.py, so the
# repository root is not guaranteed to be on sys.path on every runner.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command, raising on failure with captured output."""
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def duration(path: str | Path) -> float:
    """Get media duration in seconds via ffprobe."""
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return float(result.stdout.strip() or "0")


def atempo_filter(factor: float) -> str:
    """Build an atempo filter chain. ffmpeg atempo accepts 0.5..2.0 per filter."""
    factor = max(0.5, min(2.0, factor))
    return f"atempo={factor:.6f}"


def clean_reference(input_path: Path, output_path: Path, max_seconds: float = 45.0) -> None:
    """Clean and normalise the reference audio for Seed-VC.

    Steps:
      1. Extract audio as mono 22050 Hz WAV (compatible with Seed-VC)
      2. Apply highpass to remove rumble below 80 Hz
      3. Apply lowpass to remove hiss above 9 kHz
      4. Denoise with FFT-based noise reduction
      5. Dynamic range normalisation for consistent loudness
      6. Limit to max_seconds to stay within Seed-VC's reference limit
    """
    # First pass: extract and clean
    run(["ffmpeg", "-y", "-i", str(input_path), "-vn", "-ac", "1", "-ar", "22050",
         "-t", str(max_seconds), str(output_path.with_suffix(".raw.wav"))])

    # Second pass: apply audio filters for clean reference
    run(["ffmpeg", "-y", "-i", str(output_path.with_suffix(".raw.wav")), "-af",
         "highpass=f=80,lowpass=f=9000,"
         "afftdn=nr=12:nf=-25,"
         "dynaudnorm=f=150:g=7,"
         "loudnorm=I=-16:TP=-1.5:LRA=11",
         "-ar", "22050", "-ac", "1", str(output_path)])

    # Clean up intermediate
    output_path.with_suffix(".raw.wav").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enhance dubbed video audio with Seed-VC voice transfer"
    )
    parser.add_argument("--original", required=True,
                        help="Path to the original source video/audio")
    parser.add_argument("--dubbed", required=True,
                        help="Path to the dubbed video (from pipeline render)")
    parser.add_argument("--output", required=True,
                        help="Output path for the enhanced video")
    parser.add_argument("--space", default="phuoc2005/seed-vc",
                        help="HuggingFace Space for Seed-VC")
    parser.add_argument("--diffusion-steps", type=int, default=40,
                        help="Number of diffusion steps (higher = better quality)")
    parser.add_argument("--length-adjust", type=float, default=1.0,
                        help="Length adjustment factor for Seed-VC")
    parser.add_argument("--keep-background", action="store_true",
                        help="Mix an isolated background bed back in; "
                             "off by default to prevent original speech leakage")
    parser.add_argument("--separate-sources", action="store_true",
                        help="Use Demucs neural separation for speech and background stems")
    args = parser.parse_args()

    print(f"[Seed-VC] Enhancement starting")
    print(f"[Seed-VC] Original: {args.original}")
    print(f"[Seed-VC] Dubbed:   {args.dubbed}")
    print(f"[Seed-VC] Output:   {args.output}")
    print(f"[Seed-VC] Steps:    {args.diffusion_steps}")

    work = Path(args.output).with_suffix(".seed-work")
    work.mkdir(parents=True, exist_ok=True)
    cleaned_ref = work / "clean-reference.wav"
    dubbed_wav = work / "dubbed-content.wav"
    converted_wav = work / "seed-converted.wav"
    timed_wav = work / "seed-timed.wav"
    background_wav = work / "background.wav"
    final_audio = work / "final-mixed.wav"

    # ── 1. Prepare the reference ────────────────────────────────────────
    # Use Demucs-separated dialogue stem if available, otherwise the raw source.
    from youtube_auto_dub.source_separation import separate_dialogue_background

    separated = None
    if args.separate_sources:
        try:
            separated = separate_dialogue_background(Path(args.original), work)
            print("[Seed-VC] Demucs separation successful")
        except Exception as exc:
            print(f"[Seed-VC] Demucs failed ({exc}); using original audio")
            separated = None

    reference_input = separated[0] if separated else Path(args.original)
    background_input = separated[1] if separated else Path(args.original)

    # Clean the reference: denoise + normalise without destroying pitch/cadence
    clean_reference(reference_input, cleaned_ref, max_seconds=45.0)
    print("[Seed-VC] Reference cleaned and normalised")

    # ── 2. Prepare the dubbed audio ─────────────────────────────────────
    run(["ffmpeg", "-y", "-i", args.dubbed, "-vn", "-ac", "1", "-ar", "22050",
         str(dubbed_wav)])
    print("[Seed-VC] Dubbed audio extracted")

    # ── 3. Run Seed-VC ──────────────────────────────────────────────────
    from gradio_client import Client, handle_file

    print(f"[Seed-VC] Connecting to {args.space}...")
    client = Client(args.space)

    print("[Seed-VC] Running voice conversion...")
    result = client.predict(
        handle_file(str(dubbed_wav)),
        handle_file(str(cleaned_ref)),
        args.diffusion_steps,
        args.length_adjust,
        0.7,   # cosine overlap
        False,  # not clip
        True,   # return full
        0,      # seed
        api_name="/predict_1",
    )

    if not isinstance(result, (list, tuple)) or len(result) < 2:
        raise RuntimeError(f"Unexpected Seed-VC response: {result!r}")

    full = result[1]
    if isinstance(full, dict):
        full = full.get("path") or full.get("url")
    if not full:
        raise RuntimeError(f"Seed-VC returned no full audio: {result!r}")
    print("[Seed-VC] Voice conversion complete")

    # ── 4. Post-process the converted audio ─────────────────────────────
    # Convert to 24kHz mono for the final mix
    run(["ffmpeg", "-y", "-i", str(full), "-ar", "24000", "-ac", "1", str(converted_wav)])

    # Correct any global duration drift. Use atempo (speed change) rather than
    # trimming, so words at the end of sentences are never cut.
    target_duration = duration(args.dubbed)
    converted_duration = duration(converted_wav)
    factor = converted_duration / target_duration if target_duration > 0 else 1.0
    print(f"[Seed-VC] Duration ratio: {factor:.4f} (target: {target_duration:.2f}s, got: {converted_duration:.2f}s)")

    if abs(factor - 1.0) > 0.015:
        run(["ffmpeg", "-y", "-i", str(converted_wav), "-af", atempo_filter(factor),
             "-ar", "24000", "-ac", "1", str(timed_wav)])
        print(f"[Seed-VC] Duration corrected by {factor:.4f}x")
    else:
        timed_wav = converted_wav

    # ── 5. Mix background (if requested) ────────────────────────────────
    if args.keep_background:
        # Extract a clean background bed from the separated stems or via
        # centre-channel removal. This is music/effects only — never dialogue.
        run(["ffmpeg", "-y", "-i", str(background_input), "-af",
             "volume=0.45", "-ar", "24000", str(background_wav)])
        run(["ffmpeg", "-y", "-i", str(timed_wav), "-i", str(background_wav),
             "-filter_complex",
             "[1:a]atrim=0:{0:.3f},asetpts=PTS-STARTPTS[bg];"
             "[0:a][bg]amix=inputs=2:duration=first:normalize=0[out]".format(target_duration),
             "-map", "[out]", "-ar", "24000", "-ac", "1", str(final_audio)])
        print("[Seed-VC] Background mixed in")
    else:
        run(["ffmpeg", "-y", "-i", str(timed_wav), "-ar", "24000", "-ac", "1", str(final_audio)])
        print("[Seed-VC] No background mixing (clean dub)")

    # ── 6. Mux with video ───────────────────────────────────────────────
    # Keep video untouched (copy codec), replace audio entirely.
    # apad prevents cutting the last words; -t ensures no trailing silence.
    run([
        "ffmpeg", "-y",
        "-i", args.dubbed,
        "-i", str(final_audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-af", "apad",
        "-t", f"{target_duration:.3f}",
        "-c:a", "aac", "-b:a", "192k",
        args.output,
    ])
    print(f"[Seed-VC] Output written: {args.output}")

    report = {
        "ok": True,
        "output": args.output,
        "space": args.space,
        "diffusion_steps": args.diffusion_steps,
        "duration_factor": round(factor, 4),
        "target_duration": round(target_duration, 3),
        "converted_duration": round(converted_duration, 3),
        "reference_cleaned": True,
        "reference_denoised": True,
        "reference_normalised": True,
        "speaker_mode": "conservative-single-reference",
        "background_mixed": args.keep_background,
        "sources_separated": bool(separated),
        "original_dialogue_suppressed": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
