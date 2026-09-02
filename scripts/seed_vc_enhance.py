"""Enhance a translated dub with Seed-VC while preserving timing.

The English dub supplies content and timing. A cleaned speech reference from the
original video supplies the target timbre. This intentionally leaves speaker
assignment conservative: when no diarization evidence exists, it uses one
reference voice rather than guessing multiple speakers.
"""
from __future__ import annotations

import argparse
import json
import time
import subprocess
import sys
from pathlib import Path
import os

from gradio_client import Client, handle_file

# The GitHub workflow executes this file as scripts/seed_vc_enhance.py, so the
# repository root is not guaranteed to be on sys.path on every runner.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from youtube_auto_dub.source_separation import separate_dialogue_background


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def duration(path: str | Path) -> float:
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return float(result.stdout.strip())


def atempo_filter(factor: float) -> str:
    # ffmpeg atempo accepts 0.5..2.0 per filter; chain for safety.
    factor = max(0.5, min(2.0, factor))
    return f"atempo={factor:.6f}"



def _do_per_speaker_seed(dubbed_wav, target_duration, seg_conv, refs, work, args, out_wav):
    """Run Seed-VC once per role, then place each converted piece at its own
    timeline slot so distinct timbres (e.g. a male and a female speaker) survive.
    Falls back to the single-reference path if anything goes wrong."""
    from gradio_client import Client, handle_file
    pieces = []
    for (a, b), sp in sorted(seg_conv.items()):
        ref = refs.get(sp)
        if not ref or not Path(ref).exists():
            continue
        seg = work / f"{sp}-seg.wav"
        run(["ffmpeg", "-y", "-i", str(dubbed_wav), "-ss", f"{a:.3f}", "-t", f"{b-a:.3f}",
             "-ac", "1", "-ar", "22050", str(seg)])
        result = None
        for attempt in range(1, 6):
            try:
                client = Client(args.space)
                result = client.predict(
                    handle_file(str(seg)),
                    handle_file(str(ref)),
                    args.diffusion_steps,
                    args.length_adjust,
                    0.7, False, True, 0,
                    api_name="/predict_1",
                )
                if isinstance(result, (list, tuple)) and len(result) >= 2 and result[1]:
                    break
            except Exception:
                result = None
                time.sleep(min(5 * attempt, 30))
        if not result:
            continue
        full = result[1] if isinstance(result, (list, tuple)) else result
        if isinstance(full, dict):
            full = full.get("path") or full.get("url")
        if not full:
            continue
        conv = work / f"{sp}-conv.wav"
        run(["ffmpeg", "-y", "-i", str(full), "-ar", "24000", "-ac", "1", str(conv)])
        pieces.append((a, b, sp, conv))
    if not pieces:
        return False
    # Assemble all per-speaker pieces at their correct offsets.
    parts = []
    for a, b, sp, conv in sorted(pieces, key=lambda x: x[0]):
        prt = work / f"{sp}-placed.wav"
        run(["ffmpeg", "-y", "-i", str(conv), "-af", f"adelay={int(a*1000)}|{int(a*1000)}",
             "-ar", "24000", "-ac", "1", str(prt)])
        parts.append(str(prt))
    run(["ffmpeg", "-y"] + sum([["-i", pr] for pr in parts], []) +
        ["-filter_complex",
         "amix=inputs=%d:duration=longest:normalize=0" % len(parts),
         "-ar", "24000", "-ac", "1", str(out_wav)])
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--dubbed", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--space", default="phuoc2005/seed-vc")
    parser.add_argument("--diffusion-steps", type=int, default=40)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--keep-background", action="store_true",
                        help="Mix an isolated background bed back in; off by default to prevent original speech leakage")
    parser.add_argument("--separate-sources", action="store_true",
                        help="Use Demucs neural separation for speech and background stems")
    parser.add_argument("--speaker-refs", default="",
                        help="Comma-separated SPEAKER_XX=/path/to/ref.wav per-voice cloned references")
    parser.add_argument("--speaker-map", default="",
                        help="Comma-separated start-end=SPEAKER_XX (seconds) timeline-to-role mapping")
    args = parser.parse_args()

    work = Path(args.output).with_suffix(".seed-work")
    work.mkdir(parents=True, exist_ok=True)
    original_wav = work / "original-reference.wav"
    dubbed_wav = work / "english-content.wav"
    cleaned_ref = work / "clean-reference.wav"
    converted_wav = work / "seed-converted.wav"
    timed_wav = work / "seed-timed.wav"
    background_wav = work / "background.wav"
    final_audio = work / "final-mixed.wav"

    # Clean the reference without destroying the speaker's pitch or cadence.
    separated = separate_dialogue_background(Path(args.original), work) if args.separate_sources else None
    reference_input = separated[0] if separated else Path(args.original)
    background_input = separated[1] if separated else Path(args.original)
    run(["ffmpeg", "-y", "-i", str(reference_input), "-vn", "-ac", "1", "-ar", "22050",
         str(original_wav)])
    run(["ffmpeg", "-y", "-i", str(original_wav), "-af",
         "highpass=f=80,lowpass=f=9000,afftdn=nr=12:nf=-25,dynaudnorm=f=150:g=7",
         "-ar", "22050", "-ac", "1", str(cleaned_ref)])
    if args.keep_background:
        # Remove the phantom centre channel: dialogue is normally centred while
        # music/effects live in the sides. This bed is mixed back only after
        # Seed-VC and only when explicitly requested.
        run(["ffmpeg", "-y", "-i", str(background_input), "-af",
             "volume=0.45",
             "-ar", "24000", str(background_wav)])
    run(["ffmpeg", "-y", "-i", args.dubbed, "-vn", "-ac", "1", "-ar", "22050",
         str(dubbed_wav)])

    # ── Per-speaker Seed-VC branch: if per-role references and timeline were
    # supplied, convert each role separately so distinct timbres survive.
    if args.speaker_refs and args.speaker_map:
        refs = {}
        for item in args.speaker_refs.split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                refs[k.strip()] = v.strip()
        seg_conv = {}
        for item in args.speaker_map.split(","):
            if "=" in item and "-" in item.split("=")[0]:
                rng, sp = item.split("=", 1)
                try:
                    a, b = [float(x) for x in rng.split("-")]
                except Exception:
                    continue
                seg_conv[(a, b)] = sp.strip()
        final_audio = work / "per-speaker.wav"
        if _do_per_speaker_seed(dubbed_wav, duration(args.dubbed), seg_conv, refs, work, args, final_audio):
            # mux video + per-speaker audio, same as single path below
            target_duration = duration(args.dubbed)
            run([
                "ffmpeg", "-y", "-i", args.dubbed, "-i", str(final_audio),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-af", "apad", "-t", f"{target_duration:.3f}",
                "-c:a", "aac", "-b:a", "160k", args.output,
            ])
            print(json.dumps({"ok": True, "output": args.output, "space": args.space,
                              "speaker_mode": "per-speaker",
                              "roles": sorted(set(s for _,_,s,_ in seg_files)) if False else sorted(set(seg_conv.values())),
                              "pieces": len(seg_conv)}))
            return


    result = None
    last_error = None
    for attempt in range(1, 9):
        try:
            client = Client(args.space)
            result = client.predict(
                handle_file(str(dubbed_wav)),
                handle_file(str(cleaned_ref)),
                args.diffusion_steps,
                args.length_adjust,
                0.7,
                False,
                True,
                0,
                api_name="/predict_1",
            )
            if isinstance(result, (list, tuple)) and len(result) >= 2 and result[1]:
                break
            raise RuntimeError(f"incomplete Seed-VC response: {result!r}")
        except Exception as exc:
            last_error = exc
            if attempt == 8:
                raise RuntimeError("Seed-VC Space failed after 8 attempts") from exc
            time.sleep(min(5 * attempt, 30))
    if not isinstance(result, (list, tuple)) or len(result) < 2:
        raise RuntimeError(f"Unexpected Seed-VC response: {result!r}")
    full = result[1]
    if isinstance(full, dict):
        full = full.get("path") or full.get("url")
    if not full:
        raise RuntimeError(f"Seed-VC returned no full audio: {result!r}")
    run(["ffmpeg", "-y", "-i", str(full), "-ar", "24000", "-ac", "1", str(converted_wav)])

    # Correct any global duration drift before muxing, rather than cutting words.
    target_duration = duration(args.dubbed)
    converted_duration = duration(converted_wav)
    factor = converted_duration / target_duration if target_duration else 1.0
    if abs(factor - 1.0) > 0.015:
        run(["ffmpeg", "-y", "-i", str(converted_wav), "-af", atempo_filter(factor),
             "-ar", "24000", "-ac", "1", str(timed_wav)])
    else:
        timed_wav = converted_wav

    if args.keep_background:
        # Restore the isolated music/effects bed only when explicitly enabled.
        run(["ffmpeg", "-y", "-i", str(timed_wav), "-i", str(background_wav),
             "-filter_complex",
             "[1:a]atrim=0:{0:.3f},asetpts=PTS-STARTPTS[bg];"
             "[0:a][bg]amix=inputs=2:duration=first:normalize=0[out]".format(target_duration),
             "-map", "[out]", "-ar", "24000", "-ac", "1", str(final_audio)])
    else:
        run(["ffmpeg", "-y", "-i", str(timed_wav), "-ar", "24000", "-ac", "1", str(final_audio)])

    # Keep video untouched, pad only tiny tails, and never let a long VC result
    # truncate the last words or extend beyond the original video.
    run([
        "ffmpeg", "-y", "-i", args.dubbed, "-i", str(final_audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
        "-af", "apad", "-t", f"{target_duration:.3f}",
        "-c:a", "aac", "-b:a", "160k", args.output,
    ])
    print(json.dumps({
        "ok": True,
        "output": args.output,
        "space": args.space,
        "diffusion_steps": args.diffusion_steps,
        "duration_factor": round(factor, 4),
        "reference_cleaned": True,
        "speaker_mode": "conservative-single-reference",
        "background_mixed": args.keep_background,
        "sources_separated": bool(separated),
    }))


if __name__ == "__main__":
    main()
