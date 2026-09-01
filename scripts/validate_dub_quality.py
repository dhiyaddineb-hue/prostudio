#!/usr/bin/env python3
"""Fail a dub before publication when the rendered media is objectively broken."""
from __future__ import annotations
import argparse, json, math, re, subprocess
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, check=True)


def duration(path: Path) -> float:
    p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)])
    return float(p.stdout.strip())


def audio_peak_db(path: Path) -> float:
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"], text=True, capture_output=True)
    m = re.search(r"max_volume:\s*(-?[0-9.]+) dB", p.stderr)
    if not m:
        raise RuntimeError("could not measure final audio peak")
    return float(m.group(1))


def silences(path: Path, threshold: str = "-42dB", minimum: float = 0.8) -> list[dict]:
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af", f"silencedetect=noise={threshold}:d={minimum}", "-f", "null", "-"], text=True, capture_output=True)
    starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", p.stderr)]
    ends = [(float(a), float(b)) for a,b in re.findall(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)", p.stderr)]
    out=[]
    for i,(end,dur) in enumerate(ends):
        out.append({"start": starts[i] if i < len(starts) else max(0.0,end-dur), "end": end, "duration": dur})
    return out


def timeline_metrics(segments: list[dict], total: float) -> dict:
    spans=sorted((max(0.0,float(x["start"])), min(total,float(x["end"]))) for x in segments if float(x["end"])>float(x["start"]))
    if not spans or total <= 0: return {"coverage":0.0,"max_gap":total,"leading_gap":total,"trailing_gap":total}
    merged=[]
    for a,b in spans:
        if merged and a <= merged[-1][1]: merged[-1]=(merged[-1][0],max(merged[-1][1],b))
        else: merged.append((a,b))
    covered=sum(b-a for a,b in merged)
    gaps=[merged[i+1][0]-merged[i][1] for i in range(len(merged)-1)]
    leading=merged[0][0]; trailing=max(0.0,total-merged[-1][1])
    return {"coverage":covered/total,"max_gap":max(gaps+[leading,trailing,0.0]),"internal_max_gap":max(gaps+[0.0]),"leading_gap":leading,"trailing_gap":trailing}


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True); ap.add_argument("--video",type=Path,required=True)
    ap.add_argument("--segments",type=Path,required=True); ap.add_argument("--language-report",type=Path)
    ap.add_argument("--policy",choices=["safe","balanced","strict"],default="strict")
    ap.add_argument("--report",type=Path,required=True)
    a=ap.parse_args()
    limits={
      "safe":{"duration":1.5,"peak":-0.1,"extra_silence":3.5,"segment_gap":8.0,"coverage":0.65,"asr_confidence":0.25},
      "balanced":{"duration":1.0,"peak":-0.3,"extra_silence":2.5,"segment_gap":6.0,"coverage":0.72,"asr_confidence":0.40},
      "strict":{"duration":0.75,"peak":-0.5,"extra_silence":1.5,"segment_gap":4.0,"coverage":0.78,"asr_confidence":0.50},
    }[a.policy]
    source_dur=duration(a.source); final_dur=duration(a.video); peak=audio_peak_db(a.video)
    source_sil=silences(a.source); final_sil=silences(a.video)
    source_long=max([x["duration"] for x in source_sil]+[0.0]); final_long=max([x["duration"] for x in final_sil]+[0.0])
    source_silence_total=sum(x["duration"] for x in source_sil)
    source_active_ratio=max(0.0,min(1.0,1.0-source_silence_total/max(source_dur,0.001)))
    required_coverage=min(limits["coverage"],max(0.30,source_active_ratio-0.05))
    segdoc=json.loads(a.segments.read_text(encoding="utf-8")); tm=timeline_metrics(segdoc.get("segments",[]),source_dur)
    trusted_timing=segdoc.get("transcript_source") in {"sidecar", "provided"}
    confidences=[float(x.get("confidence",0.0)) for x in segdoc.get("segments",[])]
    mean_asr_confidence=sum(confidences)/max(len(confidences),1)
    language={}
    if a.language_report and a.language_report.exists(): language=json.loads(a.language_report.read_text(encoding="utf-8"))
    checks={
      "duration_match": abs(final_dur-source_dur) <= limits["duration"],
      "no_clipping": peak <= limits["peak"],
      "language_valid": bool(language.get("valid",False)),
      "silence_not_added": final_long <= max(3.0,source_long+limits["extra_silence"]),
      "segment_coverage": trusted_timing or tm["coverage"] >= required_coverage,
      "segment_gaps": tm["internal_max_gap"] <= max(limits["segment_gap"],source_long+limits["extra_silence"]),
      "segments_present": len(segdoc.get("segments",[])) > 0,
      "asr_confidence": trusted_timing or mean_asr_confidence >= limits["asr_confidence"],
    }
    report={"ok":all(checks.values()),"policy":a.policy,"checks":checks,"metrics":{
      "source_duration":round(source_dur,3),"transcript_source":segdoc.get("transcript_source","asr"),"source_active_ratio":round(source_active_ratio,4),"required_segment_coverage":round(required_coverage,4),"final_duration":round(final_dur,3),"duration_delta":round(final_dur-source_dur,3),
      "peak_db":peak,"source_longest_silence":round(source_long,3),"final_longest_silence":round(final_long,3),
      "segment_count":len(segdoc.get("segments",[])),"mean_asr_confidence":round(mean_asr_confidence,4),**{k:round(v,4) for k,v in tm.items()}},"limits":limits,"language":language}
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report["ok"]: raise SystemExit("quality gate failed: "+", ".join(k for k,v in checks.items() if not v))

if __name__=="__main__": main()
