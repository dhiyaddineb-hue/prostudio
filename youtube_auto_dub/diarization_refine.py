"""Conservative post-processing for neural speaker diarization turns."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RefinedTurn:
    start: float
    end: float
    speaker: str
    @property
    def duration(self): return max(0.0, self.end - self.start)

def refine_turns(turns, min_turn=0.24, merge_gap=0.18):
    """Merge adjacent same-speaker fragments and absorb tiny glitches."""
    ordered = sorted((RefinedTurn(float(a), float(b), str(s)) for a,b,s in turns if b > a), key=lambda t:t.start)
    merged=[]
    for t in ordered:
        if merged and t.speaker == merged[-1].speaker and t.start - merged[-1].end <= merge_gap:
            prev=merged.pop(); merged.append(RefinedTurn(prev.start, max(prev.end,t.end), prev.speaker))
        else: merged.append(t)
    changed=True
    while changed:
        changed=False; out=[]
        for i,t in enumerate(merged):
            if t.duration < min_turn and i>0 and i+1<len(merged) and merged[i-1].speaker == merged[i+1].speaker:
                left,right=merged[i-1],merged[i+1]
                out[-1]=RefinedTurn(left.start, right.start, left.speaker)
                changed=True
            else: out.append(t)
        merged=out
    return [t for t in merged if t.duration >= min_turn]

def resolve_overlaps(turns):
    """Split overlaps at their midpoint; never double-count audio."""
    out=[]
    for t in sorted(turns,key=lambda x:x.start):
        if out and t.start < out[-1].end and t.speaker != out[-1].speaker:
            mid=(t.start+out[-1].end)/2
            prev=out.pop(); out.append(RefinedTurn(prev.start,mid,prev.speaker)); out.append(RefinedTurn(mid,t.end,t.speaker))
        else: out.append(t)
    return [t for t in out if t.end>t.start]

def assign_segment(start,end,turns,min_overlap=.45):
    scores={}
    for t in turns:
        overlap=max(0.0,min(end,t.end)-max(start,t.start))
        if overlap: scores[t.speaker]=scores.get(t.speaker,0.0)+overlap
    if not scores: return None,0.0
    speaker,overlap=max(scores.items(),key=lambda x:x[1])
    return (speaker, round(min(1.0,overlap/max(end-start,.001)),3)) if overlap/max(end-start,.001)>=min_overlap else (None,round(overlap/max(end-start,.001),3))

def stats(turns):
    return {s: round(sum(t.duration for t in turns if t.speaker==s),2) for s in sorted({t.speaker for t in turns})}
