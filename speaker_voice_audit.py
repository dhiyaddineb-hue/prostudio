from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import soundfile as sf

def run(cmd): subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
def main():
 p=argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('--out',default='voice-audit'); p.add_argument('--hf-token',required=True); a=p.parse_args()
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True); wav=out/'dialogue.wav'
 run(['ffmpeg','-y','-i',a.source,'-vn','-ac','1','-ar','16000',str(wav)])
 from pyannote.audio import Pipeline
 import torch
 pipe=Pipeline.from_pretrained('pyannote/speaker-diarization-community-1',token=a.hf_token); pipe.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
 diar=pipe(str(wav)); ann=getattr(diar,'speaker_diarization',diar); turns=[]
 for t,_,s in ann.itertracks(yield_label=True): turns.append((float(t.start),float(t.end),str(s)))
 speakers={}
 for x,y,s in turns:
  if y-x>speakers.get(s,(0,0,0))[2]: speakers[s]=(x,y,y-x)
 if len(speakers)<2: raise RuntimeError(f'Expected at least 2 speakers, detected {len(speakers)}: {speakers}')
 from voxcpm import VoxCPM
 model=VoxCPM.from_pretrained('openbmb/VoxCPM2',load_denoiser=False)
 report={'detected_speakers':len(speakers),'speakers':{}}
 labels=['الرجل','المرأة']
 for i,(s,(start,end,dur)) in enumerate(sorted(speakers.items(),key=lambda kv:kv[1][0])):
  ref=out/f'speaker_{i+1}_reference.wav'; test=out/f'speaker_{i+1}_arabic_test.wav'
  run(['ffmpeg','-y','-i',str(wav),'-ss',str(start),'-t',str(min(dur,12)),'-ar','16000','-ac','1',str(ref)])
  audio=model.generate(text=f'هذا اختبار مستقل لصوت {labels[i] if i<2 else "المتحدث"}.',reference_wav_path=str(ref),cfg_value=2.0,inference_timesteps=10)
  sf.write(test,audio,int(model.tts_model.sample_rate))
  report['speakers'][s]={'label':labels[i] if i<2 else f'speaker_{i+1}','start':start,'end':end,'reference':ref.name,'test_audio':test.name}
 (out/'voice_audit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
