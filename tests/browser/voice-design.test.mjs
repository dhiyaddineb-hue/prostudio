/**
 * Tests for the browser voice designer.
 *
 * docs/voice-design.js reimplements youtube_auto_dub/voice_design.py so the
 * page can run on GitHub Pages, which has no backend. Two implementations of
 * the same feature drift silently, so this asserts they agree: the Arabic
 * description parser must produce byte-identical settings to the Python one.
 *
 * The DSP is checked against synthesised tones of known pitch, because the
 * property that matters is easy to get wrong invisibly -- a time stretch must
 * change duration and leave pitch alone.
 *
 * Run: node tests/browser/voice-design.test.mjs
 */

import { parseDescription, clampSpec, isNeutral, timeStretch, measurePitch,
         measureBrightness, cloneSpec, PRESETS } from '../../docs/voice-design.js';
let pass=0, fail=0;
const ok=(c,m)=>{ c?pass++:(fail++,console.log('FAIL:',m)); };
const near=(a,b,t,m)=>ok(Math.abs(a-b)<=t, `${m} (${a.toFixed(3)} vs ${b} ±${t})`);

// description parity with Python
const deep=parseDescription('رجل عميق');
ok(deep.pitch<-3 && deep.body>3, 'deep male lowers pitch, adds body');
ok(parseDescription('طفل').pitch > parseDescription('شاب').pitch, 'child above youth');
ok(Math.abs(parseDescription('صوت عميق جدا').pitch) > Math.abs(parseDescription('صوت عميق').pitch), 'intensity scales');
ok(isNeutral(parseDescription('قطة تقود دراجة')), 'unknown text -> neutral');
ok(parseDescription('رَجُل عَمِيق').pitch === parseDescription('رجل عميق').pitch, 'diacritics ignored');
const py={pitch:-8.0,rate:0.8,body:8.0,warmth:2.0,clarity:0.0,air:-1.0};
const js=parseDescription('رجل عميق مهيب بطيء');
ok(JSON.stringify(js)===JSON.stringify(py), `matches Python exactly: ${JSON.stringify(js)}`);

// clamping
const w=clampSpec({pitch:99,rate:99,body:99,warmth:-99});
ok(w.pitch===12&&w.rate===1.6&&w.body===8&&w.warmth===-8,'clamped to range');

// synth helpers
const SR=16000;
const tone=(f,sec,sr=SR)=>{const n=sr*sec,a=new Float32Array(n);
  for(let i=0;i<n;i++){const t=i/sr;a[i]=0.6*Math.sin(2*Math.PI*f*t)+0.3*Math.sin(4*Math.PI*f*t)+0.15*Math.sin(6*Math.PI*f*t);}
  return a;};

// pitch measurement accuracy
for(const f of [90,120,150,200,300]){
  const m=measurePitch(tone(f,1.0),SR);
  near(m.f0,f,f*0.03,`YIN detects ${f}Hz`);
}
ok(measurePitch(new Float32Array(SR),SR).f0===0,'silence -> 0 Hz (no false reading)');

// time stretch: length changes, pitch must NOT
for(const factor of [0.6,0.8,1.25,2.0]){
  const src=tone(150,1.0);
  const out=timeStretch(src,SR,factor);
  near(out.length/src.length,factor,0.02,`stretch x${factor} length`);
  near(measurePitch(out,SR).f0,150,6,`stretch x${factor} keeps pitch`);
}
ok(timeStretch(tone(150,0.3),SR,1).length===SR*0.3,'factor 1 is a no-op');

// brightness + clone
const dull=tone(120,1.0);
const bright=new Float32Array(SR);
for(let i=0;i<SR;i++){const t=i/SR;bright[i]=0.5*Math.sin(2*Math.PI*120*t)+0.4*Math.sin(2*Math.PI*3000*t);}
ok(measureBrightness(bright,SR) > measureBrightness(dull,SR),'brightness ranks correctly');
near(cloneSpec(200,0,100,0).pitch,12,0.01,'clone matches octave exactly');
try{ cloneSpec(0,1,120,1); ok(false,'silent reference must throw'); }
catch{ ok(true,'silent reference refused'); }

// presets all do something
for(const [n,s] of Object.entries(PRESETS)) ok(!isNeutral(s), `preset ${n} is not a no-op`);


// ── agreement with the Python/Praat measurements ────────────────────────
// The page reports pitch next to figures produced by parselmouth, so the two
// must not tell different stories about the same voice. They will never match
// exactly -- YIN and Praat are different algorithms -- so this pins the
// disagreement rather than pretending it is zero. If a change makes the
// browser detector worse, this fails.
import { readFileSync, existsSync } from 'fs';
import { execFileSync } from 'child_process';

const PRAAT = {           // measured by scripts/audition.py via parselmouth
  F4_voice05_masri: 130, R5_voice09_masri: 91,  F1_voice00_fusha: 133,
  R1_voice05_fusha: 124, F2_voice00_masri: 108, R3_voice05_shami: 105,
  F3_voice00_shami: 125, R2_voice05_masri: 123, F5_voice13_masri: 102,
  R4_voice00_masri: 176,
};

let ffmpeg = null;
try {
  ffmpeg = execFileSync(
    '.venv/bin/python',
    ['-c', 'from youtube_auto_dub.ffmpeg_bin import ffmpeg_exe;print(ffmpeg_exe())'],
    { encoding: 'utf8' },
  ).trim();
} catch { /* no venv: the corpus check is skipped, unit tests still ran */ }

if (ffmpeg && existsSync('docs/samples')) {
  const errors = [];
  for (const [name, praat] of Object.entries(PRAAT)) {
    const mp3 = `docs/samples/${name}.mp3`;
    if (!existsSync(mp3)) continue;
    execFileSync(ffmpeg, ['-y', '-v', 'error', '-i', mp3, '-ac', '1',
                          '-ar', '16000', '-f', 'f32le', '/tmp/_vd.raw']);
    const raw = readFileSync('/tmp/_vd.raw');
    const pcm = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
    const { f0 } = measurePitch(pcm, 16000);
    errors.push(Math.abs(12 * Math.log2(f0 / praat)));
  }
  if (errors.length) {
    errors.sort((a, b) => a - b);
    const median = errors[errors.length >> 1];
    const within = errors.filter(e => e <= 1).length;
    ok(median <= 1.0, `median disagreement with Praat is ${median.toFixed(2)} st (want <= 1.0)`);
    ok(within >= errors.length * 0.6,
       `${within}/${errors.length} clips agree within a semitone (want >= 60%)`);
    ok(Math.max(...errors) <= 5.0,
       `worst clip is ${Math.max(...errors).toFixed(2)} st out (want <= 5.0)`);
    console.log(`\n[corpus] median ${median.toFixed(2)} st, ${within}/${errors.length} within 1 st`);
  }
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
