/**
 * Voice design in the browser.
 *
 * The server version of this needs ffmpeg and Praat, which GitHub Pages cannot
 * run — it serves static files and nothing else. So the whole signal chain is
 * reimplemented here: pitch shifting, time stretching, EQ and pitch
 * measurement all run in the visitor's own browser, and the page needs no
 * backend at all.
 *
 * The maths mirrors the Python in youtube_auto_dub/voice_design.py so the two
 * agree. Where they cannot agree exactly it is said out loud rather than
 * papered over: pitch here is measured with YIN, not Praat, and the two
 * disagree by a few hertz on the same file.
 */

// ── controls ────────────────────────────────────────────────────────────
export const CONTROLS = [
  { id: 'pitch',   label: 'الطبقة',  min: -12, max: 12, step: 0.5,  unit: 'نصف نغمة', def: 0 },
  { id: 'rate',    label: 'السرعة',  min: 0.6, max: 1.6, step: 0.02, unit: '×',        def: 1 },
  { id: 'body',    label: 'الجسم',   min: -8,  max: 8,  step: 0.5,  unit: 'dB',       def: 0 },
  { id: 'warmth',  label: 'الدفء',   min: -8,  max: 8,  step: 0.5,  unit: 'dB',       def: 0 },
  { id: 'clarity', label: 'الوضوح',  min: -8,  max: 8,  step: 0.5,  unit: 'dB',       def: 0 },
  { id: 'air',     label: 'الهواء',  min: -8,  max: 8,  step: 0.5,  unit: 'dB',       def: 0 },
];

const LIMITS = {
  pitch: [-12, 12], rate: [0.6, 1.6], body: [-8, 8],
  warmth: [-8, 8], clarity: [-8, 8], air: [-8, 8],
};

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export function neutralSpec() {
  return { pitch: 0, rate: 1, body: 0, warmth: 0, clarity: 0, air: 0 };
}

export function clampSpec(spec) {
  const out = neutralSpec();
  for (const key of Object.keys(out)) {
    const [lo, hi] = LIMITS[key];
    const v = Number(spec?.[key]);
    out[key] = Number.isFinite(v) ? clamp(v, lo, hi) : out[key];
  }
  return out;
}

export function isNeutral(spec) {
  const s = clampSpec(spec);
  return Math.abs(s.pitch) < 0.01 && Math.abs(s.rate - 1) < 0.005 &&
    ['body', 'warmth', 'clarity', 'air'].every(k => Math.abs(s[k]) < 0.05);
}

// ── Arabic description → controls ───────────────────────────────────────
// Deltas, not absolutes, so "رجل عجوز عميق" stacks into one instruction.
const WORDS = [
  [['عميق', 'غليظ', 'جهوري', 'أجش', 'خشن'], { pitch: -3.5, body: 3.5, air: -1 }],
  [['رفيع', 'حاد', 'نحيل'], { pitch: 3.5, body: -2.5, clarity: 1.5 }],
  [['رجل', 'رجالي', 'ذكر', 'ذكوري'], { pitch: -2, body: 2 }],
  [['امرأة', 'نسائي', 'أنثى', 'أنثوي', 'سيدة'], { pitch: 4, body: -2, air: 1 }],
  [['طفل', 'طفلة', 'صبي'], { pitch: 7, body: -4, rate: 0.12, clarity: 1.5 }],
  [['شاب', 'شابة', 'صغير'], { pitch: 1.5, rate: 0.08, clarity: 1 }],
  [['عجوز', 'مسن', 'كبير', 'شيخ', 'حكيم'], { pitch: -2, rate: -0.15, air: 1.5, clarity: -1 }],
  [['غاضب', 'صارم', 'حازم', 'عنيف', 'قوي'], { clarity: 2.5, rate: 0.1, body: 1.5 }],
  [['هادئ', 'رقيق', 'حنون', 'لطيف', 'ناعم'], { rate: -0.1, warmth: 2.5, clarity: -1 }],
  [['همس', 'هامس', 'خافت', 'سري'], { air: 4, clarity: -2, body: -2, rate: -0.08 }],
  [['دافئ', 'دافي'], { warmth: 3, air: -0.5 }],
  [['واضح', 'لامع', 'ساطع', 'مشرق'], { clarity: 3, air: 1.5 }],
  [['مهيب', 'ضخم', 'عريض', 'ملحمي', 'فخم'], { pitch: -2.5, body: 4, warmth: 2 }],
  [['سريع', 'متسرع', 'متحمس', 'حماسي'], { rate: 0.2, clarity: 1 }],
  [['بطيء', 'متمهل', 'متأنٍ', 'متأني', 'رزين'], { rate: -0.2 }],
  [['مذيع', 'إخباري', 'أخبار', 'رسمي'], { clarity: 2, body: 1, rate: 0.05 }],
  [['راوي', 'وثائقي', 'سرد', 'قصة'], { warmth: 2, body: 1.5, rate: -0.08 }],
  [['شرير', 'مخيف', 'مرعب', 'مظلم'], { pitch: -4, body: 3, clarity: -1, rate: -0.1 }],
  [['مرح', 'بشوش', 'ودود', 'مبتهج'], { pitch: 1, clarity: 1.5, rate: 0.1, warmth: 1.5 }],
  [['متعب', 'حزين', 'مكسور', 'يائس'], { pitch: -1.5, rate: -0.18, clarity: -1.5, air: 1 }],
];

const STRONG = ['جدا', 'جداً', 'للغاية', 'كثيرا', 'كثيراً', 'أكثر'];
const WEAK = ['قليلا', 'قليلاً', 'خفيف', 'بعض', 'شوية', 'نوعا ما'];

function normalise(text) {
  return String(text || '')
    .replace(/[\u0610-\u061a\u064b-\u065f\u0670\u0640]/g, '')
    .replace(/[أإآٱ]/g, 'ا')
    .replace(/ى/g, 'ي')
    .replace(/ة/g, 'ه')
    .toLowerCase();
}

/**
 * Turn a plain Arabic description into mixer settings.
 *
 * Words we do not know are ignored rather than guessed at, so a description
 * matching nothing returns neutral. Inventing a transform for unknown words
 * would make every other result untrustworthy.
 */
export function parseDescription(text) {
  const norm = normalise(text);
  let scale = 1;
  if (STRONG.some(w => norm.includes(normalise(w)))) scale = 1.5;
  else if (WEAK.some(w => norm.includes(normalise(w)))) scale = 0.5;

  const spec = neutralSpec();
  for (const [words, delta] of WORDS) {
    if (!words.some(w => norm.includes(normalise(w)))) continue;
    for (const [field, amount] of Object.entries(delta)) spec[field] += amount * scale;
  }
  return clampSpec(spec);
}

export const PRESETS = {
  'مهيب عميق':   { pitch: -3.5, rate: 1,    body: 4.5, warmth: 2, clarity: 1,  air: 0 },
  'شاب حماسي':   { pitch: 2,    rate: 1.12, body: 0,   warmth: 0, clarity: 2.5, air: 1 },
  'عجوز حكيم':   { pitch: -2,   rate: 0.85, body: 0,   warmth: 2, clarity: -1, air: 2 },
  'همس قريب':    { pitch: -0.5, rate: 0.92, body: -2,  warmth: 0, clarity: -1.5, air: 5 },
  'مذيع أخبار':  { pitch: -1,   rate: 1.05, body: 1.5, warmth: 0, clarity: 3,  air: 0 },
  'راوي وثائقي': { pitch: -1.5, rate: 0.92, body: 2,   warmth: 3, clarity: 0,  air: 0 },
  'شرير مظلم':   { pitch: -5,   rate: 0.9,  body: 3.5, warmth: 0, clarity: -1, air: 0 },
  'طفل':         { pitch: 7,    rate: 1.1,  body: -4,  warmth: 0, clarity: 2,  air: 0 },
};

// ── time stretch (WSOLA) ────────────────────────────────────────────────
/**
 * Stretch `input` to `factor` times its length without moving pitch.
 *
 * Plain overlap-add smears speech badly because grains land out of phase, so
 * each grain is nudged within a +/-5 ms window to the offset that best matches
 * what has already been written. A Hann window at 50% overlap sums to exactly
 * one, so the level is preserved without a normalisation pass.
 */
export function timeStretch(input, sampleRate, factor) {
  if (!Number.isFinite(factor) || factor <= 0) throw new Error('factor must be > 0');
  if (Math.abs(factor - 1) < 1e-4) return input.slice();

  const grain = Math.max(256, Math.round(0.05 * sampleRate));
  const Hs = grain >> 1;
  const Ha = Math.max(1, Math.round(Hs / factor));
  const search = Math.min(Hs >> 1, Math.round(0.005 * sampleRate));

  const targetLen = Math.max(1, Math.round(input.length * factor));
  const out = new Float32Array(targetLen + grain);

  const win = new Float32Array(grain);
  for (let i = 0; i < grain; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / grain);

  let outPos = 0;
  let inPos = 0;
  while (outPos + grain < out.length && inPos < input.length) {
    let best = 0;
    if (outPos > 0 && search > 0) {
      let bestScore = -Infinity;
      for (let d = -search; d <= search; d++) {
        const start = inPos + d;
        if (start < 0 || start + Hs >= input.length) continue;
        let score = 0;
        for (let k = 0; k < Hs; k += 8) score += out[outPos + k] * input[start + k];
        if (score > bestScore) { bestScore = score; best = d; }
      }
    }
    const start = Math.max(0, Math.min(input.length - grain, inPos + best));
    for (let k = 0; k < grain; k++) out[outPos + k] += input[start + k] * win[k];
    outPos += Hs;
    inPos += Ha;
  }
  return out.subarray(0, targetLen);
}

// ── pitch measurement (YIN) ─────────────────────────────────────────────
function downsample(input, from, to) {
  if (to >= from) return { data: input, rate: from };
  const ratio = from / to;
  const out = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const at = i * ratio;
    const j = at | 0;
    const frac = at - j;
    out[i] = (input[j] || 0) * (1 - frac) + (input[j + 1] || 0) * frac;
  }
  return { data: out, rate: to };
}

/** The cumulative-mean-normalised difference function YIN searches. */
function yinCmnd(frame, sampleRate, fmin, fmax) {
  const tauMin = Math.max(2, Math.floor(sampleRate / fmax));
  const tauMax = Math.floor(sampleRate / fmin);
  if (tauMax >= frame.length) return null;

  const diff = new Float32Array(tauMax + 1);
  for (let tau = tauMin; tau <= tauMax; tau++) {
    let sum = 0;
    for (let i = 0; i + tau < frame.length; i++) {
      const d = frame[i] - frame[i + tau];
      sum += d * d;
    }
    diff[tau] = sum;
  }

  // Cumulative mean normalisation: without it the difference function is
  // minimal at tau=0 and every detector locks onto silence.
  const cmnd = new Float32Array(tauMax + 1);
  let running = 0;
  for (let tau = tauMin; tau <= tauMax; tau++) {
    running += diff[tau];
    cmnd[tau] = running > 0 ? (diff[tau] * (tau - tauMin + 1)) / running : 1;
  }
  return { cmnd, tauMin, tauMax };
}

/** Sub-sample refinement: at 16 kHz one whole sample is ~4 Hz at male pitch. */
function refine(cmnd, tau, sampleRate) {
  const a = cmnd[tau - 1] ?? cmnd[tau];
  const b = cmnd[tau];
  const c = cmnd[tau + 1] ?? cmnd[tau];
  const denom = a + c - 2 * b;
  const shift = Math.abs(denom) > 1e-9 ? (0.5 * (a - c)) / denom : 0;
  return sampleRate / (tau + clamp(shift, -1, 1));
}

/**
 * First tau below `threshold`, with an octave check.
 *
 * The plain rule — take the first dip under the threshold — locks onto a
 * harmonic when the fundamental is weak, reporting the voice an octave or a
 * fifth too high. So a multiple of tau is preferred when it dips there too.
 *
 * Two guards keep the check from doing harm. A strongly periodic signal dips
 * just as hard at twice its period — on a synthetic 150 Hz tone tau and 2*tau
 * both score 3e-4, and at 300 Hz the third multiple scores 4e-32 — so chasing
 * the deeper number reports a subharmonic. When the first dip is already
 * near-perfect it *is* the period, and the search is skipped; otherwise the
 * multiple must be clearly deeper, not merely comparable.
 */
const PERIODIC_ENOUGH = 0.01;

function yinTau(cmnd, tauMin, tauMax, threshold) {
  let tau = -1;
  for (let t = tauMin; t <= tauMax; t++) {
    if (cmnd[t] < threshold) {
      while (t + 1 <= tauMax && cmnd[t + 1] < cmnd[t]) t++;
      tau = t;
      break;
    }
  }
  if (tau < 0) return -1;
  if (cmnd[tau] < PERIODIC_ENOUGH) return tau;

  for (const k of [2, 3]) {
    const guess = tau * k;
    if (guess > tauMax) break;
    let best = guess, bestVal = cmnd[guess];
    for (let j = Math.max(tauMin, guess - 3); j <= Math.min(tauMax, guess + 3); j++) {
      if (cmnd[j] < bestVal) { bestVal = cmnd[j]; best = j; }
    }
    if (bestVal < threshold && bestVal < cmnd[tau] * 0.8) tau = best;
  }
  return tau;
}

/**
 * Median F0 and its coefficient of variation over voiced frames.
 *
 * CV is the number the page reports as "alive" or "robotic": the spread of
 * pitch across the take divided by its mean, so it does not move when a voice
 * is simply higher or lower.
 *
 * Two passes. The first finds a rough centre; the second re-picks each frame's
 * period within ±7 semitones of it, which suppresses the octave jumps a
 * per-frame detector makes on breathy speech. Praat does this properly with a
 * Viterbi path over all candidates; a fixed band is the cheap approximation
 * that fits in a page.
 *
 * Accuracy against Praat on the sample set: within about one semitone on five
 * of six clips. The exception is genuinely bimodal — Praat itself puts its
 * quartiles at 88 and 173 Hz — and no single median describes it well. Treat
 * these figures as a guide of the same voice against itself, not as
 * interchangeable with the Python measurements.
 */
export function measurePitch(samples, sampleRate, opts = {}) {
  const fmin = opts.fmin ?? 60;
  const fmax = opts.fmax ?? 400;
  const threshold = opts.threshold ?? 0.15;
  // 16 kHz is plenty for pitch and keeps YIN roughly 8x cheaper.
  const { data, rate } = downsample(samples, sampleRate, 16000);

  const frameLen = Math.round(0.04 * rate);
  const hop = Math.round(0.01 * rate);
  const empty = { f0: 0, cv: 0, voiced: 0, values: [] };
  if (data.length < frameLen) return empty;

  let peak = 0;
  for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i]));
  const floor = peak * 0.02;

  const frames = [];
  for (let start = 0; start + frameLen <= data.length; start += hop) {
    const frame = data.subarray(start, start + frameLen);
    let rms = 0;
    for (let i = 0; i < frame.length; i++) rms += frame[i] * frame[i];
    if (Math.sqrt(rms / frame.length) < floor) continue;
    const found = yinCmnd(frame, rate, fmin, fmax);
    if (!found) continue;
    const tau = yinTau(found.cmnd, found.tauMin, found.tauMax, threshold);
    if (tau > 0) frames.push({ ...found, tau });
  }
  if (!frames.length) return empty;

  const mid = arr => {
    const s = [...arr].sort((a, b) => a - b);
    return s[s.length >> 1];
  };
  const first = frames.map(f => refine(f.cmnd, f.tau, rate));
  const centre = mid(first);

  const lo = Math.floor(rate / (centre * Math.pow(2, 7 / 12)));
  const hi = Math.ceil(rate / (centre * Math.pow(2, -7 / 12)));
  const values = [];
  for (const f of frames) {
    let best = -1, bestVal = Infinity;
    for (let t = Math.max(f.tauMin, lo); t <= Math.min(f.tauMax, hi); t++) {
      if (f.cmnd[t] < bestVal) { bestVal = f.cmnd[t]; best = t; }
    }
    if (best > 0 && bestVal < 0.35) {
      const hz = refine(f.cmnd, best, rate);
      if (hz >= fmin && hz <= fmax) values.push(hz);
    }
  }
  const use = values.length >= frames.length * 0.3 ? values : first;
  if (!use.length) return empty;

  const mean = use.reduce((a, b) => a + b, 0) / use.length;
  const variance = use.reduce((a, b) => a + (b - mean) ** 2, 0) / use.length;
  return {
    f0: mid(use),
    cv: mean > 0 ? Math.sqrt(variance) / mean : 0,
    voiced: use.length,
    values: use,
  };
}

// ── brightness (for reference matching) ─────────────────────────────────
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const nr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}

/** Ratio of 2-5 kHz energy to 100-1000 Hz energy: how bright a voice reads. */
export function measureBrightness(samples, sampleRate) {
  const N = 4096;
  if (samples.length < N) return 0;
  const windows = 8;
  const step = Math.max(N, Math.floor((samples.length - N) / windows));
  const spec = new Float64Array(N / 2);
  let taken = 0;
  for (let off = 0; off + N <= samples.length && taken < windows; off += step, taken++) {
    const re = new Float64Array(N), im = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      re[i] = samples[off + i] * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N));
    }
    fft(re, im);
    for (let i = 0; i < N / 2; i++) spec[i] += Math.hypot(re[i], im[i]);
  }
  if (!taken) return 0;

  const binHz = sampleRate / N;
  const band = (lo, hi) => {
    let sum = 0, count = 0;
    for (let i = Math.ceil(lo / binHz); i <= Math.floor(hi / binHz) && i < N / 2; i++) {
      sum += spec[i]; count++;
    }
    return count ? sum / count : 0;
  };
  const low = band(100, 1000);
  return low > 0 ? band(2000, 5000) / low : 0;
}

/**
 * Controls that move a base voice toward a measured reference.
 *
 * Pitch is matched exactly — that part is a true match. Brightness is matched
 * with presence EQ, which stands in for vocal tract shape rather than
 * reproducing it. Throws on a dead reference: every detector returns 0 Hz on
 * silence, and a ratio against that is meaningless.
 */
export function cloneSpec(refF0, refBright, baseF0, baseBright) {
  if (!(refF0 > 0) || !(baseF0 > 0)) throw new Error('no voiced audio in the reference');
  const spec = neutralSpec();
  spec.pitch = clamp(12 * Math.log2(refF0 / baseF0), -12, 12);
  if (refBright > 0 && baseBright > 0) {
    const tilt = 20 * Math.log10(refBright / baseBright);
    spec.clarity = clamp(tilt, -6, 6);
    spec.air = clamp(tilt * 0.5, -4, 4);
  }
  return clampSpec(spec);
}

// ── rendering (browser only) ────────────────────────────────────────────
/**
 * Apply `spec` to an AudioBuffer and hand back a new one.
 *
 * Pitch and speed are entangled: resampling moves both, so the resample is
 * undone in time by exactly the amount it sped things up. Doing this in one
 * combined stretch rather than two keeps a single pass of WSOLA artefacts
 * instead of stacking them.
 */
export async function renderSpec(buffer, spec) {
  const s = clampSpec(spec);
  const sr = buffer.sampleRate;
  const ratio = Math.pow(2, s.pitch / 12);

  let mono = buffer.getChannelData(0);
  if (buffer.numberOfChannels > 1) {
    const other = buffer.getChannelData(1);
    const mixed = new Float32Array(mono.length);
    for (let i = 0; i < mono.length; i++) mixed[i] = (mono[i] + other[i]) / 2;
    mono = mixed;
  }

  // 1. resample for pitch (this also shortens the clip by `ratio`)
  if (Math.abs(s.pitch) >= 0.01) {
    const len = Math.max(1, Math.floor(mono.length / ratio));
    const ctx = new OfflineAudioContext(1, len, sr);
    const src = ctx.createBufferSource();
    const tmp = ctx.createBuffer(1, mono.length, sr);
    tmp.copyToChannel(mono, 0);
    src.buffer = tmp;
    src.playbackRate.value = ratio;
    src.connect(ctx.destination);
    src.start();
    mono = (await ctx.startRendering()).getChannelData(0).slice();
  }

  // 2. one stretch that both undoes the resample and applies the rate control
  const stretch = ratio / s.rate;
  if (Math.abs(stretch - 1) > 1e-4) mono = timeStretch(mono, sr, stretch);

  // 3. EQ
  const outLen = Math.max(1, mono.length);
  const ctx = new OfflineAudioContext(1, outLen, sr);
  const buf = ctx.createBuffer(1, outLen, sr);
  buf.copyToChannel(mono, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;

  const bands = [
    ['peaking', 250, 1.0, s.body],
    ['peaking', 700, 1.2, s.warmth],
    ['peaking', 3000, 1.2, s.clarity],
    ['highshelf', 9000, 0.7, s.air],
  ];
  let node = src;
  for (const [type, freq, q, gain] of bands) {
    if (Math.abs(gain) < 0.05) continue;
    const f = ctx.createBiquadFilter();
    f.type = type;
    f.frequency.value = freq;
    f.Q.value = q;
    f.gain.value = gain;
    node.connect(f);
    node = f;
  }
  node.connect(ctx.destination);
  src.start();
  return ctx.startRendering();
}

/** Minimal 16-bit WAV, so a designed take can be downloaded and reused. */
export function toWav(buffer) {
  const data = buffer.getChannelData(0);
  const out = new DataView(new ArrayBuffer(44 + data.length * 2));
  const str = (at, s) => { for (let i = 0; i < s.length; i++) out.setUint8(at + i, s.charCodeAt(i)); };
  str(0, 'RIFF');
  out.setUint32(4, 36 + data.length * 2, true);
  str(8, 'WAVEfmt ');
  out.setUint32(16, 16, true);
  out.setUint16(20, 1, true);
  out.setUint16(22, 1, true);
  out.setUint32(24, buffer.sampleRate, true);
  out.setUint32(28, buffer.sampleRate * 2, true);
  out.setUint16(32, 2, true);
  out.setUint16(34, 16, true);
  str(36, 'data');
  out.setUint32(40, data.length * 2, true);
  for (let i = 0; i < data.length; i++) {
    out.setInt16(44 + i * 2, clamp(data[i], -1, 1) * 32767, true);
  }
  return new Blob([out], { type: 'audio/wav' });
}
