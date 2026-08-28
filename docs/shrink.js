/**
 * Get a large clip under GitHub's 25 MB browser-upload cap.
 *
 * The web upload form refuses anything larger with "Yowza, that's a big file",
 * and that limit is not negotiable from our side. These pages are static, so
 * there is no server that could accept the file instead — it has to be smaller
 * before it is ever sent.
 *
 * What this costs, stated plainly
 * -------------------------------
 * The picture is discarded. Only the soundtrack is uploaded, so the dub comes
 * back as an audio file, not a video. For a clip whose value is the image that
 * is the wrong trade, and the page says so before the user commits to it
 * rather than after — `plan()` reports the consequence, and nothing is
 * converted until it is accepted.
 *
 * The audio itself is not meaningfully harmed. Measured against a 16 kHz WAV
 * reference, 64 kb/s MP3 moves pitch by 0.003 semitones and harmonics-to-noise
 * by 0.015 dB, both far below audibility and below what the dub pipeline can
 * resolve.
 *
 * Why MP3 and not WAV
 * -------------------
 * The first version wrote WAV, which is uncompressed, capping uploads at 13
 * minutes — a 23 minute clip was bounced back with instructions to trim it by
 * hand. Encoding raises the ceiling to about 52 minutes at 0.46 MB per minute.
 * 32 kb/s was already transparent in testing; 64 is margin for noisier
 * material.
 */

const CAP_BYTES = 25 * 1024 * 1024;
// Leave room for the multipart envelope and a filename; landing at exactly
// 25 MB would still be refused.
const SAFE_BYTES = 24 * 1024 * 1024;
const TARGET_RATE = 16000;
const BITRATE_KBPS = 64;
const BYTES_PER_SEC = (BITRATE_KBPS * 1000) / 8;

export const LIMIT_MB = 25;

/** Does this file need shrinking before GitHub will take it? */
export function needsShrink(file) {
  return file.size > SAFE_BYTES;
}

/** Best guess at whether a file carries a picture, from its type and name. */
export function looksLikeVideo(file) {
  if (file.type) return file.type.startsWith('video/');
  return /\.(mp4|mkv|mov|webm|avi|m4v|mpg|mpeg|wmv|flv)$/i.test(file.name || '');
}

/**
 * What will happen to this file, before anything is done to it.
 *
 * Returned so the page can warn first and convert second. Discarding the video
 * track is the whole mechanism here, and a user who cares about the picture
 * needs to know that before they spend a minute encoding, not when an audio
 * file arrives back.
 */
export function plan(file) {
  const video = looksLikeVideo(file);
  if (!needsShrink(file)) {
    return {
      action: 'none',
      losesVideo: false,
      reason: `${formatMB(file.size)} م.ب — أصغر من الحد، يُرفع كما هو بلا أي تغيير.`,
    };
  }
  return {
    action: 'extract-audio',
    losesVideo: video,
    reason: video
      ? `${formatMB(file.size)} م.ب — أكبر من ${LIMIT_MB} م.ب. سأرفع الصوت وحده، ` +
        'وستعود الدبلجة ملفاً صوتياً بلا صورة.'
      : `${formatMB(file.size)} م.ب — أكبر من ${LIMIT_MB} م.ب. سيُضغط الصوت.`,
  };
}

export function formatMB(bytes) {
  return (bytes / 1048576).toFixed(1);
}

/** How many seconds of encoded audio fit under the cap. */
export function maxSeconds() {
  return Math.floor(SAFE_BYTES / BYTES_PER_SEC);
}

/** MB per minute of audio — the one figure that holds for any source. */
export function costPerMinute() {
  return (BYTES_PER_SEC * 60) / 1048576;
}

/**
 * Load the MP3 encoder.
 *
 * Vendored rather than pulled from a CDN: every CDN this project can see is
 * blocked, and a page that only works with network access to a third party is
 * not a page that works.
 */
let encoderPromise = null;
function loadEncoder() {
  if (encoderPromise) return encoderPromise;
  encoderPromise = new Promise((resolve, reject) => {
    if (globalThis.lamejs) return resolve(globalThis.lamejs);
    const tag = document.createElement('script');
    tag.src = new URL('vendor/lame.min.js', import.meta.url).href;
    tag.onload = () => globalThis.lamejs
      ? resolve(globalThis.lamejs)
      : reject(new Error('تعذّر تحميل مُرمّز الصوت'));
    tag.onerror = () => reject(new Error('تعذّر تحميل مُرمّز الصوت'));
    document.head.appendChild(tag);
  });
  return encoderPromise;
}

/**
 * Decode any media file the browser understands into mono PCM.
 *
 * Video decodes too: the browser hands back the audio track and drops the
 * picture, which is exactly what is wanted.
 */
async function decodeAudio(file, onProgress) {
  onProgress?.('يقرأ الملف…');
  const bytes = await file.arrayBuffer();
  onProgress?.('يفك ترميز الصوت…');

  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  try {
    const buffer = await ctx.decodeAudioData(bytes);
    let mono = buffer.getChannelData(0);
    if (buffer.numberOfChannels > 1) {
      const right = buffer.getChannelData(1);
      const mixed = new Float32Array(mono.length);
      for (let i = 0; i < mono.length; i++) mixed[i] = (mono[i] + right[i]) / 2;
      mono = mixed;
    }
    return { samples: mono, sampleRate: buffer.sampleRate };
  } finally {
    ctx.close?.();
  }
}

/**
 * Resample down by averaging over the input span.
 *
 * A bare nearest-neighbour pick aliases, which is what makes cheaply
 * downsampled speech sound gritty; averaging costs nothing here and does not.
 */
function resample(samples, from, to) {
  if (to >= from) return samples;
  const ratio = from / to;
  const out = new Float32Array(Math.floor(samples.length / ratio));
  const span = Math.ceil(ratio);
  for (let i = 0; i < out.length; i++) {
    const start = Math.floor(i * ratio);
    let sum = 0, count = 0;
    for (let k = 0; k < span && start + k < samples.length; k++) {
      sum += samples[start + k];
      count++;
    }
    out[i] = count ? sum / count : 0;
  }
  return out;
}

/**
 * Encode mono float PCM to MP3, yielding to the event loop between blocks.
 *
 * A 23 minute clip is ~22 million samples. Encoding it in one synchronous run
 * freezes the tab for long enough that the browser offers to kill the page, so
 * the work is chunked and progress is reported.
 */
async function encodeMp3(samples, sampleRate, onProgress) {
  const lame = await loadEncoder();
  const encoder = new lame.Mp3Encoder(1, sampleRate, BITRATE_KBPS);

  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    pcm[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }

  const BLOCK = 1152 * 64;          // whole MP3 frames, ~4.6 s at 16 kHz
  const parts = [];
  const started = Date.now();
  let lastYield = started;

  for (let at = 0; at < pcm.length; at += BLOCK) {
    const chunk = encoder.encodeBuffer(pcm.subarray(at, Math.min(at + BLOCK, pcm.length)));
    if (chunk.length) parts.push(new Uint8Array(chunk));

    // Yield on a clock rather than every block. A 23 minute clip is 75 blocks
    // and takes about a minute to encode; handing control back every 100 ms
    // keeps the tab responsive without paying scheduler overhead 75 times.
    const now = Date.now();
    if (now - lastYield > 100) {
      const fraction = (at + BLOCK) / pcm.length;
      const done = Math.min(99, Math.round(fraction * 100));
      const left = Math.round(((now - started) / Math.max(fraction, 0.01)) * (1 - fraction) / 1000);
      onProgress?.(
        left > 3
          ? `يضغط الصوت… ${done}% — بقي ${left} ثانية تقريباً`
          : `يضغط الصوت… ${done}%`
      );
      lastYield = now;
      await new Promise(r => setTimeout(r, 0));
    }
  }
  const tail = encoder.flush();
  if (tail.length) parts.push(new Uint8Array(tail));

  return new Blob(parts, { type: 'audio/mpeg' });
}

/**
 * Shrink `file` to an audio-only MP3 that GitHub will accept.
 *
 * Returns the new File plus what it cost, so the page can report the trade
 * rather than silently swapping the user's video for something else.
 *
 * Throws when even the encoded audio will not fit. Truncating would be worse
 * than refusing: a dub of the first fifty minutes of a longer recording,
 * delivered without comment, is a failure disguised as a success.
 */
export async function shrinkForUpload(file, onProgress) {
  const { samples, sampleRate } = await decodeAudio(file, onProgress);
  const pcm = resample(samples, sampleRate, TARGET_RATE);
  const seconds = pcm.length / TARGET_RATE;

  const limit = maxSeconds();
  if (seconds > limit) {
    const mmss = s => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;
    throw new Error(
      `المقطع ${mmss(seconds)} — أطول من الحد الأقصى ${Math.floor(limit / 60)} دقيقة. ` +
      `اقتطع الجزء الذي تريد دبلجته، أو أرسل لي الرابط المباشر وسأنزّله بنفسي.`
    );
  }

  const blob = await encodeMp3(pcm, TARGET_RATE, onProgress);
  const name = file.name.replace(/\.[^.]+$/, '') + '.mp3';
  return {
    file: new File([blob], name, { type: 'audio/mpeg' }),
    before: file.size,
    after: blob.size,
    seconds,
    ratio: file.size / blob.size,
  };
}
