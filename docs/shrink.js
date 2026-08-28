/**
 * Get a large clip under GitHub's 25 MB browser-upload cap.
 *
 * The web upload form refuses anything larger with "Yowza, that's a big file",
 * and that limit is not negotiable from our side. But dubbing only ever reads
 * the audio, so the picture can go.
 *
 * What that saves depends entirely on the source's video bitrate, so the page
 * quotes a cost rather than a ratio. Measured on the same 25 s clip re-encoded
 * at several qualities: an already-compressed Instagram file shrinks 7.9x, a
 * 720p encode 10x, 1080p 20x, and a 12 Mb/s master 47x. The audio side is
 * constant at 1.83 MB per minute, which is the number worth telling someone.
 *
 * Everything happens in the visitor's browser. There is no server to accept a
 * large upload, so the file must get smaller before it is ever sent.
 *
 * WAV is written rather than MP3 because a browser cannot encode MP3 without
 * shipping an encoder, and 16 kHz mono WAV is already small enough: one minute
 * costs about 1.9 MB, so roughly thirteen minutes of speech fits in the cap.
 * 16 kHz keeps every formant that matters for speech — the transcription and
 * pitch work here both downsample to 16 kHz anyway — so the extra bytes of a
 * higher rate would buy nothing.
 */

const CAP_BYTES = 25 * 1024 * 1024;
// Leave room for the multipart envelope and a filename; landing at exactly
// 25 MB would still be refused.
const SAFE_BYTES = 24 * 1024 * 1024;
const TARGET_RATE = 16000;

export const LIMIT_MB = 25;

/** Does this file need shrinking before GitHub will take it? */
export function needsShrink(file) {
  return file.size > SAFE_BYTES;
}

export function formatMB(bytes) {
  return (bytes / 1048576).toFixed(1);
}

/**
 * Decode any media file the browser understands into mono PCM.
 *
 * Video files decode too: the browser hands back the audio track and discards
 * the picture, which is exactly what is wanted.
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
 * Resample by linear interpolation.
 *
 * Cheap, and adequate going *down* in rate for speech. It does alias, so a
 * gentle average over the input span is taken per output sample rather than a
 * bare nearest-neighbour pick, which is what makes downsampled speech sound
 * gritty.
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

/** 16-bit PCM WAV. */
function toWav(samples, sampleRate) {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const text = (at, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(at + i, s.charCodeAt(i));
  };
  text(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, 'WAVEfmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, v * 32767, true);
  }
  return new Blob([view], { type: 'audio/wav' });
}

/** How many seconds of 16 kHz mono WAV fit under the cap. */
export function maxSeconds() {
  return Math.floor((SAFE_BYTES - 44) / (TARGET_RATE * 2));
}

/**
 * Shrink `file` to an audio-only WAV that GitHub will accept.
 *
 * Returns the new File plus what it cost, so the page can report the trade
 * rather than silently swapping the user's video for something else.
 *
 * Throws when the audio alone still will not fit. Truncating would be worse
 * than refusing: a dub of the first eleven minutes of a twenty minute talk,
 * delivered without comment, is a failure disguised as a success.
 */
export async function shrinkForUpload(file, onProgress) {
  const { samples, sampleRate } = await decodeAudio(file, onProgress);
  onProgress?.('يحوّل إلى صوت…');

  const pcm = resample(samples, sampleRate, TARGET_RATE);
  const seconds = pcm.length / TARGET_RATE;
  const blob = toWav(pcm, TARGET_RATE);

  if (blob.size > SAFE_BYTES) {
    const limit = maxSeconds();
    throw new Error(
      `الصوت وحده ${formatMB(blob.size)} م.ب — أطول من الحد. ` +
      `الحد الأقصى ${Math.floor(limit / 60)} دقيقة تقريباً، ومقطعك ` +
      `${Math.floor(seconds / 60)}:${String(Math.round(seconds % 60)).padStart(2, '0')}. ` +
      `اقتطع الجزء الذي تريد دبلجته وأرسله.`
    );
  }

  const name = file.name.replace(/\.[^.]+$/, '') + '.wav';
  return {
    file: new File([blob], name, { type: 'audio/wav' }),
    before: file.size,
    after: blob.size,
    seconds,
    ratio: file.size / blob.size,
  };
}
