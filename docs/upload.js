/**
 * Send a file straight to the studio, splitting it on the way.
 *
 * The earlier flow cut a large file into parts, handed each one back as a
 * download, and asked the user to save them all and drag them into GitHub's
 * upload form. That is work a machine should do. When the studio is running,
 * the browser can post the parts to it directly: no saving, no re-uploading,
 * and no 25 MB cap, because that cap belongs to GitHub's web form rather than
 * to HTTP.
 *
 * Parts are still used even though the studio would accept the whole file.
 * They give progress that means something on a slow connection, and a dropped
 * request costs one 8 MB part instead of a two gigabyte upload. The server
 * assembles them on arrival.
 *
 * Nothing is re-encoded at any stage. Blob.slice() hands back a lazy byte view,
 * so a large file never lands in memory, and the reassembled file is identical
 * to the original — each part carries a SHA-256 that the server checks before
 * writing it.
 */

// Smaller than the split-for-GitHub size: these are network requests, not
// files a human handles, so smaller parts mean smoother progress and a cheaper
// retry.
const PART_BYTES = 8 * 1024 * 1024;
const RETRIES = 3;

export function formatMB(bytes) {
  return (bytes / 1048576).toFixed(1);
}

/** Is a studio reachable from this page? */
export async function studioReachable(base = '') {
  try {
    const res = await fetch(`${base}/api/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(3000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

async function sha256(blob) {
  const digest = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer());
  return [...new Uint8Array(digest)]
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

async function postPart(base, body, attempt = 1) {
  try {
    const res = await fetch(`${base}/api/upload/part`, { method: 'POST', body });
    if (res.ok) return res.json();
    // A rejected part is worth retrying; a rejected filename is not.
    if (res.status === 400) throw new Error((await res.json()).detail || 'اسم ملف غير صالح');
    throw new Error(`الخادم ردّ ${res.status}`);
  } catch (err) {
    if (attempt >= RETRIES) throw err;
    await new Promise(r => setTimeout(r, 400 * attempt));
    return postPart(base, body, attempt + 1);
  }
}

/**
 * Upload `file` to the studio, part by part.
 *
 * `onProgress({ sent, total, percent, part, parts })` fires after each part.
 * Parts already on the server from an interrupted run are skipped.
 */
export async function uploadToStudio(file, onProgress, base = '') {
  const total = Math.max(1, Math.ceil(file.size / PART_BYTES));

  let have = new Set();
  try {
    const res = await fetch(`${base}/api/upload/status/${encodeURIComponent(file.name)}`);
    if (res.ok) {
      const status = await res.json();
      if (status.complete) {
        onProgress?.({ sent: file.size, total: file.size, percent: 100, part: total, parts: total });
        return { path: `inbox/${file.name}`, size: file.size, resumed: true };
      }
      have = new Set(status.have || []);
    }
  } catch {
    // No status endpoint is not fatal; just upload everything.
  }

  let sent = 0;
  let last = null;
  for (let i = 1; i <= total; i++) {
    const start = (i - 1) * PART_BYTES;
    const blob = file.slice(start, Math.min(start + PART_BYTES, file.size));

    if (have.has(i)) {
      sent += blob.size;
      onProgress?.({ sent, total: file.size, percent: Math.round((sent / file.size) * 100), part: i, parts: total, skipped: true });
      continue;
    }

    const body = new FormData();
    body.append('chunk', blob, `${i}`);
    body.append('name', file.name);
    body.append('index', String(i));
    body.append('total', String(total));
    body.append('sha256', await sha256(blob));

    last = await postPart(base, body);
    sent += blob.size;
    onProgress?.({
      sent,
      total: file.size,
      percent: Math.round((sent / file.size) * 100),
      part: i,
      parts: total,
    });
  }

  if (!last?.complete) {
    throw new Error('اكتمل الإرسال لكن الخادم لم يؤكد تجميع الملف');
  }
  return { path: last.path, size: last.size, resumed: false };
}
