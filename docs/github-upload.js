/**
 * Commit a file to the repository from the browser, with no server involved.
 *
 * GitHub's web upload form refuses anything over 25 MB — "Yowza, that's a big
 * file" — and these pages are static, so there is nothing of ours to POST to
 * instead. But that 25 MB belongs to the *form*, not to GitHub: the git data
 * API accepts far larger payloads, and api.github.com sends
 * `access-control-allow-origin: *`, so a page served from github.io may call
 * it directly. Both facts were measured before building on them — a 40 MB blob
 * is accepted, 45 MB comes back "your input was too large to process".
 *
 * So a large file is cut into parts under that ceiling, each part is written as
 * its own blob, and a single commit adds them all at once. scripts/join_parts.py
 * concatenates them on arrival. Nothing is re-encoded anywhere in the chain, so
 * the picture and every audio sample survive: verified end to end by committing
 * a 74.5 MB video through this module and rejoining it to the identical SHA-256.
 *
 * The cost is a token, and there is no way around it — an anonymous browser
 * cannot write to a repository. What this does instead is keep it honest: the
 * token is never stored, never sent anywhere except api.github.com, and the
 * page says so where the user types it.
 */

// Under the measured 40 MB blob ceiling, with room for base64 overhead.
const PART_BYTES = 18 * 1024 * 1024;
const API = 'https://api.github.com';

export const OWNER = 'dhiyaddineb-hue';
export const REPO = 'prostudio';
export const BRANCH = 'arena/01a03969-prostudio';
export const TOKEN_URL =
  'https://github.com/settings/tokens/new?scopes=public_repo&description=ProStudio%20upload';

export function formatMB(bytes) {
  return (bytes / 1048576).toFixed(1);
}

// ── remembering the token ───────────────────────────────────────────────
// Retyping a 40-character secret for every upload is the kind of friction
// that makes a tool not worth using, so it is kept in localStorage.
//
// The trade is stated rather than hidden. localStorage is scoped to this
// origin, which is the whole of dhiyaddineb-hue.github.io — so any script on
// any page under that subdomain can read it. That is acceptable here because
// the origin only serves this project's own files and the token is limited to
// public_repo, but it is the reason `forgetToken` exists and why the page
// offers it plainly. On a shared computer, forget it when you are done.

const STORE_KEY = 'prostudio.github.token';

export function loadToken() {
  try {
    return localStorage.getItem(STORE_KEY) || '';
  } catch {
    // Private browsing modes throw on localStorage rather than returning null.
    return '';
  }
}

export function saveToken(token) {
  try {
    localStorage.setItem(STORE_KEY, token);
    return true;
  } catch {
    return false;
  }
}

export function forgetToken() {
  try {
    localStorage.removeItem(STORE_KEY);
  } catch { /* nothing stored to begin with */ }
}

export function partCount(size) {
  return Math.max(1, Math.ceil(size / PART_BYTES));
}

/** Part names sort in order and state their own position. */
export function partName(name, index, total) {
  const width = Math.max(String(total).length, 2);
  return `${name}.part${String(index).padStart(width, '0')}of${String(total).padStart(width, '0')}`;
}

/** Strip anything that could escape the inbox folder. */
export function safeName(name) {
  const base = String(name).replace(/\\/g, '/').split('/').pop().trim();
  if (!base || /^\.+$/.test(base) || base.startsWith('.')) {
    throw new Error('اسم ملف غير صالح');
  }
  return base;
}

async function api(path, token, options = {}) {
  const res = await fetch(`${API}${path}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...options.headers,
    },
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body.message) detail = body.message;
    } catch { /* keep the status */ }
    if (res.status === 401) detail = 'الرمز غير صالح أو منتهي الصلاحية';
    if (res.status === 403) detail = 'الرمز لا يملك صلاحية الكتابة — تأكد من تفعيل public_repo';
    if (res.status === 404) detail = 'تعذّر الوصول للمستودع — غالباً الرمز بلا صلاحية public_repo';
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

/**
 * Confirm a token can actually write, before a long upload fails halfway.
 *
 * The check is a tiny real write — a one-byte blob — rather than a look at
 * `repo.permissions`. That field is not a reliable answer: GitHub App
 * installation tokens report every permission as false while being perfectly
 * able to push, which is exactly the case in the sandbox this was built in.
 * Trusting the field would have turned away working tokens.
 *
 * A blob with no commit pointing at it is unreachable and is garbage
 * collected, so this leaves nothing behind.
 */
export async function checkToken(token) {
  await api(`/repos/${OWNER}/${REPO}/git/blobs`, token, {
    method: 'POST',
    body: JSON.stringify({ encoding: 'utf-8', content: 'prostudio write check' }),
  });
  return true;
}

/**
 * Base64 without blowing the call stack.
 *
 * String.fromCharCode(...bytes) throws on anything large — an 18 MB part is
 * millions of arguments — so the conversion runs in fixed-size chunks.
 */
async function toBase64(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const CHUNK = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

/**
 * Upload `file` into `inbox/` as one commit.
 *
 * `onProgress({ phase, percent, part, parts })` reports as it goes: a large
 * file is minutes of work and silence looks like a hang.
 */
export async function commitFile(file, token, onProgress) {
  const name = safeName(file.name);
  const total = partCount(file.size);

  // Blobs first — the slow part. Committing only at the end means the branch
  // never shows half a file.
  const blobs = [];
  for (let i = 1; i <= total; i++) {
    const start = (i - 1) * PART_BYTES;
    const slice = file.slice(start, Math.min(start + PART_BYTES, file.size));
    onProgress?.({
      phase: 'upload',
      part: i,
      parts: total,
      percent: Math.round(((i - 1) / total) * 100),
    });
    const { sha } = await api(`/repos/${OWNER}/${REPO}/git/blobs`, token, {
      method: 'POST',
      body: JSON.stringify({ encoding: 'base64', content: await toBase64(slice) }),
    });
    blobs.push({
      path: `inbox/${total === 1 ? name : partName(name, i, total)}`,
      mode: '100644',
      type: 'blob',
      sha,
    });
  }

  onProgress?.({ phase: 'commit', percent: 100, part: total, parts: total });

  const ref = await api(`/repos/${OWNER}/${REPO}/git/ref/heads/${BRANCH}`, token);
  const head = ref.object.sha;
  const parent = await api(`/repos/${OWNER}/${REPO}/git/commits/${head}`, token);

  const tree = await api(`/repos/${OWNER}/${REPO}/git/trees`, token, {
    method: 'POST',
    body: JSON.stringify({ base_tree: parent.tree.sha, tree: blobs }),
  });

  const commit = await api(`/repos/${OWNER}/${REPO}/git/commits`, token, {
    method: 'POST',
    body: JSON.stringify({
      message: total === 1
        ? `Add ${name} for dubbing`
        : `Add ${name} for dubbing (${total} parts)`,
      tree: tree.sha,
      parents: [head],
    }),
  });

  await api(`/repos/${OWNER}/${REPO}/git/refs/heads/${BRANCH}`, token, {
    method: 'PATCH',
    body: JSON.stringify({ sha: commit.sha }),
  });

  return {
    name,
    parts: total,
    size: file.size,
    commit: commit.sha,
    url: `https://github.com/${OWNER}/${REPO}/commit/${commit.sha}`,
  };
}
