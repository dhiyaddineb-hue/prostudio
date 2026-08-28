/**
 * Split a file into parts small enough for GitHub's browser upload form.
 *
 * The form refuses anything over 25 MB, and these pages are static so there is
 * no server that could take a big file instead. The previous answer was to
 * throw away the picture and upload only the soundtrack, which works but
 * returns a dub with no video — the wrong trade for anything worth watching.
 *
 * Splitting keeps everything. The file is cut on byte boundaries and the parts
 * are rejoined byte-for-byte on the other side, so nothing is re-encoded and
 * nothing is lost: not the picture, not the resolution, not a single sample of
 * audio. A rejoined file is bit-identical to the original, which is checked
 * rather than assumed — every part carries a SHA-256 and the join refuses on
 * any mismatch.
 *
 * The parts are not playable on their own. Only part one has the container
 * header; the rest are raw continuation bytes. That is fine because they are
 * never meant to be opened, only reassembled, but it does mean a lost part
 * cannot be worked around — hence the manifest, which names exactly what is
 * missing instead of failing obscurely.
 *
 * Memory stays bounded. Blob.slice() returns a lazy view rather than a copy,
 * so a 2 GB file never lands in RAM; only the part currently being hashed is
 * read, one at a time.
 */

// 25 MB is the hard limit; 20 leaves room for the multipart envelope and keeps
// a margin in case GitHub counts the encoded size rather than the raw bytes.
const PART_BYTES = 20 * 1024 * 1024;
const SAFE_SINGLE = 24 * 1024 * 1024;

export const LIMIT_MB = 25;
export const PART_MB = PART_BYTES / 1048576;

export function formatMB(bytes) {
  return (bytes / 1048576).toFixed(1);
}

/** Does this file need splitting at all? */
export function needsSplit(file) {
  return file.size > SAFE_SINGLE;
}

/** How many parts a file of this size becomes. */
export function partCount(size) {
  return Math.max(1, Math.ceil(size / PART_BYTES));
}

/**
 * Part filenames sort in order and state their own position.
 *
 * The index is zero-padded so a plain alphabetical sort — what a file listing
 * gives, and what the join script relies on — puts part 10 after part 9 rather
 * than after part 1.
 */
export function partName(originalName, index, total) {
  const width = String(total).length;
  const n = String(index).padStart(Math.max(width, 2), '0');
  const t = String(total).padStart(Math.max(width, 2), '0');
  return `${originalName}.part${n}of${t}`;
}

async function sha256(blob) {
  const digest = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer());
  return [...new Uint8Array(digest)]
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Cut `file` into uploadable parts plus a manifest describing them.
 *
 * `onProgress` is called with a human message as each part is hashed, because
 * hashing a multi-gigabyte file takes long enough that silence looks like a
 * hang.
 *
 * The manifest is a separate small JSON file. The join also works without it,
 * falling back to a size check, but with it the reassembly is verified against
 * the original bytes.
 */
export async function splitFile(file, onProgress) {
  const total = partCount(file.size);
  const parts = [];
  const hashes = [];

  for (let i = 0; i < total; i++) {
    const start = i * PART_BYTES;
    const blob = file.slice(start, Math.min(start + PART_BYTES, file.size));
    onProgress?.(`يجهّز الجزء ${i + 1} من ${total}…`);
    hashes.push(await sha256(blob));
    parts.push(new File([blob], partName(file.name, i + 1, total), {
      type: 'application/octet-stream',
    }));
    // Let the browser paint between parts; hashing is synchronous inside
    // subtle.digest and a large file would otherwise freeze the tab.
    await new Promise(r => setTimeout(r, 0));
  }

  const manifest = {
    original: file.name,
    size: file.size,
    parts: total,
    part_bytes: PART_BYTES,
    sha256: hashes,
    created: new Date().toISOString(),
  };

  const manifestFile = new File(
    [JSON.stringify(manifest, null, 2)],
    `${file.name}.parts.json`,
    { type: 'application/json' },
  );

  return { parts, manifest: manifestFile, total, size: file.size };
}
