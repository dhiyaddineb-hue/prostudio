/**
 * ProStudio — the complete studio, served from GitHub Pages, backed by GitHub.
 *
 * There is no server of ours. Everything here talks to api.github.com from the
 * browser with the user's own token:
 *   - the library and the dubs are read from the branch tree (one call, ETag
 *     cached) and rendered as cards; new uploads or new dubs appear on the next
 *     poll without any manual step;
 *   - uploads are committed as blobs + one commit into library/<slug>/ (large
 *     files are split into <=18 MB parts that the workflow rejoins);
 *   - dubbing dispatches the "Run YouTube Auto Dub" workflow; runs are followed
 *     live (steps from the jobs API, per-chunk progress from the checkpoint
 *     manifest in the draft release);
 *   - deletions are single commits that remove a whole folder, and they only
 *     happen after the user types the folder name in a confirmation box.
 */
import {
  OWNER, REPO, BRANCH, TOKEN_URL, loadToken, saveToken, forgetToken, checkToken, formatMB,
} from './github-upload.js';

const API = 'https://api.github.com';
const REPO_URL = `https://github.com/${OWNER}/${REPO}`;
const RAW = `https://raw.githubusercontent.com/${OWNER}/${REPO}/${encodeURIComponent(BRANCH)}/`;
const WORKFLOW = 'dub.yml';
const PART_BYTES = 18 * 1024 * 1024;           // measured blob ceiling minus base64 overhead
const VIDEO_RE = /\.(mp4|mkv|webm|mov)$/i;
const AUDIO_RE = /\.(wav|mp3|m4a|flac|ogg|aac)$/i;
const SPEAKER_RE = /^[A-Za-z0-9_.-]+$/;
const DEFAULTS_KEY = 'prostudio.studio.defaults';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const rawUrl = (path) => RAW + path.split('/').map(encodeURIComponent).join('/');
const fmtDate = (iso) => (iso ? new Date(iso).toLocaleString('ar-EG', { hour12: false }) : '—');
const fmtDur = (s) => (s == null ? '—' : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`);
const setStatus = (el, msg, kind = '') => { el.textContent = msg; el.className = `status ${kind}`; };

const state = {
  token: '',
  tree: [], treeEtag: null, treeSha: null,
  blobCache: new Map(),         // blob sha -> parsed json
  library: [], dubs: [], runs: [], voiceBank: [],
  releases: [], releasesAt: 0,
  manifests: new Map(),         // run id -> {counts,total,state,at}
  filterLibrary: 'all', searchLibrary: '', searchDubs: '',
  polling: null, busy: false, loaded: false,
};

// ───────────────────────────────────────────── API
async function api(path, opts = {}) {
  const headers = {
    Accept: 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28', ...(opts.headers || {}),
  };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (opts.body && !headers['Content-Type']) headers['Content-Type'] = 'application/json';
  const res = await fetch(`${API}${path}`, { ...opts, headers });
  const remaining = res.headers.get('x-ratelimit-remaining');
  if (remaining != null) {
    const chip = $('rateChip');
    chip.textContent = `API: ${remaining}`;
    chip.className = `chip ${Number(remaining) < 100 ? 'warn' : 'muted'}`;
  }
  if (res.status === 304) return { notModified: true, res };
  if (!res.ok) {
    let detail = `GitHub API ${res.status}`;
    try { const body = await res.json(); if (body.message) detail = `${detail}: ${body.message}`; } catch { /* keep */ }
    if (res.status === 401) detail = 'الرمز غير صالح أو منتهي';
    if (res.status === 403 && !state.token) detail = 'حدّ الطلبات بدون رمز انتهى — أدخل رمز GitHub في الإعدادات';
    const err = new Error(detail); err.status = res.status; throw err;
  }
  if (res.status === 204) return null;
  if (opts.raw) return res;
  return res.json();
}
const needToken = () => { if (!state.token) throw new Error('هذه العملية تحتاج رمز GitHub (الإعدادات)'); };

async function blobJson(sha) {
  if (state.blobCache.has(sha)) return state.blobCache.get(sha);
  let parsed = null;
  try {
    const res = await api(`/repos/${OWNER}/${REPO}/git/blobs/${sha}`, { headers: { Accept: 'application/vnd.github.raw+json' }, raw: true });
    parsed = await res.json();
  } catch { parsed = null; }
  state.blobCache.set(sha, parsed);
  return parsed;
}

// ───────────────────────────────────────────── git plumbing (uploads + deletions)
async function toBase64(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const CHUNK = 0x8000; let binary = '';
  for (let i = 0; i < bytes.length; i += CHUNK) binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  return btoa(binary);
}
async function sha256Hex(blob) {
  const digest = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer());
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}
async function createBlob(content, encoding) {
  const { sha } = await api(`/repos/${OWNER}/${REPO}/git/blobs`, { method: 'POST', body: JSON.stringify({ content, encoding }) });
  return sha;
}
/** One commit: `additions` = [{path, sha}] blobs already created, `deletions` = [path]. */
async function commitChanges({ message, additions = [], deletions = [] }) {
  needToken();
  const ref = await api(`/repos/${OWNER}/${REPO}/git/ref/heads/${BRANCH}`);
  const head = ref.object.sha;
  const parent = await api(`/repos/${OWNER}/${REPO}/git/commits/${head}`);
  const tree = [
    ...additions.map((a) => ({ path: a.path, mode: '100644', type: 'blob', sha: a.sha })),
    ...deletions.map((p) => ({ path: p, mode: '100644', type: 'blob', sha: null })),
  ];
  const newTree = await api(`/repos/${OWNER}/${REPO}/git/trees`, { method: 'POST', body: JSON.stringify({ base_tree: parent.tree.sha, tree }) });
  const commit = await api(`/repos/${OWNER}/${REPO}/git/commits`, { method: 'POST', body: JSON.stringify({ message, tree: newTree.sha, parents: [head] }) });
  await api(`/repos/${OWNER}/${REPO}/git/refs/heads/${BRANCH}`, { method: 'PATCH', body: JSON.stringify({ sha: commit.sha }) });
  return commit.sha;
}

// ───────────────────────────────────────────── tree -> library / dubs
async function refreshTree(force = false) {
  const headers = {};
  if (state.treeEtag && !force) headers['If-None-Match'] = state.treeEtag;
  const res = await fetch(`${API}/repos/${OWNER}/${REPO}/git/trees/${encodeURIComponent(BRANCH)}?recursive=1`, {
    headers: { Accept: 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28', ...(state.token ? { Authorization: `Bearer ${state.token}` } : {}), ...headers },
  });
  if (res.status === 304) return false;
  if (!res.ok) throw new Error(`تعذر قراءة شجرة المستودع (${res.status})`);
  state.treeEtag = res.headers.get('etag');
  const data = await res.json();
  state.tree = (data.tree || []).filter((x) => x.type === 'blob');
  state.treeSha = data.sha;
  await buildCollections();
  return true;
}

function groupFolders(prefix) {
  const groups = new Map();
  for (const entry of state.tree) {
    if (!entry.path.startsWith(prefix)) continue;
    const rest = entry.path.slice(prefix.length);
    const slash = rest.indexOf('/');
    if (slash < 0) continue;                       // files directly under the prefix (e.g. .gitkeep)
    const slug = rest.slice(0, slash);
    if (!groups.has(slug)) groups.set(slug, []);
    groups.get(slug).push({ ...entry, name: rest.slice(slash + 1) });
  }
  return groups;
}

async function buildCollections() {
  const lib = [];
  for (const [slug, files] of groupFolders('library/')) {
    const metaFile = files.find((f) => f.name === 'meta.json');
    const meta = metaFile ? (await blobJson(metaFile.sha)) || {} : {};
    const whole = files.find((f) => VIDEO_RE.test(f.name) && !/\.part\d+of\d+$/i.test(f.name));
    const parts = files.filter((f) => /\.part\d+of\d+$/i.test(f.name)).sort((a, b) => a.name.localeCompare(b.name));
    const sourcePath = whole ? whole.path : parts.length ? parts[0].path.replace(/\.part\d+of\d+$/i, '') : null;
    lib.push({
      slug, meta, files, whole, parts, sourcePath,
      size: whole ? whole.size : parts.reduce((a, p) => a + p.size, 0),
      title: meta.title || slug,
      voicesJson: files.find((f) => f.name === 'voices.json') || null,
      voiceSamples: files.filter((f) => f.name.startsWith('voices/') && AUDIO_RE.test(f.name)),
    });
  }
  lib.sort((a, b) => String(b.meta.uploaded_at || '').localeCompare(String(a.meta.uploaded_at || '')) || a.slug.localeCompare(b.slug));
  state.library = lib;

  const dubs = [];
  for (const [slug, files] of groupFolders('dubs/')) {
    const metaFile = files.find((f) => f.name === 'meta.json');
    const meta = metaFile ? (await blobJson(metaFile.sha)) || {} : {};
    const videos = files.filter((f) => VIDEO_RE.test(f.name)).sort((a, b) => b.name.localeCompare(a.name));
    const versions = (meta.versions || []).filter((v) => videos.some((f) => f.name === v.file));
    for (const f of videos) if (!versions.some((v) => v.file === f.name)) versions.push({ file: f.name, run_id: (f.name.match(/(\d+)\.mp4$/) || [])[1], size: f.size });
    dubs.push({ slug, meta, files, videos, versions, latest: versions[0] || null });
  }
  dubs.sort((a, b) => String(b.meta.updated_at || '').localeCompare(String(a.meta.updated_at || '')) || a.slug.localeCompare(b.slug));
  state.dubs = dubs;
  state.voiceBank = state.tree.filter((f) => /^voices\/[^/]+$/.test(f.path) && AUDIO_RE.test(f.path)).map((f) => ({ ...f, name: f.path.slice(7) }));
}

// ───────────────────────────────────────────── runs (live)
function runSource(run) {
  // run-name: "<task> · <source_path> · run <n>"
  const parts = String(run.display_title || run.name || '').split(' · ');
  return parts.length >= 2 ? parts[1].trim() : '';
}
function runSlug(run) {
  const src = runSource(run);
  const m = src.match(/^library\/([^/]+)\//);
  if (m) return m[1];
  return src.split('/').pop()?.replace(/\.(mp4|mkv|webm|mov|part0)$/i, '') || '';
}
const isActive = (run) => ['queued', 'in_progress', 'waiting', 'pending', 'requested'].includes(run.status);

async function refreshRuns() {
  const data = await api(`/repos/${OWNER}/${REPO}/actions/runs?branch=${encodeURIComponent(BRANCH)}&per_page=25`);
  state.runs = (data.workflow_runs || []).filter((r) => (r.path || '').endsWith(`/${WORKFLOW}`));
  await Promise.all(state.runs.filter(isActive).map(enrichActiveRun));
}

async function enrichActiveRun(run) {
  try {
    const jobs = await api(`/repos/${OWNER}/${REPO}/actions/runs/${run.id}/jobs`);
    const job = (jobs.jobs || []).find((j) => j.status !== 'completed' || j.conclusion !== 'skipped') || jobs.jobs?.[0];
    const step = job?.steps?.find((s) => s.status === 'in_progress') || job?.steps?.filter((s) => s.status === 'completed').pop();
    run._step = step ? step.name : (job ? job.status : '—');
    run._jobStarted = job?.started_at;
  } catch { run._step = '—'; }
  if (!state.token) return;
  try {
    const slug = runSlug(run);
    if (!slug) return;
    if (Date.now() - state.releasesAt > 20000) {
      state.releases = (await api(`/repos/${OWNER}/${REPO}/releases?per_page=100`)).filter((r) => r.draft && r.tag_name.startsWith('checkpoint-'));
      state.releasesAt = Date.now();
    }
    const candidates = state.releases.filter((r) => r.tag_name.startsWith(`checkpoint-${slug}-`)).sort((a, b) => b.updated_at.localeCompare(a.updated_at));
    const release = candidates[0];
    const asset = release?.assets.find((a) => a.name === 'checkpoint-manifest.json');
    if (!asset) return;
    const cached = state.manifests.get(run.id);
    if (cached && cached.assetUpdated === asset.updated_at) { run._progress = cached; return; }
    const res = await api(`/repos/${OWNER}/${REPO}/releases/assets/${asset.id}`, { headers: { Accept: 'application/octet-stream' }, raw: true });
    const manifest = await res.json();
    const counts = {};
    for (const c of manifest.chunks || []) counts[c.status || 'pending'] = (counts[c.status || 'pending'] || 0) + 1;
    const progress = { counts, total: (manifest.chunks || []).length, state: manifest.state, assetUpdated: asset.updated_at, tag: release.tag_name,
      chunks: (manifest.chunks || []).map((c) => ({ index: c.index, status: c.status || 'pending' })) };
    state.manifests.set(run.id, progress);
    run._progress = progress;
  } catch { /* progress is a bonus; the step name is still shown */ }
}

// ───────────────────────────────────────────── reports, subtitles, details
const srtTime = (s) => { const ms = Math.round(s * 1000); const h = Math.floor(ms / 3600000), m = Math.floor((ms % 3600000) / 60000), sec = Math.floor((ms % 60000) / 1000), r = ms % 1000; return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')},${String(r).padStart(3, '0')}`; };
function segmentsToSrt(segments, field) {
  return segments.filter((s) => String(s[field] || '').trim()).map((s, i) => `${i + 1}\n${srtTime(+s.start)} --> ${srtTime(+s.end)}\n${String(s[field]).trim()}\n`).join('\n');
}
function downloadText(name, text) {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' }); const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 2000);
}
async function openDubDetails(dub, version) {
  const base = version.file.replace(/\.mp4$/i, '');
  const find = (suffix) => dub.files.find((f) => f.name === `${base}.${suffix}.json`);
  openModal(`${dub.slug} · ${version.file}`, '<p class="hint">جارٍ قراءة التقارير…</p>');
  const [quality, segments, language] = await Promise.all(['quality', 'segments', 'language'].map((s) => { const f = find(s); return f ? blobJson(f.sha) : Promise.resolve(null); }));
  const lib = state.library.find((it) => it.slug === dub.slug);
  const segs = segments?.segments || [];
  const checks = quality?.checks ? Object.entries(quality.checks).map(([k, v]) => `<span class="chip ${v ? 'ok' : 'err'}">${esc(k)}</span>`).join('') : '<span class="chip muted">لا تقرير</span>';
  const players = `<div class="compare"><div><h4>الأصل${lib?.whole ? '' : ' (غير متاح للمعاينة)'}</h4>${lib?.whole ? `<video controls preload="metadata" src="${rawUrl(lib.whole.path)}"></video>` : ''}</div><div><h4>المدبلج</h4><video controls preload="metadata" src="${rawUrl(`dubs/${dub.slug}/${version.file}`)}"></video></div></div>`;
  const rows = segs.slice(0, 400).map((s) => `<tr><td class="mono">${fmtDur(+s.start)}</td><td dir="auto">${esc(s.source_text || '')}</td><td dir="auto">${esc(s.translated_text || '')}</td></tr>`).join('');
  const lang = language ? `<span class="chip ${language.valid ? 'ok' : 'err'}">اللغة ${esc(language.detected || language.language || '')} ${language.confidence ? Math.round(language.confidence * 100) + '%' : ''}</span>` : '';
  const cov = segments?.asr_timeline?.uncovered_speech;
  const covChip = cov?.measured ? `<span class="chip ${cov.longest_after_seconds > 1 ? 'warn' : 'ok'}">كلام غير مفرَّغ: ${cov.after_seconds}s</span>` : '';
  $('modalBody').innerHTML = `${players}
    <div class="chips" style="margin:10px 0">${checks}${lang}${covChip}${segments?.translation?.engine ? `<span class="chip muted">${esc(segments.translation.engine)}</span>` : ''}</div>
    <div class="kv"><b>التشغيل</b><a href="${REPO_URL}/actions/runs/${esc(version.run_id || '')}" target="_blank" rel="noreferrer">#${esc(version.run_id || '')}</a><b>المدة</b><span>${fmtDur(version.duration)}</span><b>الحجم</b><span>${version.size ? formatMB(version.size) + ' MB' : '—'}</span><b>الإعدادات</b><span class="mono">${esc(Object.entries(version.settings || {}).map(([k, v]) => `${k}=${v}`).join(' '))}</span></div>
    <div class="row" style="margin:10px 0"><button class="btn small" id="srtTarget" ${segs.length ? '' : 'disabled'}>تحميل ترجمة SRT (الهدف)</button><button class="btn small" id="srtSource" ${segs.length ? '' : 'disabled'}>تحميل نص الأصل SRT</button><a class="btn small" href="${rawUrl(`dubs/${dub.slug}/${version.file}`)}" download>تحميل الفيديو</a></div>
    ${segs.length ? `<div class="tablewrap"><table class="segs"><thead><tr><th>الوقت</th><th>الأصل</th><th>الترجمة</th></tr></thead><tbody>${rows}</tbody></table></div>` : '<p class="hint">لا يوجد تقرير مقاطع لهذه النسخة.</p>'}`;
  $('srtTarget')?.addEventListener('click', () => downloadText(`${dub.slug}-${version.run_id || 'dub'}.${segments?.target_language || 'target'}.srt`, segmentsToSrt(segs, 'translated_text')));
  $('srtSource')?.addEventListener('click', () => downloadText(`${dub.slug}-${version.run_id || 'dub'}.${segments?.source_language || 'source'}.srt`, segmentsToSrt(segs, 'source_text')));
}
async function dispatchYoutube(url) {
  needToken();
  if (!/^https?:\/\/(www\.|m\.)?(youtube\.com|youtu\.be)\//i.test(url)) throw new Error('أدخل رابط يوتيوب صالحاً');
  const d = defaults();
  const inputs = {
    task: 'dub', source_path: '', youtube_url: url, source_lang: $('ytLang').value.trim() || 'ar',
    voice: d.voice, tts_engine: d.tts_engine, target_lang: d.target_lang, mode: 'both', gender: d.gender, model: d.model,
    bg_music: String(!!d.bg_music), diarize: String(!!d.diarize), separate_sources: String(!!d.separate_sources), no_vad: 'false',
    seed_vc: String(!!d.seed_vc), lip_sync: 'false', lip_sync_backend: 'wav2lip', profile: d.profile, quality: d.quality,
    chunk_seconds: String(d.chunk_seconds), speaker_voices_path: '', validate_content: String(!!d.validate_content),
  };
  await api(`/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW}/dispatches`, { method: 'POST', body: JSON.stringify({ ref: BRANCH, inputs }) });
}
function chunkGrid(progress) {
  if (!progress?.chunks?.length) return '';
  return `<div class="chunkgrid" title="حالة كل مقطع">${progress.chunks.map((c) => `<i class="ck ${esc(c.status || 'pending')}" title="#${c.index} ${esc(c.status || 'pending')}"></i>`).join('')}</div>`;
}

// ───────────────────────────────────────────── rendering
function libraryStatusFor(item) {
  const active = state.runs.find((r) => isActive(r) && runSlug(r) === item.slug);
  if (active) return { key: 'running', label: `قيد الدبلجة · ${active._step || active.status}`, run: active };
  const dub = state.dubs.find((d) => d.slug === item.slug);
  if (dub && dub.versions.length) return { key: 'dubbed', label: `مدبلج · ${dub.versions.length} نسخة`, dub };
  return { key: 'undubbed', label: 'غير مدبلج' };
}

function renderLibrary() {
  const grid = $('libraryGrid');
  const q = state.searchLibrary.trim().toLowerCase();
  const items = state.library.filter((it) => {
    const st = libraryStatusFor(it);
    if (state.filterLibrary !== 'all' && st.key !== state.filterLibrary) return false;
    return !q || it.slug.toLowerCase().includes(q) || String(it.title).toLowerCase().includes(q);
  });
  $('countLibrary').textContent = state.library.length;
  const banner = state.token ? '' : '<div class="empty warn">الصفحة في وضع القراءة فقط. أدخل رمز GitHub في «الإعدادات» لتفعيل الرفع والدبلجة والحذف والمتابعة التفصيلية.</div>';
  if (!items.length) { grid.innerHTML = `${banner}<div class="empty">${state.library.length ? 'لا نتائج مطابقة.' : (state.loaded ? 'المكتبة فارغة — ارفع أول فيديو من الأعلى أو أدخل رابط يوتيوب.' : 'جارٍ تحميل المكتبة…')}</div>`; return; }
  grid.innerHTML = banner;
  for (const it of items) {
    const st = libraryStatusFor(it);
    const card = document.createElement('article'); card.className = 'card';
    const player = it.whole ? `<video preload="metadata" src="${rawUrl(it.whole.path)}#t=0.5" muted playsinline></video>`
      : `<div class="noplay">مقسّم إلى ${it.parts.length} أجزاء (${formatMB(it.size)} MB) — يُجمَع تلقائياً عند الدبلجة</div>`;
    const chipClass = st.key === 'running' ? 'info' : st.key === 'dubbed' ? 'ok' : 'warn';
    const prog = st.run?._progress;
    card.innerHTML = `${player}<div class="body">
      <h3>${esc(it.title)}</h3>
      <div class="meta"><span class="mono">library/${esc(it.slug)}/</span><span>${formatMB(it.size)} MB</span><span>${fmtDur(it.meta.duration)}</span><span>${fmtDate(it.meta.uploaded_at)}</span></div>
      <div class="chips"><span class="chip ${chipClass}">${esc(st.label)}</span>${it.meta.source_lang ? `<span class="chip muted">${esc(it.meta.source_lang)}</span>` : ''}</div>
      ${prog ? `<div class="progress" title="${prog.counts.completed || 0}/${prog.total} مقطع"><span style="width:${prog.total ? Math.round(((prog.counts.completed || 0) * 100) / prog.total) : 0}%"></span></div>` : ''}
      <div class="buttons">
        ${it.whole ? `<button class="btn small act-preview">معاينة</button>` : ''}
        <button class="btn small primary act-dub" ${st.key === 'running' ? 'disabled' : ''}>${st.key === 'dubbed' ? 'دبلجة نسخة جديدة' : 'دبلجة'}</button>
        ${st.key === 'dubbed' ? `<button class="btn small act-godubs">النسخ المدبلجة</button>` : ''}
        ${st.key === 'running' ? `<button class="btn small act-goruns">متابعة</button>` : ''}
        <button class="btn small act-voices">الأصوات${it.voicesJson ? ' ✓' : ''}</button>
        <button class="btn small act-checkpoints">نقاط الاستئناف</button>
        <a class="btn small" href="${REPO_URL}/tree/${encodeURIComponent(BRANCH)}/library/${encodeURIComponent(it.slug)}" target="_blank" rel="noreferrer">GitHub</a>
        <button class="btn small danger act-delete">حذف</button>
      </div></div>`;
    card.querySelector('.act-preview')?.addEventListener('click', () => openPlayer(it.title, rawUrl(it.whole.path), it));
    card.querySelector('.act-dub').addEventListener('click', () => openDubDialog(it));
    card.querySelector('.act-godubs')?.addEventListener('click', () => { $('searchDubs').value = it.slug; state.searchDubs = it.slug; showTab('dubs'); renderDubs(); });
    card.querySelector('.act-goruns')?.addEventListener('click', () => showTab('runs'));
    card.querySelector('.act-voices').addEventListener('click', () => openVoicesDialog(it));
    card.querySelector('.act-checkpoints').addEventListener('click', () => openCheckpointDialog(it));
    card.querySelector('.act-delete').addEventListener('click', () => confirmDeleteFolder(`library/${it.slug}/`, it.slug, 'الفيديو الأصلي وكل ملفاته'));
    grid.appendChild(card);
  }
}

function renderDubs() {
  const grid = $('dubsGrid');
  const q = state.searchDubs.trim().toLowerCase();
  const items = state.dubs.filter((d) => !q || d.slug.toLowerCase().includes(q));
  $('countDubs').textContent = state.dubs.length;
  if (!items.length) { grid.innerHTML = `<div class="empty">${state.dubs.length ? 'لا نتائج مطابقة.' : (state.loaded ? 'لا توجد دبلجة منشورة بعد. عند اكتمال أي تشغيل تظهر نسخته هنا تلقائياً.' : 'جارٍ التحميل…')}</div>`; return; }
  grid.innerHTML = '';
  for (const d of items) {
    const latest = d.latest;
    const latestPath = latest ? `dubs/${d.slug}/${latest.file}` : null;
    const card = document.createElement('article'); card.className = 'card';
    card.innerHTML = `${latestPath ? `<video preload="metadata" src="${rawUrl(latestPath)}#t=0.5" muted playsinline></video>` : '<div class="noplay">لا ملف</div>'}<div class="body">
      <h3>${esc(d.slug)}</h3>
      <div class="meta"><span class="mono">dubs/${esc(d.slug)}/</span><span>${d.versions.length} نسخة</span>${latest?.duration ? `<span>${fmtDur(latest.duration)}</span>` : ''}<span>${fmtDate(d.meta.updated_at || latest?.published_at)}</span></div>
      <div class="chips">${latest?.quality_ok === true ? '<span class="chip ok">اجتاز بوابة الجودة</span>' : latest?.quality_ok === false ? '<span class="chip err">فشل بوابة الجودة</span>' : ''}${latest?.language?.valid ? `<span class="chip ok">اللغة ${esc(latest.language.detected || '')}</span>` : ''}${latest?.translation_engine ? `<span class="chip muted">${esc(latest.translation_engine)}</span>` : ''}</div>
      <ul class="versions">${d.versions.map((v) => `<li><span class="mono">${esc(v.file)}</span><span>${v.size ? formatMB(v.size) + ' MB' : ''}</span><span class="muted">${fmtDate(v.published_at)}</span>
        <button class="btn small act-play" data-file="${esc(v.file)}">معاينة</button>
        <button class="btn small act-details" data-file="${esc(v.file)}">التفاصيل والترجمة</button>
        <a class="btn small" href="${rawUrl(`dubs/${d.slug}/${v.file}`)}" download>تحميل</a>
        ${v.run_id ? `<a class="btn small" href="${REPO_URL}/actions/runs/${esc(v.run_id)}" target="_blank" rel="noreferrer">السجل</a>` : ''}
        <button class="btn small danger act-delver" data-file="${esc(v.file)}">حذف النسخة</button></li>`).join('')}</ul>
      <div class="buttons">
        <a class="btn small" href="${REPO_URL}/tree/${encodeURIComponent(BRANCH)}/dubs/${encodeURIComponent(d.slug)}" target="_blank" rel="noreferrer">GitHub</a>
        <button class="btn small danger act-delete">حذف المجلد كاملاً</button>
      </div></div>`;
    card.querySelectorAll('.act-play').forEach((b) => b.addEventListener('click', () => openPlayer(`${d.slug} · ${b.dataset.file}`, rawUrl(`dubs/${d.slug}/${b.dataset.file}`))));
    card.querySelectorAll('.act-delver').forEach((b) => b.addEventListener('click', () => confirmDeleteVersion(d, b.dataset.file)));
    card.querySelectorAll('.act-details').forEach((b) => b.addEventListener('click', () => openDubDetails(d, d.versions.find((v) => v.file === b.dataset.file) || { file: b.dataset.file })));
    card.querySelector('.act-delete').addEventListener('click', () => confirmDeleteFolder(`dubs/${d.slug}/`, d.slug, 'كل النسخ المدبلجة لهذا الفيديو'));
    grid.appendChild(card);
  }
}

function renderRuns() {
  const list = $('runsList');
  const active = state.runs.filter(isActive).length;
  $('countRuns').textContent = active ? `${active} نشط` : state.runs.length;
  if (!state.runs.length) { list.innerHTML = `<div class="empty">${state.loaded ? 'لا توجد تشغيلات بعد.' : 'جارٍ التحميل…'}</div>`; return; }
  list.innerHTML = '';
  for (const run of state.runs) {
    const cls = run.status === 'completed' ? (run.conclusion || 'completed') : run.status;
    const started = run.run_started_at || run.created_at;
    const elapsed = Math.max(0, ((run.status === 'completed' ? new Date(run.updated_at) : new Date()) - new Date(started)) / 1000);
    const prog = run._progress;
    const bar = prog && prog.total ? `<div class="progress" title="${prog.counts.completed || 0}/${prog.total} مقطع مكتمل"><span style="width:${Math.round(((prog.counts.completed || 0) * 100) / prog.total)}%"></span></div>
      <div class="sub">${prog.counts.completed || 0}/${prog.total} مقطع مكتمل${prog.counts.failed ? ` · ${prog.counts.failed} فاشل` : ''}${prog.counts.processing ? ` · ${prog.counts.processing} قيد المعالجة` : ''} · <span class="mono">${esc(prog.state || '')}</span></div>${chunkGrid(prog)}` : '';
    const row = document.createElement('div'); row.className = `run ${cls}`;
    row.innerHTML = `<div class="dot"></div><div>
      <div class="title">${esc(runSource(run) || run.display_title)} <span class="sub">· #${run.run_number}</span></div>
      <div class="sub">${esc(statusLabel(run))}${isActive(run) && run._step ? ` · الخطوة: ${esc(run._step)}` : ''} · بدأ ${fmtDate(started)} · ${fmtDur(elapsed)} دقيقة</div>${bar}</div>
      <div class="buttons"><a class="btn small" href="${run.html_url}" target="_blank" rel="noreferrer">السجل</a>${isActive(run) ? '<button class="btn small danger act-cancel">إلغاء</button>' : ''}</div>`;
    row.querySelector('.act-cancel')?.addEventListener('click', () => confirmCancel(run));
    list.appendChild(row);
  }
}
function statusLabel(run) {
  if (run.status !== 'completed') return { queued: 'في الانتظار', in_progress: 'يعمل الآن', waiting: 'ينتظر', pending: 'معلّق', requested: 'مطلوب' }[run.status] || run.status;
  return { success: 'نجح', failure: 'فشل', cancelled: 'أُلغي', timed_out: 'انتهت المهلة', skipped: 'تُخطّي' }[run.conclusion] || run.conclusion || 'انتهى';
}

function renderAll() { renderLibrary(); renderDubs(); renderRuns(); renderVoiceBank(); $('clockChip').textContent = `آخر تحديث ${new Date().toLocaleTimeString('ar-EG', { hour12: false })}`; }

// ───────────────────────────────────────────── modal helpers
function openModal(title, bodyHtml) { $('modalTitle').textContent = title; $('modalBody').innerHTML = bodyHtml; $('modal').classList.remove('hidden'); }
function closeModal() { $('modal').classList.add('hidden'); $('modalBody').innerHTML = ''; }
function openPlayer(title, url, item) {
  const kv = item ? `<div class="kv"><b>المجلد</b><span class="mono">library/${esc(item.slug)}/</span><b>الحجم</b><span>${formatMB(item.size)} MB</span><b>المدة</b><span>${fmtDur(item.meta.duration)}</span><b>SHA-256</b><span class="mono">${esc((item.meta.sha256 || '').slice(0, 16))}…</span></div>` : '';
  openModal(title, `<video controls autoplay playsinline src="${url}"></video>${kv}<p><a class="btn small" href="${url}" download>تحميل</a></p>`);
}

/** Every destructive action passes through here: the user must type the name. */
function confirmTyped({ title, message, expect, onConfirm }) {
  openModal(title, `<div class="confirm"><p>${message}</p><p>للتأكيد اكتب: <code>${esc(expect)}</code></p><input id="confirmInput" placeholder="${esc(expect)}"><div class="row"><button id="confirmGo" class="btn danger" disabled>تنفيذ الحذف</button><button id="confirmNo" class="btn">إلغاء</button></div><div id="confirmStatus" class="status"></div></div>`);
  const input = $('confirmInput'), go = $('confirmGo');
  input.addEventListener('input', () => { go.disabled = input.value.trim() !== expect; });
  $('confirmNo').onclick = closeModal;
  go.onclick = async () => {
    go.disabled = true; setStatus($('confirmStatus'), 'جارٍ التنفيذ…', 'info');
    try { await onConfirm(); closeModal(); await fullRefresh(true); }
    catch (e) { setStatus($('confirmStatus'), e.message, 'err'); go.disabled = false; }
  };
}
function confirmDeleteFolder(prefix, slug, what) {
  const paths = state.tree.filter((f) => f.path.startsWith(prefix)).map((f) => f.path);
  const size = state.tree.filter((f) => f.path.startsWith(prefix)).reduce((a, f) => a + f.size, 0);
  confirmTyped({
    title: `حذف ${prefix}`,
    message: `سيُحذف ${what}: ${paths.length} ملفاً (${formatMB(size)} MB) في commit واحد. لا يمكن التراجع من هذه الصفحة.`,
    expect: slug,
    onConfirm: () => commitChanges({ message: `Delete ${prefix} from the studio (confirmed by typing "${slug}")`, deletions: paths }),
  });
}
function confirmDeleteVersion(dub, file) {
  const base = file.replace(/\.mp4$/i, '');
  const paths = dub.files.filter((f) => f.name === file || f.name.startsWith(`${base}.`)).map((f) => f.path);
  confirmTyped({
    title: `حذف النسخة ${file}`,
    message: `سيُحذف الملف وتقاريره (${paths.length} ملفات) ويُحدَّث meta.json.`,
    expect: base,
    onConfirm: async () => {
      const meta = { ...(dub.meta || {}), versions: (dub.meta.versions || []).filter((v) => v.file !== file), updated_at: new Date().toISOString() };
      const sha = await createBlob(JSON.stringify(meta, null, 2) + '\n', 'utf-8');
      await commitChanges({ message: `Delete dub version ${file} (confirmed in the studio)`, deletions: paths, additions: [{ path: `dubs/${dub.slug}/meta.json`, sha }] });
    },
  });
}
function confirmCancel(run) {
  confirmTyped({
    title: `إلغاء التشغيل #${run.run_number}`, message: 'سيتوقف التشغيل الآن؛ ما اكتمل من مقاطع يبقى محفوظاً في نقاط الاستئناف ويُستكمَل لاحقاً.', expect: String(run.run_number),
    onConfirm: async () => { needToken(); await api(`/repos/${OWNER}/${REPO}/actions/runs/${run.id}/cancel`, { method: 'POST' }); },
  });
}

// ───────────────────────────────────────────── dubbing
function defaults() {
  let saved = {}; try { saved = JSON.parse(localStorage.getItem(DEFAULTS_KEY) || '{}'); } catch { /* fresh */ }
  return {
    target_lang: 'en', voice: 'en-US-AndrewMultilingualNeural', tts_engine: 'voxcpm', gender: 'male', model: 'medium',
    profile: 'seed_quota_voxcpm', quality: 'balanced', chunk_seconds: '10', seed_vc: true, separate_sources: true,
    bg_music: false, diarize: false, validate_content: true, ...saved,
  };
}
function readDefaultsForm() {
  return {
    target_lang: $('dTarget').value.trim() || 'en', voice: $('dVoice').value.trim(), tts_engine: $('dEngine').value, gender: $('dGender').value,
    model: $('dModel').value, profile: $('dProfile').value, quality: $('dQuality').value, chunk_seconds: $('dChunk').value.trim() || '10',
    seed_vc: $('dSeed').checked, separate_sources: $('dSeparate').checked, bg_music: $('dBg').checked, diarize: $('dDiarize').checked, validate_content: $('dValidate').checked,
  };
}
function fillDefaultsForm(d) {
  $('dTarget').value = d.target_lang; $('dVoice').value = d.voice; $('dEngine').value = d.tts_engine; $('dGender').value = d.gender; $('dModel').value = d.model;
  $('dProfile').value = d.profile; $('dQuality').value = d.quality; $('dChunk').value = d.chunk_seconds; $('dSeed').checked = d.seed_vc; $('dSeparate').checked = d.separate_sources;
  $('dBg').checked = d.bg_music; $('dDiarize').checked = d.diarize; $('dValidate').checked = d.validate_content;
}
async function dispatchDub(item, overrides = {}) {
  needToken();
  const d = { ...defaults(), ...overrides };
  const inputs = {
    task: 'dub', source_path: item.sourcePath, youtube_url: '', source_lang: item.meta.source_lang || 'ar',
    voice: d.voice, tts_engine: d.tts_engine, target_lang: d.target_lang, mode: 'both', gender: d.gender, model: d.model,
    bg_music: String(!!d.bg_music), diarize: String(!!d.diarize), separate_sources: String(!!d.separate_sources), no_vad: 'false',
    seed_vc: String(!!d.seed_vc), lip_sync: 'false', lip_sync_backend: 'wav2lip', profile: d.profile, quality: d.quality,
    chunk_seconds: String(d.chunk_seconds), speaker_voices_path: d.speaker_voices_path || '', validate_content: String(!!d.validate_content),
  };
  await api(`/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW}/dispatches`, { method: 'POST', body: JSON.stringify({ ref: BRANCH, inputs }) });
}
function openDubDialog(item) {
  const d = defaults();
  openModal(`دبلجة ${item.title}`, `<div class="confirm">
    <div class="kv"><b>المصدر</b><span class="mono">${esc(item.sourcePath || '')}</span><b>لغة المصدر</b><span>${esc(item.meta.source_lang || 'ar')}</span></div>
    <div class="settings-grid">
      <label>لغة الهدف <input id="xTarget" value="${esc(d.target_lang)}" dir="ltr"></label>
      <label>الجودة <select id="xQuality"><option ${d.quality === 'balanced' ? 'selected' : ''}>balanced</option><option ${d.quality === 'strict' ? 'selected' : ''}>strict</option><option ${d.quality === 'safe' ? 'selected' : ''}>safe</option></select></label>
      <label>الجنس <select id="xGender"><option value="male" ${d.gender === 'male' ? 'selected' : ''}>ذكر</option><option value="female" ${d.gender === 'female' ? 'selected' : ''}>أنثى</option></select></label>
      <label class="toggle"><input type="checkbox" id="xSeed" ${d.seed_vc ? 'checked' : ''}> Seed-VC</label>
      <label class="toggle"><input type="checkbox" id="xBg" ${d.bg_music ? 'checked' : ''}> موسيقى الخلفية</label>
      ${item.voicesJson ? `<label class="toggle"><input type="checkbox" id="xVoices" checked> استخدام خريطة الأصوات (voices.json)</label>` : ''}
    </div>
    <p class="hint">باقي الإعدادات من صفحة الإعدادات (المحرك ${esc(d.tts_engine)}، النموذج ${esc(d.model)}، الملف ${esc(d.profile)}). التشغيل يظهر في تبويب التشغيلات خلال ثوانٍ، والنسخة تظهر في «المدبلجة» تلقائياً عند الانتهاء.</p>
    <div class="row"><button id="xGo" class="btn primary">ابدأ الدبلجة</button><button id="xNo" class="btn">إلغاء</button></div><div id="xStatus" class="status"></div></div>`);
  $('xNo').onclick = closeModal;
  $('xGo').onclick = async () => {
    $('xGo').disabled = true; setStatus($('xStatus'), 'جارٍ تشغيل الدبلجة في GitHub Actions…', 'info');
    try {
      await dispatchDub(item, { target_lang: $('xTarget').value.trim() || 'en', quality: $('xQuality').value, gender: $('xGender').value, seed_vc: $('xSeed').checked, bg_music: $('xBg').checked,
        speaker_voices_path: item.voicesJson && $('xVoices')?.checked ? `library/${item.slug}/voices.json` : '' });
      setStatus($('xStatus'), 'انطلق التشغيل. انتقل إلى «التشغيلات» للمتابعة.', 'ok');
      setTimeout(async () => { closeModal(); await refreshRuns(); renderAll(); showTab('runs'); }, 1500);
    } catch (e) { setStatus($('xStatus'), e.message, 'err'); $('xGo').disabled = false; }
  };
}

// ───────────────────────────────────────────── voices & characters
const defaultProfile = (speaker) => ({ speaker, label: speaker, reference_mode: 'source', reference_path: '', tts_engine: defaults().tts_engine, voice: '', voice_conversion: defaults().seed_vc ? 'seed-vc' : 'none', style: 'natural', gender: defaults().gender, approved: false });

async function releaseFor(slug) {
  if (!state.token) return null;
  if (Date.now() - state.releasesAt > 20000) {
    state.releases = (await api(`/repos/${OWNER}/${REPO}/releases?per_page=100`)).filter((r) => r.draft && r.tag_name.startsWith('checkpoint-'));
    state.releasesAt = Date.now();
  }
  return state.releases.filter((r) => r.tag_name.startsWith(`checkpoint-${slug}-`)).sort((a, b) => b.updated_at.localeCompare(a.updated_at))[0] || null;
}
async function manifestOf(release) {
  const asset = release?.assets.find((a) => a.name === 'checkpoint-manifest.json');
  if (!asset) return null;
  const res = await api(`/repos/${OWNER}/${REPO}/releases/assets/${asset.id}`, { headers: { Accept: 'application/octet-stream' }, raw: true });
  return res.json();
}
function speakerCard(speaker, p, item) {
  const sel = (v, cur) => (v === cur ? ' selected' : '');
  const sample = item.voiceSamples.find((f) => f.name.replace(/^voices\//, '').replace(/\.[^.]+$/, '') === speaker);
  const bank = state.voiceBank.map((b) => `<option value="../../voices/${esc(b.name)}"${sel(`../../voices/${b.name}`, p.reference_path)}>${esc(b.name)}</option>`).join('');
  return `<article class="character" data-speaker="${esc(speaker)}">
    <div class="charhead"><h4>${esc(speaker)}</h4><button class="btn small danger act-rmspeaker" title="إزالة من الخريطة">إزالة</button></div>
    <div class="fields">
      <label>الاسم الظاهر <input data-key="label" value="${esc(p.label || speaker)}"></label>
      <label>مصدر الصوت <select data-key="reference_mode"><option value="source"${sel('source', p.reference_mode || 'source')}>من الفيديو نفسه</option><option value="custom"${sel('custom', p.reference_mode)}>عيّنة صوتية مخصّصة</option><option value="synthetic"${sel('synthetic', p.reference_mode)}>صوت اصطناعي جاهز</option></select></label>
      <label>محرك الصوت <select data-key="tts_engine"><option value="voxcpm"${sel('voxcpm', p.tts_engine || 'voxcpm')}>VoxCPM</option><option value="xtts"${sel('xtts', p.tts_engine)}>XTTS</option><option value="qwen"${sel('qwen', p.tts_engine)}>Qwen</option><option value="edge"${sel('edge', p.tts_engine)}>Edge-TTS</option></select></label>
      <label>تحويل الهوية <select data-key="voice_conversion"><option value="seed-vc"${sel('seed-vc', p.voice_conversion || 'seed-vc')}>Seed-VC</option><option value="none"${sel('none', p.voice_conversion)}>بدون</option></select></label>
      <label>الجنس <select data-key="gender"><option value="male"${sel('male', p.gender || 'male')}>ذكر</option><option value="female"${sel('female', p.gender)}>أنثى</option></select></label>
      <label>الأسلوب <input data-key="style" value="${esc(p.style || 'natural')}"></label>
      <label>صوت Edge الجاهز (للوضع الاصطناعي) <input data-key="voice" value="${esc(p.voice || '')}" dir="ltr"></label>
      <label class="wide">مسار العيّنة (نسبةً إلى voices.json) <input data-key="reference_path" value="${esc(p.reference_path || '')}" dir="ltr"></label>
      <label class="wide">اختيار من بنك الأصوات <select class="bankpick"><option value="">—</option>${bank}</select></label>
      <label class="wide">رفع عيّنة صوتية لهذا المتحدث (wav/mp3/m4a، 6–20 ثانية كلام نقي) <input type="file" class="samplefile" accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg"></label>
      ${sample ? `<div class="wide"><audio controls preload="none" src="${rawUrl(sample.path)}"></audio><span class="hint mono">${esc(sample.name)}</span></div>` : ''}
    </div>
    <label class="toggle"><input type="checkbox" data-key="approved" ${p.approved ? 'checked' : ''}> اعتماد إعداد هذا المتحدث</label>
  </article>`;
}
async function openVoicesDialog(item) {
  openModal(`الأصوات والشخصيات · ${item.title}`, '<p class="hint">جارٍ القراءة…</p>');
  const doc = item.voicesJson ? (await blobJson(item.voicesJson.sha)) || {} : {};
  const profiles = { ...(doc.speakers || {}) };
  let detected = [];
  try {
    const manifest = await manifestOf(await releaseFor(item.slug));
    detected = Object.keys(manifest?.voice_profiles || {});
    if (!detected.length && manifest?.chunks) detected = [...new Set(manifest.chunks.map((c) => c.speaker).filter(Boolean))];
  } catch { /* no checkpoint yet */ }
  for (const s of detected) if (!profiles[s]) profiles[s] = defaultProfile(s);
  if (!Object.keys(profiles).length) profiles.SPEAKER_00 = defaultProfile('SPEAKER_00');
  const render = () => {
    $('modalBody').innerHTML = `<div class="confirm">
      <p class="hint">خريطة الأصوات تُحفَظ في <code>library/${esc(item.slug)}/voices.json</code> وتُستخدم عند الدبلجة (خيار في نافذة الدبلجة). المتحدثون المكتشفون من آخر تحليل: ${detected.length ? detected.map(esc).join(', ') : 'لا يوجد تحليل بعد — الافتراضي متحدث واحد SPEAKER_00'}. لا تُضِف متحدثين غير مكتشفين إلا مع تفعيل تمييز المتحدثين، وإلا يرفضهم التشغيل.</p>
      <div class="character-grid">${Object.entries(profiles).map(([s, p]) => speakerCard(s, p, item)).join('')}</div>
      <div class="row"><input id="newSpeaker" placeholder="SPEAKER_01" dir="ltr"><button id="addSpeaker" class="btn small">إضافة متحدث</button></div>
      <div class="row"><button id="saveVoices" class="btn primary">حفظ الخريطة (والعيّنات المرفوعة)</button>${item.voicesJson ? '<button id="deleteVoices" class="btn danger">حذف الخريطة</button>' : ''}<span id="voicesStatus" class="status"></span></div></div>`;
    $('modalBody').querySelectorAll('.bankpick').forEach((sel) => sel.addEventListener('change', () => { const card = sel.closest('.character'); if (sel.value) { card.querySelector('[data-key=reference_path]').value = sel.value; card.querySelector('[data-key=reference_mode]').value = 'custom'; } }));
    $('modalBody').querySelectorAll('.samplefile').forEach((inp) => inp.addEventListener('change', () => { const card = inp.closest('.character'); if (inp.files[0]) { const ext = (inp.files[0].name.match(/\.[^.]+$/) || ['.wav'])[0].toLowerCase(); card.querySelector('[data-key=reference_path]').value = `voices/${card.dataset.speaker}${ext}`; card.querySelector('[data-key=reference_mode]').value = 'custom'; } }));
    $('modalBody').querySelectorAll('.act-rmspeaker').forEach((b) => b.addEventListener('click', () => { delete profiles[b.closest('.character').dataset.speaker]; render(); }));
    $('addSpeaker').onclick = () => { const s = $('newSpeaker').value.trim(); if (!SPEAKER_RE.test(s)) return setStatus($('voicesStatus'), 'معرّف المتحدث: أحرف لاتينية وأرقام و _ . - فقط', 'err'); profiles[s] = defaultProfile(s); render(); };
    $('deleteVoices')?.addEventListener('click', () => confirmTyped({ title: 'حذف خريطة الأصوات', message: `سيُحذف voices.json وكل العيّنات في library/${esc(item.slug)}/voices/.`, expect: item.slug, onConfirm: () => commitChanges({ message: `Remove voice map of ${item.slug} (confirmed in the studio)`, deletions: [item.voicesJson.path, ...item.voiceSamples.map((f) => f.path)] }) }));
    $('saveVoices').onclick = async () => {
      const status = $('voicesStatus'); $('saveVoices').disabled = true;
      try {
        needToken();
        const speakers = {}; const additions = [];
        for (const card of $('modalBody').querySelectorAll('.character')) {
          const s = card.dataset.speaker; const data = { speaker: s };
          for (const el of card.querySelectorAll('[data-key]')) data[el.dataset.key] = el.type === 'checkbox' ? el.checked : el.value.trim();
          const file = card.querySelector('.samplefile').files[0];
          if (file) {
            if (file.size > PART_BYTES) throw new Error(`عيّنة ${s} أكبر من 18 MB — قصّها أولاً`);
            setStatus(status, `رفع عيّنة ${s}…`, 'info');
            additions.push({ path: `library/${item.slug}/${data.reference_path}`, sha: await createBlob(await toBase64(file), 'base64') });
          }
          if (data.reference_mode === 'custom' && !data.reference_path) throw new Error(`${s}: العيّنة المخصّصة تحتاج مساراً أو ملفاً`);
          if (data.reference_mode === 'synthetic' && data.tts_engine === 'edge' && !data.voice) throw new Error(`${s}: الصوت الاصطناعي عبر Edge يحتاج اسم صوت`);
          if (!data.approved) throw new Error(`${s}: يجب اعتماد كل متحدث قبل الحفظ — التشغيل يرفض خريطة غير معتمدة`);
          speakers[s] = data;
        }
        additions.push({ path: `library/${item.slug}/voices.json`, sha: await createBlob(JSON.stringify({ version: 1, updated_at: new Date().toISOString(), speakers }, null, 2) + '\n', 'utf-8') });
        setStatus(status, 'حفظ في المستودع…', 'info');
        await commitChanges({ message: `Voice map for ${item.slug}: ${Object.keys(speakers).join(', ')}`, additions });
        setStatus(status, 'تم الحفظ. عند الدبلجة فعّل «استخدام خريطة الأصوات».', 'ok');
        await fullRefresh(true);
      } catch (e) { setStatus(status, e.message, 'err'); } finally { $('saveVoices').disabled = false; }
    };
  };
  render();
}

// ───────────────────────────────────────────── checkpoints (stages + audio comparisons)
const STAGE_ORDER = ['analysis', 'translation', 'tts', 'seed_vc', 'timing_fit', 'content_validation', 'audio_mix', 'video_render', 'checkpoint_upload'];
const STAGE_LABEL = { analysis: 'التحليل', translation: 'الترجمة', tts: 'الصوت', seed_vc: 'Seed-VC', timing_fit: 'التوقيت', content_validation: 'الكلمات', audio_mix: 'المزج', video_render: 'الرندر', checkpoint_upload: 'الرفع' };
async function openCheckpointDialog(item) {
  openModal(`نقاط الاستئناف · ${item.title}`, '<p class="hint">جارٍ القراءة من الإصدار المسودّ…</p>');
  try {
    needToken();
    const release = await releaseFor(item.slug);
    if (!release) { $('modalBody').innerHTML = '<p class="hint">لا توجد نقاط استئناف لهذا الفيديو (لم يبدأ تشغيل بعد أو حُذفت).</p>'; return; }
    const manifest = (await manifestOf(release)) || { chunks: [] };
    const chunks = manifest.chunks || [];
    const counts = {}; for (const c of chunks) counts[c.status || 'pending'] = (counts[c.status || 'pending'] || 0) + 1;
    const size = release.assets.reduce((a, x) => a + x.size, 0);
    const sym = { success: '✓', failed: '✗', pending: '…', skipped: '—' };
    const rows = chunks.map((c) => `<tr><td class="mono">${String(c.index).padStart(4, '0')}</td><td class="mono">${fmtDur(+c.start)}–${fmtDur(+c.end)}</td><td class="mono">${esc(c.status || 'pending')}</td>${STAGE_ORDER.map((s) => { const st = c.checklist?.[s]?.state || 'pending'; return `<td class="stage-${st}" title="${esc(c.checklist?.[s]?.error || '')}">${sym[st] || st}</td>`; }).join('')}<td><button class="btn small act-audio" data-index="${c.index}">صوت</button></td></tr>`).join('');
    $('modalBody').innerHTML = `<div class="kv"><b>الإصدار</b><span class="mono">${esc(release.tag_name)}</span><b>الحالة</b><span class="mono">${esc(manifest.state || '')}</span><b>المقاطع</b><span>${counts.completed || 0}/${chunks.length} مكتمل${counts.failed ? ` · ${counts.failed} فاشل` : ''}</span><b>الحجم</b><span>${formatMB(size)} MB · ${release.assets.length} ملف</span><b>Seed-VC</b><span>${manifest.seed_quota_fallback?.active ? 'تراجع إلى VoxCPM (الحصة نفدت)' : 'مفعّل'}</span><b>الترجمة</b><span class="mono">${esc(manifest.translation?.engine || 'google')}${manifest.translation?.model ? ' · ' + esc(manifest.translation.model) : ''}</span></div>
      <div class="tablewrap" style="margin-top:10px"><table class="segs stage-table"><thead><tr><th>#</th><th>الزمن</th><th>الحالة</th>${STAGE_ORDER.map((s) => `<th>${STAGE_LABEL[s]}</th>`).join('')}<th></th></tr></thead><tbody>${rows}</tbody></table></div>
      <div id="audioCompare" class="compare-audio"></div>
      <div class="row" style="margin-top:10px"><button id="delCheckpoints" class="btn danger">حذف نقاط الاستئناف (الإصدار)</button><span class="hint">الحذف لا يمسّ الفيديو ولا النسخ المدبلجة المنشورة؛ يعيد الدبلجة القادمة من الصفر.</span></div>`;
    $('modalBody').querySelectorAll('.act-audio').forEach((b) => b.addEventListener('click', () => renderAudioCompare(release, +b.dataset.index)));
    $('delCheckpoints').onclick = () => confirmTyped({
      title: 'حذف نقاط الاستئناف', message: `سيُحذف الإصدار المسودّ ${esc(release.tag_name)} (${formatMB(size)} MB).`, expect: item.slug,
      onConfirm: async () => { await api(`/repos/${OWNER}/${REPO}/releases/${release.id}`, { method: 'DELETE' }); state.releasesAt = 0; },
    });
  } catch (e) { $('modalBody').innerHTML = `<p class="status err">${esc(e.message)}</p>`; }
}
async function renderAudioCompare(release, index) {
  const box = $('audioCompare'); if (!box) return;
  const labels = { original: 'الأصل', before_seed_vc: 'قبل Seed-VC', after_seed_vc: 'بعد Seed-VC', final: 'النهائي' };
  const prefix = `chunk-${String(index).padStart(4, '0')}-preview-`;
  const assets = release.assets.filter((a) => a.name.startsWith(prefix) && a.name.endsWith('.mp3'));
  if (!assets.length) { box.innerHTML = `<p class="hint">لا معاينات صوتية للمقطع ${index}.</p>`; return; }
  box.innerHTML = `<h4>المقطع ${index}</h4><div class="row" id="audioSlots">${assets.map((a) => `<div class="audio-slot" data-id="${a.id}"><span class="hint">${esc(labels[a.name.slice(prefix.length, -4)] || a.name)}</span><br><button class="btn small">تشغيل</button></div>`).join('')}</div>`;
  box.querySelectorAll('.audio-slot button').forEach((b) => b.addEventListener('click', async () => {
    const slot = b.parentElement; b.disabled = true;
    try { const res = await api(`/repos/${OWNER}/${REPO}/releases/assets/${slot.dataset.id}`, { headers: { Accept: 'application/octet-stream' }, raw: true }); const blob = await res.blob(); const audio = document.createElement('audio'); audio.controls = true; audio.autoplay = true; audio.src = URL.createObjectURL(blob); b.replaceWith(audio); }
    catch (e) { b.disabled = false; b.textContent = e.message; }
  }));
}

// ───────────────────────────────────────────── voice bank
function renderVoiceBank() {
  const grid = $('voicesGrid'); if (!grid) return;
  $('countVoices').textContent = state.voiceBank.length;
  if (!state.voiceBank.length) { grid.innerHTML = `<div class="empty">${state.loaded ? 'بنك الأصوات فارغ — ارفع عيّنة صوتية (6–20 ثانية كلام نقي لمتحدث واحد).' : 'جارٍ التحميل…'}</div>`; return; }
  grid.innerHTML = '';
  for (const v of state.voiceBank) {
    const card = document.createElement('article'); card.className = 'card';
    card.innerHTML = `<div class="body"><h3 class="mono">${esc(v.name)}</h3><audio controls preload="none" src="${rawUrl(v.path)}"></audio><div class="meta"><span>${formatMB(v.size)} MB</span><span class="mono">voices/${esc(v.name)}</span></div>
      <div class="buttons"><a class="btn small" href="${rawUrl(v.path)}" download>تحميل</a><button class="btn small danger act-del">حذف</button></div></div>`;
    card.querySelector('.act-del').addEventListener('click', () => confirmTyped({ title: `حذف العيّنة ${v.name}`, message: 'ستُحذف من بنك الأصوات؛ خرائط الأصوات التي تشير إليها ستفشل حتى تُعدَّل.', expect: v.name, onConfirm: () => commitChanges({ message: `Remove voice sample ${v.name} (confirmed in the studio)`, deletions: [v.path] }) }));
    grid.appendChild(card);
  }
}
async function uploadVoiceSample() {
  const status = $('voiceStatus'); const file = $('voiceFile').files[0];
  if (!file) return setStatus(status, 'اختر ملفاً صوتياً أولاً', 'err');
  try { needToken(); } catch (e) { return setStatus(status, e.message, 'err'); }
  if (file.size > PART_BYTES) return setStatus(status, 'العيّنة أكبر من 18 MB — قصّها إلى 20 ثانية تقريباً', 'err');
  const ext = (file.name.match(/\.[^.]+$/) || ['.wav'])[0].toLowerCase();
  const name = (slugify($('voiceName').value || file.name) || 'voice') + ext;
  $('voiceUpload').disabled = true;
  try {
    setStatus(status, 'رفع العيّنة…', 'info');
    const sha = await createBlob(await toBase64(file), 'base64');
    await commitChanges({ message: `Add voice sample ${name} to the voice bank`, additions: [{ path: `voices/${name}`, sha }] });
    setStatus(status, `تمت الإضافة: voices/${name}`, 'ok'); $('voiceFile').value = ''; $('voiceName').value = '';
    await fullRefresh(true);
  } catch (e) { setStatus(status, e.message, 'err'); } finally { $('voiceUpload').disabled = false; }
}

// ───────────────────────────────────────────── upload
let chosen = null;
function slugify(name) {
  const base = String(name).replace(/\\/g, '/').split('/').pop().replace(/\.[^.]+$/, '');
  let slug = base.normalize('NFKD').replace(/[\u0300-\u036f]/g, '').replace(/[^A-Za-z0-9._-]+/g, '-').replace(/-{2,}/g, '-').replace(/^[-.]+|[-.]+$/g, '').toLowerCase();
  if (!/[a-z0-9]/.test(slug)) slug = `video-${new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '')}`;
  return slug.slice(0, 60);
}
function probeVideo(file) {
  return new Promise((resolve) => {
    const v = document.createElement('video'); v.preload = 'metadata'; v.muted = true;
    const url = URL.createObjectURL(file);
    const done = (meta) => { URL.revokeObjectURL(url); resolve(meta); };
    v.onloadedmetadata = () => done({ duration: Number.isFinite(v.duration) ? Math.round(v.duration * 100) / 100 : null, width: v.videoWidth || null, height: v.videoHeight || null });
    v.onerror = () => done({ duration: null, width: null, height: null });
    v.src = url;
  });
}
function showFile() {
  $('fileName').textContent = chosen ? `${chosen.name} · ${formatMB(chosen.size)} MB` : '';
  if (chosen) { $('slugInput').value = slugify(chosen.name); if (!$('titleInput').value) $('titleInput').value = chosen.name.replace(/\.[^.]+$/, ''); }
}
async function uploadChosen() {
  const status = $('uploadStatus');
  if (!chosen) return setStatus(status, 'اختر ملف فيديو أولاً', 'err');
  try { needToken(); } catch (e) { return setStatus(status, e.message, 'err'); }
  let slug = slugify($('slugInput').value || chosen.name);
  if (state.library.some((it) => it.slug === slug)) {
    let n = 2; while (state.library.some((it) => it.slug === `${slug}-${n}`)) n++;
    slug = `${slug}-${n}`;
  }
  const ext = (chosen.name.match(/\.(mp4|mkv|webm|mov)$/i) || ['.mp4'])[0].toLowerCase();
  const sourceName = `source${ext}`;
  const btn = $('uploadBtn'); btn.disabled = true;
  try {
    setStatus(status, 'حساب SHA-256 وقراءة البيانات…', 'info');
    const [sha256, probe] = await Promise.all([sha256Hex(chosen), probeVideo(chosen)]);
    const total = Math.max(1, Math.ceil(chosen.size / PART_BYTES));
    const additions = [];
    for (let i = 1; i <= total; i++) {
      setStatus(status, total === 1 ? 'رفع الملف…' : `رفع الجزء ${i}/${total}…`, 'info');
      const slice = chosen.slice((i - 1) * PART_BYTES, Math.min(i * PART_BYTES, chosen.size));
      const sha = await createBlob(await toBase64(slice), 'base64');
      const width = Math.max(2, String(total).length);
      const name = total === 1 ? sourceName : `${sourceName}.part${String(i).padStart(width, '0')}of${String(total).padStart(width, '0')}`;
      additions.push({ path: `library/${slug}/${name}`, sha });
    }
    const meta = {
      slug, title: $('titleInput').value.trim() || chosen.name, original_name: chosen.name, size: chosen.size, sha256,
      duration: probe.duration, width: probe.width, height: probe.height, parts: total, source: `library/${slug}/${sourceName}`,
      source_lang: $('srcLangInput').value.trim() || 'ar', uploaded_at: new Date().toISOString(), status: 'uploaded', uploaded_with: 'studio',
    };
    additions.push({ path: `library/${slug}/meta.json`, sha: await createBlob(JSON.stringify(meta, null, 2) + '\n', 'utf-8') });
    setStatus(status, 'إنشاء الـ commit…', 'info');
    await commitChanges({ message: `Add ${slug} to the library (${formatMB(chosen.size)} MB${total > 1 ? `, ${total} parts` : ''})`, additions });
    setStatus(status, `تم الرفع إلى library/${slug}/`, 'ok');
    await fullRefresh(true);
    if ($('dubAfterUpload').checked) {
      const item = state.library.find((it) => it.slug === slug) || { slug, meta, sourcePath: meta.source, title: meta.title };
      await dispatchDub(item);
      setStatus(status, `تم الرفع وانطلقت الدبلجة — تابعها في «التشغيلات».`, 'ok');
      setTimeout(async () => { await refreshRuns(); renderAll(); }, 4000);
    }
    chosen = null; $('file').value = ''; $('titleInput').value = ''; showFile();
  } catch (e) { setStatus(status, e.message, 'err'); }
  finally { btn.disabled = false; }
}

// ───────────────────────────────────────────── refresh loop
async function fullRefresh(force = false) {
  if (state.busy) return; state.busy = true;
  try {
    const [changed] = await Promise.all([refreshTree(force), refreshRuns()]);
    state.loaded = true;
    if (changed || force) setStatus($('libraryStatus'), `${state.library.length} فيديو في المكتبة · ${state.dubs.length} مدبلج`, 'ok');
    renderAll();
    setStatus($('runsStatus'), `${state.runs.filter(isActive).length} تشغيل نشط من ${state.runs.length}`, 'ok');
  } catch (e) {
    setStatus($('libraryStatus'), e.message, 'err'); setStatus($('runsStatus'), e.message, 'err');
  } finally { state.busy = false; }
}
function schedule() {
  clearTimeout(state.polling);
  if (!$('autoRefresh').checked) return;
  const active = state.runs.some(isActive);
  const seconds = !state.token ? 90 : active ? 15 : 40;
  $('runsInterval').textContent = seconds;
  state.polling = setTimeout(async () => { await fullRefresh(false); schedule(); }, seconds * 1000);
}

// ───────────────────────────────────────────── tabs + wiring
function showTab(name) {
  document.querySelectorAll('.tab[data-tab]').forEach((t) => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.panel').forEach((p) => p.classList.toggle('active', p.id === `tab-${name}`));
  location.hash = name;
}
function setConnection() {
  const chip = $('connChip');
  if (state.token) { chip.textContent = 'متصل بالرمز'; chip.className = 'chip ok'; }
  else { chip.textContent = 'قراءة فقط — بدون رمز'; chip.className = 'chip warn'; }
  document.querySelectorAll('#uploadBtn').forEach((b) => { b.disabled = !state.token; });
}

function init() {
  $('repoName').textContent = `${OWNER}/${REPO}@${BRANCH}`;
  $('repoLink').href = `${REPO_URL}/tree/${encodeURIComponent(BRANCH)}`;
  $('actionsLink').href = `${REPO_URL}/actions`;
  $('tokenLink').href = TOKEN_URL;
  $('preflightLink').href = `${REPO_URL}/actions/workflows/translation-preflight.yml`;
  state.token = loadToken(); $('token').value = state.token; setConnection();
  fillDefaultsForm(defaults());

  document.querySelectorAll('.tab[data-tab]').forEach((t) => t.addEventListener('click', () => showTab(t.dataset.tab)));
  const initial = location.hash.replace('#', ''); if (['library', 'dubs', 'runs', 'voices', 'settings'].includes(initial)) showTab(initial);
  $('voiceUpload').onclick = uploadVoiceSample;
  $('modalClose').onclick = closeModal; $('modal').addEventListener('click', (e) => { if (e.target === $('modal')) closeModal(); });
  $('refreshNow').onclick = () => fullRefresh(true);
  $('autoRefresh').onchange = schedule;
  $('searchLibrary').oninput = (e) => { state.searchLibrary = e.target.value; renderLibrary(); };
  $('filterLibrary').onchange = (e) => { state.filterLibrary = e.target.value; renderLibrary(); };
  $('searchDubs').oninput = (e) => { state.searchDubs = e.target.value; renderDubs(); };

  const drop = $('drop'), file = $('file');
  drop.onclick = () => file.click();
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
  drop.ondragleave = () => drop.classList.remove('over');
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove('over'); if (e.dataTransfer.files[0]) { chosen = e.dataTransfer.files[0]; showFile(); } };
  file.onchange = () => { chosen = file.files[0]; showFile(); };
  $('uploadBtn').onclick = uploadChosen;
  $('ytGo').onclick = async () => {
    const status = $('ytStatus'); $('ytGo').disabled = true; setStatus(status, 'جارٍ تشغيل الدبلجة من الرابط…', 'info');
    try { await dispatchYoutube($('ytUrl').value.trim()); setStatus(status, 'انطلق التشغيل — تابعه في «التشغيلات»؛ الفيديو المدبلج سيظهر في «المدبلجة» باسم مأخوذ من الملف.', 'ok'); $('ytUrl').value = ''; setTimeout(async () => { await refreshRuns(); renderAll(); }, 4000); }
    catch (e) { setStatus(status, e.message, 'err'); } finally { $('ytGo').disabled = false; }
  };

  $('checkToken').onclick = async () => {
    const t = $('token').value.trim(); if (!t) return setStatus($('tokenStatus'), 'أدخل الرمز أولاً', 'err');
    setStatus($('tokenStatus'), 'جارٍ فحص صلاحية الكتابة…', 'info');
    try { await checkToken(t); saveToken(t); state.token = t; setConnection(); setStatus($('tokenStatus'), 'الرمز صالح للكتابة وتم حفظه في هذا المتصفح', 'ok'); await fullRefresh(true); schedule(); }
    catch (e) { setStatus($('tokenStatus'), e.message, 'err'); }
  };
  $('forgetToken').onclick = () => { forgetToken(); state.token = ''; $('token').value = ''; setConnection(); setStatus($('tokenStatus'), 'تم حذف الرمز من هذا المتصفح', 'ok'); };
  $('saveDefaults').onclick = () => { localStorage.setItem(DEFAULTS_KEY, JSON.stringify(readDefaultsForm())); setStatus($('defaultsStatus'), 'تم الحفظ', 'ok'); };
  $('runPreflight').onclick = async () => {
    try { needToken(); await api(`/repos/${OWNER}/${REPO}/actions/workflows/translation-preflight.yml/dispatches`, { method: 'POST', body: JSON.stringify({ ref: BRANCH, inputs: { source_lang: 'ar', target_lang: $('dTarget').value.trim() || 'en' } }) }); setStatus($('preflightStatus'), 'انطلق الفحص — النتيجة في Actions خلال دقيقة', 'ok'); }
    catch (e) { setStatus($('preflightStatus'), e.message, 'err'); }
  };

  fullRefresh(true).then(schedule);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) { fullRefresh(false); schedule(); } });
}
init();
