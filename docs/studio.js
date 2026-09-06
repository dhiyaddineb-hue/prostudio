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
  filterLibrary: 'all', searchLibrary: '', searchDubs: '', currentProjectSlug: '',
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
    voice: d.voice, tts_engine: d.tts_engine, target_lang: d.target_lang, gender: d.gender, model: d.model,
    bg_music: String(!!d.bg_music), diarize: String(!!d.diarize), separate_sources: String(!!d.separate_sources), no_vad: 'false',
    seed_vc: String(!!d.seed_vc), profile: d.profile, quality: d.quality,
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
        <button class="btn small primary act-project">فتح المشروع</button>
        ${it.whole ? `<button class="btn small act-preview">معاينة</button>` : ''}
        <button class="btn small act-dub" ${st.key === 'running' ? 'disabled' : ''}>${st.key === 'dubbed' ? 'دبلجة نسخة جديدة' : 'دبلجة'}</button>
        ${st.key === 'dubbed' ? `<button class="btn small act-godubs">النسخ المدبلجة</button>` : ''}
        ${st.key === 'running' ? `<button class="btn small act-goruns">متابعة</button>` : ''}
        <button class="btn small act-voices">الأصوات${it.voicesJson ? ' ✓' : ''}</button>
        <button class="btn small act-checkpoints">نقاط الاستئناف</button>
        <a class="btn small" href="${REPO_URL}/tree/${encodeURIComponent(BRANCH)}/library/${encodeURIComponent(it.slug)}" target="_blank" rel="noreferrer">GitHub</a>
        <button class="btn small danger act-delete">حذف</button>
      </div></div>`;
    card.querySelector('.act-project').addEventListener('click', () => openProjectWorkspace(it));
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


function renderOverview() {
  if (!$('overviewLibrary')) return;
  const activeRuns = state.runs.filter(isActive);
  const activeSlugs = new Set(activeRuns.map(runSlug).filter(Boolean));
  const dubbedSlugs = new Set(state.dubs.map((d) => d.slug));
  const waiting = state.library.filter((v) => !dubbedSlugs.has(v.slug) && !activeSlugs.has(v.slug)).length;
  $('overviewLibrary').textContent = state.library.length;
  $('overviewUndubbed').textContent = waiting;
  $('overviewDubbed').textContent = state.dubs.length;
  $('overviewActive').textContent = activeRuns.length;
  $('overviewVoices').textContent = state.voiceBank.length;
  $('overviewHeroValue').textContent = state.library.length;
  $('overviewHeroLabel').textContent = state.library.length === 1 ? 'مشروع' : 'مشاريع';

  const projects = $('overviewProjects');
  projects.innerHTML = state.library.length ? state.library.map((item) => {
    const status = libraryStatusFor(item); const run = status.run; const progress = run?._progress;
    const percent = progress?.total ? Math.round(((progress.counts.completed || 0) * 100) / progress.total) : (status.key === 'dubbed' ? 100 : 0);
    const next = status.key === 'running' ? 'متابعة العمل' : status.key === 'dubbed' ? 'فتح ومراجعة' : item.voicesJson ? 'بدء الدبلجة' : 'إعداد الشخصيات';
    return `<article class="project-row ${status.key}" data-slug="${esc(item.slug)}">
      <button class="project-main act-open-project" data-slug="${esc(item.slug)}"><span class="project-state-dot"></span><span><b>${esc(item.title)}</b><small class="mono">${esc(item.slug)}</small></span></button>
      <div class="project-path"><span>${item.voicesJson ? 'الأصوات جاهزة' : 'الأصوات غير معتمدة'}</span><span>${state.dubs.find((d) => d.slug === item.slug)?.versions.length || 0} نسخة</span><span>${fmtDur(item.meta.duration)}</span></div>
      <div class="project-progress"><span><i style="width:${percent}%"></i></span><small>${status.key === 'running' ? `${percent}%` : esc(status.label)}</small></div>
      <button class="btn small primary act-open-project" data-slug="${esc(item.slug)}">${next}</button>
    </article>`;
  }).join('') : `<div class="empty">${state.loaded ? 'لا توجد مشاريع. أنشئ أول مشروع من فيديو أو رابط يوتيوب.' : 'جارٍ تحميل المشاريع…'}</div>`;
  projects.querySelectorAll('.act-open-project').forEach((button) => button.addEventListener('click', () => {
    const item = state.library.find((value) => value.slug === button.dataset.slug); if (item) openProjectWorkspace(item);
  }));

  const shownRuns = state.runs.slice(0, 4);
  $('overviewRunsList').innerHTML = shownRuns.length ? shownRuns.map((run) => {
    const prog = run._progress; const percent = prog?.total ? Math.round(((prog.counts.completed || 0) * 100) / prog.total) : (run.conclusion === 'success' ? 100 : 0); const active = isActive(run);
    return `<div class="overview-item"><div><b>${esc(runSource(run) || run.display_title || `تشغيل #${run.run_number}`)}</b><small>${esc(statusLabel(run))} · ${fmtDate(run.run_started_at || run.created_at)}</small>${active || percent ? `<div class="mini-progress"><span style="width:${percent}%"></span></div>` : ''}</div><a class="overview-item-status ${active ? 'active' : ''}" href="${run.html_url}" target="_blank" rel="noreferrer">${active ? `${percent}% · مباشر` : `#${run.run_number}`}</a></div>`;
  }).join('') : `<div class="empty compact-empty">${state.loaded ? 'لا توجد تشغيلات.' : 'جارٍ التحميل…'}</div>`;
  const recent = state.dubs.map((d) => ({ ...d, stamp: d.latest?.published_at || d.meta?.updated_at || '' })).sort((a, b) => String(b.stamp).localeCompare(String(a.stamp))).slice(0, 4);
  $('overviewRecentList').innerHTML = recent.length ? recent.map((d) => `<div class="overview-item"><div><b>${esc(d.meta?.title || d.slug)}</b><small>${d.versions.length} نسخة · ${fmtDate(d.stamp)}</small></div><button class="btn small overview-open-dub" data-slug="${esc(d.slug)}">فتح</button></div>`).join('') : `<div class="empty compact-empty">${state.loaded ? 'لا توجد نتائج بعد.' : 'جارٍ التحميل…'}</div>`;
  $('overviewRecentList').querySelectorAll('.overview-open-dub').forEach((button) => button.addEventListener('click', () => { state.searchDubs = button.dataset.slug; $('searchDubs').value = button.dataset.slug; showTab('dubs'); renderDubs(); }));
}

function renderAll() { renderLibrary(); renderDubs(); renderRuns(); renderVoiceBank(); renderOverview(); $('clockChip').textContent = `آخر تحديث ${new Date().toLocaleTimeString('ar-EG', { hour12: false })}`; }

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
    task: 'dub', source_path: item.sourcePath, youtube_url: '', source_lang: d.source_lang || item.meta.source_lang || 'ar',
    voice: d.voice, tts_engine: d.tts_engine, target_lang: d.target_lang, gender: d.gender, model: d.model,
    bg_music: String(!!d.bg_music), diarize: String(!!d.diarize), separate_sources: String(!!d.separate_sources), no_vad: String(!!d.no_vad),
    seed_vc: String(!!d.seed_vc), profile: d.profile, quality: d.quality,
    chunk_seconds: String(d.chunk_seconds), speaker_voices_path: d.speaker_voices_path || '', validate_content: String(!!d.validate_content),
    analysis_only: String(!!d.analysis_only),
  };
  await api(`/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW}/dispatches`, { method: 'POST', body: JSON.stringify({ ref: BRANCH, inputs }) });
}
function resumeOverrides(item, manifest) {
  const config = manifest?.config || {};
  return {
    source_lang: config.source_lang || item.meta.source_lang || 'ar',
    target_lang: config.target_lang || defaults().target_lang,
    tts_engine: config.tts_engine || defaults().tts_engine,
    voice: config.voice || defaults().voice,
    gender: config.gender || defaults().gender,
    model: config.model || defaults().model,
    separate_sources: config.separate_sources ?? defaults().separate_sources,
    bg_music: config.preserve_background ?? defaults().bg_music,
    diarize: config.diarize ?? defaults().diarize,
    no_vad: config.no_vad ?? false,
    seed_vc: config.seed_vc ?? defaults().seed_vc,
    chunk_seconds: config.max_seconds || defaults().chunk_seconds,
    speaker_voices_path: item.voicesJson ? `library/${item.slug}/voices.json` : '',
    validate_content: true,
    profile: manifest?.seed_quota_policy === 'voxcpm' ? 'seed_quota_voxcpm' : defaults().profile,
    quality: defaults().quality,
  };
}
async function resumeProject(item, manifest) {
  await dispatchDub(item, resumeOverrides(item, manifest));
}
async function openDubDialog(item) {
  state.currentProjectSlug = item.slug;
  $('dubTitle').textContent = `دبلجة ${item.title}`;
  $('dubBody').innerHTML = '<div class="empty">جارٍ تحميل إعدادات المشروع والشخصيات…</div>';
  showTab('dub');
  try {
    const d = defaults(); const release = await releaseFor(item.slug); const manifest = release ? await manifestOf(release) : null;
    const doc = item.voicesJson ? (await blobJson(item.voicesJson.sha)) || {} : {};
    const detected = [...new Set([...(manifest?.detected_speakers || []), ...Object.keys(manifest?.voice_profiles || {}), ...(manifest?.chunks || []).map((chunk) => chunk.speaker).filter(Boolean)])];
    const speakerIds = detected.length ? detected : ['SPEAKER_00'];
    const profiles = {};
    for (const speaker of speakerIds) profiles[speaker] = { ...defaultProfile(speaker), ...(doc.speakers?.[speaker] || {}), speaker };
    const analysisReady = detected.length > 0 || manifest?.state === 'analysis_completed_waiting_for_voice_approval';
    $('dubBody').innerHTML = `<div class="dub-config">
      <div class="dub-steps"><div class="done"><i>1</i><span>المصدر</span></div><div class="current"><i>2</i><span>الإعداد والصوت</span></div><div><i>3</i><span>التشغيل</span></div><div><i>4</i><span>المراجعة</span></div></div>
      <section class="surface dub-section"><div class="surface-head"><div><p class="eyebrow">الأساسيات</p><h3>ماذا تريد أن تنتج؟</h3></div><span class="chip ${analysisReady ? 'ok' : 'warn'}">${analysisReady ? `${speakerIds.length} شخصية مكتشفة` : 'لم تُحلل الشخصيات بعد'}</span></div>
        <div class="dub-form-grid">
          <label>لغة المصدر <input id="xSource" value="${esc(manifest?.config?.source_lang || item.meta.source_lang || 'ar')}" dir="ltr"></label>
          <label>لغة الدبلجة <input id="xTarget" value="${esc(manifest?.config?.target_lang || d.target_lang)}" dir="ltr"></label>
          <label>محرك الصوت الافتراضي <select id="xEngine"><option value="voxcpm">VoxCPM</option><option value="xtts">XTTS</option><option value="qwen">Qwen</option><option value="edge">Edge-TTS</option></select></label>
          <label>الصوت الجاهز الافتراضي <input id="xVoice" value="${esc(manifest?.config?.voice || d.voice)}" dir="ltr"></label>
          <label>مستوى الجودة <select id="xQuality"><option value="balanced">متوازن</option><option value="strict">صارم</option><option value="safe">آمن</option></select></label>
          <label>نموذج التفريغ <select id="xModel"><option value="medium">Medium — أدق</option><option value="small">Small — أسرع</option><option value="base">Base</option><option value="tiny">Tiny</option></select></label>
        </div>
      </section>
      <section class="surface dub-section"><div class="surface-head"><div><p class="eyebrow">الشخصيات</p><h3>صوت مستقل لكل شخصية</h3><p class="hint">اختر المصدر والمحرك وتحويل Seed‑VC والأسلوب، ثم اعتمد كل شخصية.</p></div>${!analysisReady ? '<button id="xAnalyzeTop" class="btn primary">كشف الشخصيات أولاً</button>' : ''}</div>
        <div id="dubCharacters" class="character-grid">${speakerIds.map((speaker) => speakerCard(speaker, profiles[speaker], item)).join('')}</div>
      </section>
      <section class="surface dub-section"><div class="surface-head"><div><p class="eyebrow">الجودة والخلفية</p><h3>خيارات التنفيذ</h3></div></div>
        <div class="option-grid">
          <label class="option-toggle"><input type="checkbox" id="xSeparate" ${manifest?.config?.separate_sources ?? d.separate_sources ? 'checked' : ''}><span><b>فصل الحوار عن الخلفية</b><small>يحافظ على الموسيقى والمؤثرات بعيداً عن الكلام الأصلي.</small></span></label>
          <label class="option-toggle"><input type="checkbox" id="xBg" ${manifest?.config?.preserve_background ?? d.bg_music ? 'checked' : ''}><span><b>الاحتفاظ بالموسيقى والمؤثرات</b><small>يمزج الخلفية مع الدبلجة النهائية.</small></span></label>
          <label class="option-toggle"><input type="checkbox" id="xDiarize" ${manifest?.config?.diarize ?? d.diarize ? 'checked' : ''}><span><b>تمييز المتحدثين</b><small>مطلوب للفيديو متعدد الشخصيات.</small></span></label>
          <label class="option-toggle"><input type="checkbox" id="xValidate" ${d.validate_content ? 'checked' : ''}><span><b>التحقق من عدم فقدان الكلام</b><small>يعيد فحص كل عبارة بعد توليدها.</small></span></label>
        </div>
        <details class="advanced-options"><summary>خيارات متقدمة</summary><div class="dub-form-grid">
          <label>ملف التشغيل <select id="xProfile"><option value="seed_quota_voxcpm">Seed‑VC ثم VoxCPM عند نفاد الحصة</option><option value="safe">آمن</option><option value="high_quality_single">جودة عالية — متحدث واحد</option><option value="multi_speaker_cinematic">سينمائي متعدد الشخصيات</option></select></label>
          <label>الحد الأقصى للمقطع <select id="xChunk"><option value="10">10 ثوانٍ</option><option value="8">8 ثوانٍ</option><option value="6">6 ثوانٍ</option><option value="4">4 ثوانٍ</option></select></label>
          <label>الجنس الافتراضي <select id="xGender"><option value="male">ذكر</option><option value="female">أنثى</option></select></label>
          <label class="option-toggle compact-option"><input type="checkbox" id="xNoVad"><span><b>تعطيل VAD</b><small>للكلام المتواصل الذي يُسقطه الكشف.</small></span></label>
        </div></details>
      </section>
      <div class="dub-submit"><div><b>${analysisReady ? 'الإعداد جاهز للمراجعة' : 'ابدأ بتحليل الشخصيات إذا كان الفيديو متعدد المتحدثين'}</b><small>كل النتائج الناجحة تحفظ في نقاط الاستئناف، والفاشل فقط يعاد.</small></div><div><button id="xAnalyze" class="btn">تحليل الشخصيات فقط</button><button id="xGo" class="btn primary">حفظ الأصوات وبدء الدبلجة</button></div></div>
      <div id="xStatus" class="status"></div>
    </div>`;
    $('xEngine').value = manifest?.config?.tts_engine || d.tts_engine; $('xQuality').value = d.quality; $('xModel').value = manifest?.config?.model || d.model; $('xProfile').value = d.profile; $('xChunk').value = String(manifest?.config?.max_seconds || d.chunk_seconds); $('xGender').value = manifest?.config?.gender || d.gender;
    const characterRoot = $('dubCharacters');
    characterRoot.querySelectorAll('.act-rmspeaker').forEach((button) => button.remove());
    characterRoot.querySelectorAll('.bankpick').forEach((select) => select.addEventListener('change', () => { const card = select.closest('.character'); if (select.value) { card.querySelector('[data-key=reference_path]').value = select.value; card.querySelector('[data-key=reference_mode]').value = 'custom'; } }));
    characterRoot.querySelectorAll('.samplefile').forEach((input) => input.addEventListener('change', async () => {
      const card = input.closest('.character'); const file = input.files[0]; const preview = card.querySelector('.local-sample-preview'); if (!file) return;
      const ext = (file.name.match(/\.[^.]+$/) || ['.wav'])[0].toLowerCase(); const stamp = new Date().toISOString().replace(/[-:TZ.]/g, '').slice(0, 12);
      card.querySelector('[data-key=reference_path]').value = `voices/${card.dataset.speaker}-${stamp}${ext}`; card.querySelector('[data-key=reference_mode]').value = 'custom';
      try { const duration = await probeAudioDuration(file); input.dataset.duration = String(duration); const ok = duration >= 6 && duration <= 20; const url = URL.createObjectURL(file); preview.innerHTML = `<audio controls src="${url}"></audio><span class="status ${ok ? 'ok' : 'err'}">${duration.toFixed(1)} ثانية · ${ok ? 'صالحة' : 'المطلوب 6–20 ثانية'}</span>`; preview.classList.remove('hidden'); }
      catch (error) { preview.innerHTML = `<span class="status err">${esc(error.message)}</span>`; preview.classList.remove('hidden'); }
    }));
    const values = () => ({ source_lang: $('xSource').value.trim() || 'ar', target_lang: $('xTarget').value.trim() || 'en', tts_engine: $('xEngine').value, voice: $('xVoice').value.trim(), quality: $('xQuality').value, model: $('xModel').value, gender: $('xGender').value, profile: $('xProfile').value, chunk_seconds: $('xChunk').value, separate_sources: $('xSeparate').checked, bg_music: $('xBg').checked, diarize: $('xDiarize').checked, no_vad: $('xNoVad').checked, seed_vc: [...characterRoot.querySelectorAll('[data-key=voice_conversion]')].some((select) => select.value === 'seed-vc'), validate_content: $('xValidate').checked });
    const analyze = async () => { const status = $('xStatus'); setStatus(status, 'جارٍ تشغيل تحليل المصدر والشخصيات فقط…', 'info'); $('xAnalyze').disabled = true; $('xAnalyzeTop')?.setAttribute('disabled', ''); try { await dispatchDub(item, { ...values(), analysis_only: true, speaker_voices_path: '' }); setStatus(status, 'انطلق التحليل. عند انتهائه افتح المشروع ثم عد إلى إعداد الدبلجة لتظهر الشخصيات.', 'ok'); setTimeout(async () => { await refreshRuns(); renderAll(); showTab('runs'); }, 1400); } catch (error) { setStatus(status, error.message, 'err'); $('xAnalyze').disabled = false; $('xAnalyzeTop')?.removeAttribute('disabled'); } };
    $('xAnalyze').onclick = analyze; $('xAnalyzeTop')?.addEventListener('click', analyze);
    $('xGo').onclick = async () => {
      const status = $('xStatus'); const config = values(); if (config.diarize && !analysisReady) return setStatus(status, 'الفيديو متعدد الشخصيات: نفّذ «تحليل الشخصيات فقط» أولاً، ثم عيّن صوت كل شخصية.', 'err');
      $('xGo').disabled = true;
      try {
        needToken(); const speakers = {}; const additions = [];
        for (const card of characterRoot.querySelectorAll('.character')) {
          const speaker = card.dataset.speaker; const data = { speaker };
          for (const element of card.querySelectorAll('[data-key]')) data[element.dataset.key] = element.type === 'checkbox' ? element.checked : element.value.trim();
          const file = card.querySelector('.samplefile').files[0];
          if (file) { if (file.size > PART_BYTES) throw new Error(`${speaker}: العينة أكبر من 18 MB`); const duration = Number(card.querySelector('.samplefile').dataset.duration) || await probeAudioDuration(file); if (duration < 6 || duration > 20) throw new Error(`${speaker}: يجب أن تكون العينة بين 6 و20 ثانية`); data.reference_duration = Math.round(duration * 100) / 100; additions.push({ path: `library/${item.slug}/${data.reference_path}`, sha: await createBlob(await toBase64(file), 'base64') }); }
          if (data.reference_mode === 'custom' && !data.reference_path) throw new Error(`${speaker}: اختر أو ارفع عينة صوتية`); if (data.reference_mode === 'synthetic' && data.tts_engine === 'edge' && !data.voice) throw new Error(`${speaker}: أدخل اسم صوت Edge`); if (!data.approved) throw new Error(`${speaker}: اعتمد إعداد الشخصية قبل البدء`); speakers[speaker] = data;
        }
        setStatus(status, 'حفظ إعدادات الشخصيات…', 'info'); additions.push({ path: `library/${item.slug}/voices.json`, sha: await createBlob(JSON.stringify({ version: 1, updated_at: new Date().toISOString(), speakers }, null, 2) + '\n', 'utf-8') }); await commitChanges({ message: `Approve dubbing voices for ${item.slug}`, additions });
        setStatus(status, 'تشغيل الدبلجة…', 'info'); await dispatchDub(item, { ...config, analysis_only: false, speaker_voices_path: `library/${item.slug}/voices.json` }); setStatus(status, 'انطلقت الدبلجة. تابع المقاطع من صفحة التشغيلات أو مساحة المشروع.', 'ok'); setTimeout(async () => { await refreshRuns(); renderAll(); showTab('runs'); }, 1400);
      } catch (error) { setStatus(status, error.message, 'err'); $('xGo').disabled = false; }
    };
  } catch (error) { $('dubBody').innerHTML = `<div class="empty"><span class="status err">${esc(error.message)}</span></div>`; }
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
  const sample = item.voiceSamples.find((f) => p.reference_path && f.name === p.reference_path) || item.voiceSamples.find((f) => f.name.replace(/^voices\//, '').replace(/\.[^.]+$/, '') === speaker);
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
      <div class="wide local-sample-preview hidden"></div>
      ${sample ? `<div class="wide saved-sample"><span class="hint">العينة المحفوظة</span><audio controls preload="none" src="${rawUrl(sample.path)}"></audio><span class="hint mono">${esc(sample.name)}</span></div>` : ''}
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
    $('modalBody').querySelectorAll('.samplefile').forEach((inp) => inp.addEventListener('change', async () => {
      const card = inp.closest('.character'); const file = inp.files[0]; const preview = card.querySelector('.local-sample-preview');
      if (!file) { preview.classList.add('hidden'); return; }
      const ext = (file.name.match(/\.[^.]+$/) || ['.wav'])[0].toLowerCase();
      const stamp = new Date().toISOString().replace(/[-:TZ.]/g, '').slice(0, 12);
      card.querySelector('[data-key=reference_path]').value = `voices/${card.dataset.speaker}-${stamp}${ext}`;
      card.querySelector('[data-key=reference_mode]').value = 'custom';
      try {
        const duration = await probeAudioDuration(file); inp.dataset.duration = String(duration);
        const ok = duration >= 6 && duration <= 20; const url = URL.createObjectURL(file);
        preview.innerHTML = `<audio controls preload="metadata" src="${url}"></audio><span class="status ${ok ? 'ok' : 'err'}">${esc(file.name)} · ${duration.toFixed(1)} ثانية${ok ? ' · صالحة مبدئياً' : ' · يجب أن تكون بين 6 و20 ثانية'}</span>`;
        preview.classList.remove('hidden');
        if (!ok) setStatus($('voicesStatus'), `${card.dataset.speaker}: مدة العينة ${duration.toFixed(1)} ثانية؛ المطلوب 6–20 ثانية.`, 'err');
      } catch (error) { inp.dataset.duration = ''; preview.innerHTML = `<span class="status err">تعذر فحص الملف: ${esc(error.message)}</span>`; preview.classList.remove('hidden'); }
    }));
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
            const duration = Number(card.querySelector('.samplefile').dataset.duration) || await probeAudioDuration(file);
            if (duration < 6 || duration > 20) throw new Error(`${s}: مدة العينة ${duration.toFixed(1)} ثانية؛ يجب أن تكون بين 6 و20 ثانية`);
            data.reference_duration = Math.round(duration * 100) / 100;
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


async function openProjectWorkspace(item) {
  state.currentProjectSlug = item.slug;
  $('projectTitle').textContent = item.title;
  $('projectBody').innerHTML = '<div class="empty">جارٍ تحميل المشروع ونقاط الاستئناف…</div>';
  showTab('project');
  try {
    const release = await releaseFor(item.slug);
    const manifest = release ? await manifestOf(release) : null;
    const chunks = manifest?.chunks || [];
    const completed = chunks.filter((c) => c.status === 'completed').length;
    const failed = chunks.filter((c) => c.status === 'failed' || Object.values(c.checklist || {}).some((s) => s.state === 'failed')).length;
    const pending = Math.max(0, chunks.length - completed - failed);
    const progress = chunks.length ? Math.round((completed * 100) / chunks.length) : 0;
    const speakers = [...new Set(chunks.map((c) => c.speaker || 'SPEAKER_00'))];
    const dub = state.dubs.find((d) => d.slug === item.slug); const latest = dub?.latest;
    const active = state.runs.find((r) => isActive(r) && runSlug(r) === item.slug);
    const stateText = active ? 'تعمل الدبلجة الآن' : manifest?.state === 'failed_resumable' ? 'متوقف وقابل للاستئناف' : manifest?.state === 'completed_waiting_for_cleanup_approval' ? 'مكتمل وجاهز للمراجعة' : release ? 'محفوظ في نقطة استئناف' : 'مشروع جديد';
    const stageTotals = {};
    for (const name of STAGE_ORDER) stageTotals[name] = { success: 0, failed: 0, pending: 0, skipped: 0 };
    for (const chunk of chunks) for (const name of STAGE_ORDER) { const value = chunk.checklist?.[name]?.state || 'pending'; stageTotals[name][value] = (stageTotals[name][value] || 0) + 1; }
    const stageDone = (name) => chunks.length && (stageTotals[name].success + stageTotals[name].skipped === chunks.length);
    const flow = [
      ['المصدر', true], ['الشخصيات', !!item.voicesJson], ['التحليل', chunks.length > 0], ['الصوت', stageDone('tts')], ['المراجعة', stageDone('content_validation')], ['التصدير', !!latest],
    ];
    const nextAction = active ? 'متابعة التشغيل الحالي' : !item.voicesJson ? 'إعداد الشخصيات والأصوات' : !manifest ? 'بدء الدبلجة' : completed < chunks.length ? 'استئناف المقاطع غير المكتملة' : !latest ? 'إعادة فتح التشغيل والنشر' : 'مراجعة النسخة النهائية';
    const rows = chunks.map((chunk) => `<tr class="${chunk.status === 'failed' ? 'failed-row' : ''}"><td class="mono">${String(chunk.index).padStart(4, '0')}</td><td class="mono">${fmtDur(+chunk.start)}–${fmtDur(+chunk.end)}</td><td>${esc(chunk.speaker || 'SPEAKER_00')}</td><td>${esc(chunk.status || 'pending')}</td>${STAGE_ORDER.map((name) => { const value = chunk.checklist?.[name]?.state || 'pending'; return `<td class="stage-${value}" title="${esc(chunk.checklist?.[name]?.error || '')}"><span class="stage-dot"></span></td>`; }).join('')}<td>${release ? `<button class="btn small project-audio" data-index="${chunk.index}">استماع</button>` : ''}</td></tr>`).join('');
    const voiceProfiles = manifest?.voice_profiles || {};
    const characters = speakers.length ? speakers.map((speaker) => { const profile = voiceProfiles[speaker] || {}; return `<article class="project-character"><div><span class="avatar">${esc((profile.label || speaker).slice(0, 2))}</span><div><b>${esc(profile.label || speaker)}</b><small class="mono">${esc(speaker)}</small></div></div><dl><dt>المصدر</dt><dd>${esc(profile.reference_mode || 'source')}</dd><dt>المحرك</dt><dd>${esc(profile.tts_engine || manifest?.config?.tts_engine || '—')}</dd><dt>التحويل</dt><dd>${esc(profile.voice_conversion || '—')}</dd></dl></article>`; }).join('') : '<div class="empty compact-empty">لم يبدأ تحليل الشخصيات بعد. يمكنك إعداد متحدث افتراضي الآن.</div>';
    const errors = manifest?.errors || [];
    $('projectBody').innerHTML = `<div class="project-screen">
      <div class="project-command"><div><span class="project-status ${active ? 'active' : failed ? 'failed' : completed && chunks.length ? 'done' : ''}">${esc(stateText)}</span><h3>${esc(item.title)}</h3><p class="mono">library/${esc(item.slug)}/ · ${fmtDur(item.meta.duration)} · ${formatMB(item.size)} MB</p></div><div class="project-command-actions">${active ? `<a class="btn primary" href="${active.html_url}" target="_blank" rel="noreferrer">فتح التشغيل المباشر</a>` : manifest && completed < chunks.length ? '<button id="projectResume" class="btn primary">استئناف غير المكتمل</button>' : '<button id="projectStart" class="btn primary">بدء الدبلجة</button>'}<button id="projectVoices" class="btn">الشخصيات والأصوات</button><button id="projectMore" class="btn">خيارات المشروع</button></div></div>
      <div class="workflow-track">${flow.map(([label,done],index) => `<div class="workflow-step ${done ? 'done' : index === flow.findIndex((x) => !x[1]) ? 'current' : ''}"><i>${done ? '✓' : index + 1}</i><span>${label}</span></div>`).join('')}</div>
      <div class="next-action"><span>الإجراء التالي</span><b>${nextAction}</b><small>لن يُعاد أي مقطع مكتمل عند الاستئناف.</small></div>
      <div class="project-content-grid"><section class="project-source surface">${item.whole ? `<video controls preload="metadata" src="${rawUrl(item.whole.path)}#t=0.5"></video>` : `<div class="noplay">الفيديو مقسّم إلى ${item.parts.length} أجزاء ويُجمع عند التشغيل.</div>`}<div class="workspace-metrics"><div><strong>${chunks.length}</strong><span>المقاطع</span></div><div class="ok"><strong>${completed}</strong><span>مكتملة</span></div><div class="err"><strong>${failed}</strong><span>فاشلة</span></div><div><strong>${pending}</strong><span>معلّقة</span></div></div></section>
      <section class="surface project-output"><div class="surface-head"><div><p class="eyebrow">المخرج</p><h3>آخر نسخة</h3></div>${latest ? `<button id="projectOutputDetails" class="btn small">التقرير والترجمة</button>` : ''}</div>${latest ? `<video controls preload="metadata" src="${rawUrl(`dubs/${item.slug}/${latest.file}`)}"></video><p class="hint">${esc(latest.file)} · ${latest.quality_ok === false ? 'فشلت الجودة' : latest.quality_ok === true ? 'اجتازت الجودة' : 'بانتظار تقرير الجودة'}</p>` : '<div class="empty compact-empty">لا توجد نسخة مدبلجة منشورة بعد.</div>'}</section></div>
      <section class="surface project-characters"><div class="surface-head"><div><p class="eyebrow">الشخصيات</p><h3>تعيين الأصوات</h3></div><button id="projectVoices2" class="btn small">تعديل الأصوات</button></div><div class="project-character-grid">${characters}</div></section>
      <section class="surface project-checklist"><div class="surface-head"><div><p class="eyebrow">المراجعة التفصيلية</p><h3>Checklist المقاطع</h3></div><div class="legend"><span><i class="stage-dot success"></i> ناجح</span><span><i class="stage-dot failed"></i> فاشل</span><span><i class="stage-dot pending"></i> معلّق</span></div></div>${chunks.length ? `<div class="tablewrap project-table"><table class="segs stage-table"><thead><tr><th>#</th><th>الوقت</th><th>الشخصية</th><th>الحالة</th>${STAGE_ORDER.map((name) => `<th>${STAGE_LABEL[name]}</th>`).join('')}<th>الصوت</th></tr></thead><tbody>${rows}</tbody></table></div><div id="projectAudioCompare" class="compare-audio"></div>` : '<div class="empty compact-empty">تظهر مراحل كل مقطع بعد تشغيل التحليل الأول.</div>'}</section>
      ${errors.length ? `<section class="surface project-errors"><div class="surface-head"><div><p class="eyebrow">التشخيص</p><h3>الأخطاء الأخيرة</h3></div><span class="chip err">${errors.length}</span></div><ul>${errors.slice(-8).reverse().map((error) => `<li><b>المقطع ${error.chunk ?? '—'}</b><span>${esc(error.message)}</span></li>`).join('')}</ul></section>` : ''}
      <div id="projectStatus" class="status"></div>
    </div>`;
    const manageVoices = () => openVoicesDialog(item); $('projectVoices').onclick = manageVoices; $('projectVoices2').onclick = manageVoices;
    $('projectStart')?.addEventListener('click', () => openDubDialog(item));
    $('projectMore').onclick = () => openCheckpointDialog(item);
    $('projectResume')?.addEventListener('click', async () => { const button = $('projectResume'); button.disabled = true; setStatus($('projectStatus'), 'جارٍ إرسال الاستئناف بالإعدادات المحفوظة…', 'info'); try { await resumeProject(item, manifest); setStatus($('projectStatus'), 'انطلق الاستئناف؛ المقاطع المكتملة لن تُعاد.', 'ok'); setTimeout(async () => { await refreshRuns(); renderAll(); showTab('runs'); }, 1400); } catch (error) { setStatus($('projectStatus'), error.message, 'err'); button.disabled = false; } });
    $('projectOutputDetails')?.addEventListener('click', () => openDubDetails(dub, latest));
    $('projectBody').querySelectorAll('.project-audio').forEach((button) => button.addEventListener('click', () => renderAudioCompare(release, +button.dataset.index, 'projectAudioCompare')));
  } catch (error) { $('projectBody').innerHTML = `<div class="empty"><span class="status err">${esc(error.message)}</span></div>`; }
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
      <div class="row" style="margin-top:10px">${(counts.completed || 0) < chunks.length ? '<button id="resumeIncomplete" class="btn primary">استئناف غير المكتمل فقط</button>' : ''}<button id="delCheckpoints" class="btn danger">حذف نقاط الاستئناف (الإصدار)</button><span id="checkpointStatus" class="status"></span></div>
      <p class="hint">الحذف لا يمسّ الفيديو ولا النسخ المدبلجة المنشورة؛ يعيد الدبلجة القادمة من الصفر.</p>`;
    $('modalBody').querySelectorAll('.act-audio').forEach((b) => b.addEventListener('click', () => renderAudioCompare(release, +b.dataset.index)));
    $('resumeIncomplete')?.addEventListener('click', async () => {
      const button = $('resumeIncomplete'); button.disabled = true; setStatus($('checkpointStatus'), 'جارٍ إرسال الاستئناف بالإعدادات المحفوظة…', 'info');
      try { await resumeProject(item, manifest); setStatus($('checkpointStatus'), 'انطلق الاستئناف؛ ستُتخطى المقاطع المكتملة.', 'ok'); setTimeout(async () => { closeModal(); await refreshRuns(); renderAll(); showTab('runs'); }, 1500); }
      catch (error) { setStatus($('checkpointStatus'), error.message, 'err'); button.disabled = false; }
    });
    $('delCheckpoints').onclick = () => confirmTyped({
      title: 'حذف نقاط الاستئناف', message: `سيُحذف الإصدار المسودّ ${esc(release.tag_name)} (${formatMB(size)} MB).`, expect: item.slug,
      onConfirm: async () => { await api(`/repos/${OWNER}/${REPO}/releases/${release.id}`, { method: 'DELETE' }); state.releasesAt = 0; },
    });
  } catch (e) { $('modalBody').innerHTML = `<p class="status err">${esc(e.message)}</p>`; }
}
async function renderAudioCompare(release, index, targetId = 'audioCompare') {
  const box = $(targetId); if (!box) return;
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


let voicePreviewUrl = '';
function probeAudioDuration(file) {
  return new Promise((resolve, reject) => {
    const audio = document.createElement('audio'); const url = URL.createObjectURL(file);
    const done = () => { URL.revokeObjectURL(url); audio.removeAttribute('src'); };
    audio.preload = 'metadata';
    audio.onloadedmetadata = () => { const duration = Number(audio.duration); done(); Number.isFinite(duration) && duration > 0 ? resolve(duration) : reject(new Error('مدة صوت غير صالحة')); };
    audio.onerror = () => { done(); reject(new Error('المتصفح لم يستطع قراءة هذا الملف الصوتي')); };
    audio.src = url;
  });
}
async function previewVoiceFile(file) {
  const panel = $('voicePreviewPanel'); const preview = $('voicePreview');
  if (voicePreviewUrl) URL.revokeObjectURL(voicePreviewUrl);
  if (!file) { voicePreviewUrl = ''; preview.removeAttribute('src'); panel.classList.add('hidden'); return; }
  voicePreviewUrl = URL.createObjectURL(file); preview.src = voicePreviewUrl; $('voicePreviewName').textContent = file.name;
  try {
    const duration = await probeAudioDuration(file); $('voiceFile').dataset.duration = String(duration);
    const ok = duration >= 6 && duration <= 20;
    $('voicePreviewMeta').textContent = `${duration.toFixed(1)} ثانية · ${formatMB(file.size)} MB · ${ok ? 'صالحة مبدئياً' : 'خارج المدة المطلوبة 6–20 ثانية'}`;
    $('voicePreviewMeta').className = `status ${ok ? 'ok' : 'err'}`;
  } catch (error) { $('voiceFile').dataset.duration = ''; $('voicePreviewMeta').textContent = error.message; $('voicePreviewMeta').className = 'status err'; }
  panel.classList.remove('hidden');
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
  let duration;
  try { duration = Number($('voiceFile').dataset.duration) || await probeAudioDuration(file); }
  catch (error) { return setStatus(status, error.message, 'err'); }
  if (duration < 6 || duration > 20) return setStatus(status, `مدة العينة ${duration.toFixed(1)} ثانية؛ يجب أن تكون بين 6 و20 ثانية`, 'err');
  const ext = (file.name.match(/\.[^.]+$/) || ['.wav'])[0].toLowerCase();
  let baseName = slugify($('voiceName').value || file.name) || 'voice';
  let name = baseName + ext; let version = 2;
  while (state.voiceBank.some((sample) => sample.name === name)) name = `${baseName}-${version++}${ext}`;
  $('voiceUpload').disabled = true;
  try {
    setStatus(status, 'رفع العيّنة…', 'info');
    const sha = await createBlob(await toBase64(file), 'base64');
    await commitChanges({ message: `Add voice sample ${name} to the voice bank`, additions: [{ path: `voices/${name}`, sha }] });
    setStatus(status, `تمت الإضافة: voices/${name} · ${duration.toFixed(1)} ثانية`, 'ok'); $('voiceFile').value = ''; $('voiceFile').dataset.duration = ''; $('voiceName').value = ''; previewVoiceFile(null);
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
  document.querySelectorAll('.tab[data-tab]').forEach((t) => {
    const active = t.dataset.tab === name;
    t.classList.toggle('active', active);
    t.setAttribute('aria-selected', String(active));
  });
  document.querySelectorAll('.panel').forEach((p) => p.classList.toggle('active', p.id === `tab-${name}`));
  location.hash = name;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
function setConnection() {
  const chip = $('connChip');
  if (state.token) { chip.textContent = 'متصل بالرمز'; chip.className = 'chip ok'; }
  else { chip.textContent = 'قراءة فقط — بدون رمز'; chip.className = 'chip warn'; }
  const readiness = $('overviewConn');
  if (readiness) {
    readiness.textContent = state.token ? 'متصل وجاهز للرفع والتشغيل والحذف.' : 'وضع القراءة فقط؛ أضف رمز GitHub لتفعيل التحكم.';
    readiness.className = `readiness-state ${state.token ? 'ok' : ''}`;
  }
  document.querySelectorAll('#uploadBtn, #voiceUpload, #ytGo, #runPreflight').forEach((b) => { b.disabled = !state.token; });
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
  document.querySelectorAll('[data-jump]').forEach((button) => button.addEventListener('click', () => {
    if (button.dataset.filter) { state.filterLibrary = button.dataset.filter; $('filterLibrary').value = button.dataset.filter; }
    showTab(button.dataset.jump);
    renderAll();
  }));
  const requireConnection = () => {
    if (state.token) return true;
    showTab('settings');
    setStatus($('tokenStatus'), 'أضف رمز GitHub أولاً لتفعيل التحكم من الصفحة.', 'info');
    setTimeout(() => $('token').focus(), 120);
    return false;
  };
  $('quickUpload').onclick = () => { if (!requireConnection()) return; showTab('library'); $('file').click(); };
  $('libraryUploadShortcut').onclick = () => { if (!requireConnection()) return; $('file').click(); };
  $('quickYoutube').onclick = () => { if (!requireConnection()) return; showTab('library'); setTimeout(() => { $('ytUrl').focus(); $('uploader').scrollIntoView({ behavior: 'smooth' }); }, 120); };
  $('quickVoice').onclick = () => { if (!requireConnection()) return; showTab('voices'); setTimeout(() => $('voiceName').focus(), 120); };
  $('quickRuns').onclick = () => showTab('runs');
  $('overviewRefresh').onclick = () => fullRefresh(true);
  $('projectBack').onclick = () => { state.currentProjectSlug = ''; showTab('overview'); };
  $('dubBack').onclick = () => { const item = state.library.find((value) => value.slug === state.currentProjectSlug); item ? openProjectWorkspace(item) : showTab('overview'); };
  const initial = location.hash.replace('#', ''); if (['overview', 'library', 'dubs', 'runs', 'voices', 'settings'].includes(initial)) showTab(initial); else showTab('overview');
  $('voiceUpload').onclick = uploadVoiceSample;
  $('voiceFile').onchange = () => previewVoiceFile($('voiceFile').files[0]);
  $('modalClose').onclick = closeModal; $('modal').addEventListener('click', (e) => { if (e.target === $('modal')) closeModal(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !$('modal').classList.contains('hidden')) closeModal(); });
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
