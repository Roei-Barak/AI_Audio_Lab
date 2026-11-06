// Connection to the Lyra processing server (server/lyra_server). When /api/health answers, the app runs in
// "server" mode: imports, processing, renders and saves go to the server. Otherwise it stays in demo mode.
import { demoProject } from './demo';
import { store, type RemoteSaveResult } from './store';
import type { Job, Lang, Profile, Project, Res, Source } from './types';

const BASE: string = (import.meta.env?.VITE_LYRA_API as string | undefined) ?? '';
const tokenKey = 'lyra.v1.serverToken';
const token = () => {
  try {
    return localStorage.getItem(tokenKey);
  } catch {
    return null;
  }
};

export class ApiError extends Error {
  constructor(public status: number, message: string, public body?: unknown) {
    super(message);
  }
}

export async function api<T = unknown>(path: string, init: RequestInit = {}, timeoutMs = 15000): Promise<T> {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  const headers: Record<string, string> = { ...(init.headers as Record<string, string>) };
  if (init.body && !(init.body instanceof FormData)) headers['Content-Type'] = 'application/json';
  const tk = token();
  if (tk) headers.Authorization = 'Bearer ' + tk;
  try {
    const r = await fetch(BASE + path, { ...init, headers, signal: ctl.signal });
    const ct = r.headers.get('content-type') || '';
    const body = ct.includes('json') ? await r.json() : await r.text();
    if (!r.ok) throw new ApiError(r.status, (body as { detail?: string })?.detail || 'שגיאת שרת ' + r.status, body);
    return body as T;
  } finally {
    clearTimeout(t);
  }
}

export const isServer = () => store.get().backend.mode === 'server';
export const fileUrl = (pid: string, name: string, download = false) => BASE + `/api/projects/${pid}/files/${name}` + (download ? '?download=1' : '');

interface ServerJob {
  id: string;
  projectId: string;
  kind: 'process' | 'render';
  name: string;
  profile: Profile;
  stems: 2 | 4;
  status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
  stage: number;
  pct: number;
  attempt: number;
  error: string | null;
  crash: boolean;
  log: string[];
  estimate?: boolean;
  createdAt: number;
  startedAt: number | null;
  endedAt: number | null;
}

const toJob = (j: ServerJob): Job => ({
  id: j.id,
  serverId: j.id,
  kind: j.kind,
  name: j.name,
  profile: j.profile,
  stems: j.stems,
  status: j.status === 'queued' ? 'running' : j.status,
  stage: Math.min(j.stage, 5),
  pct: j.status === 'done' ? 100 : j.pct,
  attempt: j.attempt,
  startedAt: (j.startedAt || j.createdAt) * 1000,
  endedAt: j.endedAt ? j.endedAt * 1000 : undefined,
  error: j.error,
  crash: j.crash,
  log: j.log,
  estimate: j.estimate,
});

// ---------- health ----------
async function health() {
  try {
    const h = await api<{ engine: 'ai' | 'mock'; gpu: { name: string | null; vramGB: number | null; device: string }; host: string }>('/api/health', {}, 4000);
    store.setBackend({ mode: 'server', reachable: true, engine: h.engine, gpu: h.gpu, host: h.host });
  } catch {
    store.setBackend({ reachable: false });
  }
}

// ---------- saves ----------
store.setRemoteSaver(async (p: Project, force: boolean): Promise<RemoteSaveResult> => {
  try {
    const r = await api<Project>(`/api/projects/${p.id}`, { method: 'PUT', body: JSON.stringify({ baseRev: p.serverRev, project: p, force }) });
    return { ok: true, serverRev: r.serverRev! };
  } catch (e) {
    if (e instanceof ApiError && e.status === 409) return { conflict: (e.body as { current: Project }).current };
    if (e instanceof ApiError) return { error: e.message };
    return { offline: true };
  }
});

// ---------- media ----------
function attachMedia(p: Project) {
  if (p.serverRev == null || !p.processed) return;
  store.setMediaRemote({
    url: p.source.hasVideo ? fileUrl(p.id, 'source') : '',
    type: p.source.hasVideo ? 'video/mp4' : 'audio/wav',
    name: p.source.name,
    audioUrl: fileUrl(p.id, 'instrumental'),
    remote: true,
  });
}

// ---------- jobs ----------
let pollTimer: ReturnType<typeof setInterval> | undefined;
function poll() {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    const j = store.get().job;
    if (!j?.serverId) return clearInterval(pollTimer);
    try {
      const sj = await api<ServerJob>('/api/jobs/' + j.serverId, {}, 5000);
      store.setBackend({ reachable: true });
      const nj = toJob(sj);
      store.setJob(nj);
      if (nj.status !== 'running') {
        clearInterval(pollTimer);
        if (nj.status === 'done') await refreshProject(sj.projectId, sj.kind === 'process');
      }
    } catch (e) {
      if (!(e instanceof ApiError)) store.setBackend({ reachable: false });
    }
  }, 1000);
}

/** Pulls the server copy after a job. Local unsaved edits win; server-only fields are merged. */
async function refreshProject(pid: string, fresh: boolean) {
  const sp = await api<Project>('/api/projects/' + pid);
  const S = store.get();
  if (S.project?.id !== pid) return;
  const local = S.project;
  const clean = S.save.state === 'saved' || S.save.state === 'idle';
  if (fresh && !local.lines.some((l) => l.manual)) store.adoptProject(sp, { resetHistory: true });
  else if (clean) store.adoptProject(sp);
  else {
    store.adoptProject({ ...local, serverRev: sp.serverRev, renderedRev: sp.renderedRev, suggestion: sp.suggestion, source: sp.source, stems: sp.stems, processed: sp.processed });
    store.saveNow(); // push the edits made while the server was busy
  }
  attachMedia(store.get().project!);
}

export async function startProcessing(source: Source, cfg: { profile: Profile; stems: 2 | 4; language: Lang; res: Res }, file?: File, simulateCrash = false) {
  const doc = demoProject(source);
  doc.processing = { profile: cfg.profile, stems: cfg.stems, language: cfg.language };
  doc.export.res = cfg.res;
  doc.lines = [];
  doc.history = [{ at: Date.now(), label: 'פרויקט נוצר' }];
  if (source.kind === 'youtube') {
    const info = await api<{ title: string; artist: string; duration: number }>('/api/sources/youtube', { method: 'POST', body: JSON.stringify({ url: source.url }) }, 30000);
    doc.title = info.title;
    doc.artist = info.artist;
    doc.source = { ...source, name: info.title, duration: info.duration || source.duration };
  } else {
    doc.title = source.name.replace(/\.[^.]+$/, '');
    doc.artist = '';
  }
  let p = await api<Project>('/api/projects', { method: 'POST', body: JSON.stringify({ project: doc, source: doc.source }) });
  if (file) {
    const fd = new FormData();
    fd.append('file', file);
    p = await api<Project>(`/api/projects/${p.id}/upload`, { method: 'POST', body: fd }, 24 * 3600 * 1000);
  }
  store.adoptProject(p, { resetHistory: true });
  const sj = await api<ServerJob>('/api/jobs', { method: 'POST', body: JSON.stringify({ projectId: p.id, kind: 'process', profile: cfg.profile, stems: cfg.stems, language: cfg.language, simulateCrash }) });
  store.setJob(toJob(sj));
  poll();
  return p;
}

export async function submitJob(kind: 'process' | 'render', opts: { profile?: Profile; stems?: 2 | 4; language?: Lang; rerun?: boolean } = {}) {
  const p = store.get().project;
  if (!p || p.serverRev == null) throw new Error('הפרויקט לא נמצא בשרת');
  if (opts.rerun) store.edit('הרצת AI מחדש כהצעה', (P) => void (P.rerun = true));
  await flushSave();
  const sj = await api<ServerJob>('/api/jobs', {
    method: 'POST',
    body: JSON.stringify({ projectId: p.id, kind, profile: opts.profile ?? p.processing.profile, stems: opts.stems ?? p.processing.stems, language: opts.language ?? p.processing.language }),
  });
  store.setJob(toJob(sj));
  poll();
}

export async function jobAction(action: 'cancel' | 'retry' | 'retry-fast' | 'remove') {
  const j = store.get().job;
  if (!j?.serverId) return;
  if (action === 'remove') {
    await api('/api/jobs/' + j.serverId, { method: 'DELETE' });
    store.setJob(null);
    return;
  }
  const sj = await api<ServerJob>(`/api/jobs/${j.serverId}/${action === 'cancel' ? 'cancel' : 'retry'}`, { method: 'POST', body: JSON.stringify({ fast: action === 'retry-fast' }) });
  store.setJob(toJob(sj));
  poll();
}

store.setJobHook(jobAction);

export async function listJobs(): Promise<Job[]> {
  return (await api<ServerJob[]>('/api/jobs')).map(toJob);
}

/** Waits for the debounced save to reach the server. */
export async function flushSave() {
  const s0 = store.get().save;
  if (s0.state !== 'saving' && s0.pending === 0 && s0.state !== 'local' && s0.state !== 'failed') return s0.state;
  if (s0.state !== 'saving') store.saveNow();
  for (let i = 0; i < 60; i++) {
    const st = store.get().save.state;
    if (st !== 'saving') return st;
    await new Promise((r) => setTimeout(r, 100));
  }
  return store.get().save.state;
}

export async function listProjects() {
  return api<{ id: string; title: string; artist: string; lines: number; updatedAt: number }[]>('/api/projects');
}

export async function openProject(pid: string) {
  const p = await api<Project>('/api/projects/' + pid);
  store.adoptProject(p, { resetHistory: true });
  attachMedia(p);
  const jobs = await api<ServerJob[]>('/api/jobs');
  const mine = jobs.filter((j) => j.projectId === pid).pop();
  store.setJob(mine ? toJob(mine) : null);
  if (mine && (mine.status === 'running' || mine.status === 'queued')) poll();
}

export async function deleteRemoteProject(pid: string) {
  await api('/api/projects/' + pid, { method: 'DELETE' });
}

/** MP4 is fresh when the last render matches the current saved revision. */
export const renderFresh = (p: Project) => p.renderedRev != null && p.renderedRev === p.serverRev;

// boot: detect the server, resume polling an active job, re-attach media
if (typeof window !== 'undefined' && !import.meta.env?.VITEST) {
  void health().then(() => {
    const S = store.get();
    if (!isServer()) return;
    if (S.project?.serverRev != null) {
      attachMedia(S.project);
      if (S.job?.serverId) poll();
    }
  });
  // poll every 5s while connected; back off to 30s when no server answers (demo mode)
  const loop = () => setTimeout(async () => {
    await health();
    loop();
  }, store.get().backend.reachable ? 5000 : 30000);
  loop();
}
