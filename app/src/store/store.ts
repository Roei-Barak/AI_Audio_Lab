// Shared project store: ONE project model used by Simple Mode, the lyrics editor, karaoke
// preview and exports. Persists project JSON to localStorage (never media).
// With a Lyra server reachable (backend.mode 'server') saves go to the server with revision-based
// conflict detection and jobs run there; otherwise everything is simulated in the browser ("demo").
import { useSyncExternalStore } from 'react';
import { hhmmss, lineSpan } from '../lib/format';
import { THEIRS_LINE_INDEX, THEIRS_LINE_TEXT, OTHER_USER, demoProject, makeWords } from './demo';
import type { Backend, Job, Media, Mode, Profile, Project, SaveInfo, Sim, Source, StoreState } from './types';

export const KEYS = { project: 'lyra.v1.project', job: 'lyra.v1.job', ui: 'lyra.v1.ui' } as const;

const UNDO_LIMIT = 60;
const HISTORY_LIMIT = 40;

function read<T>(key: string): T | null {
  try {
    return JSON.parse(localStorage.getItem(key) || 'null') as T | null;
  } catch {
    return null;
  }
}
function write(key: string, v: unknown) {
  try {
    if (v === null) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(v));
  } catch {
    /* storage unavailable: state stays in memory */
  }
}

const clone = <T,>(x: T): T => JSON.parse(JSON.stringify(x));

/** Project as it was before a run of live previews; the next edit() is undone back to it. */
let previewBase: Project | null = null;
let undoStack: { p: Project; label: string }[] = [];
let redoStack: { p: Project; label: string }[] = [];

function initialState(): StoreState {
  const project = read<Project>(KEYS.project);
  const job = read<Job>(KEYS.job);
  const ui = read<{ mode?: Mode; savedAt?: string }>(KEYS.ui);
  const ok = !!(project && Array.isArray(project.lines));
  return {
    project: ok ? project : null,
    job,
    mode: ui?.mode ?? 'simple',
    save: { state: ui?.savedAt ? 'saved' : 'idle', at: ui?.savedAt ?? null, pending: 0, error: '' },
    sim: { offline: false, failNext: false, conflictNext: false, crash: false, conn: 'local' },
    canUndo: false,
    canRedo: false,
    media: null,
    restored: ok,
    bgCatchUp: null,
    backend: { mode: 'demo', reachable: false },
    conflictTheirs: null,
  };
}

let S: StoreState = initialState();
const listeners = new Set<() => void>();
const conflictListeners = new Set<() => void>();

function set(patch: Partial<StoreState>) {
  S = { ...S, ...patch, canUndo: undoStack.length > 0, canRedo: redoStack.length > 0 };
  listeners.forEach((f) => f());
}

const persistUI = () => write(KEYS.ui, { mode: S.mode, savedAt: S.save.at });
const persistJob = () => write(KEYS.job, S.job ? { ...S.job, updatedAt: Date.now() } : null);
const setSave = (save: Partial<SaveInfo>) => set({ save: { ...S.save, ...save } });

let saveTimer: ReturnType<typeof setTimeout> | undefined;

export type RemoteSaveResult = { ok: true; serverRev: number } | { conflict: Project } | { offline: true } | { error: string };
/** Injected by the server module; saves a server-backed project with optimistic concurrency. */
/** Injected by the server module: queue actions for jobs that run on the server. */
let jobHook: ((a: 'cancel' | 'retry' | 'retry-fast' | 'remove') => Promise<void>) | null = null;
const viaServer = (a: 'cancel' | 'retry' | 'retry-fast' | 'remove') => {
  if (!S.job?.serverId || !jobHook) return false;
  jobHook(a).catch((e) => listeners.forEach(() => console.error(e)));
  return true;
};
let remoteSave: ((p: Project, force: boolean) => Promise<RemoteSaveResult>) | null = null;
const isRemote = () => S.backend.mode === 'server' && S.project?.serverRev != null && !!remoteSave;
/** While the server processes this project it owns the document; local edits wait and are pushed afterwards. */
const remoteBusy = () => !!(S.job?.serverId && S.job.status === 'running' && S.job.kind === 'process');

async function pushRemote(force = false) {
  const p = S.project!;
  const r = await remoteSave!(p, force);
  if (S.project?.id !== p.id) return;
  if ('ok' in r) {
    // keep newer local edits; only adopt the new server revision
    set({ project: { ...S.project!, serverRev: r.serverRev }, conflictTheirs: null });
    write(KEYS.project, S.project);
    setSave({ state: 'saved', at: hhmmss(), pending: 0, error: '' });
  } else if ('conflict' in r) {
    set({ conflictTheirs: r.conflict });
    setSave({ state: 'conflict', at: hhmmss() });
    conflictListeners.forEach((f) => f());
  } else if ('offline' in r) {
    set({ backend: { ...S.backend, reachable: false } });
    setSave({ state: 'local', at: hhmmss() });
  } else setSave({ state: 'failed', error: r.error });
  persistUI();
}

/** Writes the project to browser storage, honouring the failure/offline/conflict simulations. */
function doSave(manual: boolean) {
  clearTimeout(saveTimer);
  if (!S.project) return;
  setSave({ state: 'saving' });
  saveTimer = setTimeout(
    () => {
      if (S.sim.failNext) {
        set({ sim: { ...S.sim, failNext: false } });
        return setSave({ state: 'failed', error: 'כשל כתיבה מדומה' });
      }
      try {
        localStorage.setItem(KEYS.project, JSON.stringify(S.project));
      } catch (e) {
        return setSave({ state: 'failed', error: String((e as Error).message || e) });
      }
      if (isRemote() && remoteBusy()) {
        setSave({ state: 'local', at: hhmmss() });
        return persistUI();
      }
      if (isRemote()) return void pushRemote();
      if (S.sim.offline) {
        setSave({ state: 'local', at: hhmmss() });
        return persistUI();
      }
      if (S.sim.conflictNext) {
        set({ sim: { ...S.sim, conflictNext: false } });
        setSave({ state: 'conflict', at: hhmmss() });
        persistUI();
        conflictListeners.forEach((f) => f());
        return;
      }
      setSave({ state: 'saved', at: hhmmss(), pending: 0, error: '' });
      persistUI();
    },
    manual ? 300 : 650,
  );
}

function ensure(): Project {
  if (!S.project) {
    const p = demoProject();
    write(KEYS.project, p);
    set({ project: p });
  }
  return S.project!;
}

export const STAGES = [
  'מייבא את השיר',
  'מפריד שירה ומוזיקה',
  'מתמלל את השירה',
  'מסנכרן את המילים',
  'מכין את סרטון הקריוקי',
] as const;

export const store = {
  get: () => S,
  subscribe(f: () => void) {
    listeners.add(f);
    return () => {
      listeners.delete(f);
    };
  },
  onConflict(f: () => void) {
    conflictListeners.add(f);
    return () => {
      conflictListeners.delete(f);
    };
  },
  ensure,

  /**
   * Applies an edit to a draft copy of the project. Return `false` from `fn` to abort.
   * Every edit is undoable, logged in history and autosaved.
   */
  edit(label: string, fn: (p: Project) => void | boolean): boolean {
    ensure();
    const before = previewBase ?? S.project!;
    previewBase = null;
    const draft = clone(S.project!);
    if (fn(draft) === false) return false;
    draft.rev = (draft.rev || 1) + 1;
    draft.history = [{ at: Date.now(), label }, ...(draft.history || [])].slice(0, HISTORY_LIMIT);
    undoStack = [...undoStack, { p: before, label }].slice(-UNDO_LIMIT);
    redoStack = [];
    set({ project: draft, save: { ...S.save, pending: S.save.pending + 1 } });
    doSave(false);
    return true;
  },
  /** Live (non-undoable, unsaved) preview of a change, e.g. while dragging a slider. */
  preview(fn: (p: Project) => void) {
    if (!previewBase) previewBase = ensure();
    const draft = clone(S.project!);
    fn(draft);
    set({ project: draft });
  },
  undo() {
    const u = undoStack.pop();
    if (!u || !S.project) return;
    redoStack = [{ p: S.project, label: u.label }, ...redoStack];
    set({ project: u.p, save: { ...S.save, pending: S.save.pending + 1 } });
    doSave(false);
  },
  redo() {
    const r = redoStack.shift();
    if (!r || !S.project) return;
    undoStack = [...undoStack, { p: S.project, label: r.label }];
    set({ project: r.p, save: { ...S.save, pending: S.save.pending + 1 } });
    doSave(false);
  },
  saveNow: () => doSave(true),
  retrySave: () => doSave(true),

  resolveConflict(side: 'mine' | 'theirs') {
    if (!S.project) return;
    if (isRemote() && S.conflictTheirs) {
      if (side === 'theirs') {
        undoStack = [...undoStack, { p: S.project, label: 'לפני טעינת הגרסה מהשרת' }];
        set({ project: S.conflictTheirs, conflictTheirs: null });
        write(KEYS.project, S.project);
        setSave({ state: 'saved', at: hhmmss(), pending: 0, error: '' });
      } else {
        setSave({ state: 'saving' });
        void pushRemote(true);
      }
      return;
    }
    const p = clone(S.project);
    if (side === 'theirs') {
      const l = p.lines[THEIRS_LINE_INDEX];
      if (l) {
        const [a, b] = lineSpan(l);
        l.words = makeWords(a, b, THEIRS_LINE_TEXT);
      }
    }
    p.history = [
      { at: Date.now(), label: side === 'theirs' ? `התנגשות: נטענה הגרסה של ${OTHER_USER}` : 'התנגשות: נשמרה הגרסה שלי' },
      ...p.history,
    ].slice(0, HISTORY_LIMIT);
    write(KEYS.project, p);
    set({ project: p, save: { state: 'saved', at: hhmmss(), pending: 0, error: '' } });
    persistUI();
  },

  setSim<K extends keyof Sim>(k: K, v: Sim[K]) {
    const wasOffline = S.sim.offline;
    set({ sim: { ...S.sim, [k]: v } });
    if (k === 'offline' && wasOffline && !v && S.save.state === 'local') doSave(true);
  },
  setBackend(b: Partial<Backend>) {
    const wasReachable = S.backend.reachable;
    set({ backend: { ...S.backend, ...b } });
    if (!wasReachable && S.backend.reachable && S.save.state === 'local' && isRemote()) doSave(true);
  },
  setRemoteSaver(fn: typeof remoteSave) {
    remoteSave = fn;
  },
  /** Replaces the open project with a server copy (new import, finished job, opening from the library). */
  adoptProject(p: Project, opts: { resetHistory?: boolean } = {}) {
    if (opts.resetHistory) {
      undoStack = [];
      redoStack = [];
    }
    write(KEYS.project, p);
    set({ project: p, restored: false });
  },
  setJob(job: Job | null) {
    set({ job });
    persistJob();
  },
  setMediaRemote(media: Media | null) {
    if (S.media && !S.media.remote) URL.revokeObjectURL(S.media.url);
    set({ media });
  },
  setMode(mode: Mode) {
    set({ mode });
    persistUI();
  },
  setMedia(file: File | null) {
    if (S.media && !S.media.remote) URL.revokeObjectURL(S.media.url);
    const media: Media | null = file ? { url: URL.createObjectURL(file), type: file.type, name: file.name } : null;
    set({ media });
  },
  newProject(source: Source, cfg?: { profile: Profile; stems: 2 | 4; language: Project['processing']['language']; res: Project['export']['res'] }) {
    const p = demoProject(source);
    if (cfg) {
      p.processing = { profile: cfg.profile, stems: cfg.stems, language: cfg.language };
      p.export.res = cfg.res;
      p.stems = cfg.stems === 4 ? ['vocals', 'drums', 'bass', 'other'] : ['vocals', 'instrumental'];
    }
    undoStack = [];
    redoStack = [];
    write(KEYS.project, p);
    set({ project: p, save: { state: 'saved', at: hhmmss(), pending: 0, error: '' } });
    persistUI();
    return p;
  },
  /** Deletes the open project and its job (media is never stored). */
  deleteProject() {
    write(KEYS.project, null);
    write(KEYS.job, null);
    undoStack = [];
    redoStack = [];
    set({ project: null, job: null, save: { state: 'idle', at: null, pending: 0, error: '' } });
  },
  reset() {
    Object.values(KEYS).forEach((k) => write(k, null));
    undoStack = [];
    redoStack = [];
    set({ project: null, job: null, mode: 'simple', restored: false, save: { state: 'idle', at: null, pending: 0, error: '' } });
  },

  // ---- simulated processing job (one heavy GPU job at a time) ----
  startJob(cfg: { name: string; profile: Profile; stems: 2 | 4 }) {
    set({ job: { id: 'j' + Date.now().toString(36), ...cfg, status: 'running', stage: 0, pct: 0, attempt: 1, startedAt: Date.now() } });
    persistJob();
  },
  setJobHook(fn: typeof jobHook) {
    jobHook = fn;
  },
  cancelJob() {
    if (viaServer('cancel')) return;
    if (!S.job || S.job.status !== 'running') return;
    set({ job: { ...S.job, status: 'cancelled' } });
    persistJob();
  },
  /** Resumes from the stage where the job stopped; completed stages are kept. */
  retryJob(fast = false) {
    if (viaServer(fast ? 'retry-fast' : 'retry')) return;
    if (!S.job) return;
    set({ job: { ...S.job, status: 'running', pct: 0, attempt: S.job.attempt + 1, profile: fast ? 'מהיר' : S.job.profile } });
    persistJob();
  },
  removeJob() {
    if (S.job?.serverId && S.job.status !== 'running' && viaServer('remove')) return;
    set({ job: null });
    persistJob();
  },
  /** Advances the simulated job by `dt` ms. Exposed for tests. */
  tick(dt: number) {
    const j = S.job;
    if (!j || j.status !== 'running' || S.sim.offline || j.serverId) return;
    let { stage, pct } = j;
    pct += (j.profile === 'מהיר' ? 11 : 6) * Math.max(1, dt / 200);
    if (S.sim.crash && j.attempt === 1 && stage === 2 && pct >= 38) {
      set({ job: { ...j, pct: 38, status: 'failed' } });
      return persistJob();
    }
    if (pct >= 100) {
      stage++;
      pct = 0;
    }
    if (stage >= STAGES.length) {
      ensure();
      set({ job: { ...j, stage: STAGES.length, pct: 100, status: 'done', endedAt: Date.now() } });
      return persistJob();
    }
    set({ job: { ...j, stage, pct } });
    persistJob();
  },
};

// The processing server keeps working while the UI is closed: catch up on load.
(function catchUp() {
  const j = S.job;
  if (!j || j.status !== 'running' || !j.updatedAt || j.serverId) return;
  let left = Math.min(10 * 60000, Date.now() - j.updatedAt);
  if (left < 3000) return;
  const from = jobPct(j);
  while (left > 0 && S.job?.status === 'running') {
    store.tick(1000);
    left -= 1000;
  }
  set({ bgCatchUp: { from, to: S.job ? jobPct(S.job) : 100 } });
})();

// Never lose edits on refresh/close: flush a pending debounced save to browser storage.
function flushPending() {
  if (S.project && (S.save.state === 'saving' || S.save.pending > 0)) {
    try {
      localStorage.setItem(KEYS.project, JSON.stringify(S.project));
    } catch {
      /* ignore */
    }
  }
}
if (typeof window !== 'undefined') {
  window.addEventListener('pagehide', flushPending);
  window.addEventListener('beforeunload', flushPending);
}

let lastTick = Date.now();
if (typeof window !== 'undefined' && !import.meta.env?.VITEST) {
  setInterval(() => {
    const now = Date.now();
    store.tick(Math.min(5000, now - lastTick));
    lastTick = now;
  }, 200);
}

export function useStore(): StoreState {
  return useSyncExternalStore(store.subscribe, store.get, store.get);
}

/** Overall job progress 0..100. */
export function jobPct(j: Job) {
  return (
    j.status === 'done' ? 100 : Math.round((j.stage * 100 + j.pct) / STAGES.length)
  );
}
