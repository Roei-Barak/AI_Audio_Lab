// Workspace data around the open project: versions, outputs, waiting jobs, settings, team and session.
// Persisted to localStorage. Private product: no public signup, admin-made one-time invites, no emails.
import { useSyncExternalStore } from 'react';
import { uid } from './demo';
import { store } from './store';
import type { Lang, Line, Profile, SubtitleStyle } from './types';

export type Role = 'admin' | 'editor' | 'viewer';
export const ROLE_L: Record<Role, string> = { admin: 'מנהל', editor: 'עורך', viewer: 'צופה' };

export interface Version {
  id: string;
  at: number;
  label: string;
  rev: number;
  lines: Line[];
  style: SubtitleStyle;
}
export interface Output {
  id: string;
  at: number;
  kind: 'srt' | 'txt' | 'json' | 'lrc' | 'mp4';
  name: string;
  simulated: boolean;
  lines: number;
}
export interface Waiting {
  id: string;
  name: string;
  pipe: string;
}
export interface User {
  id: string;
  name: string;
  user: string;
  role: Role;
}
export interface Invite {
  id: string;
  name: string;
  role: Role;
  token: string;
  expiresAt: number;
  used: boolean;
}
export interface Settings {
  defaultProfile: Profile;
  defaultStems: 2 | 4;
  language: Lang;
  autosave: boolean;
  model: 'large-v3' | 'medium';
  separator: 'demucs' | 'mdx';
  gpuSlots: 1;
  outputDir: string;
  apiToken: string | null;
}
export interface WS {
  versions: Version[];
  outputs: Output[];
  waiting: Waiting[];
  users: User[];
  invites: Invite[];
  settings: Settings;
  session: { userId: string | null };
}

const KEY = 'lyra.v1.workspace';
const ME = 'u_daniel';

const defaults = (): WS => ({
  versions: [],
  outputs: [],
  waiting: [
    { id: 'q1', name: 'Neon Rain', pipe: 'תמלול · Whisper large-v3' },
    { id: 'q2', name: 'Lighthouse', pipe: 'עיבוד מלא · מהיר · 2 ערוצים' },
    { id: 'q3', name: 'שקיעה בנמל', pipe: 'סנכרון מחדש כהצעה · שורות 12–18' },
  ],
  users: [
    { id: ME, name: 'דניאל כהן', user: 'daniel', role: 'admin' },
    { id: 'u_michal', name: 'מיכל לוי', user: 'michal', role: 'editor' },
    { id: 'u_yoav', name: 'יואב פרץ', user: 'yoav', role: 'viewer' },
  ],
  invites: [],
  settings: { defaultProfile: 'מהיר', defaultStems: 2, language: 'auto', autosave: true, model: 'large-v3', separator: 'demucs', gpuSlots: 1, outputDir: '~/LyraStudio/outputs', apiToken: null },
  session: { userId: ME },
});

function load(): WS {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) || 'null');
    if (raw) return { ...defaults(), ...raw, settings: { ...defaults().settings, ...raw.settings } };
  } catch {
    /* ignore */
  }
  return defaults();
}

let W: WS = load();
const listeners = new Set<() => void>();
function set(patch: Partial<WS>) {
  W = { ...W, ...patch };
  try {
    localStorage.setItem(KEY, JSON.stringify(W));
  } catch {
    /* ignore */
  }
  listeners.forEach((f) => f());
}

export const ws = {
  get: () => W,
  subscribe(f: () => void) {
    listeners.add(f);
    return () => {
      listeners.delete(f);
    };
  },
  me: () => W.users.find((u) => u.id === W.session.userId) ?? null,

  // ---- versions (manual save points) ----
  saveVersion(label: string) {
    const p = store.get().project;
    if (!p) return;
    const v: Version = { id: uid('v'), at: Date.now(), label, rev: p.rev, lines: JSON.parse(JSON.stringify(p.lines)), style: { ...p.style } };
    set({ versions: [v, ...W.versions].slice(0, 30) });
  },
  restoreVersion(id: string) {
    const v = W.versions.find((x) => x.id === id);
    if (!v) return false;
    ws.saveVersion('לפני שחזור');
    return store.edit('שוחזרה גרסה: ' + v.label, (P) => {
      P.lines = JSON.parse(JSON.stringify(v.lines));
      P.style = { ...v.style };
    });
  },

  clearVersions: () => set({ versions: [] }),

  // ---- outputs ----
  addOutput(kind: Output['kind'], simulated = false) {
    const p = store.get().project;
    if (!p) return;
    const o: Output = { id: uid('o'), at: Date.now(), kind, name: 'yareach-shel-august' + (kind === 'mp4' ? '_' + p.export.res : '') + '.' + kind, simulated, lines: p.lines.length };
    set({ outputs: [o, ...W.outputs].slice(0, 50) });
  },
  removeOutput: (id: string) => set({ outputs: W.outputs.filter((o) => o.id !== id) }),

  // ---- waiting queue ----
  moveWaiting(id: string, d: -1 | 1) {
    const a = [...W.waiting];
    const i = a.findIndex((x) => x.id === id);
    const j = i + d;
    if (i < 0 || j < 0 || j >= a.length) return;
    [a[i], a[j]] = [a[j], a[i]];
    set({ waiting: a });
  },
  removeWaiting: (id: string) => set({ waiting: W.waiting.filter((x) => x.id !== id) }),

  // ---- settings ----
  setSettings: (patch: Partial<Settings>) => set({ settings: { ...W.settings, ...patch } }),

  // ---- team & invites ----
  setRole(id: string, role: Role) {
    if (id === ME && role !== 'admin') return false; // keep at least one admin
    set({ users: W.users.map((u) => (u.id === id ? { ...u, role } : u)) });
    return true;
  },
  removeUser(id: string) {
    if (id === ME) return false;
    set({ users: W.users.filter((u) => u.id !== id) });
    return true;
  },
  createInvite(name: string, role: Role, days: number, host: string): Invite {
    const token = Math.random().toString(36).slice(2, 6) + '-' + Math.random().toString(36).slice(2, 6);
    const inv: Invite = { id: uid('i'), name, role, token, expiresAt: Date.now() + days * 86400000, used: false };
    set({ invites: [inv, ...W.invites] });
    void host;
    return inv;
  },
  revokeInvite: (id: string) => set({ invites: W.invites.filter((x) => x.id !== id) }),
  /** Accepts a one-time invite: creates the user and burns the token. */
  acceptInvite(token: string, user: string): User | string {
    const inv = W.invites.find((x) => x.token === token.trim());
    if (!inv) return 'הקישור לא תקין או שבוטל.';
    if (inv.used) return 'הקישור כבר שומש. בקש מהמנהל קישור חדש.';
    if (inv.expiresAt < Date.now()) return 'תוקף הקישור פג.';
    if (!/^[a-z0-9._-]{3,}$/i.test(user)) return 'שם משתמש: לפחות 3 תווים באנגלית, ספרות, נקודה או מקף.';
    if (W.users.some((u) => u.user === user)) return 'שם המשתמש תפוס.';
    const u: User = { id: uid('u'), name: inv.name, user, role: inv.role };
    set({ users: [...W.users, u], invites: W.invites.map((x) => (x.id === inv.id ? { ...x, used: true } : x)), session: { userId: u.id } });
    return u;
  },

  // ---- session (simulated local auth) ----
  login(user: string, password: string): true | string {
    const u = W.users.find((x) => x.user === user.trim());
    if (!u || password.length < 4) return 'שם משתמש או סיסמה שגויים.';
    set({ session: { userId: u.id } });
    return true;
  },
  logout: () => set({ session: { userId: null } }),

  reset() {
    try {
      localStorage.removeItem(KEY);
    } catch {
      /* ignore */
    }
    W = defaults();
    listeners.forEach((f) => f());
  },
};

export function useWS(): WS {
  return useSyncExternalStore(ws.subscribe, ws.get, ws.get);
}

export const inviteUrl = (token: string, remote: boolean) => (remote ? 'https://lyra.home' : 'http://192.168.1.20:8443') + '/join/' + token;
