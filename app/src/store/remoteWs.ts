// Server-backed workspace: accounts, invites, settings, versions and outputs come from the Lyra server when
// it is reachable. Every function falls back to the browser-local workspace (demo mode).
import { api, isServer } from './server';
import { store } from './store';
import type { Line, SubtitleStyle } from './types';
import { ws, type Invite, type Output, type Role, type Settings, type User, type Version } from './workspace';

type SUser = { id: string; name: string; user: string; role: Role };
type SInvite = { id: string; name: string; role: Role; expiresAt: number; used: boolean; token?: string };

const remote = () => isServer() && ws.get().remote.on;
const toInvite = (i: SInvite): Invite => ({ id: i.id, name: i.name, role: i.role, token: i.token ?? '', expiresAt: i.expiresAt * 1000, used: i.used });

/** Called when the server becomes reachable: loads the session and shared workspace data. */
export async function syncAccount() {
  if (!isServer()) return;
  try {
    const st = await api<{ authRequired: boolean; needsSetup: boolean; user: SUser | null }>('/api/auth/status');
    ws._setRemote({ remote: { on: true, needsSetup: st.needsSetup, checked: true }, session: { userId: st.user?.id ?? null }, users: st.user ? ws.get().users.filter((u) => u.id === st.user!.id).concat(ws.get().users.some((u) => u.id === st.user!.id) ? [] : [st.user]) : [] });
    if (st.user) await refreshShared(st.user.role === 'admin');
  } catch {
    /* server went away: keep current state */
  }
}

async function refreshShared(isAdmin: boolean) {
  const [users, settings, outputs] = await Promise.all([
    api<SUser[]>('/api/users'),
    api<Partial<Settings>>('/api/settings'),
    api<(Output & { projectId: string; title: string })[]>('/api/outputs'),
  ]);
  const invites = isAdmin ? (await api<SInvite[]>('/api/invites')).map(toInvite) : [];
  ws._setRemote({ users, invites, settings: { ...ws.get().settings, ...settings }, outputs: outputs.map((o) => ({ ...o, name: (o.title || 'lyra') + '.' + o.kind })) });
  const p = store.get().project;
  if (p?.serverRev != null) await refreshVersions(p.id);
}

export async function refreshVersions(pid: string) {
  if (!remote()) return;
  const vs = await api<{ id: string; at: number; label: string; rev: number; lineTexts: string[] }[]>(`/api/projects/${pid}/versions`);
  // keep line texts for the diff; full lines are fetched on restore
  ws._setRemote({ versions: vs.map((v) => ({ id: v.id, at: v.at, label: v.label, rev: v.rev, style: {} as SubtitleStyle, lines: v.lineTexts.map((t, i) => ({ id: 'v' + i, start: 0, end: 0, speaker: '1', manual: false, words: t.split(' ').map((w, j) => ({ id: 'vw' + j, w, s: 0, e: 0, conf: 'high', manual: false })) })) as Line[] })) });
}

// ---------- auth ----------
export async function login(user: string, password: string): Promise<true | string> {
  if (!remote()) return ws.login(user, password);
  try {
    await api('/api/auth/login', { method: 'POST', body: JSON.stringify({ user, password }) });
    await syncAccount();
    return true;
  } catch (e) {
    return (e as Error).message;
  }
}

export async function setupAdmin(name: string, user: string, password: string): Promise<true | string> {
  try {
    await api('/api/auth/setup', { method: 'POST', body: JSON.stringify({ name, user, password }) });
    await syncAccount();
    return true;
  } catch (e) {
    return (e as Error).message;
  }
}

export async function logout() {
  if (!remote()) return ws.logout();
  await api('/api/auth/logout', { method: 'POST' }).catch(() => {});
  ws._setRemote({ session: { userId: null } });
}

export async function acceptInvite(token: string, user: string, password: string): Promise<User | string> {
  if (!remote()) {
    if (password.length < 8) return 'סיסמה: לפחות 8 תווים.';
    return ws.acceptInvite(token.split('/join/').pop() || '', user);
  }
  try {
    const r = await api<{ user: SUser }>('/api/auth/accept', { method: 'POST', body: JSON.stringify({ token, user, password }) });
    await syncAccount();
    return r.user;
  } catch (e) {
    return (e as Error).message;
  }
}

export async function resetPassword(token: string, password: string): Promise<true | string> {
  try {
    await api('/api/auth/reset', { method: 'POST', body: JSON.stringify({ token, password }) });
    await syncAccount();
    return true;
  } catch (e) {
    return (e as Error).message;
  }
}

// ---------- team ----------
export async function createInvite(name: string, role: Role, days: number): Promise<Invite> {
  if (!remote()) return ws.createInvite(name, role, days, '');
  const i = await api<SInvite>('/api/invites', { method: 'POST', body: JSON.stringify({ name, role, days }) });
  await refreshShared(true);
  return toInvite(i);
}
export async function revokeInvite(id: string) {
  if (!remote()) return ws.revokeInvite(id);
  await api('/api/invites/' + id, { method: 'DELETE' });
  await refreshShared(true);
}
export async function setRole(id: string, role: Role): Promise<true | string> {
  if (!remote()) return ws.setRole(id, role) || 'חייב להישאר לפחות מנהל אחד.';
  try {
    await api('/api/users/' + id, { method: 'PATCH', body: JSON.stringify({ role }) });
    await refreshShared(true);
    return true;
  } catch (e) {
    return (e as Error).message;
  }
}
export async function removeUser(id: string) {
  if (!remote()) return void ws.removeUser(id);
  await api('/api/users/' + id, { method: 'DELETE' });
  await refreshShared(true);
}
export async function resetLink(id: string): Promise<string> {
  const r = await api<{ token: string }>(`/api/users/${id}/reset`, { method: 'POST' });
  return r.token;
}

// ---------- settings / versions / outputs ----------
export async function setSettings(patch: Partial<Settings>) {
  ws.setSettings(patch);
  if (remote() && !('apiToken' in patch)) await api('/api/settings', { method: 'PUT', body: JSON.stringify(patch) }).catch(() => {});
}

export async function saveVersion(label: string) {
  const p = store.get().project;
  if (!p) return;
  if (!(remote() && p.serverRev != null)) return ws.saveVersion(label);
  await api(`/api/projects/${p.id}/versions`, { method: 'POST', body: JSON.stringify({ label, rev: p.serverRev, lines: p.lines, style: p.style }) });
  await refreshVersions(p.id);
}

export async function restoreVersion(id: string) {
  const p = store.get().project;
  if (!(remote() && p?.serverRev != null)) return ws.restoreVersion(id);
  const v = await api<Version>(`/api/projects/${p.id}/versions/${id}`);
  await saveVersion('לפני שחזור');
  return store.edit('שוחזרה גרסה: ' + v.label, (P) => {
    P.lines = JSON.parse(JSON.stringify(v.lines));
    if (v.style && Object.keys(v.style).length) P.style = { ...P.style, ...v.style };
  });
}

export async function clearVersions() {
  const p = store.get().project;
  if (!(remote() && p?.serverRev != null)) return ws.clearVersions();
  await api(`/api/projects/${p.id}/versions`, { method: 'DELETE' });
  await refreshVersions(p.id);
}

export async function recordOutput(kind: Output['kind'] | 'vocals' | 'instrumental' | 'ass', simulated = false) {
  const p = store.get().project;
  if (remote() && p?.serverRev != null && !simulated) {
    await api('/api/outputs', { method: 'POST', body: JSON.stringify({ projectId: p.id, kind }) }).catch(() => {});
    await refreshShared(ws.me()?.role === 'admin').catch(() => {});
  } else if (kind !== 'vocals' && kind !== 'instrumental' && kind !== 'ass') ws.addOutput(kind, simulated);
  else ws.addOutput(kind as Output['kind'], simulated);
}

// keep account state in step with the connection
let wasServer = false;
store.subscribe(() => {
  const now = isServer() && store.get().backend.reachable;
  if (now && !wasServer) void syncAccount();
  wasServer = now;
});
