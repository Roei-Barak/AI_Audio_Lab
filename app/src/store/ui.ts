// App navigation, modal and toast state. Routes are mirrored in location.hash.
import { useSyncExternalStore } from 'react';
import { store } from './store';

export type StateKind = 'background' | 'offline' | 'crash' | 'failed' | 'recovery' | 'emptyDashboard' | 'emptyLibrary' | 'loading' | 'forbidden' | 'notfound';
export type Route =
  | 'simple' | 'options' | 'queue' | 'subs' | 'subs-lyrics' | 'subs-word' | 'karaoke'
  | 'dashboard' | 'library' | 'project' | 'project-versions' | 'audio' | 'video' | 'export' | 'outputs' | 'team'
  | 'settings' | 'settings-general' | 'settings-models' | 'settings-storage' | 'settings-api'
  | 'login' | 'invite' | 'forgot' | `state:${StateKind}`;
export type ModalKind = 'conflict' | 'airerun' | 'shortcuts' | 'share' | 'invite' | 'find' | 'delete' | 'restore';
export type SimpleStep = 'input' | 'progress' | 'preview' | 'result';

export const STATE_KINDS: StateKind[] = ['background', 'offline', 'crash', 'failed', 'recovery', 'emptyDashboard', 'emptyLibrary', 'loading', 'forbidden', 'notfound'];
export const ADV_ROUTES: Route[] = [
  'options', 'queue', 'subs', 'subs-lyrics', 'subs-word', 'karaoke',
  'dashboard', 'library', 'project', 'project-versions', 'audio', 'video', 'export', 'outputs', 'team',
  'settings', 'settings-general', 'settings-models', 'settings-storage', 'settings-api',
  'login', 'invite', 'forgot', ...STATE_KINDS.map((k) => `state:${k}` as Route),
];
const STEP_KEY = 'lyra.v1.simpleStep';
const ADV_KEY = 'lyra.v1.advRoute';

interface UIState {
  route: Route;
  modal: ModalKind | null;
  /** argument for the open modal, e.g. a version id to restore */
  modalArg: string | null;
  toast: string;
  simpleStep: SimpleStep;
}

const safeGet = (k: string) => {
  try {
    return localStorage.getItem(k);
  } catch {
    return null;
  }
};
const safeSet = (k: string, v: string) => {
  try {
    localStorage.setItem(k, v);
  } catch {
    /* ignore */
  }
};

function initial(): UIState {
  const h = typeof location !== 'undefined' ? decodeURIComponent(location.hash.slice(1)) : '';
  const S = store.get();
  let route: Route = S.mode === 'advanced' ? ((safeGet(ADV_KEY) as Route) || 'dashboard') : 'simple';
  if (h === 'simple' || ADV_ROUTES.includes(h as Route)) route = h as Route;
  if (!ADV_ROUTES.includes(route) && route !== 'simple') route = 'simple';
  let simpleStep = (safeGet(STEP_KEY) as SimpleStep) || 'input';
  if (S.job && S.job.status !== 'done' && simpleStep !== 'input') simpleStep = 'progress';
  if ((simpleStep === 'preview' || simpleStep === 'result') && !S.project) simpleStep = 'input';
  return { route, modal: null, modalArg: null, toast: '', simpleStep };
}

let U: UIState = initial();
const listeners = new Set<() => void>();
function set(patch: Partial<UIState>) {
  U = { ...U, ...patch };
  listeners.forEach((f) => f());
}

let toastTimer: ReturnType<typeof setTimeout> | undefined;

export const ui = {
  get: () => U,
  subscribe(f: () => void) {
    listeners.add(f);
    return () => {
      listeners.delete(f);
    };
  },
  go(route: Route) {
    const mode = route === 'simple' ? 'simple' : 'advanced';
    if (store.get().mode !== mode) store.setMode(mode);
    if (route !== 'simple' && !['login', 'invite', 'forgot'].includes(route)) safeSet(ADV_KEY, route);
    set({ route, modal: null });
    try {
      history.replaceState(null, '', '#' + route);
    } catch {
      /* ignore */
    }
  },
  setStep(simpleStep: SimpleStep) {
    safeSet(STEP_KEY, simpleStep);
    set({ simpleStep });
  },
  open: (modal: ModalKind, modalArg: string | null = null) => set({ modal, modalArg }),
  close: () => set({ modal: null, modalArg: null }),
  toast(text: string) {
    set({ toast: text });
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => set({ toast: '' }), 3200);
  },
  resetKeys() {
    try {
      localStorage.removeItem(STEP_KEY);
      localStorage.removeItem(ADV_KEY);
    } catch {
      /* ignore */
    }
  },
};

store.onConflict(() => ui.open('conflict'));

export function useUI(): UIState {
  return useSyncExternalStore(ui.subscribe, ui.get, ui.get);
}
