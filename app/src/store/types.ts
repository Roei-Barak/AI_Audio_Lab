export type Conf = 'high' | 'med' | 'low';
export type Speaker = '1' | '2' | 'duet';

export interface Word {
  id: string;
  w: string;
  s: number;
  e: number;
  conf: Conf;
  manual: boolean;
}

export interface Line {
  id: string;
  start: number;
  end: number;
  speaker: Speaker;
  words: Word[];
  manual: boolean;
}

export type Template = 'classic' | 'bold' | 'minimal';
export type HiColor = 'accent' | 'soft' | 'light' | 'deep';
export type Position = 'top' | 'middle' | 'bottom';
export type Background = 'video' | 'blur' | 'solid';
export type Aspect = '16:9' | '9:16' | '1:1';
export type Profile = 'מהיר' | 'איכותי';
export type Lang = 'auto' | 'he' | 'en';
export type Res = '720p' | '1080p';

export interface Source {
  kind: 'youtube' | 'file';
  url?: string;
  name: string;
  size?: number;
  duration: number;
  hasVideo: boolean;
}

export interface SubtitleStyle {
  template: Template;
  color: HiColor;
  size: number;
  position: Position;
  outline: boolean;
  shadow: boolean;
  aspect: Aspect;
  background: Background;
}

export interface StemMix {
  vol: number;
  mute: boolean;
  solo: boolean;
}

export interface HistoryEntry {
  at: number;
  label: string;
}

export interface Project {
  id: string;
  title: string;
  artist: string;
  createdAt: number;
  source: Source;
  processing: { profile: Profile; stems: 2 | 4; language: Lang };
  stems: string[];
  /** 'word' only when word timings were actually produced; never invented. */
  timing: 'word' | 'line';
  lines: Line[];
  style: SubtitleStyle;
  audio: { track: 'instrumental'; originalMuted: boolean; mix?: Record<string, StemMix> };
  export: { format: 'mp4'; res: Res };
  history: HistoryEntry[];
  rev: number;
  /** server-side revision used for conflict detection (absent for a browser-only demo project) */
  serverRev?: number;
  /** serverRev of the last MP4 render */
  renderedRev?: number;
  /** AI re-run output kept as a suggestion when manual fixes exist */
  suggestion?: { at: number; lines: Line[] } | null;
  processed?: boolean;
  rerun?: boolean;
}

export type JobStatus = 'running' | 'failed' | 'cancelled' | 'done';

export interface Job {
  id: string;
  name: string;
  profile: Profile;
  stems: 2 | 4;
  status: JobStatus;
  /** 0..4 = current stage, 5 = finished */
  stage: number;
  pct: number;
  attempt: number;
  startedAt: number;
  endedAt?: number;
  /** last time the job state was written; used to catch up after the UI was closed */
  updatedAt?: number;
  /** set when the job runs on the Lyra server */
  serverId?: string;
  kind?: 'process' | 'render';
  error?: string | null;
  crash?: boolean;
  log?: string[];
}

export type SaveState = 'idle' | 'saving' | 'saved' | 'local' | 'failed' | 'conflict';

export interface SaveInfo {
  state: SaveState;
  at: string | null;
  pending: number;
  error: string;
}

export interface Sim {
  offline: boolean;
  failNext: boolean;
  conflictNext: boolean;
  crash: boolean;
  conn: 'local' | 'remote';
}

export interface Media {
  url: string;
  type: string;
  name: string;
  /** separate audio track (server instrumental) played in sync with a muted video */
  audioUrl?: string;
  remote?: boolean;
}

export interface Backend {
  mode: 'demo' | 'server';
  engine?: 'ai' | 'mock';
  gpu?: { name: string | null; vramGB: number | null; device: string };
  host?: string;
  /** last health check reached the server */
  reachable: boolean;
}

export type Mode = 'simple' | 'advanced';

export interface StoreState {
  project: Project | null;
  job: Job | null;
  mode: Mode;
  save: SaveInfo;
  sim: Sim;
  canUndo: boolean;
  canRedo: boolean;
  media: Media | null;
  restored: boolean;
  /** the job kept running while the UI was closed */
  bgCatchUp: { from: number; to: number } | null;
  backend: Backend;
  /** server copy received on a 409 conflict */
  conflictTheirs: Project | null;
}
