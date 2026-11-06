// Shared playback clock for the karaoke preview, editors and word loop.
// Uses a real <video> element when the user uploaded playable local media; otherwise a
// simulated clock (clearly labelled "סימולציית ניגון" in the UI).
import { useSyncExternalStore } from 'react';
import { store } from './store';

interface PlayState {
  t: number;
  playing: boolean;
  loop: [number, number] | null;
}

let P: PlayState = { t: 0, playing: false, loop: null };
const listeners = new Set<() => void>();
let video: HTMLVideoElement | null = null;
/** separate instrumental track (server mode); when present it is the master clock and the video is muted */
let audio: HTMLAudioElement | null = null;

function set(patch: Partial<PlayState>) {
  P = { ...P, ...patch };
  listeners.forEach((f) => f());
}

const hasMedia = () => !!(store.get().media && (audio || video));
const clockEl = (): HTMLMediaElement | null => audio || video;
const els = (): HTMLMediaElement[] => [audio, video].filter((x): x is HTMLMediaElement => !!x && !!x.src);

function duration(): number {
  const c = clockEl();
  if (hasMedia() && c && c.duration && isFinite(c.duration)) return c.duration;
  return store.get().project?.source.duration || 66;
}

export const playback = {
  get: () => P,
  subscribe(f: () => void) {
    listeners.add(f);
    return () => {
      listeners.delete(f);
    };
  },
  duration,
  registerVideo(el: HTMLVideoElement | null) {
    video = el;
  },
  registerAudio(el: HTMLAudioElement | null) {
    audio = el;
  },
  seek(t: number, play = false) {
    t = Math.max(0, Math.min(duration(), t));
    if (hasMedia()) els().forEach((e) => (e.currentTime = t));
    set({ t });
    if (play) playback.play();
  },
  play() {
    if (P.t >= duration() - 0.05) playback.seek(0);
    if (hasMedia()) els().forEach((e) => e.play().catch(() => {}));
    set({ playing: true });
  },
  pause() {
    els().forEach((e) => e.pause());
    set({ playing: false });
  },
  toggle() {
    if (P.playing) playback.pause();
    else playback.play();
  },
  setLoop(loop: [number, number] | null) {
    set({ loop });
    if (loop) playback.seek(loop[0], true);
  },
};

let last = typeof performance !== 'undefined' ? performance.now() : 0;
function frame(now: number) {
  const dt = (now - last) / 1000;
  last = now;
  if (P.playing) {
    const c = clockEl();
    let t = hasMedia() && c ? c.currentTime : P.t + dt;
    // keep the muted video locked to the audio clock (no drift between picture and instrumental)
    if (audio && video && video.src && Math.abs(video.currentTime - t) > 0.15) video.currentTime = t;
    const d = duration();
    if (P.loop && t >= P.loop[1]) {
      t = P.loop[0];
      if (hasMedia()) els().forEach((e) => (e.currentTime = t));
    }
    if (t >= d) {
      set({ t: d });
      playback.pause();
    } else set({ t });
  }
  requestAnimationFrame(frame);
}
if (typeof window !== 'undefined' && typeof requestAnimationFrame !== 'undefined') requestAnimationFrame(frame);

export function usePlayback(): PlayState {
  return useSyncExternalStore(playback.subscribe, playback.get, playback.get);
}
