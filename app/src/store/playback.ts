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

function set(patch: Partial<PlayState>) {
  P = { ...P, ...patch };
  listeners.forEach((f) => f());
}

const hasMedia = () => !!(store.get().media && video);

function duration(): number {
  if (hasMedia() && video!.duration && isFinite(video!.duration)) return video!.duration;
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
  seek(t: number, play = false) {
    t = Math.max(0, Math.min(duration(), t));
    if (hasMedia()) video!.currentTime = t;
    set({ t });
    if (play) playback.play();
  },
  play() {
    if (P.t >= duration() - 0.05) playback.seek(0);
    if (hasMedia()) video!.play().catch(() => {});
    set({ playing: true });
  },
  pause() {
    video?.pause();
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
    let t = hasMedia() ? video!.currentTime : P.t + dt;
    const d = duration();
    if (P.loop && t >= P.loop[1]) {
      t = P.loop[0];
      if (hasMedia()) video!.currentTime = t;
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
