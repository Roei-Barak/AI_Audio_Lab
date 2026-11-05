import { r2 } from '../lib/format';
import type { Conf, Line, Project, Source, Word } from './types';

type Raw = [number, number, string, Record<string, Conf>?];

// Demo transcription (stands in for the AI output until a processing backend is connected).
const RAW: Raw[] = [
  [12.0, 16.2, 'הלילה יורד לאט על העיר'],
  [16.8, 21.0, 'ואני סופר את האורות בדרך'],
  [21.6, 25.4, 'כל חלון שנדלק הוא סיפור'],
  [26.0, 29.2, 'ואת בכל אחד מהם'],
  [30.0, 34.6, 'ירח של אוגוסט, תישאר עוד רגע', { 'אוגוסט,': 'low' }],
  [35.0, 38.8, 'תאיר לי את הדרך הביתה'],
  [39.4, 43.8, 'ירח של אוגוסט, אל תלך מכאן'],
  [45.0, 51.4, 'בקיץ של 1999 שרנו Hey Jude עד 3 בבוקר', { '1999': 'low', Jude: 'med', '3': 'med' }],
  [52.0, 55.6, 'וכל אחד נשא את שמך', { 'נשא': 'low' }],
  [56.4, 61.0, 'ירח של אוגוסט, תישאר עוד רגע'],
];

export const uid = (p: string) => p + Math.random().toString(36).slice(2, 10);

/** Distributes word timings across a span proportionally to word length. */
export function makeWords(s: number, e: number, text: string, conf?: Record<string, Conf>): Word[] {
  const ws = text.split(/\s+/).filter(Boolean);
  const tot = ws.reduce((a, w) => a + w.length + 1, 0);
  const span = e - s - 0.08 * ws.length;
  let t = s;
  return ws.map((w) => {
    const d = (span * (w.length + 1)) / tot;
    const o: Word = { id: uid('w'), w, s: r2(t), e: r2(t + d), conf: conf?.[w] ?? 'high', manual: false };
    t += d + 0.08;
    return o;
  });
}

export function makeLine(s: number, e: number, text: string, conf?: Record<string, Conf>): Line {
  return { id: uid('l'), start: s, end: e, speaker: '1', words: makeWords(s, e, text, conf), manual: false };
}

export const DEMO_YT_SOURCE: Source = {
  kind: 'youtube',
  url: 'https://youtu.be/lyra-demo',
  name: 'ירח של אוגוסט (Official Video)',
  duration: 66,
  hasVideo: true,
};

export function demoProject(source: Source = DEMO_YT_SOURCE): Project {
  return {
    id: uid('prj_'),
    title: 'ירח של אוגוסט',
    artist: 'נועה ברק',
    createdAt: Date.now(),
    source,
    processing: { profile: 'מהיר', stems: 2, language: 'auto' },
    stems: ['vocals', 'instrumental'],
    timing: 'word',
    lines: RAW.map((r) => makeLine(r[0], r[1], r[2], r[3])),
    style: {
      template: 'classic',
      color: 'accent',
      size: 44,
      position: 'bottom',
      outline: true,
      shadow: true,
      aspect: '16:9',
      background: 'video',
    },
    audio: { track: 'instrumental', originalMuted: true },
    export: { format: 'mp4', res: '1080p' },
    history: [{ at: Date.now(), label: 'נוצר מתמלול AI (מדומה)' }],
    rev: 1,
  };
}

/** Simulated alternate AI transcription, used for the "AI re-run as suggestion" comparison. */
export const AI_ALTERNATES: Record<string, string> = {
  'נשא': 'נושא',
  '1999': '99',
  '3': 'שלוש',
  'אוגוסט,': 'אוגוסט',
};

/** The other collaborator's version of line 7 in the simulated conflict. */
export const THEIRS_LINE_INDEX = 6;
export const THEIRS_LINE_TEXT = 'ירח של אוגוסט, אל תלך מפה';
export const OTHER_USER = 'מיכל לוי';
