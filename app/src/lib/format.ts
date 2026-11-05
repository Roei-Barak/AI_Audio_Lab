import type { Line, Project, Word } from '../store/types';

export const r2 = (x: number) => Math.round(x * 100) / 100;

/** m:ss.cc */
export function fmt(t: number): string {
  t = Math.max(0, t);
  return Math.floor(t / 60) + ':' + (t % 60).toFixed(2).padStart(5, '0');
}

/** m:ss */
export const fmtShort = (t: number) => fmt(t).split('.')[0];

/** hh:mm:ss,mmm */
export function fmtSrt(t: number): string {
  const ms = Math.max(0, Math.round(t * 1000));
  return (
    [Math.floor(ms / 3600000), Math.floor(ms / 60000) % 60, Math.floor(ms / 1000) % 60]
      .map((x) => String(x).padStart(2, '0'))
      .join(':') +
    ',' +
    String(ms % 1000).padStart(3, '0')
  );
}

export const lineText = (l: Pick<Line, 'words'>) => l.words.map((w) => w.w).join(' ');

export const lineSpan = (l: Line): [number, number] =>
  l.words.length ? [l.words[0].s, l.words[l.words.length - 1].e] : [l.start, l.end];

export const HEB = /[֐-׿]/;
const LATIN = /[A-Za-z]/;

/**
 * Groups consecutive Latin-script words (and numbers adjacent to them) into LTR runs so a
 * row of per-word chips laid out in an RTL container reads in the correct order
 * (e.g. "Hey Jude" stays "Hey Jude" inside a Hebrew line).
 */
export function bidiRuns<T extends Pick<Word, 'w'>>(words: T[]): { ltr: boolean; items: T[] }[] {
  const kind = (w: string) => (HEB.test(w) ? 'rtl' : LATIN.test(w) ? 'ltr' : 'neutral');
  const out: { ltr: boolean; items: T[] }[] = [];
  words.forEach((w, i) => {
    let k = kind(w.w);
    if (k === 'neutral') {
      const prev = i > 0 ? kind(words[i - 1].w) : 'rtl';
      const next = i < words.length - 1 ? kind(words[i + 1].w) : 'rtl';
      k = prev === 'ltr' && next === 'ltr' ? 'ltr' : 'rtl';
    }
    const ltr = k === 'ltr';
    const last = out[out.length - 1];
    if (ltr && last && last.ltr) last.items.push(w);
    else out.push({ ltr, items: [w] });
  });
  return out;
}

export function toSrt(p: Project): string {
  return p.lines
    .map((l, i) => {
      const [a, b] = lineSpan(l);
      return i + 1 + '\n' + fmtSrt(a) + ' --> ' + fmtSrt(b) + '\n' + lineText(l) + '\n';
    })
    .join('\n');
}

export function toTxt(p: Project): string {
  return p.title + ' / ' + p.artist + '\n\n' + p.lines.map(lineText).join('\n') + '\n';
}

export function toJson(p: Project): string {
  return JSON.stringify(
    {
      format: 'lyra-project',
      version: 4,
      exportedAt: new Date().toISOString(),
      note: 'Media files (video, stems) are not included.',
      project: p,
    },
    null,
    2,
  );
}

/** Enhanced LRC: line timestamps plus per-word <mm:ss.xx> marks when word timing exists. */
export function toLrc(p: Project): string {
  const ts = (t: number) => {
    t = Math.max(0, t);
    return String(Math.floor(t / 60)).padStart(2, '0') + ':' + (t % 60).toFixed(2).padStart(5, '0');
  };
  const head = '[ti:' + p.title + ']\n[ar:' + p.artist + ']\n';
  return head + p.lines.map((l) => '[' + ts(lineSpan(l)[0]) + ']' + (p.timing === 'word' ? l.words.map((w) => '<' + ts(w.s) + '>' + w.w).join(' ') : lineText(l))).join('\n') + '\n';
}

export type ExportKind = 'srt' | 'txt' | 'json' | 'lrc';

const EXPORTERS: Record<ExportKind, [(p: Project) => string, string]> = {
  srt: [toSrt, 'application/x-subrip'],
  txt: [toTxt, 'text/plain'],
  json: [toJson, 'application/json'],
  lrc: [toLrc, 'text/plain'],
};

/** Real client-side download built from the current (edited) project data. */
export function downloadExport(p: Project, kind: ExportKind, basename = 'yareach-shel-august') {
  const [fn, mime] = EXPORTERS[kind];
  const blob = new Blob(['﻿' + fn(p)], { type: mime + ';charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = basename + '.' + kind;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(a.href);
    a.remove();
  }, 800);
}

export const hhmmss = () => {
  const d = new Date();
  return [d.getHours(), d.getMinutes(), d.getSeconds()].map((x) => String(x).padStart(2, '0')).join(':');
};

export const hhmm = (at: number) => {
  const d = new Date(at);
  return String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
};
