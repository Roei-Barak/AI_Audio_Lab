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

/** ASS karaoke: \\kf word sweeps when word timing exists; same output as the server renderer. */
export function toAss(p: Project): string {
  const st = p.style;
  const [W, H] = st.aspect === '9:16' ? [1080, 1920] : st.aspect === '1:1' ? [1080, 1080] : [1920, 1080];
  const size = Math.round(st.size * (H / 1080) * (st.template === 'bold' ? 1.2 : 1) * (st.aspect !== '16:9' ? 1.6 : 1));
  const align = { top: 8, middle: 5, bottom: 2 }[st.position];
  const hi = { accent: '&H00D98491', soft: '&H00FAF0F5', light: '&H00FEF5F3', deep: '&H00B05A72' }[st.color];
  const base = st.template === 'minimal' ? '&H00A0A0A0' : '&H00FFFFFF';
  const t = (x: number) => {
    x = Math.max(0, x);
    const h = Math.floor(x / 3600), m = Math.floor((x % 3600) / 60), sec = x % 60;
    return h + ':' + String(m).padStart(2, '0') + ':' + sec.toFixed(2).padStart(5, '0');
  };
  const esc = (w: string) => w.replace(/[{}]/g, '').replace(/\\/g, '/');
  const head = `[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nPlayResX: ${W}\nPlayResY: ${H}\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Karaoke,Noto Sans Hebrew,${size},${hi},${base},&H00101010,&H90000000,-1,0,0,0,100,100,0,0,1,${st.outline ? 3 : 0},${st.shadow ? 2 : 0},${align},60,60,${Math.round(H * (align === 2 ? 0.09 : 0.14))},1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n`;
  const ev = p.lines.filter((l) => l.words.length).map((l) => {
    const start = Math.max(0, l.words[0].s - 0.4);
    let text: string;
    if (p.timing === 'word') {
      let cur = start;
      text = l.words.map((w) => {
        const lead = w.s - cur > 0.01 ? `{\\k${Math.round((w.s - cur) * 100)}}` : '';
        cur = w.e;
        return lead + `{\\kf${Math.max(1, Math.round((w.e - w.s) * 100))}}` + esc(w.w);
      }).join(' ');
    } else text = l.words.map((w) => esc(w.w)).join(' ');
    return `Dialogue: 0,${t(start)},${t(l.words[l.words.length - 1].e + 0.3)},Karaoke,,0,0,0,,${text}`;
  });
  return head + ev.join('\n') + '\n';
}

/**
 * Demo-mode fallback for pasted lyrics: keeps existing word timings in order where the word count allows and
 * spreads the rest across the old line spans. The server aligns against the real transcription instead.
 */
export function alignLyricsLocal(text: string, p: Project): Line[] {
  const rows = text.split('\n').map((l) => l.trim().split(/\s+/).filter(Boolean)).filter((l) => l.length);
  const old = p.lines.flatMap((l) => l.words);
  const t0 = old[0]?.s ?? 1, t1 = old[old.length - 1]?.e ?? (p.source.duration || 60) - 1;
  const total = rows.reduce((a, r) => a + r.length, 0);
  let k = 0;
  return rows.map((ws, i) => {
    const words = ws.map((w, j) => {
      const src = total === old.length ? old[k] : null;
      const s = src ? src.s : t0 + ((t1 - t0) * k) / total;
      const e = src ? src.e : t0 + ((t1 - t0) * (k + 0.85)) / total;
      k++;
      return { id: 'w' + i + '_' + j + Math.random().toString(36).slice(2, 6), w, s: r2(s), e: r2(e), conf: src ? ('med' as const) : ('low' as const), manual: false };
    });
    return { id: 'l' + i + Math.random().toString(36).slice(2, 6), start: words[0].s, end: words[words.length - 1].e, speaker: '1' as const, words, manual: true };
  });
}

export type ExportKind = 'srt' | 'txt' | 'json' | 'lrc' | 'ass';

const EXPORTERS: Record<ExportKind, [(p: Project) => string, string]> = {
  srt: [toSrt, 'application/x-subrip'],
  txt: [toTxt, 'text/plain'],
  json: [toJson, 'application/json'],
  lrc: [toLrc, 'text/plain'],
  ass: [toAss, 'text/x-ssa'],
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
