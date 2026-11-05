import { useEffect, useMemo, useRef, useState } from 'react';
import { KaraokePreview } from '../../components/KaraokePreview';
import { WordChips } from '../../components/controls';
import { fmt, hhmm, lineSpan, lineText, r2 } from '../../lib/format';
import { makeWords, uid } from '../../store/demo';
import { playback, usePlayback } from '../../store/playback';
import { store, useStore } from '../../store/store';
import type { Line, Project, Speaker, Word } from '../../store/types';
import { ui, type Route } from '../../store/ui';
import { saveCheckpoint } from '../../lib/actions';

export type EditorMode = 'timeline' | 'lyrics' | 'word';
const ROUTE_OF: Record<EditorMode, Route> = { timeline: 'subs', lyrics: 'subs-lyrics', word: 'subs-word' };
const SPK: Record<Speaker, string> = { '1': 'זמר/ת 1', '2': 'זמר/ת 2', duet: 'דואט' };
const NEXT_SPK: Record<Speaker, Speaker> = { '1': '2', '2': 'duet', duet: '1' };
const CONF_L = { high: 'גבוה', med: 'בינוני', low: 'נמוך' };

type DragKind = 'clip' | 'left' | 'right' | 'word';
interface Drag {
  kind: DragKind;
  line: string;
  word?: string;
  x0: number;
  dx: number;
}

/** Pure drag application, shared by live preview and commit. */
export function applyDrag(l: Line, d: Pick<Drag, 'kind' | 'dx' | 'word'>, linked: boolean) {
  const dx = d.dx;
  if (d.kind === 'clip') {
    const c = Math.max(-l.start, dx);
    l.start = r2(l.start + c);
    l.end = r2(l.end + c);
    if (linked)
      l.words.forEach((w) => {
        w.s = r2(w.s + c);
        w.e = r2(w.e + c);
      });
  }
  if (d.kind === 'left') l.start = r2(Math.min(l.end - 0.3, Math.max(0, l.start + dx)));
  if (d.kind === 'right') l.end = r2(Math.max(l.start + 0.3, l.end + dx));
  if (d.kind === 'word') {
    const i = l.words.findIndex((w) => w.id === d.word);
    const w = l.words[i];
    if (!w) return;
    const pv = l.words[i - 1];
    const nx = l.words[i + 1];
    const lo = pv ? pv.e - w.s : -w.s;
    const hi = nx ? nx.s - w.e : 99;
    const c = Math.max(lo, Math.min(hi, dx));
    w.s = r2(w.s + c);
    w.e = r2(w.e + c);
    w.manual = true;
  }
}

interface ZDrag {
  word: string;
  kind: 'move' | 's' | 'e';
  x0: number;
  dt: number;
  k: number;
}

/** Word Zoom drag: move or resize a word without overlapping its neighbours. */
export function constrainWord(l: Line, w: Word, d: Pick<ZDrag, 'kind' | 'dt'>): Word {
  const i = l.words.findIndex((x) => x.id === w.id);
  const pv = l.words[i - 1];
  const nx = l.words[i + 1];
  const lo = pv ? pv.e : 0;
  const hi = nx ? nx.s : 99999;
  if (d.kind === 'move') {
    const c = Math.max(lo - w.s, Math.min(hi - w.e, d.dt));
    return { ...w, s: r2(w.s + c), e: r2(w.e + c) };
  }
  if (d.kind === 's') return { ...w, s: r2(Math.min(w.e - 0.03, Math.max(lo, w.s + d.dt))) };
  return { ...w, e: r2(Math.max(w.s + 0.03, Math.min(hi, w.e + d.dt))) };
}

export function LyricsStudio({ mode }: { mode: EditorMode }) {
  const { project: p, canUndo, canRedo } = useStore();
  const { t, loop } = usePlayback();
  const [sel, setSel] = useState<string | null>(null);
  const [selW, setSelW] = useState<string | null>(null);
  const [wDraft, setWDraft] = useState('');
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [pps, setPps] = useState(70);
  const [linked, setLinked] = useState(true);
  const [freeWords, setFreeWords] = useState(true);
  const [drag, setDrag] = useState<Drag | null>(null);
  const tlRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<Drag | null>(null);
  const [zDrag, setZDrag] = useState<ZDrag | null>(null);
  const zDragRef = useRef<ZDrag | null>(null);
  const zDragged = (l: Line, w: Word, d: ZDrag | null = zDrag): Word => (d ? constrainWord(l, w, d) : w);

  const setMode = (m: EditorMode) => ui.go(ROUTE_OF[m]);

  // Default selection: the mixed Hebrew/English/numbers line, 4th word.
  useEffect(() => {
    const P = store.ensure();
    if (sel && P.lines.some((l) => l.id === sel)) return;
    const l = P.lines[7] || P.lines[0];
    const w = l.words[3] || l.words[0];
    setSel(l.id);
    setSelW(w?.id ?? null);
    setWDraft(w?.w ?? '');
  }, [p, sel]);

  // Drafts reset whenever project data changes underneath (undo/redo, other views).
  useEffect(() => setDrafts({}), [p?.rev, p]);

  const selLine = p?.lines.find((l) => l.id === sel);
  const selWord = selLine?.words.find((w) => w.id === selW);
  useEffect(() => {
    if (selWord) setWDraft(selWord.w);
  }, [selWord?.id, selWord?.w]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (mode !== 'timeline') return;
    const el = tlRef.current;
    if (el && selLine) el.scrollLeft = Math.max(0, selLine.start * pps - 240);
  }, [mode, sel, pps]); // eslint-disable-line react-hooks/exhaustive-deps

  const editLine = (label: string, fn: (l: Line, P: Project) => void | boolean) =>
    store.edit(label, (P) => {
      const l = P.lines.find((x) => x.id === sel);
      if (!l) return false;
      return fn(l, P);
    });
  const editWord = (label: string, fn: (w: Word, i: number, l: Line) => void | boolean) =>
    editLine(label, (l) => {
      const i = l.words.findIndex((w) => w.id === selW);
      if (i < 0) return false;
      const r = fn(l.words[i], i, l);
      if (r === false) return false;
      l.manual = true;
    });

  const pickWord = (l: Line, w: Word) => {
    setSel(l.id);
    setSelW(w.id);
    setWDraft(w.w);
    playback.seek(w.s - 0.1);
  };
  const toggleLoop = () => {
    if (loop || !selWord) return playback.setLoop(null);
    playback.setLoop([Math.max(0, selWord.s - 0.15), selWord.e + 0.15]);
  };
  const splitWord = () =>
    editWord('פיצול מילה', (w, i, l) => {
      const ch = Array.from(w.w);
      if (ch.length < 2) return false;
      const k = Math.ceil(ch.length / 2);
      const mid = r2((w.s + w.e) / 2);
      const b: Word = { ...w, id: uid('w'), w: ch.slice(k).join(''), s: mid, manual: true };
      w.w = ch.slice(0, k).join('');
      w.e = mid;
      w.manual = true;
      l.words.splice(i + 1, 0, b);
    });
  const mergeWord = () =>
    editWord('מיזוג מילים', (w, i, l) => {
      const b = l.words[i + 1];
      if (!b) return false;
      const rank = { low: 0, med: 1, high: 2 };
      w.w = w.w + b.w;
      w.e = b.e;
      w.conf = rank[b.conf] < rank[w.conf] ? b.conf : w.conf;
      w.manual = true;
      l.words.splice(i + 1, 1);
    });
  const cycleSpeaker = () => editLine('החלפת דובר', (l) => void (l.speaker = NEXT_SPK[l.speaker]));
  const nudge = (k: 's' | 'e', dv: number) =>
    editWord('שינוי תזמון מילה', (w, i, l) => {
      const pv = l.words[i - 1];
      const nx = l.words[i + 1];
      if (k === 's') w.s = r2(Math.min(w.e - 0.03, Math.max(pv ? pv.e : 0, w.s + dv)));
      else w.e = r2(Math.max(w.s + 0.03, Math.min(nx ? nx.s : 9999, w.e + dv)));
      w.manual = true;
    });

  // Keyboard shortcuts
  const keyRef = useRef<(e: KeyboardEvent) => void>();
  keyRef.current = (e: KeyboardEvent) => {
    if (/INPUT|TEXTAREA|SELECT/.test((e.target as HTMLElement).tagName)) return;
    const mod = e.metaKey || e.ctrlKey;
    if (mod && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      return e.shiftKey ? store.redo() : store.undo();
    }
    if (mod && e.key.toLowerCase() === 'y') {
      e.preventDefault();
      return store.redo();
    }
    if (mod || e.altKey) return;
    if (e.key === '1') return setMode('timeline');
    if (e.key === '2') return setMode('lyrics');
    if (e.key === '3') return setMode('word');
    if (e.key === '?') return ui.open('shortcuts');
    if (e.key === 'd' || e.key === 'D') return cycleSpeaker();
    if (mode !== 'word' || !selLine) return;
    const i = selLine.words.findIndex((w) => w.id === selW);
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      // RTL: left arrow = next word in reading order
      const ni = Math.max(0, Math.min(selLine.words.length - 1, i + (e.key === 'ArrowLeft' ? 1 : -1)));
      pickWord(selLine, selLine.words[ni]);
    }
    if (e.key === 'l' || e.key === 'L') toggleLoop();
    if (e.key === 's' || e.key === 'S') splitWord();
    if (e.key === 'm' || e.key === 'M') mergeWord();
  };
  useEffect(() => {
    const f = (e: KeyboardEvent) => keyRef.current?.(e);
    window.addEventListener('keydown', f);
    return () => {
      window.removeEventListener('keydown', f);
      playback.setLoop(null);
    };
  }, []);

  // Timeline drag
  const startDrag = (e: React.PointerEvent, d: Omit<Drag, 'x0' | 'dx'>) => {
    e.stopPropagation();
    e.preventDefault();
    dragRef.current = { ...d, x0: e.clientX, dx: 0 };
    setDrag(dragRef.current);
    const mv = (ev: PointerEvent) => {
      if (!dragRef.current) return;
      dragRef.current = { ...dragRef.current, dx: (ev.clientX - dragRef.current.x0) / pps };
      setDrag(dragRef.current);
    };
    const up = () => {
      window.removeEventListener('pointermove', mv);
      window.removeEventListener('pointerup', up);
      const d2 = dragRef.current;
      dragRef.current = null;
      setDrag(null);
      if (!d2) return;
      const P = store.get().project!;
      const l = P.lines.find((x) => x.id === d2.line)!;
      if (Math.abs(d2.dx) < 0.02) {
        if (d2.kind === 'word') pickWord(l, l.words.find((x) => x.id === d2.word)!);
        else {
          setSel(d2.line);
          playback.seek(l.start);
        }
        return;
      }
      const lbl = { clip: linked ? 'הזזת מקטע + מילים' : 'הזזת מקטע בלבד', left: 'שינוי התחלת מקטע', right: 'שינוי סיום מקטע', word: 'הזזת מילה' }[d2.kind];
      store.edit(lbl, (P2) => {
        const l2 = P2.lines.find((x) => x.id === d2.line);
        if (!l2) return false;
        applyDrag(l2, d2, linked);
        l2.manual = true;
      });
      setSel(d2.line);
    };
    window.addEventListener('pointermove', mv);
    window.addEventListener('pointerup', up);
  };

  const dur = p?.source.duration || 66;
  const eff = (l: Line): Line => {
    if (!drag || drag.line !== l.id) return l;
    const c: Line = JSON.parse(JSON.stringify(l));
    applyDrag(c, drag, linked);
    return c;
  };

  const wave = useMemo(() => {
    if (!p) return [];
    const out: number[] = [];
    for (let i = 0; i < Math.floor(dur * 4); i++) {
      const tt = i / 4;
      const sing = p.lines.some((l) => tt >= l.start && tt <= l.end);
      out.push(Math.round((sing ? 18 : 4) + (sing ? 30 : 6) * Math.abs(Math.sin(i * 1.7) * Math.cos(i * 0.31))));
    }
    return out;
  }, [p, dur]);

  if (!p) return null;

  const outN = p.lines.reduce((a, l) => a + l.words.filter((w) => w.s < l.start - 0.01 || w.e > l.end + 0.01).length, 0);
  const zl = selLine || p.lines[0];
  const zs = lineSpan(zl);
  const z0 = Math.min(zl.start, zs[0]) - 0.4;
  const z1 = Math.max(zl.end, zs[1]) + 0.4;
  const zd = z1 - z0;
  const zp = (x: number) => ((x - z0) / zd) * 100;

  return (
    <div dir="rtl" data-screen-label="Lyrics Studio" style={{ height: '100%', display: 'flex', flexDirection: 'column', background: 'var(--color-bg)', overflow: 'hidden' }}>
      <header className="no-scrollbar" style={{ minHeight: 56, flex: 'none', display: 'flex', alignItems: 'center', gap: 12, padding: '0 20px', overflowX: 'auto', whiteSpace: 'nowrap', borderBottom: '1px solid color-mix(in srgb, var(--color-text) 7%, transparent)' }}>
        <span style={{ fontSize: 17 }}>עורך מילים</span>
        <span className="tag tag-neutral">{p.title}</span>
        <div className="seg" role="tablist" aria-label="מצב עורך">
          {([['timeline', 'ציר זמן', 'ph ph-align-left', '1'], ['lyrics', 'מילים תחילה', 'ph ph-text-align-right', '2'], ['word', 'זום מילה', 'ph ph-magnifying-glass-plus', '3']] as const).map(([k, l, i, kk]) => (
            <button key={k} role="tab" className="seg-opt" aria-selected={mode === k} aria-pressed={mode === k} onClick={() => setMode(k)} style={{ fontSize: 13 }}>
              <i className={i} />
              {l}
              <span className="dim" style={{ fontSize: 10.5 }}>{kk}</span>
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          <button className="btn btn-ghost btn-icon" onClick={() => store.undo()} disabled={!canUndo} title="Undo ⌘Z" aria-label="Undo"><i className="ph ph-arrow-counter-clockwise" /></button>
          <button className="btn btn-ghost btn-icon" onClick={() => store.redo()} disabled={!canRedo} title="Redo ⇧⌘Z" aria-label="Redo"><i className="ph ph-arrow-clockwise" /></button>
        </div>
        <span className="dim hide-narrow" style={{ fontSize: 12, maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {p.history[0] ? 'אחרון: ' + p.history[0].label : ''}
        </span>
        <div style={{ marginInlineStart: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
          <button className="btn btn-ghost" onClick={() => ui.open('shortcuts')} aria-label="קיצורי מקלדת"><i className="ph ph-keyboard" /></button>
          <button className="btn btn-ghost" onClick={() => ui.open('airerun')}><i className="ph ph-sparkle" />הרץ AI מחדש (הצעה)</button>
          <button className="btn btn-ghost" onClick={() => ui.open('find')} aria-label="חיפוש והחלפה" title="חיפוש והחלפה ⌘F"><i className="ph ph-magnifying-glass" /></button>
          <button className="btn btn-ghost" onClick={() => { saveCheckpoint(); ui.toast('נשמר · נוצרה נקודת שמירה'); }}><i className="ph ph-floppy-disk" />שמור</button>
          <button className="btn btn-secondary" onClick={() => ui.go('karaoke')}><i className="ph ph-video-camera" />לעיצוב הקריוקי</button>
        </div>
      </header>
      <div style={{ flex: 1, minHeight: 0, display: 'flex' }}>
        <main style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          {mode === 'timeline' && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '10px 20px', fontSize: 12.5, flexWrap: 'wrap' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                  <input type="checkbox" checked={linked} onChange={() => setLinked(!linked)} style={{ accentColor: 'var(--color-accent)' }} />
                  הזזת מקטע מזיזה את המילים שלו
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                  <input type="checkbox" checked={freeWords} onChange={() => setFreeWords(!freeWords)} style={{ accentColor: 'var(--color-accent)' }} />
                  הזזת מילים בנפרד
                </label>
                <span className="dim">גרור מקטע להזזה · גרור קצה לשינוי אורך · לחץ על הסרגל לקפיצה</span>
                <div style={{ marginInlineStart: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <i className="ph ph-magnifying-glass-minus" />
                  <input type="range" aria-label="זום" min={20} max={240} step={10} value={pps} onChange={(e) => setPps(parseInt(e.target.value, 10))} style={{ width: 140, accentColor: 'var(--color-accent)' }} />
                  <i className="ph ph-magnifying-glass-plus" />
                  <span className="num" style={{ width: 62 }}>{pps} px/s</span>
                </div>
              </div>
              {outN > 0 && (
                <div role="alert" className="alert" style={{ margin: '0 20px 8px', padding: '8px 12px', fontSize: 12.5, alignItems: 'center' }}>
                  <i className="ph ph-warning" style={{ color: 'var(--color-accent-200)' }} />
                  {outN} מילים מחוץ לגבולות המקטע שלהן. הן עדיין יופיעו בכתוביות לפי הזמן שלהן.
                </div>
              )}
              {/* Time flows left-to-right like every audio tool; the rest of the UI is RTL. */}
              <div ref={tlRef} dir="ltr" className="panel" style={{ flex: 1, minHeight: 0, overflow: 'auto', margin: '0 20px 16px' }}>
                <div style={{ position: 'relative', width: Math.round(dur * pps + 40), height: '100%', minHeight: 300 }}>
                  <div
                    onPointerDown={(e) => {
                      const r = e.currentTarget.getBoundingClientRect();
                      playback.seek((e.clientX - r.left) / pps);
                    }}
                    style={{ position: 'absolute', left: 0, right: 0, top: 0, height: 28, cursor: 'pointer', borderBottom: '1px solid var(--color-divider)' }}
                  >
                    {Array.from({ length: Math.floor(dur) + 1 }, (_, s) => (
                      <div key={s} style={{ position: 'absolute', left: s * pps, bottom: 0, height: s % 5 === 0 ? 14 : 6, borderLeft: '1px solid var(--color-neutral-600)' }}>
                        {s % 5 === 0 && <span className="dim" style={{ position: 'absolute', left: 4, bottom: 6, fontSize: 10.5 }}>{Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0')}</span>}
                      </div>
                    ))}
                  </div>
                  <TrackLabel top={36}>שורות</TrackLabel>
                  {p.lines.map((l0) => {
                    const l = eff(l0);
                    const on = l.id === sel;
                    return (
                      <div
                        key={l.id}
                        data-testid="clip"
                        onPointerDown={(e) => startDrag(e, { kind: 'clip', line: l.id })}
                        style={{ position: 'absolute', left: Math.round(l.start * pps), width: Math.max(12, Math.round((l.end - l.start) * pps)), top: 52, height: 46, borderRadius: 6, cursor: 'grab', background: on ? 'var(--color-accent-900)' : 'var(--color-neutral-800)', boxShadow: on ? 'inset 0 0 0 1px var(--color-accent-500)' : 'inset 0 0 0 1px var(--color-neutral-700)', overflow: 'hidden', touchAction: 'none' }}
                      >
                        <div onPointerDown={(e) => startDrag(e, { kind: 'left', line: l.id })} style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 8, cursor: 'ew-resize', background: 'color-mix(in srgb, var(--color-accent) 30%, transparent)' }} />
                        <div dir="auto" style={{ padding: '5px 12px', fontSize: 12.5, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', pointerEvents: 'none' }}>{lineText(l)}</div>
                        <div className="muted" style={{ padding: '0 12px', fontSize: 10.5, pointerEvents: 'none' }}>{SPK[l.speaker]}</div>
                        <div onPointerDown={(e) => startDrag(e, { kind: 'right', line: l.id })} style={{ position: 'absolute', right: 0, top: 0, bottom: 0, width: 8, cursor: 'ew-resize', background: 'color-mix(in srgb, var(--color-accent) 30%, transparent)' }} />
                      </div>
                    );
                  })}
                  <TrackLabel top={108}>מילים</TrackLabel>
                  {p.lines.flatMap((l0) => {
                    const l = eff(l0);
                    return l.words.map((w) => {
                      const on = w.id === selW;
                      const outside = w.s < l.start - 0.01 || w.e > l.end + 0.01;
                      return (
                        <div
                          key={w.id}
                          dir="auto"
                          onPointerDown={(e) => (freeWords ? startDrag(e, { kind: 'word', line: l.id, word: w.id }) : pickWord(l, w))}
                          style={{ position: 'absolute', left: Math.round(w.s * pps), width: Math.max(6, Math.round((w.e - w.s) * pps) - 1), top: 124, height: 34, borderRadius: 5, cursor: freeWords ? 'grab' : 'pointer', background: outside ? 'color-mix(in srgb, var(--color-accent-800) 60%, var(--color-bg))' : on ? 'var(--color-accent-800)' : 'var(--color-neutral-700)', boxShadow: w.conf === 'low' && !w.manual ? 'inset 0 -2px 0 var(--color-accent-300)' : on ? 'inset 0 0 0 1px var(--color-accent-300)' : 'none', fontSize: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', whiteSpace: 'nowrap', touchAction: 'none' }}
                        >
                          {w.w}
                        </div>
                      );
                    });
                  })}
                  <TrackLabel top={170}>Vocals</TrackLabel>
                  <div style={{ position: 'absolute', left: 0, right: 0, top: 186, height: 56, display: 'flex', alignItems: 'center', gap: 2, overflow: 'hidden', opacity: 0.6 }}>
                    {wave.map((b, i) => (
                      <div key={i} style={{ flex: 'none', width: Math.max(1, pps / 4 - 2), height: b, borderRadius: 2, background: 'var(--color-accent-700)' }} />
                    ))}
                  </div>
                  <div style={{ position: 'absolute', top: 0, bottom: 0, left: Math.round(t * pps), width: 2, background: 'var(--color-accent)', pointerEvents: 'none' }} />
                </div>
              </div>
            </>
          )}

          {mode === 'lyrics' && (
            <div style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '12px 20px 20px', display: 'flex', flexDirection: 'column', gap: 4 }}>
              {p.lines.map((l, i) => (
                <LyricsRow
                  key={l.id}
                  l={l}
                  i={i}
                  on={l.id === sel}
                  draft={drafts[l.id] ?? lineText(l)}
                  setDraft={(v) => setDrafts((d) => ({ ...d, [l.id]: v }))}
                  pick={() => l.id !== sel && setSel(l.id)}
                  setSel={setSel}
                />
              ))}
            </div>
          )}

          {mode === 'word' && (
            <div style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '14px 20px 20px', display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }} role="tablist" aria-label="שורות">
                {p.lines.map((l, i) => (
                  <button
                    key={l.id}
                    dir="auto"
                    className="pill sm"
                    aria-pressed={l.id === sel}
                    onClick={() => {
                      setSel(l.id);
                      if (l.words[0]) pickWord(l, l.words[0]);
                    }}
                    style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', fontSize: 12 }}
                  >
                    {i + 1} · {lineText(l)}
                  </button>
                ))}
              </div>
              <div dir="ltr" data-zoom className="panel" style={{ position: 'relative', height: 150, overflow: 'hidden', flex: 'none' }}>
                <div style={{ position: 'absolute', left: 0, right: 0, top: 0, height: 22, borderBottom: '1px solid var(--color-divider)' }}>
                  {Array.from({ length: Math.floor((z1 - Math.ceil(z0 * 2) / 2) * 2) + 1 }, (_, k) => Math.ceil(z0 * 2) / 2 + k * 0.5).map((s) => (
                    <span key={s} className="dim" style={{ position: 'absolute', left: zp(s) + '%', top: 4, fontSize: 10, borderLeft: '1px solid var(--color-neutral-700)', paddingLeft: 3 }}>{s.toFixed(1)}</span>
                  ))}
                </div>
                <div style={{ position: 'absolute', left: 0, right: 0, top: 96, height: 44, display: 'flex', alignItems: 'center', gap: 1, opacity: 0.55 }}>
                  {Array.from({ length: 160 }, (_, i) => {
                    const tt = z0 + (zd * i) / 160;
                    const on = zl.words.some((w) => tt >= w.s && tt <= w.e);
                    return <div key={i} style={{ flex: 1, height: Math.round((on ? 35 : 6) + (on ? 60 : 10) * Math.abs(Math.sin(i * 0.9) * Math.cos(i * 0.23))) + '%', borderRadius: 1, background: 'var(--color-accent-700)' }} />;
                  })}
                </div>
                {zl.words.map((w0) => {
                  const w = zDrag && zDrag.word === w0.id ? zDragged(zl, w0) : w0;
                  const on = w.id === selW;
                  const now = t >= w.s && t < w.e;
                  const startZ = (e: React.PointerEvent<HTMLElement>, kind: 'move' | 's' | 'e') => {
                    e.stopPropagation();
                    e.preventDefault();
                    const box = (e.currentTarget.closest('[data-zoom]') as HTMLElement).getBoundingClientRect();
                    const d = { word: w0.id, kind, x0: e.clientX, dt: 0, k: zd / box.width };
                    zDragRef.current = d;
                    setZDrag(d);
                    const mv = (ev: PointerEvent) => {
                      if (!zDragRef.current) return;
                      zDragRef.current = { ...zDragRef.current, dt: (ev.clientX - zDragRef.current.x0) * zDragRef.current.k };
                      setZDrag(zDragRef.current);
                    };
                    const up = () => {
                      window.removeEventListener('pointermove', mv);
                      window.removeEventListener('pointerup', up);
                      const d2 = zDragRef.current;
                      zDragRef.current = null;
                      setZDrag(null);
                      if (!d2 || Math.abs(d2.dt) < 0.01) return pickWord(zl, w0);
                      setSel(zl.id);
                      setSelW(w0.id);
                      store.edit(d2.kind === 'move' ? 'הזזת מילה' : 'שינוי תזמון מילה', (P) => {
                        const l = P.lines.find((x) => x.id === zl.id);
                        const ww = l?.words.find((x) => x.id === w0.id);
                        if (!l || !ww) return false;
                        Object.assign(ww, zDragged(l, ww, d2), { manual: true });
                        l.manual = true;
                      });
                    };
                    window.addEventListener('pointermove', mv);
                    window.addEventListener('pointerup', up);
                  };
                  return (
                    <div
                      key={w.id}
                      dir="auto"
                      data-testid="zoom-word"
                      role="button"
                      tabIndex={0}
                      onPointerDown={(e) => startZ(e, 'move')}
                      onKeyDown={(e) => e.key === 'Enter' && pickWord(zl, w0)}
                      style={{ position: 'absolute', left: zp(w.s) + '%', width: Math.max(1, zp(w.e) - zp(w.s) - 0.3) + '%', top: 34, height: 52, borderRadius: 6, fontSize: 15, cursor: 'grab', touchAction: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', background: on ? 'var(--color-accent-800)' : now ? 'var(--color-accent-900)' : 'var(--color-neutral-700)', color: on ? 'var(--color-accent-100)' : 'var(--color-text)', boxShadow: w.conf === 'low' && !w.manual ? 'inset 0 -3px 0 var(--color-accent-300)' : on ? 'inset 0 0 0 1px var(--color-accent-300)' : 'none', overflow: 'hidden', whiteSpace: 'nowrap' }}
                    >
                      <span onPointerDown={(e) => startZ(e, 's')} title="גרור לשינוי התחלה" style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 7, cursor: 'ew-resize', background: on ? 'color-mix(in srgb, var(--color-accent) 35%, transparent)' : 'transparent' }} />
                      {w.w}
                      <span onPointerDown={(e) => startZ(e, 'e')} title="גרור לשינוי סיום" style={{ position: 'absolute', right: 0, top: 0, bottom: 0, width: 7, cursor: 'ew-resize', background: on ? 'color-mix(in srgb, var(--color-accent) 35%, transparent)' : 'transparent' }} />
                    </div>
                  );
                })}
                {loop && (
                  <div style={{ position: 'absolute', top: 22, bottom: 0, left: zp(loop[0]) + '%', width: zp(loop[1]) - zp(loop[0]) + '%', background: 'color-mix(in srgb, var(--color-accent) 12%, transparent)', borderLeft: '1px dashed var(--color-accent)', borderRight: '1px dashed var(--color-accent)', pointerEvents: 'none' }} />
                )}
                <div style={{ position: 'absolute', top: 22, bottom: 0, left: Math.max(0, Math.min(100, zp(t))) + '%', width: 2, background: 'var(--color-accent)', pointerEvents: 'none' }} />
              </div>
              <WordChips words={zl.words} selected={selW} onPick={(w) => pickWord(zl, w)} />
              {selWord && (
                <div className="panel" style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'flex-end', padding: 14, boxShadow: 'none' }}>
                  <div className="field" style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    <label htmlFor="word-edit" className="muted" style={{ fontSize: 12 }}>מילה</label>
                    <input
                      id="word-edit"
                      className="input"
                      dir="auto"
                      value={wDraft}
                      onChange={(e) => setWDraft(e.target.value)}
                      onBlur={() => {
                        const v = wDraft.trim();
                        if (!v || v === selWord.w) return;
                        if (/\s/.test(v)) return setWDraft(selWord.w);
                        editWord('תיקון מילה', (w) => {
                          w.w = v;
                          w.conf = 'high';
                          w.manual = true;
                        });
                      }}
                      onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
                      style={{ width: 180, fontSize: 16 }}
                    />
                  </div>
                  {(['s', 'e'] as const).map((k) => (
                    <div key={k} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      <span className="muted" style={{ fontSize: 12 }}>{k === 's' ? 'התחלה (ms)' : 'סיום (ms)'}</span>
                      <div dir="ltr" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <button className="btn btn-ghost" onClick={() => nudge(k, -0.05)} style={{ height: 30, fontSize: 12 }}>−50</button>
                        <button className="btn btn-ghost" onClick={() => nudge(k, -0.01)} style={{ height: 30, fontSize: 12 }}>−10</button>
                        <span className="num" style={{ width: 56, textAlign: 'center', fontSize: 13 }}>{(k === 's' ? selWord.s : selWord.e).toFixed(2)}</span>
                        <button className="btn btn-ghost" onClick={() => nudge(k, 0.01)} style={{ height: 30, fontSize: 12 }}>+10</button>
                        <button className="btn btn-ghost" onClick={() => nudge(k, 0.05)} style={{ height: 30, fontSize: 12 }}>+50</button>
                      </div>
                    </div>
                  ))}
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    <button className="btn btn-secondary" onClick={toggleLoop} aria-pressed={!!loop}><i className="ph ph-repeat" />{loop ? 'עצור לולאה' : 'לולאה על המילה'} <Kbd>L</Kbd></button>
                    <button className="btn btn-ghost" onClick={splitWord}><i className="ph ph-scissors" />פצל <Kbd>S</Kbd></button>
                    <button className="btn btn-ghost" onClick={mergeWord}><i className="ph ph-arrows-in-line-horizontal" />מזג עם הבאה <Kbd>M</Kbd></button>
                    <button className="btn btn-ghost" onClick={cycleSpeaker}><i className="ph ph-user-switch" />{SPK[zl.speaker]} <Kbd>D</Kbd></button>
                  </div>
                  <span className="muted" style={{ fontSize: 12 }}>
                    ביטחון מודל: {CONF_L[selWord.conf]}
                    {selWord.manual ? ' · תוקן ידנית' : ''} · הערכת המודל, לא דיוק מאומת
                  </span>
                </div>
              )}
              <span className="dim" style={{ fontSize: 12 }}>גרור מילה להזזה או את הקצוות לשינוי תזמון · ← → מעבר בין מילים · L לולאה · S פיצול · M מיזוג · D דובר · רווח ניגון · ⌘Z / ⇧⌘Z · 1/2/3 החלפת מצב</span>
            </div>
          )}
        </main>
        <aside className="hide-narrow" style={{ width: 400, flex: 'none', display: 'flex', flexDirection: 'column', gap: 12, padding: 14, borderInlineStart: '1px solid color-mix(in srgb, var(--color-text) 7%, transparent)', minHeight: 0, overflow: 'auto' }}>
          <div style={{ height: 290, flex: 'none' }}>
            <KaraokePreview />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: 12, borderRadius: 'var(--radius-lg)', background: 'var(--color-surface)' }}>
            <span className="kicker">שורה נבחרת</span>
            {selLine ? (
              <>
                <div dir="auto" style={{ fontSize: 15 }}>{lineText(selLine)}</div>
                <div style={{ display: 'flex', gap: 10 }}>
                  <TimeField label="התחלה" value={selLine.start} onCommit={(v) => editLine('שינוי התחלת מקטע', (l) => (v >= l.end - 0.2 || v < 0 ? false : void ((l.start = r2(v)), (l.manual = true))))} />
                  <TimeField label="סיום" value={selLine.end} onCommit={(v) => editLine('שינוי סיום מקטע', (l) => (v <= l.start + 0.2 ? false : void ((l.end = r2(v)), (l.manual = true))))} />
                </div>
                <div className="muted" style={{ fontSize: 12 }}>
                  {selLine.words.length} מילים · {SPK[selLine.speaker]}
                  {selLine.manual ? ' · תוקן ידנית' : ''} · {fmt(lineSpan(selLine)[0])}
                </div>
              </>
            ) : (
              <span className="dim" style={{ fontSize: 13 }}>בחר שורה או מקטע</span>
            )}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, padding: 12, borderRadius: 'var(--radius-lg)', background: 'var(--color-surface)' }}>
            <span className="kicker">היסטוריה</span>
            {p.history.slice(0, 7).map((h, i) => (
              <div key={i} style={{ display: 'flex', gap: 8, fontSize: 12.5 }}>
                <span dir="ltr" className="num dim">{hhmm(h.at)}</span>
                <span>{h.label}</span>
              </div>
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}

const Kbd = ({ children }: { children: React.ReactNode }) => (
  <span className="dim" style={{ fontSize: 10.5 }}>{children}</span>
);
const TrackLabel = ({ top, children }: { top: number; children: React.ReactNode }) => (
  <div className="dim" style={{ position: 'absolute', left: 8, top, fontSize: 10.5 }}>{children}</div>
);

function TimeField({ label, value, onCommit }: { label: string; value: number; onCommit: (v: number) => boolean | void }) {
  const [v, setV] = useState(value.toFixed(2));
  useEffect(() => setV(value.toFixed(2)), [value]);
  return (
    <label className="muted" style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 11.5 }}>
      {label}
      <input
        className="input num"
        dir="ltr"
        type="number"
        step={0.01}
        value={v}
        onChange={(e) => setV(e.target.value)}
        onBlur={() => {
          const n = parseFloat(v);
          if (isNaN(n) || n === value || onCommit(n) === false) setV(value.toFixed(2));
        }}
        onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
        style={{ width: 96 }}
      />
    </label>
  );
}

function LyricsRow({ l, i, on, draft, setDraft, pick, setSel }: { l: Line; i: number; on: boolean; draft: string; setDraft: (v: string) => void; pick: () => void; setSel: (id: string) => void }) {
  const txt = lineText(l);
  const ed = (label: string, fn: (P: Project, k: number) => void | boolean) => store.edit(label, (P) => fn(P, P.lines.findIndex((x) => x.id === l.id)));
  const stop = (fn: () => void) => (e: React.MouseEvent) => {
    e.stopPropagation();
    fn();
  };
  // Reorder swaps the TEXT between adjacent slots; timings stay attached to the audio.
  const swap = (j: number) =>
    ed('שינוי סדר שורות', (P, k) => {
      const o = P.lines[j];
      if (!o) return false;
      const a = P.lines[k];
      const ta = lineText(a);
      const tb = lineText(o);
      const sa = lineSpan(a);
      const sb = lineSpan(o);
      a.words = makeWords(sa[0], sa[1], tb).map((w) => ({ ...w, manual: true }));
      o.words = makeWords(sb[0], sb[1], ta).map((w) => ({ ...w, manual: true }));
      a.manual = o.manual = true;
      setSel(o.id);
    });
  const commit = () => {
    const v = draft.trim();
    if (!v || v === txt) return setDraft(txt);
    ed('עריכת שורה', (P, k) => {
      const a = P.lines[k];
      const sp = lineSpan(a);
      const nw = v.split(/\s+/);
      a.words =
        nw.length === a.words.length
          ? a.words.map((w, j) => (w.w === nw[j] ? w : { ...w, w: nw[j], manual: true, conf: 'high' }))
          : makeWords(sp[0], sp[1], v).map((w) => ({ ...w, manual: true }));
      a.manual = true;
    });
  };
  return (
    <div data-testid="lyrics-row" onClick={pick} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '6px 10px', borderRadius: 'var(--radius-md)', background: on ? 'var(--color-neutral-800)' : 'transparent', boxShadow: on ? 'inset 0 0 0 1px var(--color-accent-800)' : 'none', flexWrap: 'wrap' }}>
      <span className="dim" style={{ width: 22, fontSize: 11.5 }}>{i + 1}</span>
      <button className="btn btn-ghost num" dir="ltr" title="קפוץ לזמן" onClick={stop(() => { setSel(l.id); playback.seek(lineSpan(l)[0] - 0.2, true); })} style={{ height: 28, fontSize: 12, width: 76 }}>
        <i className="ph ph-play" />
        {fmt(lineSpan(l)[0])}
      </button>
      <input className="input" dir="auto" aria-label={'שורה ' + (i + 1)} value={draft} onFocus={pick} onChange={(e) => setDraft(e.target.value)} onBlur={commit} onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()} style={{ flex: '1 1 240px', minWidth: 0, fontSize: 14.5, background: 'transparent' }} />
      <select className="input" aria-label="דובר" value={l.speaker} onClick={(e) => e.stopPropagation()} onChange={(e) => ed('החלפת דובר', (P, k) => void (P.lines[k].speaker = e.target.value as Speaker))} style={{ width: 120, fontSize: 12.5 }}>
        <option value="1">זמר/ת 1</option>
        <option value="2">זמר/ת 2</option>
        <option value="duet">דואט</option>
      </select>
      <div style={{ display: 'flex', gap: 2 }}>
        <IconBtn icon="arrow-up" label="הזז למעלה" onClick={stop(() => swap(i - 1))} />
        <IconBtn icon="arrow-down" label="הזז למטה" onClick={stop(() => swap(i + 1))} />
        <IconBtn
          icon="scissors"
          label="פצל באמצע"
          onClick={stop(() =>
            ed('פיצול שורה', (P, k) => {
              const a = P.lines[k];
              if (a.words.length < 2) return false;
              const m = Math.ceil(a.words.length / 2);
              const bw = a.words.splice(m);
              const b: Line = { id: uid('l'), start: bw[0].s, end: a.end, speaker: a.speaker, words: bw, manual: true };
              a.end = a.words[a.words.length - 1].e;
              a.manual = true;
              P.lines.splice(k + 1, 0, b);
            }),
          )}
        />
        <IconBtn
          icon="arrows-in-line-vertical"
          label="מזג עם הבאה"
          onClick={stop(() =>
            ed('מיזוג שורות', (P, k) => {
              const a = P.lines[k];
              const b = P.lines[k + 1];
              if (!b) return false;
              a.words = a.words.concat(b.words);
              a.end = b.end;
              a.manual = true;
              P.lines.splice(k + 1, 1);
            }),
          )}
        />
        <IconBtn
          icon="plus"
          label="הוסף שורה אחרי"
          onClick={stop(() =>
            ed('הוספת שורה', (P, k) => {
              const a = P.lines[k];
              const nx = P.lines[k + 1];
              const s = r2(a.end + 0.2);
              const en = r2(Math.min(s + 2.5, nx ? nx.start - 0.1 : s + 2.5));
              if (en - s < 0.5) return false;
              const nl: Line = { id: uid('l'), start: s, end: en, speaker: a.speaker, words: makeWords(s, en, 'שורה חדשה').map((w) => ({ ...w, manual: true })), manual: true };
              P.lines.splice(k + 1, 0, nl);
              setSel(nl.id);
            }),
          )}
        />
        <IconBtn icon="trash" label="מחק שורה" onClick={stop(() => ed('מחיקת שורה', (P, k) => (P.lines.length < 2 ? false : void P.lines.splice(k, 1))))} />
      </div>
    </div>
  );
}

const IconBtn = ({ icon, label, onClick }: { icon: string; label: string; onClick: (e: React.MouseEvent) => void }) => (
  <button className="btn btn-ghost btn-icon" onClick={onClick} title={label} aria-label={label}>
    <i className={'ph ph-' + icon} />
  </button>
);
