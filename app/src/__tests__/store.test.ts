import { beforeEach, describe, expect, it, vi } from 'vitest';
import { bidiRuns, fmtSrt, toSrt, toTxt } from '../lib/format';
import { applyDrag } from '../screens/advanced/LyricsStudio';
import { KEYS, store } from '../store/store';

describe('format', () => {
  it('formats SRT timestamps', () => {
    expect(fmtSrt(0)).toBe('00:00:00,000');
    expect(fmtSrt(3725.5)).toBe('01:02:05,500');
  });
  it('keeps English runs in order inside a Hebrew line', () => {
    const runs = bidiRuns(['בקיץ', 'של', '1999', 'שרנו', 'Hey', 'Jude', 'עד'].map((w) => ({ w })));
    expect(runs.find((r) => r.ltr)!.items.map((x) => x.w)).toEqual(['Hey', 'Jude']);
  });
});

describe('shared store', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    localStorage.clear();
    store.reset();
  });

  it('edits flow into SRT/TXT, persist, and undo/redo', () => {
    store.ensure();
    store.edit('תיקון מילה', (P) => {
      const w = P.lines[7].words.find((x) => x.w === '1999')!;
      w.w = '1998';
      w.s += 0.2;
      w.manual = true;
    });
    const p = store.get().project!;
    expect(toSrt(p)).toContain('1998');
    expect(toTxt(p)).toContain('1998');
    vi.advanceTimersByTime(1000);
    expect(store.get().save.state).toBe('saved');
    expect(JSON.parse(localStorage.getItem(KEYS.project)!).lines[7].words[2].w).toBe('1998');
    store.undo();
    expect(toSrt(store.get().project!)).toContain('1999');
    store.redo();
    expect(toSrt(store.get().project!)).toContain('1998');
  });

  it('offline saves locally, then syncs; conflict and failure states', () => {
    store.ensure();
    store.setSim('offline', true);
    store.edit('a', (P) => void (P.title = 'x'));
    vi.advanceTimersByTime(1000);
    expect(store.get().save.state).toBe('local');
    store.setSim('conflictNext', true);
    store.setSim('offline', false);
    vi.advanceTimersByTime(1000);
    expect(store.get().save.state).toBe('conflict');
    store.resolveConflict('theirs');
    expect(store.get().save.state).toBe('saved');
    expect(store.get().project!.lines[6].words.map((w) => w.w).join(' ')).toBe('ירח של אוגוסט, אל תלך מפה');
    store.setSim('failNext', true);
    store.edit('b', (P) => void (P.title = 'y'));
    vi.advanceTimersByTime(1000);
    expect(store.get().save.state).toBe('failed');
    store.retrySave();
    vi.advanceTimersByTime(1000);
    expect(store.get().save.state).toBe('saved');
  });

  it('simulated crash, resume from the failed stage, completion', () => {
    store.setSim('crash', true);
    store.startJob({ name: 's', profile: 'מהיר', stems: 2 });
    for (let i = 0; i < 200 && store.get().job!.status === 'running'; i++) store.tick(200);
    expect(store.get().job).toMatchObject({ status: 'failed', stage: 2 });
    store.retryJob();
    expect(store.get().job).toMatchObject({ status: 'running', stage: 2, attempt: 2 });
    for (let i = 0; i < 200 && store.get().job!.status === 'running'; i++) store.tick(200);
    expect(store.get().job!.status).toBe('done');
  });

  it('moving a clip moves its words when linked, not otherwise', () => {
    const l = store.ensure().lines[0];
    const a = structuredClone(l);
    applyDrag(a, { kind: 'clip', dx: 1 }, true);
    expect(a.words[0].s).toBeCloseTo(l.words[0].s + 1);
    const b = structuredClone(l);
    applyDrag(b, { kind: 'clip', dx: 1 }, false);
    expect(b.words[0].s).toBe(l.words[0].s);
    expect(b.start).toBeCloseTo(l.start + 1);
  });
});

import { toLrc } from '../lib/format';
import { constrainWord } from '../screens/advanced/LyricsStudio';
import { ws } from '../store/workspace';

describe('workspace', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    localStorage.clear();
    store.reset();
    ws.reset();
  });
  it('version checkpoint restores previous lyrics (undoable)', () => {
    store.ensure();
    ws.saveVersion('v1');
    store.edit('x', (P) => void (P.lines[0].words[0].w = 'שונה'));
    ws.restoreVersion(ws.get().versions.find((v) => v.label === 'v1')!.id);
    expect(store.get().project!.lines[0].words[0].w).toBe('הלילה');
    store.undo();
    expect(store.get().project!.lines[0].words[0].w).toBe('שונה');
  });
  it('one-time invite creates a user once', () => {
    const inv = ws.createInvite('אורי', 'editor', 7, '');
    const u = ws.acceptInvite(inv.token, 'uri');
    expect(typeof u).toBe('object');
    expect(ws.acceptInvite(inv.token, 'uri2')).toMatch(/שומש/);
    expect(ws.get().users.some((x) => x.user === 'uri' && x.role === 'editor')).toBe(true);
  });
  it('admin cannot demote self; LRC has word marks', () => {
    expect(ws.setRole('u_daniel', 'viewer')).toBe(false);
    expect(toLrc(store.ensure())).toMatch(/^\[ti:ירח של אוגוסט\]/);
    expect(toLrc(store.get().project!)).toContain('<00:12.00>הלילה');
  });
  it('word zoom drag never overlaps neighbours', () => {
    const l = store.ensure().lines[0];
    const moved = constrainWord(l, l.words[1], { kind: 'move', dt: -10 });
    expect(moved.s).toBe(l.words[0].e);
  });
});

import { alignLyricsLocal, toAss } from '../lib/format';

describe('ass + pasted lyrics', () => {
  it('ASS has word sweeps and style', () => {
    localStorage.clear();
    store.reset();
    const a = toAss(store.ensure());
    expect(a).toContain('PlayResY: 1080');
    expect(a).toMatch(/\{\\kf\d+\}הלילה/);
    expect(a.split('Dialogue:').length - 1).toBe(10);
  });
  it('local lyrics keep timings when the word count matches', () => {
    const p = store.ensure();
    const text = p.lines.map((l) => l.words.map((w) => w.w).join(' ')).join('\n');
    const out = alignLyricsLocal(text, p);
    expect(out[3].words[0].s).toBe(p.lines[3].words[0].s);
    const short = alignLyricsLocal('שורה אחת\nשורה שתיים', p);
    expect(short).toHaveLength(2);
    expect(short[1].words[0].conf).toBe('low');
  });
});
