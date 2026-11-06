import { useMemo } from 'react';
import { KaraokePreview } from '../../components/KaraokePreview';
import { fmt, fmtShort, hhmm, lineSpan, lineText, r2 } from '../../lib/format';
import { playback, usePlayback } from '../../store/playback';
import { store, useStore } from '../../store/store';
import type { StemMix } from '../../store/types';
import { ui, useUI } from '../../store/ui';
import { useWS } from '../../store/workspace';
import * as RW from '../../store/remoteWs';
import { Card, KV, Page, Tabs } from './Sidebar';
import { StateView } from './SystemStates';

type PTab = 'overview' | 'files' | 'versions' | 'activity' | 'processing';
const STEM_L: Record<string, string> = { vocals: 'שירה (Vocals)', instrumental: 'ליווי (Instrumental)', drums: 'תופים', bass: 'בס', other: 'אחר' };

export function ProjectDetails() {
  const { route } = useUI();
  const { project: p, sim } = useStore();
  const { versions } = useWS();
  const tab: PTab = route === 'project-versions' ? 'versions' : ((sessionStorage.getItem('lyra.ptab') as PTab) || 'overview');
  const setTab = (t: PTab) => {
    try {
      sessionStorage.setItem('lyra.ptab', t);
    } catch {
      /* ignore */
    }
    ui.go(t === 'versions' ? 'project-versions' : 'project');
  };
  if (!p) return <StateView kind="emptyDashboard" />;
  const remote = sim.conn === 'remote';
  const manual = p.lines.reduce((a, l) => a + l.words.filter((w) => w.manual).length, 0);
  return (
    <Page
      title={p.title}
      sub={p.artist + ' · ' + fmtShort(p.source.duration)}
      actions={
        <>
          <button className="btn btn-secondary" onClick={() => ui.open('share')}><i className="ph ph-share-network" />שיתוף</button>
          <button className="btn btn-ghost" onClick={() => ui.open('delete')}><i className="ph ph-trash" />מחיקה</button>
        </>
      }
    >
      <Tabs value={tab} onChange={setTab} tabs={[['overview', 'סקירה'], ['files', 'קבצים'], ['versions', 'היסטוריית גרסאות'], ['activity', 'פעילות'], ['processing', 'הגדרות עיבוד']]} />
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
          <Card>
            <span className="kicker">מקור</span>
            <KV k="סוג" v={p.source.kind === 'youtube' ? 'YouTube (מדומה)' : 'קובץ מקומי'} />
            <KV k="שם" v={<span dir="auto">{p.source.name}</span>} />
            <KV k="משך" v={fmtShort(p.source.duration)} />
            <KV k="נוצר" v={new Date(p.createdAt).toLocaleString('he-IL')} />
          </Card>
          <Card>
            <span className="kicker">מילים</span>
            <KV k="שורות" v={p.lines.length} />
            <KV k="מילים" v={p.lines.reduce((a, l) => a + l.words.length, 0)} />
            <KV k="תיקונים ידניים" v={manual} />
            <KV k="תזמון" v={p.timing === 'word' ? 'ברמת מילה' : 'ברמת שורה'} />
            <button className="btn btn-secondary" onClick={() => ui.go('subs-word')} style={{ alignSelf: 'flex-start' }}>פתח בעורך</button>
          </Card>
          <Card>
            <span className="kicker">עיבוד</span>
            <KV k="פרופיל" v={p.processing.profile} />
            <KV k="ערוצים" v={p.stems.map((s) => STEM_L[s]).join(', ')} />
            <KV k="מעבד" v={remote ? 'שרת מרוחק · GPU 16GB' : 'המחשב הזה · GPU 16GB'} />
          </Card>
        </div>
      )}
      {tab === 'files' && (
        <Card>
          {[
            ['ph ph-film-strip', p.source.name, 'מקור', true],
            ...p.stems.map((s) => ['ph ph-waveform', s + '.wav', STEM_L[s], true] as const),
            ['ph ph-brackets-curly', 'project.json', 'נתוני פרויקט · בדפדפן', false],
          ].map(([i, n, d, server]) => (
            <div key={n as string} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 13.5, flexWrap: 'wrap' }}>
              <i className={i as string} style={{ fontSize: 18, color: 'var(--color-accent-300)' }} />
              <span dir="auto" style={{ flex: 1, minWidth: 160 }}>{n}</span>
              <span className="dim" style={{ fontSize: 12 }}>{d}</span>
              {server ? (
                <button className="btn btn-ghost" onClick={() => ui.toast('סימולציה: הקובץ נמצא בשרת העיבוד')}>{remote ? 'נגן / הורד' : 'פתח תיקייה'}</button>
              ) : (
                <button className="btn btn-ghost" onClick={() => ui.go('export')}>ייצוא</button>
              )}
            </div>
          ))}
          <span className="dim" style={{ fontSize: 12 }}>קובצי מדיה לא נשמרים בדפדפן. {remote ? 'בשרת מרוחק: ניגון והורדה.' : 'במחשב הזה: פתיחת התיקייה.'}</span>
        </Card>
      )}
      {tab === 'versions' && (
        <Card>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span className="kicker">נקודות שמירה</span>
            <button className="btn btn-secondary" style={{ marginInlineStart: 'auto' }} onClick={async () => { await RW.saveVersion('נקודת שמירה ידנית'); ui.toast('נוצרה נקודת שמירה'); }}><i className="ph ph-bookmark-simple" />צור נקודת שמירה</button>
          </div>
          {versions.length === 0 ? (
            <span className="muted" style={{ fontSize: 13 }}>אין עדיין נקודות שמירה. ⌘S או הכפתור למעלה יוצרים אחת.</span>
          ) : (
            versions.map((v) => {
              const diff = v.lines.filter((l, i) => !p.lines[i] || lineText(l) !== lineText(p.lines[i])).length + Math.max(0, p.lines.length - v.lines.length);
              return (
                <div key={v.id} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 13, flexWrap: 'wrap' }}>
                  <span dir="ltr" className="num dim">{hhmm(v.at)}</span>
                  <span style={{ flex: 1, minWidth: 140 }}>{v.label}</span>
                  <span className="dim">{diff ? diff + ' שורות שונות מהנוכחית' : 'זהה לנוכחית'}</span>
                  <button className="btn btn-ghost" disabled={!diff} onClick={() => ui.open('restore', v.id)}>שחזר</button>
                </div>
              );
            })
          )}
        </Card>
      )}
      {tab === 'activity' && (
        <Card>
          {p.history.map((h, i) => (
            <div key={i} style={{ display: 'flex', gap: 10, fontSize: 13 }}><span dir="ltr" className="num dim">{hhmm(h.at)}</span><span>{h.label}</span></div>
          ))}
        </Card>
      )}
      {tab === 'processing' && (
        <Card>
          <KV k="פרופיל" v={p.processing.profile} />
          <KV k="הפרדה" v={p.processing.stems + ' ערוצים'} />
          <KV k="שפה" v={{ auto: 'זיהוי אוטומטי', he: 'עברית', en: 'English' }[p.processing.language]} />
          <div className="muted" style={{ fontSize: 12.5 }}>הרצה מחדש לא דורסת תיקונים ידניים. הפלט החדש מוצג כהצעה להשוואה.</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button className="btn btn-primary" onClick={() => ui.open('airerun')}><i className="ph ph-sparkle" />הרץ מחדש כהצעה</button>
            <button className="btn btn-secondary" onClick={() => ui.go('options')}>שנה הגדרות ועבד שוב</button>
          </div>
        </Card>
      )}
    </Page>
  );
}

export function AudioStudio() {
  const { project: p, media } = useStore();
  if (!p) return <StateView kind="emptyDashboard" />;
  const mix = p.audio.mix || {};
  const get = (s: string): StemMix => mix[s] || { vol: 100, mute: false, solo: false };
  const anySolo = p.stems.some((s) => get(s).solo);
  const setMix = (s: string, patch: Partial<StemMix>) =>
    store.edit('מיקסר: ' + (STEM_L[s] || s), (P) => {
      P.audio.mix = { ...(P.audio.mix || {}), [s]: { ...get(s), ...patch } };
    });
  return (
    <Page title="אודיו וסטמים" sub={p.stems.length + ' ערוצים · ' + p.processing.profile}>
      <div className="alert" style={{ background: 'var(--color-surface)' }}>
        <i className="ph ph-info" style={{ color: 'var(--color-accent-300)' }} />
        {media ? 'הקובץ שהועלה מתנגן עם האודיו המקורי: הפרדת ערוצים רצה בשרת העיבוד ולא בדפדפן.' : 'הערוצים נוצרים בשרת העיבוד. הגדרות המיקסר נשמרות בפרויקט, וההשמעה כאן היא סימולציה.'}
      </div>
      <Card>
        {p.stems.map((s, i) => {
          const m = get(s);
          const audible = !m.mute && (!anySolo || m.solo);
          return (
            <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', opacity: audible ? 1 : 0.5 }}>
              <span style={{ width: 150, fontSize: 14 }}>{STEM_L[s] || s}</span>
              <div dir="ltr" style={{ flex: '1 1 240px', height: 34, display: 'flex', alignItems: 'center', gap: 2, overflow: 'hidden' }}>
                {Array.from({ length: 90 }, (_, k) => <div key={k} style={{ flex: 1, height: Math.round(((8 + 26 * Math.abs(Math.sin((k + 1) * (1.3 + i * 0.4)) * Math.cos(k * 0.21))) * m.vol) / 100) + 'px', borderRadius: 1, background: s === 'vocals' ? 'var(--color-accent-500)' : 'var(--color-neutral-500)' }} />)}
              </div>
              <button className="btn btn-ghost" aria-pressed={m.mute} onClick={() => setMix(s, { mute: !m.mute })} style={{ height: 30, fontSize: 12, color: m.mute ? 'var(--color-accent-200)' : undefined }}>M</button>
              <button className="btn btn-ghost" aria-pressed={m.solo} onClick={() => setMix(s, { solo: !m.solo })} style={{ height: 30, fontSize: 12, color: m.solo ? 'var(--color-accent-200)' : undefined }}>S</button>
              <input type="range" dir="ltr" aria-label={'עוצמה ' + (STEM_L[s] || s)} min={0} max={100} value={m.vol} onChange={(e) => store.preview((P) => void (P.audio.mix = { ...(P.audio.mix || {}), [s]: { ...get(s), vol: +e.target.value } }))} onPointerUp={(e) => setMix(s, { vol: +(e.target as HTMLInputElement).value })} onKeyUp={(e) => setMix(s, { vol: +(e.target as HTMLInputElement).value })} style={{ width: 120, accentColor: 'var(--color-accent)' }} />
              <span className="num dim" style={{ width: 36, fontSize: 12 }}>{m.vol}%</span>
            </div>
          );
        })}
      </Card>
      <Card>
        <span className="kicker">פסקול הקריוקי</span>
        <KV k="בסרטון הסופי" v={p.stems.filter((s) => s !== 'vocals').map((s) => STEM_L[s]).join(' + ') + ' · השירה המקורית מושתקת'} />
        <KV k="סנכרון" v="אורך הערוצים זהה למקור עד לדגימה · בלי היסט זמן" />
        <span className="dim" style={{ fontSize: 12 }}>ניקוי רעשים, הסרת ריוורב, הפרדת גיטרה/פסנתר וניתוח סולם אינם חלק מ־V1.</span>
      </Card>
    </Page>
  );
}

export function VideoEditor() {
  const { project: p } = useStore();
  const { t } = usePlayback();
  const dur = p?.source.duration || 66;
  const lines = useMemo(() => p?.lines ?? [], [p]);
  if (!p) return <StateView kind="emptyDashboard" />;
  const pct = (x: number) => (x / dur) * 100 + '%';
  const offset = (d: number) =>
    store.edit('היסט כל הכתוביות ' + (d > 0 ? '+' : '') + d + ' שנ׳', (P) => {
      P.lines.forEach((l) => {
        l.start = r2(Math.max(0, l.start + d));
        l.end = r2(l.end + d);
        l.words.forEach((w) => {
          w.s = r2(Math.max(0, w.s + d));
          w.e = r2(w.e + d);
        });
      });
    });
  const Track = ({ label, children }: { label: string; children: React.ReactNode }) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span className="dim" style={{ width: 90, fontSize: 12, flex: 'none' }}>{label}</span>
      <div dir="ltr" style={{ position: 'relative', flex: 1, height: 36, borderRadius: 6, background: 'var(--color-bg)' }}>{children}</div>
    </div>
  );
  return (
    <Page title="עורך וידאו" sub={p.style.aspect + ' · ' + p.export.res} actions={<button className="btn btn-secondary" onClick={() => ui.go('karaoke')}><i className="ph ph-paint-brush" />עיצוב כתוביות</button>}>
      <div style={{ height: 380 }}><KaraokePreview /></div>
      <Card>
        <div
          dir="ltr"
          onPointerDown={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            playback.seek(((e.clientX - r.left - 100) / (r.width - 100)) * dur);
          }}
          style={{ display: 'flex', flexDirection: 'column', gap: 8, position: 'relative', cursor: 'pointer' }}
        >
          <div dir="rtl" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <Track label="וידאו (רקע)"><div style={{ position: 'absolute', inset: 2, borderRadius: 5, background: 'linear-gradient(90deg, var(--color-neutral-700), var(--color-neutral-800))', fontSize: 11, padding: '0 8px', display: 'flex', alignItems: 'center' }}>{p.style.background === 'video' ? 'וידאו מקור · מושתק' : 'רקע ' + (p.style.background === 'blur' ? 'מעבר צבע' : 'אחיד')}</div></Track>
            <Track label="Instrumental"><div style={{ position: 'absolute', inset: 2, borderRadius: 5, background: 'color-mix(in srgb, var(--color-accent-800) 60%, var(--color-bg))' }} /></Track>
            <Track label="כתוביות">
              {lines.map((l) => {
                const [a, b] = lineSpan(l);
                return <div key={l.id} title={lineText(l)} style={{ position: 'absolute', top: 3, bottom: 3, left: pct(a), width: pct(b - a), borderRadius: 4, background: 'var(--color-accent-700)' }} />;
              })}
            </Track>
          </div>
          <div style={{ position: 'absolute', top: 0, bottom: 0, left: `calc(100px + (100% - 100px) * ${Math.min(1, t / dur)})`, width: 2, background: 'var(--color-accent)', pointerEvents: 'none' }} />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span className="muted" style={{ fontSize: 13 }}>היסט כל הכתוביות</span>
          {[-0.5, -0.1, 0.1, 0.5].map((d) => <button key={d} className="btn btn-ghost" onClick={() => offset(d)} style={{ height: 30, fontSize: 12 }} dir="ltr">{d > 0 ? '+' : ''}{d}s</button>)}
          <span className="dim num" style={{ marginInlineStart: 'auto', fontSize: 12 }}>{fmt(t)} / {fmt(dur)}</span>
        </div>
        <span className="dim" style={{ fontSize: 12 }}>ההיסט משנה את זמני השורות והמילים בפרויקט, ולכן מופיע גם בקובצי הייצוא. ⌘Z לביטול.</span>
      </Card>
    </Page>
  );
}
