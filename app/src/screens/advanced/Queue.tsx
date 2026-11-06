import { STAGES, jobPct, store, useStore } from '../../store/store';
import { ui } from '../../store/ui';
import { Empty } from './ProcessingOptions';
import { useWS, ws } from '../../store/workspace';
import { useEffect, useState } from 'react';
import { api, isServer, listJobs } from '../../store/server';
import type { Job } from '../../store/types';

const NAMES = (stems: number) => ['הכנת קובץ', 'הפרדה ל־' + stems + ' ערוצים', 'תמלול', 'יישור מילים', 'טיוטת כתוביות'];
const RES = ['CPU', 'GPU', 'GPU', 'GPU', 'CPU'];

export function Queue() {
  const { job, sim, project } = useStore();
  const { waiting: demoWaiting } = useWS();
  const [srvJobs, setSrvJobs] = useState<Job[] | null>(null);
  useEffect(() => {
    if (!isServer()) return;
    const f = () => listJobs().then(setSrvJobs).catch(() => {});
    f();
    const t = setInterval(f, 2000);
    return () => clearInterval(t);
  }, []);
  const waiting = srvJobs
    ? srvJobs.filter((j) => j.status === 'running' && j.id !== job?.serverId).map((j) => ({ id: j.id, name: j.name, pipe: (j.kind === 'render' ? 'רינדור MP4' : 'עיבוד מלא') + ' · ' + j.profile + ' · ' + j.stems + ' ערוצים' }))
    : demoWaiting;
  if (!job)
    return (
      <Empty>
        <span style={{ flex: 1 }}>אין עבודה בתור.</span>
        <button className="btn btn-primary" onClick={() => ui.go(project ? 'options' : 'simple')}>{project ? 'לבחירת עיבוד' : 'לייבוא'}</button>
      </Empty>
    );
  const names = NAMES(job.stems);
  const st = job.status;
  const remote = sim.conn === 'remote';
  const where = remote ? 'שרת מרוחק' : 'המחשב הזה';
  const stage = Math.min(job.stage, STAGES.length - 1);
  const curName = names[stage];
  const overall = jobPct(job);
  const gpuBusy = st === 'running' && RES[stage] === 'GPU' && !sim.offline;

  const V = {
    running: { tag: 'רץ · ' + curName, tagCls: 'tag tag-accent', icon: 'ph ph-waveform', sub: where + ' · ' + RES[stage] + ' · פרופיל ' + job.profile + (job.attempt > 1 ? ' · ניסיון ' + job.attempt : ''), eta: sim.offline ? 'מצב לא ידוע' : 'נותרו כ־' + Math.max(1, Math.round((100 - overall) / 18)) + ' דק׳', banner: sim.offline ? ['אין חיבור לשרת', 'מצב העבודה לא ידוע. הסטטוס יתעדכן כשהחיבור יחזור.'] : null },
    failed: job.serverId && !job.crash ? { tag: 'נכשל', tagCls: 'tag tag-neutral', icon: 'ph ph-warning-circle', sub: where + ' · בשלב ' + curName, eta: '—', banner: ['העיבוד נכשל בגלל המקור', job.error || ''] } : { tag: 'קרס · דורש התאוששות', tagCls: 'tag tag-accent', icon: 'ph ph-lightning-slash', sub: where + ' · נעצר בשלב ' + curName, eta: 'מושהה', banner: ['תהליך ה־GPU הופסק באופן לא צפוי', (job.serverId ? (job.error || 'שגיאה') + ' · שלב ' + curName : 'חריגה מזיכרון GPU בשלב ' + curName) + '. מה שהושלם נשמר (' + names.slice(0, job.stage).join(', ') + '), ולכן אין צורך להתחיל מחדש. זו קריסה, לא כשל בתוכן הקובץ.'] },
    cancelled: { tag: 'בוטל', tagCls: 'tag tag-neutral', icon: 'ph ph-stop-circle', sub: 'בוטל בשלב ' + curName, eta: '—', banner: ['העבודה בוטלה', 'שלבים שהושלמו נשמרו. הפעלה מחדש תמשיך מ' + curName + '.'] },
    done: { tag: 'הושלם', tagCls: 'tag tag-outline', icon: 'ph ph-check-circle', sub: where + ' · פרופיל ' + job.profile + ' · ' + job.stems + ' ערוצים', eta: 'הסתיים', banner: ['העיבוד הושלם', 'נוצרו תמלול, תזמון ברמת מילה וטיוטת כתוביות. מילים בביטחון מודל נמוך מסומנות לבדיקה (הערכת המודל, לא דיוק מאומת).'] },
  }[st];

  const logs: [string, boolean][] = [['file ok · ' + job.profile, false], ['separation · stems=' + job.stems, false]];
  if (job.stage >= 2) logs.push(['separation done · transcription', false]);
  if (st === 'failed') logs.push(['ERROR CUDA out of memory (15.6/16.0 GB) · process terminated', true], ['checkpoint kept: stages 1-' + job.stage, false]);
  if (st === 'cancelled') logs.push(['cancelled by user · checkpoint kept', true]);
  if (st === 'running') logs.push([curName + ' · ' + Math.round(job.pct) + '%' + (sim.offline ? ' · connection lost' : ''), true]);
  if (st === 'done') logs.push(['alignment done · draft subtitles ready', true]);

  return (
    <div dir="rtl" style={{ height: '100%', overflow: 'auto', background: 'var(--color-bg)' }}>
      <div className="page-pad" style={{ maxWidth: 1180, margin: '0 auto', display: 'flex', flexWrap: 'wrap', gap: 26 }}>
        <div style={{ flex: '1 1 560px', minWidth: 0, display: 'flex', flexDirection: 'column', gap: 18 }}>
          <header style={{ display: 'flex', alignItems: 'flex-end', gap: 12, flexWrap: 'wrap' }}>
            <h2 style={{ fontSize: 28, margin: 0 }}>תור עיבודי AI</h2>
            <span className="dim" style={{ fontSize: 14, paddingBottom: 4 }}>{st === 'running' ? 'משימה אחת רצה' : 'אין משימת GPU פעילה'}</span>
            <span className="tag tag-outline" style={{ marginInlineStart: 'auto' }}>{job.serverId ? 'שרת Lyra' : 'סימולציה · אין שרת עיבוד מחובר'}</span>
          </header>
          <section style={{ display: 'flex', flexDirection: 'column', gap: 14, padding: 18, borderRadius: 'var(--radius-lg)', background: 'var(--color-surface)', boxShadow: st === 'failed' ? 'inset 0 0 0 1px var(--color-accent-700)' : 'var(--shadow-sm)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div style={{ width: 44, height: 44, flex: 'none', borderRadius: 'var(--radius-md)', background: st === 'failed' ? 'var(--color-accent-800)' : 'var(--color-accent-900)', display: 'grid', placeItems: 'center', color: st === 'failed' ? 'var(--color-accent-100)' : 'var(--color-accent-300)', fontSize: 22 }}>
                <i className={V.icon} />
              </div>
              <div style={{ minWidth: 0 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                  <span dir="auto" style={{ fontSize: 16, fontWeight: 500 }}>{job.name}</span>
                  <span className={V.tagCls}>{V.tag}</span>
                </div>
                <div className="muted" style={{ fontSize: 12.5 }}>{V.sub}</div>
              </div>
              <div style={{ marginInlineStart: 'auto', textAlign: 'left' }}>
                <div style={{ fontSize: 22, fontWeight: 500, color: 'var(--color-accent-300)' }}>{overall}%</div>
                <div className="dim" style={{ fontSize: 12 }}>{V.eta}</div>
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))', gap: 8 }}>
              {names.map((l, i) => {
                const done = i < job.stage || st === 'done';
                const cur = i === job.stage && st !== 'done';
                const k = done ? 'done' : !cur ? 'todo' : st === 'running' ? 'run' : st === 'failed' ? 'fail' : 'stop';
                return (
                  <div key={l} style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                    <div style={{ height: 4, borderRadius: 2, background: 'var(--color-neutral-800)' }}>
                      <div style={{ width: done ? '100%' : cur ? Math.round(job.pct) + '%' : '0%', height: '100%', borderRadius: 2, background: k === 'fail' || k === 'stop' ? 'var(--color-neutral-400)' : 'var(--color-accent)', boxShadow: k === 'run' ? '0 0 8px var(--color-accent)' : 'none' }} />
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12.5, color: k === 'todo' ? 'var(--color-neutral-500)' : k === 'run' ? 'var(--color-accent-300)' : 'var(--color-text)' }}>
                      <i className={k === 'done' ? 'ph-bold ph-check' : k === 'run' ? 'ph ph-spinner-gap spin' : k === 'fail' ? 'ph-fill ph-lightning-slash' : k === 'stop' ? 'ph ph-stop-circle' : 'ph ph-circle-dashed'} />
                      {l}
                    </div>
                    <div className="dim" style={{ fontSize: 11.5 }}>
                      {RES[i] + (k === 'run' ? ' · ' + (job.estimate ? '~' : '') + Math.round(job.pct) + '%' + (job.estimate ? ' הערכה' : '') : k === 'fail' ? ' · קרס ב־' + Math.round(job.pct) + '%' : k === 'stop' ? ' · בוטל' : k === 'done' ? ' · הושלם' : '')}
                    </div>
                  </div>
                );
              })}
            </div>
            {V.banner && (
              <div role={st === 'failed' ? 'alert' : undefined} style={{ display: 'flex', gap: 12, padding: '12px 14px', borderRadius: 'var(--radius-md)', background: st === 'failed' ? 'color-mix(in srgb, var(--color-accent) 12%, transparent)' : 'var(--color-bg)' }}>
                <i className={st === 'failed' ? 'ph-fill ph-lightning-slash' : st === 'done' ? 'ph-fill ph-check-circle' : 'ph ph-info'} style={{ fontSize: 22, color: 'var(--color-accent-200)' }} />
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 4 }}>
                  <div style={{ fontSize: 14 }}>{V.banner[0]}</div>
                  <div style={{ fontSize: 12.5, color: 'var(--color-neutral-300)', textWrap: 'pretty' }}>{V.banner[1]}</div>
                </div>
              </div>
            )}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              {st === 'running' && <button className="btn btn-secondary" onClick={() => { store.cancelJob(); ui.toast('העבודה בוטלה · השלבים שהושלמו נשמרו'); }}><i className="ph ph-x" />בטל עבודה</button>}
              {st === 'failed' && (
                <>
                  <button className="btn btn-primary" onClick={() => { store.retryJob(false); ui.toast('ממשיך מהשלב שנעצר'); }}><i className="ph ph-play" />המשך מ{curName}</button>
                  {job.profile === 'איכותי' && <button className="btn btn-secondary" onClick={() => store.retryJob(true)}><i className="ph ph-lightning" />נסה שוב בפרופיל מהיר</button>}
                  <button className="btn btn-secondary" onClick={() => { store.removeJob(); ui.toast('העבודה בוטלה והוסרה מהתור'); }}><i className="ph ph-x" />בטל עבודה</button>
                </>
              )}
              {st === 'cancelled' && (
                <>
                  <button className="btn btn-primary" onClick={() => store.retryJob(false)}><i className="ph ph-arrow-clockwise" />הפעל מחדש מאותו שלב</button>
                  <button className="btn btn-secondary" onClick={() => { store.removeJob(); ui.toast('העבודה הוסרה מהתור'); }}><i className="ph ph-trash" />הסר מהתור</button>
                </>
              )}
              {st === 'done' && <button className="btn btn-primary" onClick={() => ui.go('subs-word')}><i className="ph ph-text-aa" />פתח לתיקון מילים</button>}
              <span className="dim" style={{ marginInlineStart: 'auto', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
                <i className="ph ph-moon" />
                אפשר לסגור את הדפדפן: העבודה ממשיכה ברקע
              </span>
            </div>
            <div dir="ltr" style={{ fontFamily: "ui-monospace,'SF Mono',Menlo,monospace", fontSize: 11.5, lineHeight: 1.7, padding: '10px 12px', borderRadius: 'var(--radius-md)', background: 'var(--color-bg)', textAlign: 'left', overflowX: 'auto' }}>
              {(job.log && job.log.length ? job.log.map((t, i) => [t, i === job.log!.length - 1] as [string, boolean]) : logs).map(([t, h], i) => (
                <div key={i} style={{ color: h ? 'var(--color-accent-300)' : 'var(--color-neutral-400)' }}>{t}</div>
              ))}
            </div>
          </section>
          <section style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div style={{ display: 'flex', alignItems: 'baseline', marginBottom: 6 }}>
              <h4 style={{ margin: 0, fontSize: 17 }}>ממתינים</h4>
              <span className="dim" style={{ marginInlineStart: 'auto', fontSize: 12 }}>משימת GPU אחת בכל רגע · יתחילו כשה־GPU יתפנה</span>
            </div>
            {waiting.length === 0 && <span className="muted" style={{ fontSize: 13 }}>אין משימות ממתינות.</span>}
            {waiting.map((w, i) => (
              <div key={w.id} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 12px', borderRadius: 'var(--radius-md)', background: 'var(--color-surface)' }}>
                <span className="dim" style={{ width: 22, fontSize: 12 }}>#{i + 2}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 14 }} dir="auto">{w.name}</div>
                  <div className="dim" style={{ fontSize: 12 }}>{w.pipe}</div>
                </div>
                <span className="tag tag-neutral">{gpuBusy ? 'ממתין ל־GPU' : i === 0 ? 'מתחיל בקרוב' : 'ממתין'}</span>
                <button className="btn btn-ghost btn-icon" aria-label={'הקדם את ' + w.name} disabled={i === 0} onClick={() => (srvJobs ? api('/api/jobs/' + w.id + '/move', { method: 'POST', body: JSON.stringify({ delta: -1 }) }) : ws.moveWaiting(w.id, -1))}><i className="ph ph-arrow-up" /></button>
                <button className="btn btn-ghost btn-icon" aria-label={'דחה את ' + w.name} disabled={i === waiting.length - 1} onClick={() => (srvJobs ? api('/api/jobs/' + w.id + '/move', { method: 'POST', body: JSON.stringify({ delta: 1 }) }) : ws.moveWaiting(w.id, 1))}><i className="ph ph-arrow-down" /></button>
                <button className="btn btn-ghost btn-icon" aria-label={'הסר את ' + w.name} onClick={async () => { if (srvJobs) await api('/api/jobs/' + w.id + '/cancel', { method: 'POST' }); else ws.removeWaiting(w.id); ui.toast('הוסר מהתור'); }}><i className="ph ph-x" /></button>
              </div>
            ))}
            <span className="dim" style={{ fontSize: 11.5 }}>{srvJobs ? 'העבודות בתור של השרת.' : 'משימות לדוגמה של שירים אחרים · הסדר נשמר.'}</span>
          </section>
        </div>
        <aside style={{ flex: '1 1 280px', maxWidth: 340, display: 'flex', flexDirection: 'column', gap: 20, paddingTop: 6 }}>
          <section style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: 14, borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-sm)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <i className={remote ? 'ph ph-globe-simple' : 'ph ph-desktop'} style={{ color: 'var(--color-accent-300)' }} />
              <h5 style={{ margin: 0 }}>{remote ? 'שרת מרוחק · lyra.home' : 'המחשב הזה'}</h5>
              <span className="tag tag-accent" style={{ marginInlineStart: 'auto' }}>{sim.offline ? 'לא זמין' : 'מחובר'}</span>
            </div>
            <div className="dim" style={{ fontSize: 12 }}>GPU 16GB VRAM · החומרה מזוהה מהמחשב בפועל</div>
            {[
              ['GPU · משימות כבדות', gpuBusy ? '1 / 1 בשימוש' : '0 / 1 · פנוי', gpuBusy ? '100%' : '0%'],
              ['VRAM', gpuBusy ? '9.8 / 16 GB' : '0.6 / 16 GB', gpuBusy ? '61%' : '4%'],
            ].map(([l, v, w]) => (
              <div key={l} style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                <div style={{ display: 'flex', fontSize: 12.5 }}><span>{l}</span><span className="muted" style={{ marginInlineStart: 'auto' }}>{v}</span></div>
                <div style={{ height: 3, borderRadius: 2, background: 'var(--color-neutral-800)' }}><div style={{ width: w, height: '100%', borderRadius: 2, background: 'var(--color-neutral-400)' }} /></div>
              </div>
            ))}
            <div className="dim" style={{ fontSize: 11.5, textWrap: 'pretty' }}>ברירת מחדל: משימה כבדה אחת על ה־GPU בכל רגע, כדי לא לחרוג מ־16GB. ייצוא SRT/TXT/JSON רץ בדפדפן.</div>
          </section>
        </aside>
      </div>
    </div>
  );
}
