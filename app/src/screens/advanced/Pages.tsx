import { useEffect, useState } from 'react';
import { listProjects, openProject } from '../../store/server';
import { Seg } from '../../components/controls';
import { exportFile, exportMp4, serverProject } from '../../lib/actions';
import { renderFresh } from '../../store/server';
import { fmtShort, hhmm, toLrc, toSrt } from '../../lib/format';
import { jobPct, store, useStore } from '../../store/store';
import type { Res } from '../../store/types';
import { ui } from '../../store/ui';
import { ROLE_L, inviteUrl, useWS, ws, type Role } from '../../store/workspace';
import { Card, KV, Page } from './Sidebar';
import { StateView } from './SystemStates';

/** Other library entries are illustrative only: this build holds one real project. */
const DEMO_LIBRARY = [
  { t: 'Paper Lanterns', a: 'Mira Hale', d: '7:51', s: 'הושלם · איכותי' },
  { t: 'Glass Houses', a: 'The Sundays After', d: '6:10', s: 'יוצא MP4' },
  { t: 'Neon Rain', a: 'Kite District', d: '3:44', s: 'בתור' },
  { t: 'שקיעה בנמל', a: 'אלון שגיא', d: '4:20', s: 'טיוטת כתוביות' },
];

const kindIcon: Record<string, string> = { srt: 'ph ph-subtitles', txt: 'ph ph-text-align-right', json: 'ph ph-brackets-curly', lrc: 'ph ph-music-notes-simple', mp4: 'ph ph-film-strip' };

export function Dashboard() {
  const { project: p, job, save } = useStore();
  const { outputs, versions } = useWS();
  if (!p) return <StateView kind="emptyDashboard" />;
  const low = p.lines.reduce((a, l) => a + l.words.filter((w) => w.conf === 'low' && !w.manual).length, 0);
  const manual = p.lines.reduce((a, l) => a + l.words.filter((w) => w.manual).length, 0);
  return (
    <Page title={'שלום, ' + (ws.me()?.name.split(' ')[0] ?? '')} sub="הפרויקטים פרטיים · משימת GPU אחת בכל רגע" actions={<button className="btn btn-primary" onClick={() => { ui.setStep('input'); ui.go('simple'); }}><i className="ph ph-plus" />קריוקי חדש</button>}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 16 }}>
        <Card>
          <span className="kicker">פרויקט פתוח</span>
          <div style={{ fontSize: 20 }}>{p.title}</div>
          <div className="muted" style={{ fontSize: 13 }}>{p.artist} · {fmtShort(p.source.duration)} · {p.lines.length} שורות</div>
          <KV k="שמירה" v={{ idle: 'אין שינויים', saving: 'שומר בדפדפן…', saved: 'נשמר בדפדפן · ' + save.at, local: 'נשמר בדפדפן · ממתין לסנכרון', failed: 'השמירה נכשלה', conflict: 'התנגשות גרסאות' }[save.state]} />
          <KV k="תיקונים ידניים" v={manual} />
          <KV k="ביטחון מודל נמוך" v={low ? low + ' מילים לבדיקה' : 'אין'} />
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button className="btn btn-secondary" onClick={() => ui.go('subs-word')}><i className="ph ph-text-aa" />עורך מילים</button>
            <button className="btn btn-ghost" onClick={() => ui.go('project')}>פרטי פרויקט</button>
          </div>
        </Card>
        <Card>
          <span className="kicker">עיבוד</span>
          {job ? (
            <>
              <div style={{ fontSize: 15 }} dir="auto">{job.name}</div>
              <div style={{ height: 6, borderRadius: 999, background: 'var(--color-neutral-800)', overflow: 'hidden' }}><div style={{ height: '100%', width: jobPct(job) + '%', background: 'var(--color-accent)' }} /></div>
              <KV k="סטטוס" v={{ running: 'רץ · ' + jobPct(job) + '%', failed: 'קרס · דורש התאוששות', cancelled: 'בוטל', done: 'הושלם' }[job.status]} />
              <button className="btn btn-ghost" onClick={() => ui.go('queue')} style={{ alignSelf: 'flex-start' }}>לתור העיבוד</button>
            </>
          ) : (
            <span className="muted" style={{ fontSize: 13 }}>אין עבודה פעילה.</span>
          )}
        </Card>
        <Card>
          <span className="kicker">תוצרים אחרונים</span>
          {outputs.length ? outputs.slice(0, 4).map((o) => (
            <div key={o.id} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
              <i className={kindIcon[o.kind]} style={{ color: 'var(--color-accent-300)' }} />
              <span dir="ltr" style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}>{o.name}</span>
              <span className="dim num">{hhmm(o.at)}</span>
            </div>
          )) : <span className="muted" style={{ fontSize: 13 }}>עדיין לא יוצאו קבצים.</span>}
          <button className="btn btn-ghost" onClick={() => ui.go('outputs')} style={{ alignSelf: 'flex-start' }}>לספריית התוצרים</button>
        </Card>
      </div>
      <Card>
        <span className="kicker">פעילות אחרונה</span>
        {p.history.slice(0, 6).map((h, i) => (
          <div key={i} style={{ display: 'flex', gap: 10, fontSize: 13 }}><span dir="ltr" className="num dim">{hhmm(h.at)}</span><span>{h.label}</span></div>
        ))}
        <span className="dim" style={{ fontSize: 12 }}>{versions.length} נקודות שמירה · ⌘S יוצר נקודה חדשה</span>
      </Card>
    </Page>
  );
}

export function Library() {
  const { project: p, backend } = useStore();
  const [q, setQ] = useState('');
  const [remote, setRemote] = useState<{ id: string; title: string; artist: string; lines: number }[] | null>(null);
  useEffect(() => {
    if (backend.mode !== 'server') return;
    listProjects().then(setRemote).catch(() => setRemote(null));
  }, [backend.mode, p?.id]);
  const rows: { id?: string; t: string; a: string; d: string; s: string; real: boolean }[] = remote
    ? remote.map((r) => ({ id: r.id, t: r.title, a: r.artist, d: '', s: (r.id === p?.id ? 'פתוח · ' : '') + r.lines + ' שורות', real: true }))
    : [
        ...(p ? [{ t: p.title, a: p.artist, d: fmtShort(p.source.duration), s: 'פתוח · ' + p.lines.length + ' שורות', real: true }] : []),
        ...DEMO_LIBRARY.map((x) => ({ ...x, real: false })),
      ];
  const filtered = rows.filter((r) => !q.trim() || (r.t + ' ' + r.a).toLowerCase().includes(q.trim().toLowerCase()));
  return (
    <Page title="ספריית שירים" sub={backend.mode === 'server' ? 'פרויקטים פרטיים · בשרת' : 'פרויקטים פרטיים · מצב הדגמה'} actions={<button className="btn btn-primary" onClick={() => { ui.setStep('input'); ui.go('simple'); }}><i className="ph ph-plus" />שיר חדש</button>}>
      <input className="input" placeholder="חיפוש לפי שם שיר או אמן" aria-label="חיפוש" value={q} onChange={(e) => setQ(e.target.value)} style={{ maxWidth: 420 }} />
      {filtered.length === 0 ? (
        <StateView kind={q ? 'emptyLibrary' : 'emptyDashboard'} inline onClear={() => setQ('')} />
      ) : (
        <div className="panel" style={{ padding: 6, display: 'flex', flexDirection: 'column', gap: 2 }}>
          {filtered.map((r) => (
            <button key={r.id || r.t} className="row-btn" onClick={async () => {
              if (r.id && r.id !== p?.id) {
                try { await openProject(r.id); } catch (e) { return ui.toast((e as Error).message); }
              }
              if (r.real) ui.go('project'); else ui.toast('פרויקט לדוגמה בלבד · במצב הדגמה יש פרויקט אמיתי אחד');
            }}>
              <i className="ph ph-music-note" style={{ fontSize: 18, color: 'var(--color-accent-300)' }} />
              <span style={{ flex: 1, minWidth: 0 }}>
                <span dir="auto">{r.t}</span>
                <span className="dim" style={{ fontSize: 12, marginInlineStart: 8 }} dir="auto">{r.a}</span>
              </span>
              <span className="num dim" style={{ fontSize: 12 }}>{r.d}</span>
              <span className={r.real ? 'tag tag-accent' : 'tag tag-neutral'}>{r.real ? r.s : 'דוגמה · ' + r.s}</span>
            </button>
          ))}
        </div>
      )}
    </Page>
  );
}

export function Outputs() {
  const { outputs } = useWS();
  const { sim } = useStore();
  const remote = sim.conn === 'remote';
  return (
    <Page title="ספריית תוצרים" sub={remote ? 'שרת מרוחק · ניגון והורדה' : 'המחשב הזה · ' + ws.get().settings.outputDir}>
      {outputs.length === 0 ? (
        <Card><span className="muted">עדיין לא יוצאו קבצים. ייצוא SRT, TXT, LRC ו־JSON מהעורך או ממסך הייצוא יופיע כאן.</span><button className="btn btn-secondary" onClick={() => ui.go('export')} style={{ alignSelf: 'flex-start' }}>למסך הייצוא</button></Card>
      ) : (
        <div className="panel" style={{ padding: 6, display: 'flex', flexDirection: 'column', gap: 2 }}>
          {outputs.map((o) => (
            <div key={o.id} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: 10, flexWrap: 'wrap' }}>
              <i className={kindIcon[o.kind]} style={{ fontSize: 18, color: 'var(--color-accent-300)' }} />
              <span dir="ltr" style={{ flex: 1, minWidth: 160 }}>{o.name}</span>
              <span className="dim" style={{ fontSize: 12 }}>{o.lines} שורות · {hhmm(o.at)}</span>
              <span className={o.simulated ? 'tag tag-neutral' : 'tag tag-outline'}>{o.simulated ? 'מדומה · דורש שרת' : 'קובץ אמיתי'}</span>
              {o.simulated ? (
                <button className="btn btn-ghost" onClick={() => ui.toast('סימולציה: אין קובץ MP4 באב־טיפוס')}>{remote ? 'נגן' : 'פתח תיקייה'}</button>
              ) : (
                <button className="btn btn-ghost" onClick={() => exportFile(o.kind as 'srt')}><i className="ph ph-download-simple" />{remote ? 'הורד' : 'הורד שוב'}</button>
              )}
              <button className="btn btn-ghost btn-icon" aria-label="הסר" onClick={() => ws.removeOutput(o.id)}><i className="ph ph-x" /></button>
            </div>
          ))}
        </div>
      )}
      <span className="dim" style={{ fontSize: 12 }}>הורדה חוזרת מפיקה את הקובץ מחדש מהנתונים העדכניים של הפרויקט.</span>
    </Page>
  );
}

export function ExportScreen() {
  const { project: p } = useStore();
  const [fmt, setFmt] = useState<'srt' | 'lrc'>('srt');
  if (!p) return <StateView kind="emptyDashboard" />;
  const preview = (fmt === 'srt' ? toSrt(p) : toLrc(p)).split('\n').slice(0, 14).join('\n');
  return (
    <Page title="ייצוא" sub={p.title}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 20, alignItems: 'flex-start' }}>
        <div style={{ flex: '1 1 380px', display: 'flex', flexDirection: 'column', gap: 16 }}>
          <Card>
            <span className="kicker">סרטון קריוקי</span>
            <KV k="פסקול" v="Instrumental · אודיו מקורי מושתק" />
            <KV k="תזמון" v={p.timing === 'word' ? 'הדגשה ברמת מילה' : 'שורות בלבד'} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span className="muted" style={{ fontSize: 13 }}>רזולוציה</span>
              <Seg value={p.export.res} onChange={(v: Res) => store.edit('רזולוציית ייצוא', (P) => void (P.export.res = v))} options={[['720p', '720p'], ['1080p', '1080p']]} />
            </div>
            <button className="btn btn-primary" onClick={() => void exportMp4()} style={{ justifyContent: 'center' }}><i className="ph ph-film-strip" />{serverProject() ? (renderFresh(p) ? 'הורד MP4' : 'רנדר MP4 בשרת') : 'ייצא MP4 (דורש שרת · מדומה)'}</button>
          </Card>
          <Card>
            <span className="kicker">קבצים אמיתיים מהנתונים הערוכים</span>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {(['srt', 'lrc', 'txt', 'json'] as const).map((k) => (
                <button key={k} className="btn btn-secondary" onClick={() => exportFile(k)}><i className="ph ph-download-simple" />{k.toUpperCase()}</button>
              ))}
            </div>
            <span className="dim" style={{ fontSize: 12 }}>LRC כולל סימני זמן למילים כשיש תזמון ברמת מילה. ASS לאפקט קריוקי מדורג הוא יכולת עתידית.</span>
          </Card>
        </div>
        <Card style={{ flex: '1 1 380px', minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}><span className="kicker">תצוגת קובץ</span><Seg value={fmt} onChange={setFmt} options={[['srt', 'SRT'], ['lrc', 'LRC']]} /></div>
          <pre dir="auto" style={{ margin: 0, fontFamily: "ui-monospace,'SF Mono',Menlo,monospace", fontSize: 12, lineHeight: 1.6, whiteSpace: 'pre-wrap', color: 'var(--color-neutral-300)', background: 'var(--color-bg)', padding: 12, borderRadius: 'var(--radius-md)' }}>{preview}</pre>
        </Card>
      </div>
    </Page>
  );
}

export function Team() {
  const { users, invites } = useWS();
  const { sim } = useStore();
  const me = ws.me();
  const isAdmin = me?.role === 'admin';
  return (
    <Page title="צוות והרשאות" sub="אין הרשמה ציבורית · הזמנות חד־פעמיות בלבד" actions={isAdmin && <button className="btn btn-primary" onClick={() => ui.open('invite')}><i className="ph ph-link" />צור קישור הזמנה</button>}>
      {!isAdmin && <div className="alert"><i className="ph ph-lock-simple" />רק מנהל יכול לשנות תפקידים וליצור הזמנות.</div>}
      <Card>
        <span className="kicker">משתמשים</span>
        {users.map((u) => (
          <div key={u.id} style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span style={{ width: 32, height: 32, borderRadius: '50%', display: 'grid', placeItems: 'center', fontSize: 12.5, background: 'var(--color-neutral-700)' }}>{u.name[0]}</span>
            <div style={{ flex: 1, minWidth: 140 }}>
              <div style={{ fontSize: 13.5 }}>{u.name}{u.id === me?.id ? ' (אתה)' : ''}</div>
              <div className="dim" dir="ltr" style={{ fontSize: 11.5, textAlign: 'right' }}>@{u.user}</div>
            </div>
            <select className="input" aria-label={'תפקיד של ' + u.name} value={u.role} disabled={!isAdmin || u.id === me?.id} onChange={(e) => ws.setRole(u.id, e.target.value as Role) && ui.toast('התפקיד עודכן')} style={{ width: 110 }}>
              {(Object.keys(ROLE_L) as Role[]).map((r) => <option key={r} value={r}>{ROLE_L[r]}</option>)}
            </select>
            {isAdmin && u.id !== me?.id && <button className="btn btn-ghost btn-icon" aria-label={'הסר את ' + u.name} onClick={() => confirm('להסיר את ' + u.name + '?') && ws.removeUser(u.id)}><i className="ph ph-user-minus" /></button>}
          </div>
        ))}
      </Card>
      <Card>
        <span className="kicker">הרשאות לפי תפקיד</span>
        <table className="table" style={{ width: '100%', fontSize: 13 }}>
          <thead><tr><th style={{ textAlign: 'start' }}>פעולה</th><th>מנהל</th><th>עורך</th><th>צופה</th></tr></thead>
          <tbody>
            {[['צפייה וניגון', 1, 1, 1], ['עריכת מילים ועיצוב', 1, 1, 0], ['הפעלת עיבוד AI', 1, 1, 0], ['ייצוא והורדה', 1, 1, 1], ['ניהול משתמשים והזמנות', 1, 0, 0], ['הגדרות שרת', 1, 0, 0]].map(([l, ...v]) => (
              <tr key={l as string}><td>{l}</td>{v.map((x, i) => <td key={i} style={{ textAlign: 'center' }}><i className={x ? 'ph-bold ph-check' : 'ph ph-minus'} style={{ color: x ? 'var(--color-accent-300)' : 'var(--color-neutral-600)' }} /></td>)}</tr>
            ))}
          </tbody>
        </table>
      </Card>
      <Card>
        <span className="kicker">קישורי הזמנה</span>
        {invites.length === 0 ? <span className="muted" style={{ fontSize: 13 }}>אין קישורים פעילים.</span> : invites.map((i) => {
          const st = i.used ? 'שומש' : i.expiresAt < Date.now() ? 'פג תוקף' : 'פעיל';
          return (
            <div key={i.id} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 13, flexWrap: 'wrap' }}>
              <span style={{ flex: 1, minWidth: 120 }}>{i.name} · {ROLE_L[i.role]}</span>
              <span dir="ltr" className="dim" style={{ fontFamily: 'ui-monospace,Menlo,monospace', fontSize: 11.5 }}>{inviteUrl(i.token, sim.conn === 'remote')}</span>
              <span className={st === 'פעיל' ? 'tag tag-accent' : 'tag tag-neutral'}>{st}</span>
              {isAdmin && <button className="btn btn-ghost" onClick={() => ws.revokeInvite(i.id)}>בטל</button>}
            </div>
          );
        })}
        <span className="dim" style={{ fontSize: 12 }}>אין שליחת מיילים. המנהל מעתיק את הקישור ושולח בעצמו. קישור עובד פעם אחת.</span>
      </Card>
    </Page>
  );
}

