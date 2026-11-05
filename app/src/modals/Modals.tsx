import { useEffect, useMemo, useRef, useState } from 'react';
import { lineText } from '../lib/format';
import { AI_ALTERNATES, OTHER_USER, THEIRS_LINE_INDEX, THEIRS_LINE_TEXT } from '../store/demo';
import { store, useStore } from '../store/store';
import { ui, useUI, type ModalKind } from '../store/ui';
import { ROLE_L, inviteUrl, useWS, ws, type Role } from '../store/workspace';
import { hhmm } from '../lib/format';

const META: Record<ModalKind, [string, string, number]> = {
  conflict: ['התנגשות גרסאות', 'ph ph-git-merge', 620],
  airerun: ['הצעת AI להשוואה', 'ph ph-git-diff', 860],
  shortcuts: ['קיצורי מקלדת', 'ph ph-keyboard', 560],
  share: ['שיתוף פרויקט', 'ph ph-share-network', 540],
  invite: ['קישור הזמנה חד־פעמי', 'ph ph-link', 480],
  find: ['חיפוש והחלפה', 'ph ph-magnifying-glass', 540],
  delete: ['מחיקת פרויקט', 'ph ph-trash', 460],
  restore: ['לשחזר גרסה?', 'ph ph-clock-counter-clockwise', 480],
};

export function ModalHost() {
  const { modal, modalArg } = useUI();
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!modal) return;
    const prev = document.activeElement as HTMLElement | null;
    ref.current?.focus();
    return () => prev?.focus?.();
  }, [modal]);
  if (!modal) return null;
  const [title, icon, w] = META[modal];
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && ui.close()}>
      <div ref={ref} tabIndex={-1} role="dialog" aria-modal="true" aria-label={title} className="modal" style={{ width: w }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <i className={icon} style={{ fontSize: 20, color: 'var(--color-accent-300)' }} />
          <h4 style={{ margin: 0, fontSize: 17 }}>{title}</h4>
          <button className="btn btn-ghost btn-icon" onClick={() => ui.close()} aria-label="סגור" style={{ marginInlineStart: 'auto', color: 'var(--color-neutral-400)' }}>
            <i className="ph ph-x" />
          </button>
        </div>
        {modal === 'conflict' && <Conflict />}
        {modal === 'airerun' && <AiRerun />}
        {modal === 'shortcuts' && <Shortcuts />}
        {modal === 'share' && <Share />}
        {modal === 'invite' && <InviteForm />}
        {modal === 'find' && <FindReplace />}
        {modal === 'delete' && <DeleteProject />}
        {modal === 'restore' && <RestoreVersion id={modalArg} />}
      </div>
    </div>
  );
}

function Conflict() {
  const { project: p } = useStore();
  const [side, setSide] = useState<'mine' | 'theirs'>('mine');
  if (!p) return null;
  const mine = p.lines[THEIRS_LINE_INDEX] ? lineText(p.lines[THEIRS_LINE_INDEX]) : '';
  const sides: ['mine' | 'theirs', string, string, string][] = [
    ['mine', 'השינויים שלי', 'נשמר בדפדפן', mine],
    ['theirs', 'של ' + OTHER_USER, 'נשמר לפני רגע', THEIRS_LINE_TEXT],
  ];
  return (
    <>
      <div style={{ fontSize: 13.5, color: 'var(--color-neutral-300)', textWrap: 'pretty' }}>
        {OTHER_USER} שמרה שינוי בשורה {THEIRS_LINE_INDEX + 1} בזמן שערכת. השינויים שלך שמורים בדפדפן. (התנגשות מדומה: אין שרת או משתמש נוסף באב־טיפוס.)
      </div>
      <div role="radiogroup" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10 }}>
        {sides.map(([k, t, m, v]) => (
          <button
            key={k}
            role="radio"
            aria-checked={side === k}
            onClick={() => setSide(k)}
            style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', gap: 8, padding: 12, borderRadius: 'var(--radius-md)', background: 'var(--color-bg)', border: 0, font: 'inherit', color: 'inherit', textAlign: 'start', boxShadow: side === k ? 'inset 0 0 0 1px var(--color-accent)' : 'none' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%' }}>
              <span style={{ width: 16, height: 16, borderRadius: '50%', boxShadow: `inset 0 0 0 ${side === k ? 5 : 1.5}px var(--color-accent)` }} />
              <span style={{ fontSize: 14 }}>{t}</span>
              <span className="dim" style={{ marginInlineStart: 'auto', fontSize: 11.5 }}>{m}</span>
            </div>
            <div style={{ display: 'flex', gap: 8, fontSize: 12.5 }}>
              <span className="dim" style={{ width: 56 }}>שורה {THEIRS_LINE_INDEX + 1}</span>
              <span dir="auto" style={{ flex: 1 }}>{v}</span>
            </div>
          </button>
        ))}
      </div>
      <div className="muted" style={{ fontSize: 12 }}>הגרסה שלא תיבחר לא נמחקת: היא נרשמת בהיסטוריית הגרסאות.</div>
      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
        <button
          className="btn btn-primary"
          onClick={() => {
            ws.saveVersion('לפני פתרון התנגשות');
            store.resolveConflict(side);
            ui.close();
            ui.toast(side === 'mine' ? 'השינויים שלך נשמרו · הגרסה של ' + OTHER_USER + ' בהיסטוריה' : 'הגרסה של ' + OTHER_USER + ' נטענה · השינויים שלך בהיסטוריה');
          }}
        >
          <i className="ph ph-check" />
          {side === 'mine' ? 'שמור את שלי' : 'קבל את של ' + OTHER_USER}
        </button>
      </div>
    </>
  );
}

interface Suggestion {
  lineId: string;
  n: number;
  current: string;
  proposed: string;
  locked: boolean;
  note: string;
  changes: { wordId: string; to: string }[];
}

/** Simulated AI re-run: the new output is shown as a suggestion; manual fixes are never offered for change. */
function AiRerun() {
  const { project: p } = useStore();
  const [dec, setDec] = useState<Record<string, 'a' | 'r'>>({});
  const rows = useMemo<Suggestion[]>(() => {
    if (!p) return [];
    const out: Suggestion[] = [];
    p.lines.forEach((l, i) => {
      const changes = l.words.filter((w) => !w.manual && AI_ALTERNATES[w.w]).map((w) => ({ wordId: w.id, to: AI_ALTERNATES[w.w] }));
      const hasAlt = l.words.some((w) => AI_ALTERNATES[w.w]);
      if (!changes.length && !(l.manual && hasAlt)) return;
      const proposed = l.words.map((w) => changes.find((c) => c.wordId === w.id)?.to ?? w.w).join(' ');
      const worst = l.words.filter((w) => changes.some((c) => c.wordId === w.id)).reduce((a, w) => (w.conf === 'low' ? 'נמוך' : a === 'נמוך' ? a : w.conf === 'med' ? 'בינוני' : a), 'גבוה');
      out.push({ lineId: l.id, n: i + 1, current: lineText(l), proposed, locked: changes.length === 0, note: changes.length ? 'ביטחון מודל: ' + worst : 'נשמר אוטומטית', changes });
    });
    return out;
  }, [p]);
  const manualCount = p ? p.lines.reduce((a, l) => a + l.words.filter((w) => w.manual).length, 0) : 0;
  const accepted = rows.filter((r) => !r.locked && dec[r.lineId] === 'a');

  return (
    <>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        <span className="tag tag-neutral">תמלול מחדש · סימולציה</span>
        <span className="tag tag-outline">שום דבר לא הוחלף אוטומטית</span>
        <span className="tag tag-accent">{manualCount} תיקונים ידניים נשמרו ולא הוצעו לשינוי</span>
      </div>
      {rows.length === 0 ? (
        <div className="muted" style={{ fontSize: 13.5 }}>אין הצעות חדשות. התמלול הנוכחי תואם להרצה החדשה.</div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <div className="dim" style={{ display: 'grid', gridTemplateColumns: '44px minmax(0,1fr) minmax(0,1fr) 140px', gap: 10, padding: '0 8px', fontSize: 11, letterSpacing: '0.06em' }}>
            <span>שורה</span>
            <span>נוכחי</span>
            <span>הצעת AI</span>
            <span />
          </div>
          {rows.map((r) => {
            const d = dec[r.lineId];
            return (
              <div key={r.lineId} style={{ display: 'grid', gridTemplateColumns: '44px minmax(0,1fr) minmax(0,1fr) 140px', gap: 10, alignItems: 'center', padding: '9px 8px', borderRadius: 'var(--radius-md)', background: d === 'a' ? 'color-mix(in srgb, var(--color-accent) 8%, transparent)' : 'var(--color-bg)' }}>
                <span className="dim" style={{ fontSize: 12 }}>{r.n}</span>
                <span dir="auto" style={{ fontSize: 13.5, color: 'var(--color-neutral-300)' }}>{r.current}</span>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <span dir="auto" style={{ fontSize: 13.5, color: r.locked ? 'var(--color-neutral-500)' : d === 'a' ? 'var(--color-accent-200)' : 'var(--color-text)' }}>
                    {r.locked ? 'ללא הצעה · השורה תוקנה ידנית' : r.proposed}
                  </span>
                  <span className="dim" style={{ fontSize: 11 }}>{r.note}</span>
                </div>
                <div style={{ display: 'flex', gap: 4, justifyContent: 'flex-end' }}>
                  {r.locked ? (
                    <span className="tag tag-neutral"><i className="ph ph-lock-simple" />תיקון ידני</span>
                  ) : (
                    <>
                      <button className="btn btn-secondary" aria-pressed={d === 'a'} onClick={() => setDec({ ...dec, [r.lineId]: 'a' })} style={{ fontSize: 12, boxShadow: d === 'a' ? 'inset 0 0 0 1px var(--color-accent)' : undefined }}>קבל</button>
                      <button className="btn btn-ghost" aria-pressed={d === 'r'} onClick={() => setDec({ ...dec, [r.lineId]: 'r' })} style={{ fontSize: 12, color: d === 'r' ? 'var(--color-accent-200)' : 'var(--color-neutral-400)' }}>דחה</button>
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
      <div className="dim" style={{ fontSize: 11.5 }}>ביטחון המודל הוא הערכה של המודל עצמו ולא דיוק מאומת. ההחלה היא עריכה אחת שאפשר לבטל (⌘Z).</div>
      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', flexWrap: 'wrap' }}>
        <button className="btn btn-secondary" onClick={() => { ui.close(); ui.toast('ההצעה נדחתה · הגרסה הנוכחית לא השתנתה'); }}>דחה הכל</button>
        <button
          className="btn btn-primary"
          disabled={!accepted.length}
          onClick={() => {
            ws.saveVersion('לפני החלת הצעות AI');
            store.edit('הצעות AI שנבחרו הוחלו (' + accepted.length + ')', (P) => {
              accepted.forEach((r) => {
                const l = P.lines.find((x) => x.id === r.lineId);
                r.changes.forEach((c) => {
                  const w = l?.words.find((x) => x.id === c.wordId);
                  if (w && !w.manual) {
                    w.w = c.to;
                    w.conf = 'med';
                  }
                });
              });
            });
            ui.close();
            ui.toast(accepted.length + ' הצעות הוחלו · אפשר לבטל עם ⌘Z');
          }}
        >
          <i className="ph ph-check" />
          החל {accepted.length} הצעות שנבחרו
        </button>
      </div>
    </>
  );
}

function Shortcuts() {
  const keys: [string, string][] = [
    ['נגן / עצור', 'Space'],
    ['מצבי עורך', '1 · 2 · 3'],
    ['מילה הבאה / קודמת', '← →'],
    ['פיצול מילה', 'S'],
    ['מיזוג עם הבאה', 'M'],
    ['החלפת דובר', 'D'],
    ['לולאה על המילה', 'L'],
    ['בטל / בצע שוב', '⌘Z · ⇧⌘Z'],
    ['שמירה ידנית', '⌘S'],
    ['חלון זה', '?'],
  ];
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '6px 20px' }}>
      {keys.map(([l, k]) => (
        <div key={l} style={{ display: 'flex', alignItems: 'center', fontSize: 13.5, padding: '6px 0' }}>
          <span>{l}</span>
          <span dir="ltr" className="tag tag-neutral" style={{ marginInlineStart: 'auto' }}>{k}</span>
        </div>
      ))}
    </div>
  );
}

function CopyField({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div style={{ display: 'flex', gap: 6 }}>
      <input className="input" dir="ltr" readOnly value={value} aria-label="קישור" onFocus={(e) => e.target.select()} style={{ flex: 1, minWidth: 0, textAlign: 'left', fontFamily: 'ui-monospace,Menlo,monospace', fontSize: 12 }} />
      <button
        className="btn btn-primary"
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(value);
          } catch {
            /* clipboard blocked: the field is selectable */
          }
          setCopied(true);
        }}
        style={{ fontSize: 12.5 }}
      >
        <i className={copied ? 'ph ph-check' : 'ph ph-copy'} />
        {copied ? 'הועתק' : 'העתק'}
      </button>
    </div>
  );
}

function InviteForm({ embedded }: { embedded?: boolean }) {
  const { sim } = useStore();
  const [name, setName] = useState('');
  const [role, setRole] = useState<Role>('editor');
  const [days, setDays] = useState(7);
  const [link, setLink] = useState<string | null>(null);
  if (ws.me()?.role !== 'admin') return <div className="alert"><i className="ph ph-lock-simple" />רק מנהל יכול ליצור הזמנות.</div>;
  if (link)
    return (
      <>
        <CopyField value={link} />
        <div className="muted" style={{ fontSize: 12 }}>הקישור מוצג פעם אחת. שלח אותו בעצמך (וואטסאפ, הודעה). אחרי שימוש אחד הוא נסגר.</div>
        {!embedded && <button className="btn btn-secondary" onClick={() => ui.close()} style={{ alignSelf: 'flex-start' }}>סיום</button>}
      </>
    );
  return (
    <>
      <label className="field" style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <span className="muted" style={{ fontSize: 12 }}>שם לזיהוי</span>
        <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="למשל: אורי אלון" />
      </label>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <span className="muted" style={{ fontSize: 12 }}>תפקיד</span>
          <select className="input" value={role} onChange={(e) => setRole(e.target.value as Role)}>{(Object.keys(ROLE_L) as Role[]).map((r) => <option key={r} value={r}>{ROLE_L[r]}</option>)}</select>
        </label>
        <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <span className="muted" style={{ fontSize: 12 }}>תוקף</span>
          <select className="input" value={days} onChange={(e) => setDays(+e.target.value)}><option value={1}>24 שעות</option><option value={7}>7 ימים</option></select>
        </label>
      </div>
      <button className="btn btn-primary" disabled={!name.trim()} style={{ alignSelf: 'flex-start' }} onClick={() => { const inv = ws.createInvite(name.trim(), role, days, ''); setLink(inviteUrl(inv.token, sim.conn === 'remote')); }}>
        <i className="ph ph-link" />צור קישור הזמנה
      </button>
    </>
  );
}

function Share() {
  const { users } = useWS();
  const { project: p } = useStore();
  const me = ws.me();
  return (
    <>
      <div className="muted" style={{ fontSize: 13 }}>{p?.title} · פרויקט פרטי. גישה לפי תפקיד בצוות.</div>
      {users.map((u) => (
        <div key={u.id} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ width: 32, height: 32, borderRadius: '50%', display: 'grid', placeItems: 'center', fontSize: 12.5, background: 'var(--color-neutral-700)' }}>{u.name[0]}</span>
          <div style={{ flex: 1, fontSize: 13.5 }}>{u.name}{u.id === me?.id ? ' (אתה)' : ''}</div>
          <span className="tag tag-neutral">{ROLE_L[u.role]}</span>
        </div>
      ))}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, padding: 12, borderRadius: 'var(--radius-md)', background: 'var(--color-bg)' }}>
        <div style={{ fontSize: 13.5 }}>למישהו שעדיין אין לו חשבון</div>
        <InviteForm embedded />
        <div className="dim" style={{ fontSize: 11.5 }}>אין שליחת מיילים ואין קישור ציבורי. הקישור חד־פעמי.</div>
      </div>
      <button className="btn btn-ghost" onClick={() => ui.go('team')} style={{ alignSelf: 'flex-start' }}>ניהול צוות והרשאות</button>
    </>
  );
}

function FindReplace() {
  const { project: p } = useStore();
  const [q, setQ] = useState('');
  const [r, setR] = useState('');
  if (!p) return null;
  const hits = q.trim() ? p.lines.flatMap((l, i) => l.words.filter((w) => w.w.includes(q.trim())).map((w) => ({ i, w }))) : [];
  return (
    <>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <input className="input" dir="auto" aria-label="חפש" placeholder="חפש" value={q} onChange={(e) => setQ(e.target.value)} style={{ flex: '1 1 160px' }} autoFocus />
        <input className="input" dir="auto" aria-label="החלף ב" placeholder="החלף ב" value={r} onChange={(e) => setR(e.target.value)} style={{ flex: '1 1 160px' }} />
      </div>
      <div className="muted" style={{ fontSize: 12.5 }}>{q.trim() ? hits.length + ' התאמות' : 'החיפוש בתוך מילים, בכל השורות. התזמונים לא משתנים.'}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 220, overflow: 'auto' }}>
        {hits.slice(0, 30).map(({ i, w }) => (
          <div key={w.id} style={{ display: 'flex', gap: 10, fontSize: 13 }}>
            <span className="dim" style={{ width: 52 }}>שורה {i + 1}</span>
            <span dir="auto">{lineText(p.lines[i])}</span>
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
        <button
          className="btn btn-primary"
          disabled={!hits.length || /\s/.test(r.trim()) || !r.trim()}
          onClick={() => {
            const n = hits.length;
            store.edit('החלפה: ' + q.trim() + ' ← ' + r.trim() + ' (' + n + ')', (P) => {
              P.lines.forEach((l) => l.words.forEach((w) => {
                if (w.w.includes(q.trim())) {
                  w.w = w.w.split(q.trim()).join(r.trim());
                  w.manual = true;
                  w.conf = 'high';
                  l.manual = true;
                }
              }));
            });
            ui.close();
            ui.toast(n + ' מילים הוחלפו · ⌘Z לביטול');
          }}
        >
          החלף הכל
        </button>
      </div>
    </>
  );
}

function DeleteProject() {
  const { project: p } = useStore();
  const [t, setT] = useState('');
  if (!p) return null;
  return (
    <>
      <div style={{ fontSize: 13.5, color: 'var(--color-neutral-300)' }}>
        הפעולה מוחקת את הפרויקט, המילים, ההיסטוריה והגדרות העיצוב מהדפדפן הזה. אי אפשר לבטל אותה. כדי לאשר, הקלד את שם הפרויקט: <b style={{ fontWeight: 500 }}>{p.title}</b>
      </div>
      <input className="input" dir="auto" aria-label="שם הפרויקט לאישור" value={t} onChange={(e) => setT(e.target.value)} />
      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
        <button className="btn btn-secondary" onClick={() => ui.close()}>ביטול</button>
        <button className="btn btn-primary" disabled={t.trim() !== p.title} onClick={() => { store.deleteProject(); ui.close(); ui.go('dashboard'); ui.toast('הפרויקט נמחק'); }}>
          <i className="ph ph-trash" />מחק לצמיתות
        </button>
      </div>
    </>
  );
}

function RestoreVersion({ id }: { id: string | null }) {
  const { versions } = useWS();
  const { project: p } = useStore();
  const v = versions.find((x) => x.id === id);
  if (!v || !p) return <div className="muted">הגרסה לא נמצאה.</div>;
  const changed = v.lines.length !== p.lines.length ? Math.abs(v.lines.length - p.lines.length) + ' שורות נוספו/נמחקו' : v.lines.filter((l, i) => lineText(l) !== lineText(p.lines[i])).length + ' שורות שונות';
  return (
    <>
      <div style={{ fontSize: 13.5, color: 'var(--color-neutral-300)' }}>
        הגרסה "{v.label}" מ־{hhmm(v.at)} · {changed} לעומת הגרסה הנוכחית. הגרסה הנוכחית תישמר קודם כנקודת שמירה, כך ששום דבר לא אובד.
      </div>
      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
        <button className="btn btn-secondary" onClick={() => ui.close()}>ביטול</button>
        <button className="btn btn-primary" onClick={() => { ws.restoreVersion(v.id); ui.close(); ui.toast('הגרסה שוחזרה · ⌘Z לביטול'); }}>
          <i className="ph ph-clock-counter-clockwise" />שחזר
        </button>
      </div>
    </>
  );
}
