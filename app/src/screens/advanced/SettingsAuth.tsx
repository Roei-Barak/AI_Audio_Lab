import { useState } from 'react';
import { Seg, Switch } from '../../components/controls';
import { store, useStore } from '../../store/store';
import { ui, useUI, type Route } from '../../store/ui';
import { ROLE_L, useWS, ws } from '../../store/workspace';
import * as RW from '../../store/remoteWs';
import { Card, KV, Page, Tabs } from './Sidebar';
import { StateView } from './SystemStates';

type STab = 'settings' | 'settings-general' | 'settings-models' | 'settings-storage' | 'settings-api';

const Row = ({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', padding: '6px 0' }}>
    <div style={{ flex: '1 1 220px' }}>
      <div style={{ fontSize: 14 }}>{label}</div>
      {hint && <div className="dim" style={{ fontSize: 12 }}>{hint}</div>}
    </div>
    {children}
  </div>
);

function storageBytes() {
  let n = 0;
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i)!;
      if (k.startsWith('lyra.')) n += (localStorage.getItem(k) || '').length * 2;
    }
  } catch {
    /* ignore */
  }
  return n;
}

export function Settings() {
  const { route } = useUI();
  const { sim, job } = useStore();
  const { settings: s } = useWS();
  const [, force] = useState(0);
  const me = ws.me();
  if (me?.role !== 'admin' && (route === 'settings' || route === 'settings-api')) return <StateView kind="forbidden" />;
  const tab = route as STab;
  const remote = sim.conn === 'remote';
  const gpuBusy = job?.status === 'running';
  return (
    <Page title="הגדרות ושרת" sub={remote ? 'שרת מרוחק · lyra.home' : 'המחשב הזה'}>
      <Tabs value={tab} onChange={(t) => ui.go(t as Route)} tabs={[['settings', 'שרת ו־Workers'], ['settings-general', 'כללי'], ['settings-models', 'מודלי AI'], ['settings-storage', 'אחסון'], ['settings-api', 'API']]} />
      {tab === 'settings' && (
        <>
          <Card>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <i className={remote ? 'ph ph-globe-simple' : 'ph ph-desktop'} style={{ color: 'var(--color-accent-300)' }} />
              <span style={{ fontSize: 15 }}>{remote ? 'שרת מרוחק · lyra.home' : 'המחשב הזה'}</span>
              <span className="tag tag-accent" style={{ marginInlineStart: 'auto' }}>{sim.offline ? 'לא זמין' : 'מחובר'}</span>
            </div>
            <KV k="GPU" v="16GB VRAM · מזוהה מהמחשב בפועל" />
            <KV k="משימות GPU כבדות במקביל" v="1 (ברירת מחדל ב־V1)" />
            <KV k="מצב" v={gpuBusy ? '1 / 1 בשימוש' : '0 / 1 · פנוי'} />
            <KV k="ייצוא SRT/TXT/LRC/JSON" v="בדפדפן, ללא GPU" />
          </Card>
          <Card>
            <span className="kicker">חיבור</span>
            <Row label="מצב חיבור" hint="במחשב הזה: פתיחת תיקיות. בשרת מרוחק: ניגון והורדה.">
              <Seg value={sim.conn} onChange={(v) => store.setSim('conn', v)} options={[['local', 'המחשב הזה'], ['remote', 'שרת מרוחק']]} />
            </Row>
            <span className="dim" style={{ fontSize: 12 }}>סימולציה: אין שרת עיבוד מחובר באב־טיפוס.</span>
          </Card>
        </>
      )}
      {tab === 'settings-general' && (
        <Card>
          <Row label="פרופיל ברירת מחדל" hint="משמש במצב הפשוט"><Seg value={s.defaultProfile} onChange={(v) => RW.setSettings({ defaultProfile: v })} options={[['מהיר', 'מהיר'], ['איכותי', 'איכותי']]} /></Row>
          <Row label="הפרדה ברירת מחדל"><Seg value={s.defaultStems} onChange={(v) => RW.setSettings({ defaultStems: v })} options={[[2, '2 ערוצים'], [4, '4 ערוצים']]} /></Row>
          <Row label="שפת תמלול"><Seg value={s.language} onChange={(v) => RW.setSettings({ language: v })} options={[['auto', 'אוטומטי'], ['he', 'עברית'], ['en', 'English']]} /></Row>
          <Row label="שמירה אוטומטית" hint="נשמר בדפדפן אחרי כל עריכה"><Switch on={s.autosave} label="שמירה אוטומטית" onChange={(v) => RW.setSettings({ autosave: v })} /></Row>
        </Card>
      )}
      {tab === 'settings-models' && (
        <Card>
          <Row label="מודל תמלול" hint="large-v3 מדויק יותר, medium מהיר יותר"><Seg value={s.model} onChange={(v) => RW.setSettings({ model: v })} options={[['large-v3', 'Whisper large-v3'], ['medium', 'Whisper medium']]} /></Row>
          <Row label="מודל הפרדה"><Seg value={s.separator} onChange={(v) => RW.setSettings({ separator: v })} options={[['demucs', 'Demucs'], ['mdx', 'MDX']]} /></Row>
          <span className="dim" style={{ fontSize: 12 }}>כל המודלים רצים מקומית על החומרה שלך. אין שירותי ענן בתשלום.</span>
        </Card>
      )}
      {tab === 'settings-storage' && (
        <Card>
          <Row label="תיקיית תוצרים" hint={remote ? 'בשרת המרוחק' : 'במחשב הזה'}>
            <input className="input" dir="ltr" value={s.outputDir} onChange={(e) => RW.setSettings({ outputDir: e.target.value })} style={{ width: 260 }} />
          </Row>
          <KV k="נתוני Lyra בדפדפן" v={(storageBytes() / 1024).toFixed(1) + ' KB · ללא קובצי מדיה'} />
          <Row label="ניקוי נקודות שמירה" hint="הפרויקט עצמו לא נמחק">
            <button className="btn btn-secondary" onClick={async () => { await RW.clearVersions(); force((n) => n + 1); ui.toast('נקודות השמירה נוקו'); }}>נקה</button>
          </Row>
        </Card>
      )}
      {tab === 'settings-api' && (
        <Card>
          <Row label="אסימון API מקומי" hint="לשימוש בסקריפטים ברשת הביתית. מוצג פעם אחת.">
            <button className="btn btn-secondary" onClick={() => RW.setSettings({ apiToken: 'lyra_' + Math.random().toString(36).slice(2, 14) })}>{s.apiToken ? 'צור מחדש' : 'צור אסימון'}</button>
          </Row>
          {s.apiToken && (
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
              <code dir="ltr" style={{ padding: '6px 10px', borderRadius: 6, background: 'var(--color-bg)', fontSize: 12.5 }}>{s.apiToken}</code>
              <button className="btn btn-ghost" onClick={() => RW.setSettings({ apiToken: null })}>בטל אסימון</button>
            </div>
          )}
          <span className="dim" style={{ fontSize: 12 }}>אין מפתחות לשירותים חיצוניים בתשלום ב־V1.</span>
        </Card>
      )}
    </Page>
  );
}

function AuthFrame({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div dir="rtl" style={{ height: '100%', overflow: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16, background: 'radial-gradient(90% 60% at 80% 0%, color-mix(in srgb, var(--color-accent-900) 55%, transparent), transparent 70%), var(--color-bg)' }}>
      <div className="panel" style={{ width: 400, maxWidth: '100%', padding: 24, display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}><i className="ph ph-microphone-stage" style={{ color: 'var(--color-accent)', fontSize: 22 }} /><span style={{ fontSize: 16, fontWeight: 500 }}>Lyra Studio</span></div>
        <h2 style={{ margin: 0, fontSize: 22, fontWeight: 500 }}>{title}</h2>
        {children}
      </div>
    </div>
  );
}

const Field = ({ label, ...p }: { label: string } & React.InputHTMLAttributes<HTMLInputElement>) => (
  <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
    <span className="muted" style={{ fontSize: 12 }}>{label}</span>
    <input className="input" {...p} />
  </label>
);

export function Login() {
  const { remote } = useWS();
  const [u, setU] = useState(remote.on ? '' : 'daniel');
  const [pw, setPw] = useState('');
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    const r = await RW.login(u, pw);
    setBusy(false);
    if (r !== true) return setErr(r);
    ui.go('simple');
    ui.toast('מחובר בתור ' + (ws.me()?.name ?? u));
  };
  return (
    <AuthFrame title="התחברות">
      <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <Field label="שם משתמש" dir="ltr" value={u} onChange={(e) => setU(e.target.value)} autoComplete="username" />
        <Field label="סיסמה" type="password" dir="ltr" value={pw} onChange={(e) => setPw(e.target.value)} autoComplete="current-password" />
        {err && <div role="alert" className="alert">{err}</div>}
        <button className="btn btn-primary" type="submit" disabled={busy} style={{ justifyContent: 'center' }}>התחבר</button>
      </form>
      <div style={{ display: 'flex', gap: 12, fontSize: 13, flexWrap: 'wrap' }}>
        <a onClick={() => ui.go('forgot')}>שכחתי סיסמה</a>
        <a onClick={() => ui.go('invite')}>יש לי קישור הזמנה</a>
      </div>
      <span className="dim" style={{ fontSize: 12 }}>
        אין הרשמה ציבורית. חשבונות נוצרים רק מהזמנה של מנהל.{remote.on ? '' : ' (מצב הדגמה: כל סיסמה באורך 4+ מתקבלת.)'}
      </span>
    </AuthFrame>
  );
}

/** First run on a new server: create the admin account. */
export function Setup() {
  const [name, setName] = useState('');
  const [u, setU] = useState('');
  const [pw, setPw] = useState('');
  const [err, setErr] = useState('');
  return (
    <AuthFrame title="הגדרה ראשונה">
      <p className="muted" style={{ margin: 0, fontSize: 13.5 }}>זה השרת שלך. צור את חשבון המנהל. משתמשים נוספים יצטרפו רק דרך קישור הזמנה שתיצור.</p>
      <form
        onSubmit={async (e) => {
          e.preventDefault();
          const r = await RW.setupAdmin(name, u.trim(), pw);
          if (r !== true) return setErr(r);
          ui.go('simple');
          ui.toast('חשבון המנהל נוצר');
        }}
        style={{ display: 'flex', flexDirection: 'column', gap: 12 }}
      >
        <Field label="שם" value={name} onChange={(e) => setName(e.target.value)} />
        <Field label="שם משתמש" dir="ltr" value={u} onChange={(e) => setU(e.target.value)} autoComplete="username" />
        <Field label="סיסמה (8 תווים לפחות)" type="password" dir="ltr" value={pw} onChange={(e) => setPw(e.target.value)} autoComplete="new-password" />
        {err && <div role="alert" className="alert">{err}</div>}
        <button className="btn btn-primary" type="submit" style={{ justifyContent: 'center' }}>צור חשבון מנהל</button>
      </form>
    </AuthFrame>
  );
}

const hashToken = () => decodeURIComponent(location.hash.split('?t=')[1] || '');

export function AcceptInvite() {
  const [token, setToken] = useState(hashToken);
  const [user, setUser] = useState('');
  const [pw, setPw] = useState('');
  const [err, setErr] = useState('');
  return (
    <AuthFrame title="קבלת הזמנה">
      <form
        onSubmit={async (e) => {
          e.preventDefault();
          if (pw.length < 8) return setErr('סיסמה: לפחות 8 תווים.');
          const r = await RW.acceptInvite(token, user.trim(), pw);
          if (typeof r === 'string') return setErr(r);
          ui.go('simple');
          ui.toast('ברוך הבא, ' + r.name + ' · ' + ROLE_L[r.role]);
        }}
        style={{ display: 'flex', flexDirection: 'column', gap: 12 }}
      >
        <Field label="קישור או קוד הזמנה" dir="ltr" value={token} onChange={(e) => setToken(e.target.value)} placeholder="https://…/join/xxxx-xxxx" />
        <Field label="שם משתמש" dir="ltr" value={user} onChange={(e) => setUser(e.target.value)} autoComplete="username" />
        <Field label="סיסמה" type="password" dir="ltr" value={pw} onChange={(e) => setPw(e.target.value)} autoComplete="new-password" />
        {err && <div role="alert" className="alert">{err}</div>}
        <button className="btn btn-primary" type="submit" style={{ justifyContent: 'center' }}>צור חשבון</button>
      </form>
      <a onClick={() => ui.go('login')} style={{ fontSize: 13 }}>חזרה להתחברות</a>
    </AuthFrame>
  );
}

export function Forgot() {
  const { remote } = useWS();
  const [token, setToken] = useState(hashToken);
  const [pw, setPw] = useState('');
  const [err, setErr] = useState('');
  return (
    <AuthFrame title="איפוס סיסמה">
      <p className="muted" style={{ margin: 0, fontSize: 14, textWrap: 'pretty' }}>
        אין שליחת מיילים. פנה למנהל הסטודיו: הוא ייצור עבורך קישור איפוס חד־פעמי (24 שעות) וישלח אותו בעצמו.
      </p>
      {remote.on && (
        <form
          onSubmit={async (e) => {
            e.preventDefault();
            const r = await RW.resetPassword(token, pw);
            if (r !== true) return setErr(r);
            ui.go('simple');
            ui.toast('הסיסמה עודכנה');
          }}
          style={{ display: 'flex', flexDirection: 'column', gap: 12 }}
        >
          <Field label="קישור האיפוס" dir="ltr" value={token} onChange={(e) => setToken(e.target.value)} />
          <Field label="סיסמה חדשה (8 תווים לפחות)" type="password" dir="ltr" value={pw} onChange={(e) => setPw(e.target.value)} autoComplete="new-password" />
          {err && <div role="alert" className="alert">{err}</div>}
          <button className="btn btn-primary" type="submit" style={{ justifyContent: 'center' }}>קבע סיסמה</button>
        </form>
      )}
      <a onClick={() => ui.go('login')} style={{ fontSize: 13 }}>חזרה להתחברות</a>
    </AuthFrame>
  );
}
