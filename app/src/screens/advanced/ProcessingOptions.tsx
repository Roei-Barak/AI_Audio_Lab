import { useState } from 'react';
import { Switch } from '../../components/controls';
import { fmtShort } from '../../lib/format';
import { store, useStore } from '../../store/store';
import type { Lang, Profile } from '../../store/types';
import { ui } from '../../store/ui';
import { isServer, submitJob } from '../../store/server';

export function ProcessingOptions() {
  const { project: p, job, sim } = useStore();
  const [profile, setProfile] = useState<Profile>(p?.processing.profile ?? 'איכותי');
  const [stems, setStems] = useState<2 | 4>(p?.processing.stems ?? 2);
  const [lang, setLang] = useState<Lang>(p?.processing.language ?? 'auto');
  const [steps, setSteps] = useState({ tr: true, al: true, sub: true });
  const running = job?.status === 'running';
  const mins = (profile === 'מהיר' ? 3 : 8) + (stems === 4 ? 1 : 0);
  const sel = (on: boolean) => ({ boxShadow: on ? 'inset 0 0 0 1px var(--color-accent)' : 'none' });
  const ring = (on: boolean) => <span style={{ width: 16, height: 16, flex: 'none', borderRadius: '50%', boxShadow: `inset 0 0 0 ${on ? 5 : 1.5}px var(--color-accent)` }} />;

  if (!p)
    return (
      <Empty>
        <span style={{ flex: 1 }}>אין עדיין פרויקט. ייבא שיר במצב הפשוט כדי להתחיל.</span>
        <button className="btn btn-primary" onClick={() => { ui.setStep('input'); ui.go('simple'); }}>לייבוא</button>
      </Empty>
    );

  const start = async () => {
    if (isServer() && p.serverRev != null) {
      try {
        store.edit('הגדרות עיבוד נבחרו', (P) => void (P.processing = { profile, stems, language: lang }));
        await submitJob('process', { profile, stems, language: lang, rerun: !!p.processed });
        ui.go('queue');
        ui.toast(p.processed ? 'עיבוד מחדש בשרת · התוצאה תוצג כהצעה' : 'נוסף לתור בשרת · משימת GPU אחת בכל רגע');
      } catch (e) {
        ui.toast((e as Error).message);
      }
      return;
    }
    store.edit('הגדרות עיבוד נבחרו', (P) => {
      P.processing = { profile, stems, language: lang };
      P.stems = stems === 4 ? ['vocals', 'drums', 'bass', 'other'] : ['vocals', 'instrumental'];
    });
    store.startJob({ name: p.source.name, profile, stems });
    ui.go('queue');
    ui.toast('נוסף לתור · משימת GPU אחת בכל רגע');
  };

  return (
    <div dir="rtl" style={{ height: '100%', overflow: 'auto', background: 'var(--color-bg)' }}>
      <div className="page-pad" style={{ maxWidth: 1180, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 18 }}>
        <h2 style={{ fontSize: 28, margin: 0 }}>בחירת עיבוד</h2>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 28, alignItems: 'flex-start' }}>
          <section style={{ flex: '1 1 520px', minWidth: 0, display: 'flex', flexDirection: 'column', gap: 18 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 14px', borderRadius: 'var(--radius-md)', background: 'var(--color-surface)', flexWrap: 'wrap' }}>
              <i className={p.source.kind === 'youtube' ? 'ph ph-youtube-logo' : 'ph ph-file-audio'} style={{ fontSize: 24, color: 'var(--color-accent-400)' }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div dir="auto" style={{ fontSize: 14 }}>{p.source.name}</div>
                <div className="dim" style={{ fontSize: 12 }}>
                  {(p.source.kind === 'youtube' ? 'YouTube' : 'קובץ מקומי') + ' · ' + fmtShort(p.source.duration) + (p.source.hasVideo ? ' · וידאו + אודיו' : ' · אודיו')}
                </div>
              </div>
              <span className="tag tag-neutral">בטווח המותר: עד 20GB ועד 3 שעות</span>
            </div>
            <Field label="פרופיל עיבוד">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10 }}>
                {([['מהיר', 'כ־3 דק׳ לשיר', 'מודלים קלים. מתאים לבדיקה מהירה ולשירים ארוכים.'], ['איכותי', 'כ־8 דק׳ לשיר', 'מודלים מלאים. הפרדה ותמלול מדויקים יותר, זמן ארוך יותר.']] as const).map(([l, t, d]) => (
                  <button key={l} aria-pressed={profile === l} onClick={() => setProfile(l)} className="panel" style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', gap: 6, padding: 14, border: 0, font: 'inherit', color: 'inherit', textAlign: 'start', ...sel(profile === l) }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%' }}>{ring(profile === l)}<span style={{ fontSize: 15 }}>{l}</span><span className="tag tag-neutral" style={{ marginInlineStart: 'auto' }}>{t}</span></div>
                    <div className="muted" style={{ fontSize: 12.5 }}>{d}</div>
                  </button>
                ))}
              </div>
            </Field>
            <Field label="הפרדת אודיו">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10 }}>
                {([[2, '2 ערוצים', ['שירה', 'ליווי']], [4, '4 ערוצים', ['שירה', 'תופים', 'בס', 'אחר']]] as const).map(([n, l, ch]) => (
                  <button key={n} aria-pressed={stems === n} onClick={() => setStems(n)} className="panel" style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', gap: 6, padding: 14, border: 0, font: 'inherit', color: 'inherit', textAlign: 'start', ...sel(stems === n) }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>{ring(stems === n)}<span style={{ fontSize: 15 }}>{l}</span></div>
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>{ch.map((c) => <span key={c} className="tag tag-neutral" style={{ padding: '1px 7px' }}>{c}</span>)}</div>
                  </button>
                ))}
              </div>
            </Field>
            <Field label="שלבים">
              {([['tr', 'תמלול', 'מתוך ערוץ השירה', 'ph ph-text-aa'], ['al', 'יישור ברמת מילה', 'זמן התחלה וסיום לכל מילה', 'ph ph-timer'], ['sub', 'טיוטת כתוביות', 'חלוקה לשורות לפי משפטים ונשימות', 'ph ph-closed-captioning']] as const).map(([k, t, d, i]) => (
                <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 12px', borderRadius: 'var(--radius-md)' }}>
                  <i className={i} style={{ fontSize: 19, color: 'var(--color-accent-400)' }} />
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 14 }}>{t}</div>
                    <div className="dim" style={{ fontSize: 12 }}>{d}</div>
                  </div>
                  <Switch on={steps[k]} label={t} onChange={(v) => setSteps({ ...steps, [k]: v })} />
                </div>
              ))}
            </Field>
            <Field label="שפת השירה">
              <select className="input" value={lang} onChange={(e) => setLang(e.target.value as Lang)} style={{ width: 200 }}>
                <option value="auto">זיהוי אוטומטי</option>
                <option value="he">עברית</option>
                <option value="en">English</option>
              </select>
            </Field>
          </section>
          <aside className="panel" style={{ flex: '1 1 300px', maxWidth: 360, display: 'flex', flexDirection: 'column', gap: 12, padding: 18 }}>
            <h4 style={{ margin: 0 }}>סיכום</h4>
            {[
              ['מעבד', (sim.conn === 'remote' ? 'שרת מרוחק' : 'המחשב הזה') + ' · GPU 16GB'],
              ['פרופיל', profile],
              ['ערוצים', String(stems)],
              ['זמן משוער', 'כ־' + mins + ' דק׳'],
              ['מקום בתור', running ? '#2' : '#1'],
            ].map(([k, v]) => (
              <div key={k} style={{ display: 'flex', fontSize: 13 }}>
                <span className="muted">{k}</span>
                <span style={{ marginInlineStart: 'auto' }}>{v}</span>
              </div>
            ))}
            <div style={{ display: 'flex', gap: 8, padding: 10, borderRadius: 'var(--radius-md)', background: 'var(--color-bg)', fontSize: 12.5, color: 'var(--color-neutral-300)' }}>
              <i className="ph ph-cpu" style={{ color: 'var(--color-accent-300)' }} />
              ה־GPU מריץ משימה כבדה אחת בכל רגע. העבודה תמתין אם משימה אחרת רצה.
            </div>
            <div className="dim" style={{ fontSize: 12 }}>תיקונים ידניים לא יידרסו. פלט AI חדש מוצג כהצעה להשוואה.</div>
            <button className="btn btn-primary btn-block" onClick={start} disabled={running} style={{ height: 42 }}>
              <i className="ph ph-queue" />
              {running ? 'עבודה כבר רצה בתור' : 'הוסף לתור העיבוד'}
            </button>
          </aside>
        </div>
      </div>
    </div>
  );
}

const Field = ({ label, children }: { label: string; children: React.ReactNode }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
    <span className="muted" style={{ fontSize: 12 }}>{label}</span>
    {children}
  </div>
);

export const Empty = ({ children }: { children: React.ReactNode }) => (
  <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16, background: 'var(--color-bg)' }}>
    <div className="panel" style={{ padding: 18, display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', maxWidth: 560 }}>
      {children}
    </div>
  </div>
);
