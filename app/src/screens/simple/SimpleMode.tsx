import { useEffect, useRef, useState } from 'react';
import { KaraokePreview } from '../../components/KaraokePreview';
import { ColorSwatches, Seg, SizeSlider, WordChips } from '../../components/controls';
import { fmt, fmtShort, lineSpan, lineText, r2, type ExportKind } from '../../lib/format';
import { previewStyle, setStyle } from '../../lib/styleEdit';
import { makeWords, DEMO_YT_SOURCE } from '../../store/demo';
import { playback, usePlayback } from '../../store/playback';
import { STAGES, jobPct, store, useStore } from '../../store/store';
import type { Lang, Profile, Res, Source } from '../../store/types';
import { ui, useUI, type SimpleStep } from '../../store/ui';
import { ws } from '../../store/workspace';
import { exportFile, exportMp4 } from '../../lib/actions';

const YT_RE = /^(https?:\/\/)?(www\.|m\.)?(youtube\.com\/(watch\?v=|shorts\/)|youtu\.be\/)[\w-]{4,}/i;

export function checkUrl(u: string): { ok: boolean; err: string; upload: boolean } {
  u = (u || '').trim();
  if (!u) return { ok: false, err: '', upload: false };
  if (!YT_RE.test(u)) return { ok: false, err: 'זה לא נראה כמו קישור YouTube תקין.', upload: false };
  if (/private|drm|blocked|members/i.test(u))
    return {
      ok: false,
      err: 'לא ניתן לייבא את המקור: הסרטון פרטי, מוגבל או מוגן. Lyra לא עוקפת הגנות. אפשר להעלות קובץ מקומי שמותר לך להשתמש בו.',
      upload: true,
    };
  return { ok: true, err: '', upload: false };
}

function useCompact() {
  const [c, setC] = useState(() => window.innerWidth < 720);
  useEffect(() => {
    const f = () => setC(window.innerWidth < 720);
    window.addEventListener('resize', f);
    return () => window.removeEventListener('resize', f);
  }, []);
  return c;
}

export function SimpleMode() {
  const { simpleStep: step } = useUI();
  const { project: p, job } = useStore();
  const compact = useCompact();

  // Leaving the screen doesn't stop the job; when it finishes, open the review step.
  const lastStatus = useRef(job?.status);
  useEffect(() => {
    if (step === 'progress' && job?.status === 'done' && lastStatus.current === 'running') {
      const t = setTimeout(() => ui.setStep('preview'), 900);
      lastStatus.current = job.status;
      return () => clearTimeout(t);
    }
    lastStatus.current = job?.status;
  }, [job?.status, step]);

  const done = !!p && (!job || job.status === 'done');
  const can: Record<SimpleStep, boolean> = { input: true, progress: !!job, preview: done, result: done };
  const order: SimpleStep[] = ['input', 'progress', 'preview', 'result'];
  const crumbs: [SimpleStep, string, string][] = [
    ['input', 'מקור', 'ph ph-link'],
    ['progress', 'עיבוד', 'ph ph-cpu'],
    ['preview', 'בדיקה', 'ph ph-play-circle'],
    ['result', 'הורדה', 'ph ph-download-simple'],
  ];

  return (
    <div
      dir="rtl"
      data-screen-label="Simple Mode"
      style={{
        width: '100%',
        height: '100%',
        overflow: 'auto',
        background: 'radial-gradient(90% 60% at 80% 0%, color-mix(in srgb, var(--color-accent-900) 55%, transparent), transparent 70%), var(--color-bg)',
      }}
    >
      <div style={{ maxWidth: 1240, margin: '0 auto', padding: compact ? '16px 16px 28px' : '28px 40px 40px', display: 'flex', flexDirection: 'column', gap: 22, boxSizing: 'border-box', minHeight: '100%' }}>
        <nav aria-label="שלבים" style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          {crumbs.map(([k, l, i], idx) => {
            const cur = step === k;
            const past = order.indexOf(step) > idx;
            return (
              <button
                key={k}
                className={'pill sm step ' + (past ? 'done' : can[k] ? '' : 'todo')}
                aria-current={cur ? 'step' : undefined}
                disabled={!can[k]}
                onClick={() => ui.setStep(k)}
                style={{ cursor: can[k] ? 'pointer' : 'default' }}
              >
                <i className={past ? 'ph-bold ph-check' : i} />
                {idx + 1} · {l}
              </button>
            );
          })}
        </nav>
        {step === 'input' && <InputStep compact={compact} />}
        {step === 'progress' && <ProgressStep compact={compact} />}
        {step === 'preview' && p && <PreviewStep compact={compact} />}
        {step === 'result' && p && <ResultStep compact={compact} />}
      </div>
    </div>
  );
}

/* ---------------- 1. Input ---------------- */

interface FileInfo {
  name: string;
  size: number;
  type: string;
  duration: number | null;
  unplayable?: boolean;
}

function InputStep({ compact }: { compact: boolean }) {
  const { project: p, job, save, sim } = useStore();
  const [src, setSrc] = useState<'yt' | 'file'>('yt');
  const [url, setUrlRaw] = useState('');
  const [err, setErr] = useState({ text: '', upload: false });
  const [file, setFile] = useState<FileInfo | null>(null);
  const [more, setMore] = useState(false);
  const [cfg, setCfg] = useState<{ stems: 2 | 4; profile: Profile; lang: Lang; res: Res }>(() => { const d = ws.get().settings; return { stems: d.defaultStems, profile: d.defaultProfile, lang: d.language, res: '1080p' }; });

  const setUrl = (u: string) => {
    const c = checkUrl(u);
    setUrlRaw(u);
    setErr({ text: c.err, upload: c.upload });
  };

  const onFile = (f: File | undefined) => {
    if (!f) return;
    if (f.size > 20 * 1024 ** 3) return setErr({ text: 'הקובץ גדול מ־20GB.', upload: false });
    store.setMedia(f);
    const info: FileInfo = { name: f.name, size: f.size, type: f.type, duration: null };
    setFile(info);
    setErr({ text: '', upload: false });
    const m = document.createElement(/^video/.test(f.type) ? 'video' : 'audio');
    m.preload = 'metadata';
    m.src = store.get().media!.url;
    m.onloadedmetadata = () => {
      if (m.duration > 3 * 3600) return setErr({ text: 'הקובץ ארוך משלוש שעות.', upload: false });
      setFile({ ...info, duration: m.duration });
    };
    m.onerror = () => setFile({ ...info, unplayable: true });
  };

  const create = () => {
    let source: Source;
    if (src === 'yt') {
      const c = checkUrl(url);
      if (!c.ok) return setErr({ text: c.err || 'הדבק קישור YouTube כדי להתחיל.', upload: c.upload });
      source = { ...DEMO_YT_SOURCE, url: url.trim() };
    } else {
      if (!file) return setErr({ text: 'בחר קובץ וידאו או אודיו.', upload: false });
      source = { kind: 'file', name: file.name, size: file.size, duration: file.duration || 66, hasVideo: /^video/.test(file.type) };
    }
    store.newProject(source, { profile: cfg.profile, stems: cfg.stems, language: cfg.lang, res: cfg.res });
    store.startJob({ name: source.name, profile: cfg.profile, stems: cfg.stems });
    ui.setStep('progress');
  };

  const hasSrc = (src === 'yt' && checkUrl(url).ok) || (src === 'file' && !!file);
  const defaults: [string, string][] = [
    ['הפרדה', cfg.stems === 2 ? '2 ערוצים · שירה / ליווי' : '4 ערוצים · שירה, תופים, בס, אחר'],
    ['פרופיל', cfg.profile],
    ['שפת תמלול', { auto: 'זיהוי אוטומטי', he: 'עברית', en: 'English' }[cfg.lang]],
    ['סנכרון', 'אוטומטי, ברמת מילה כשאפשר'],
    ['רקע', 'הווידאו המקורי'],
    ['אודיו מקורי', 'מושתק'],
    ['פסקול', 'Instrumental'],
    ['כתוביות', 'קריוקי · פעילות'],
    ['ייצוא', 'MP4 ' + cfg.res],
  ];

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 28, alignItems: 'flex-start' }}>
      <div style={{ flex: '1 1 520px', minWidth: 0, display: 'flex', flexDirection: 'column', gap: 18 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <h1 style={{ margin: 0, fontSize: compact ? 30 : 44, fontWeight: 500, lineHeight: 1.1 }}>צור קריוקי משיר</h1>
          <p style={{ margin: 0, fontSize: 15, color: 'var(--color-neutral-300)', maxWidth: 560, textWrap: 'pretty' }}>
            הדבק קישור YouTube או העלה קובץ. נפריד שירה ממוזיקה, נתמלל, נסנכרן ונכין סרטון עם המילים על גבי הווידאו המקורי.
          </p>
        </div>
        {p && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', padding: '12px 14px', borderRadius: 'var(--radius-md)', background: 'var(--color-surface)', boxShadow: 'inset 0 0 0 1px var(--color-accent-800)' }}>
            <i className="ph ph-clock-counter-clockwise" style={{ fontSize: 20, color: 'var(--color-accent-300)' }} />
            <div style={{ flex: 1, minWidth: 200, fontSize: 13.5 }}>
              יש פרויקט שמור בדפדפן: <b style={{ fontWeight: 500 }}>{p.title}</b> · {p.lines.length} שורות · שמירה אחרונה {save.at || '—'}
            </div>
            <button className="btn btn-secondary" onClick={() => ui.setStep(job && job.status !== 'done' ? 'progress' : 'preview')}>
              המשך לפרויקט
            </button>
          </div>
        )}
        <Seg
          value={src}
          size={14}
          label="סוג מקור"
          onChange={(v) => {
            setSrc(v);
            setErr({ text: '', upload: false });
          }}
          options={[
            ['yt', <><i className="ph ph-youtube-logo" />קישור YouTube</>],
            ['file', <><i className="ph ph-file-arrow-up" />קובץ מקומי</>],
          ]}
        />
        {src === 'yt' ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <input
                className="input"
                dir="ltr"
                aria-label="קישור YouTube"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && create()}
                placeholder="https://www.youtube.com/watch?v=…"
                style={{ flex: '1 1 300px', minWidth: 0, height: 46, fontSize: 15 }}
              />
              <button
                className="btn btn-secondary"
                style={{ height: 46 }}
                onClick={async () => {
                  try {
                    setUrl(await navigator.clipboard.readText());
                  } catch {
                    setErr({ text: 'הדפדפן חסם גישה ללוח. הדבק עם Ctrl+V.', upload: false });
                  }
                }}
              >
                <i className="ph ph-clipboard-text" />
                הדבק
              </button>
            </div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 12.5 }} className="dim">
              <span>לבדיקה:</span>
              <a onClick={() => setUrl('https://www.youtube.com/watch?v=lyraDemo01')}>קישור לדוגמה</a>
              <a onClick={() => setUrl('https://www.youtube.com/watch?v=private-drm-001')}>מקור מוגן / פרטי</a>
            </div>
          </div>
        ) : (
          <label style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8, padding: 28, borderRadius: 'var(--radius-lg)', cursor: 'pointer', background: 'var(--color-surface)', boxShadow: 'inset 0 0 0 1px var(--color-divider)', textAlign: 'center' }}>
            <i className="ph ph-upload-simple" style={{ fontSize: 28, color: 'var(--color-accent-300)' }} />
            <span style={{ fontSize: 15 }}>בחר קובץ וידאו או אודיו</span>
            <span className="dim" style={{ fontSize: 12.5 }}>MP4 · MOV · MKV · MP3 · WAV · FLAC · עד 20GB · עד 3 שעות</span>
            <input type="file" accept="video/*,audio/*" onChange={(e) => onFile(e.target.files?.[0])} style={{ display: 'none' }} />
          </label>
        )}
        {err.text && (
          <div role="alert" className="alert">
            <i className="ph ph-warning-circle" style={{ fontSize: 18, color: 'var(--color-accent-200)' }} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <span>{err.text}</span>
              {err.upload && <a onClick={() => { setSrc('file'); setErr({ text: '', upload: false }); }}>העלה קובץ מקומי במקום</a>}
            </div>
          </div>
        )}
        {hasSrc && (
          <div style={{ display: 'flex', gap: 14, alignItems: 'center', padding: 12, borderRadius: 'var(--radius-md)', background: 'var(--color-surface)', boxShadow: 'var(--shadow-sm)' }}>
            <div style={{ width: compact ? 88 : 132, height: compact ? 50 : 74, flex: 'none', borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'radial-gradient(100% 90% at 30% 20%, var(--color-accent-800), var(--color-neutral-900))' }}>
              <i className={src === 'yt' ? 'ph ph-youtube-logo' : file && /^video/.test(file.type) ? 'ph ph-film-strip' : 'ph ph-waveform'} style={{ fontSize: 24, color: 'var(--color-accent-200)' }} />
            </div>
            <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span dir="auto" style={{ fontSize: 15, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {src === 'yt' ? DEMO_YT_SOURCE.name : file!.name}
              </span>
              <span className="muted" style={{ fontSize: 12.5 }}>
                {src === 'yt'
                  ? 'נועה ברק · 1:06 · וידאו + אודיו'
                  : (file!.size / 1048576).toFixed(1) + 'MB · ' + (file!.duration ? fmtShort(file!.duration) : file!.unplayable ? 'הדפדפן לא יכול לנגן את הקובץ' : 'קורא משך…')}
              </span>
              <span className="dim" style={{ fontSize: 11.5 }}>
                {src === 'yt'
                  ? 'פרטי המקור מדומים. באב־טיפוס אין גישה ל־YouTube.'
                  : file!.unplayable
                    ? 'הקובץ יעובד בשרת. בתצוגה המקדימה ינוגן בסימולציה.'
                    : 'הקובץ נשאר בדפדפן ומנוגן מקומית בתצוגה המקדימה. הוא לא נשמר.'}
              </span>
            </div>
            <button
              className="btn btn-ghost btn-icon"
              aria-label="נקה"
              onClick={() => {
                if (src === 'yt') setUrl('');
                else {
                  store.setMedia(null);
                  setFile(null);
                }
              }}
            >
              <i className="ph ph-x" />
            </button>
          </div>
        )}
        <button className="btn btn-primary" onClick={create} style={{ height: 56, fontSize: 17, justifyContent: 'center' }}>
          <i className="ph ph-microphone-stage" style={{ fontSize: 20 }} />
          צור קריוקי
        </button>
        <span className="dim" style={{ fontSize: 12 }}>
          משתמשים רק במקורות שמותר לך להשתמש בהם. Lyra לא עוקפת DRM או הגבלות גישה.
        </span>
      </div>
      <aside className="panel" style={{ flex: '1 1 340px', minWidth: 0, maxWidth: 440, display: 'flex', flexDirection: 'column', gap: 12, padding: 18 }}>
        <div className="kicker">מה ייווצר</div>
        <div style={{ fontSize: 15, lineHeight: 1.5 }}>
          וידאו מקורי (מושתק) + Instrumental + מילים מסונכרנות = <b style={{ fontWeight: 500 }}>סרטון קריוקי</b>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {defaults.map(([k, v]) => (
            <div key={k} style={{ display: 'flex', gap: 10, fontSize: 13 }}>
              <span className="muted" style={{ flex: 'none', width: 130 }}>{k}</span>
              <span>{v}</span>
            </div>
          ))}
        </div>
        <button
          onClick={() => setMore(!more)}
          aria-expanded={more}
          style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 0', border: 0, background: 'none', font: 'inherit', fontSize: 13.5, color: 'var(--color-accent-300)', cursor: 'pointer', alignSelf: 'flex-start' }}
        >
          <i className={more ? 'ph ph-caret-up' : 'ph ph-caret-down'} />
          אפשרויות נוספות
        </button>
        {more && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <Opt k="הפרדה"><Seg value={cfg.stems} onChange={(stems) => setCfg({ ...cfg, stems })} options={[[2, '2 ערוצים'], [4, '4 ערוצים']]} /></Opt>
            <Opt k="פרופיל עיבוד"><Seg value={cfg.profile} onChange={(profile) => setCfg({ ...cfg, profile })} options={[['מהיר', 'מהיר'], ['איכותי', 'איכותי']]} /></Opt>
            <Opt k="שפת תמלול"><Seg value={cfg.lang} onChange={(lang) => setCfg({ ...cfg, lang })} options={[['auto', 'אוטומטי'], ['he', 'עברית'], ['en', 'English']]} /></Opt>
            <Opt k="רזולוציית ייצוא"><Seg value={cfg.res} onChange={(res) => setCfg({ ...cfg, res })} options={[['720p', '720p'], ['1080p', '1080p']]} /></Opt>
            <span className="dim" style={{ fontSize: 11.5 }}>
              2 ערוצים: שירה + ליווי. 4 ערוצים: שירה, תופים, בס, אחר (הקריוקי משתמש בכל מה שאינו שירה).
            </span>
          </div>
        )}
        <span className="dim" style={{ fontSize: 11.5 }}>
          יעובד ב{sim.conn === 'remote' ? 'שרת המרוחק' : 'מחשב הזה'} · משימת GPU אחת בכל רגע
        </span>
      </aside>
    </div>
  );
}

const Opt = ({ k, children }: { k: string; children: React.ReactNode }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
    <span className="muted" style={{ fontSize: 12.5 }}>{k}</span>
    {children}
  </div>
);

/* ---------------- 2. Progress ---------------- */

const STAGE_DETAILS = (stems: number, lang: Lang | undefined) => [
  'הורדת וידאו ואודיו מהמקור המורשה, בדיקת משך ופורמט, חילוץ ערוץ אודיו בלי לשנות היסט זמן.',
  'הפרדה ל־' + stems + ' ערוצים על ה־GPU. אורך הערוצים זהה לאורך המקור עד לדגימה.',
  'תמלול ערוץ השירה. ' + (lang && lang !== 'auto' ? 'שפה: ' + (lang === 'he' ? 'עברית' : 'English') : 'זיהוי שפה אוטומטי (עברית + English).'),
  'יישור כפוי של המילים לאודיו. זמני מילים נשמרים רק אם הופקו בפועל.',
  'השתקת האודיו המקורי, הוספת Instrumental, הטמעת כתוביות ורינדור MP4.',
];

function ProgressStep({ compact }: { compact: boolean }) {
  const { job, project: p, sim } = useStore();
  const [open, setOpen] = useState<Record<number, boolean>>({});
  if (!job)
    return (
      <div className="panel" style={{ padding: 18, display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <span style={{ flex: 1 }}>אין עבודה פעילה.</span>
        <button className="btn btn-primary" onClick={() => ui.setStep('input')}>התחל שיר חדש</button>
      </div>
    );
  const stg = job.status === 'done' ? STAGES.length : job.stage;
  const dets = STAGE_DETAILS(job.stems, p?.processing.language);
  const paused = job.status === 'running' && sim.offline;
  const title = { running: paused ? 'החיבור לשרת אבד' : 'מכין את הקריוקי…', failed: 'העיבוד נעצר', cancelled: 'העבודה בוטלה', done: 'הקריוקי מוכן' }[job.status];
  const restart = () => {
    store.removeJob();
    store.setMedia(null);
    ui.setStep('input');
  };

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 28, alignItems: 'flex-start' }}>
      <div style={{ flex: '1 1 560px', minWidth: 0, display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: 12, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <h1 style={{ margin: 0, fontSize: compact ? 24 : 32, fontWeight: 500 }}>{title}</h1>
            <span dir="auto" className="muted" style={{ fontSize: 14 }}>
              {job.name} · {job.stems} ערוצים · {job.profile}
            </span>
          </div>
          <span className="tag tag-outline" style={{ marginInlineStart: 'auto' }}>סימולציה · אין שרת עיבוד מחובר</span>
        </div>
        <div role="progressbar" aria-valuenow={jobPct(job)} aria-valuemin={0} aria-valuemax={100} style={{ height: 6, borderRadius: 999, background: 'var(--color-neutral-800)', overflow: 'hidden' }}>
          <div style={{ height: '100%', width: jobPct(job) + '%', background: 'var(--color-accent)', transition: 'width 200ms' }} />
        </div>
        <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 2, padding: 6 }}>
          {STAGES.map((l, i) => {
            const done = i < stg;
            const run = i === stg && job.status === 'running';
            const fail = i === stg && job.status === 'failed';
            const canc = i === stg && job.status === 'cancelled';
            return (
              <div key={l} style={{ display: 'flex', flexDirection: 'column', gap: 8, padding: 12, borderRadius: 'var(--radius-md)', background: run || fail ? 'var(--color-neutral-800)' : 'transparent' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <i
                    className={(done ? 'ph-fill ph-check-circle' : run ? 'ph ph-spinner-gap spin' : fail ? 'ph-fill ph-x-circle' : canc ? 'ph ph-prohibit' : 'ph ph-circle')}
                    style={{ fontSize: 20, color: done ? 'var(--color-accent-300)' : run || fail ? 'var(--color-accent-200)' : 'var(--color-neutral-600)' }}
                  />
                  <span style={{ flex: 1, fontSize: 15, color: done || run || fail ? 'var(--color-text)' : 'var(--color-neutral-500)' }}>{l}</span>
                  <span className="num muted" style={{ fontSize: 12.5 }}>
                    {done ? 'הושלם' : run ? Math.round(job.pct) + '%' : fail ? 'נכשל ב־' + Math.round(job.pct) + '%' : canc ? 'בוטל' : 'ממתין'}
                  </span>
                  <button className="btn btn-ghost" aria-expanded={!!open[i]} onClick={() => setOpen({ ...open, [i]: !open[i] })} style={{ height: 28, fontSize: 12 }}>
                    {open[i] ? 'הסתר' : 'פרטים'}
                  </button>
                </div>
                {(run || fail) && (
                  <div style={{ height: 4, borderRadius: 999, background: 'var(--color-neutral-800)', overflow: 'hidden', marginInlineStart: 32 }}>
                    <div style={{ height: '100%', width: job.pct + '%', background: fail ? 'var(--color-accent-700)' : 'var(--color-accent)' }} />
                  </div>
                )}
                {open[i] && (
                  <div className="muted" style={{ marginInlineStart: 32, fontSize: 12.5, textWrap: 'pretty' }}>
                    {dets[i]}
                  </div>
                )}
              </div>
            );
          })}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: 12 }}>
            <i className={job.status === 'done' ? 'ph-fill ph-check-circle' : 'ph ph-circle'} style={{ fontSize: 20, color: job.status === 'done' ? 'var(--color-accent-300)' : 'var(--color-neutral-600)' }} />
            <span style={{ flex: 1, fontSize: 15, color: job.status === 'done' ? 'var(--color-text)' : 'var(--color-neutral-500)' }}>מוכן לבדיקה או להורדה</span>
            {job.status === 'done' && <button className="btn btn-primary" onClick={() => ui.setStep('preview')}>לבדיקה<i className="ph ph-arrow-left" /></button>}
          </div>
        </div>
        {job.status === 'failed' && (
          <div role="alert" style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: 14, borderRadius: 'var(--radius-md)', background: 'color-mix(in srgb, var(--color-accent-800) 45%, var(--color-bg))' }}>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', fontSize: 14.5 }}>
              <i className="ph ph-lightning-slash" style={{ fontSize: 20, color: 'var(--color-accent-200)' }} />
              העיבוד קרס בשלב {STAGES[job.stage]}. השלבים הקודמים הושלמו ונשמרו.
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button className="btn btn-primary" onClick={() => store.retryJob(false)}><i className="ph ph-arrow-clockwise" />נסה שוב מהשלב</button>
              {job.profile === 'איכותי' && <button className="btn btn-secondary" onClick={() => store.retryJob(true)}>נסה בפרופיל מהיר</button>}
              <button className="btn btn-ghost" onClick={restart}>בטל והתחל שיר אחר</button>
            </div>
          </div>
        )}
        {job.status === 'cancelled' && (
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', padding: 14, borderRadius: 'var(--radius-md)', background: 'var(--color-surface)' }}>
            <i className="ph ph-prohibit" style={{ fontSize: 20, color: 'var(--color-neutral-400)' }} />
            <span style={{ flex: 1, fontSize: 14 }}>העבודה בוטלה. שלבים שהושלמו נשמרו.</span>
            <button className="btn btn-primary" onClick={() => store.retryJob(false)}>המשך מהשלב שנעצר</button>
            <button className="btn btn-ghost" onClick={restart}>התחל שיר אחר</button>
          </div>
        )}
        {job.status === 'running' && !paused && (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            <button className="btn btn-secondary" onClick={() => store.cancelJob()}><i className="ph ph-stop" />בטל</button>
            <a onClick={() => ui.go('queue')} style={{ fontSize: 13 }}>פתח בתור העיבוד</a>
            <span className="dim" style={{ fontSize: 12.5 }}>אפשר לעזוב את המסך. העבודה ממשיכה בתור כל עוד השרת פעיל.</span>
          </div>
        )}
        {paused && (
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', padding: '12px 14px', borderRadius: 'var(--radius-md)', background: 'var(--color-surface)', fontSize: 13.5 }}>
            <i className="ph ph-plugs" style={{ fontSize: 18, color: 'var(--color-accent-200)' }} />
            אין חיבור לשרת. מצב העבודה לא ידוע. הסטטוס יתעדכן כשהחיבור יחזור.
          </div>
        )}
      </div>
      <aside style={{ flex: '1 1 300px', minWidth: 0, maxWidth: 380, display: 'flex', flexDirection: 'column', gap: 10, padding: 16, borderRadius: 'var(--radius-lg)', background: 'var(--color-surface)' }}>
        <div className="kicker">משימת GPU אחת בכל רגע</div>
        <div style={{ fontSize: 13, color: 'var(--color-neutral-300)', textWrap: 'pretty' }}>
          ההפרדה והתמלול רצים על ה־GPU של {sim.conn === 'remote' ? 'השרת המרוחק' : 'המחשב הזה'} (16GB VRAM). משימות נוספות ימתינו בתור.
        </div>
        <div className="dim" style={{ fontSize: 12 }}>
          ניסיון {job.attempt} · פרופיל {job.profile} · {job.stems} ערוצים
        </div>
      </aside>
    </div>
  );
}

/* ---------------- 3. Preview & quick corrections ---------------- */

function PreviewStep({ compact }: { compact: boolean }) {
  const { project: p, canUndo, canRedo } = useStore();
  const { t } = usePlayback();
  const [sel, setSel] = useState<string | null>(null);
  const [selW, setSelW] = useState<string | null>(null);
  const [lineDraft, setLineDraft] = useState('');
  const [wordDraft, setWordDraft] = useState('');
  if (!p) return null;

  const lines = p.lines;
  const lowCount = lines.reduce((a, l) => a + l.words.filter((w) => w.conf === 'low' && !w.manual).length, 0);
  const selL = lines.find((l) => l.id === sel);
  const selWd = selL?.words.find((w) => w.id === selW);

  const editWord = (label: string, fn: (w: (typeof lines)[0]['words'][0], l: (typeof lines)[0]) => void | false) =>
    store.edit(label, (P) => {
      const l = P.lines.find((x) => x.id === sel);
      const w = l?.words.find((x) => x.id === selW);
      if (!l || !w) return false;
      if (fn(w, l) === false) return false;
      w.manual = true;
      l.manual = true;
    });
  const nudge = (k: 's' | 'e', d: number) =>
    editWord('שינוי תזמון מילה', (w, l) => {
      const i = l.words.indexOf(w);
      const prev = l.words[i - 1];
      const next = l.words[i + 1];
      if (k === 's') w.s = r2(Math.min(w.e - 0.05, Math.max(prev ? prev.e : 0, w.s + d)));
      else w.e = r2(Math.max(w.s + 0.05, Math.min(next ? next.s : 9999, w.e + d)));
    });
  const shiftLine = (d: number) =>
    store.edit('הזזת תזמון שורה', (P) => {
      const l = P.lines.find((x) => x.id === sel);
      if (!l) return false;
      l.start = r2(l.start + d);
      l.end = r2(l.end + d);
      l.words.forEach((w) => {
        w.s = r2(w.s + d);
        w.e = r2(w.e + d);
      });
      l.manual = true;
    });
  const commitLine = () => {
    const v = lineDraft.trim();
    if (!selL || !v || v === lineText(selL)) return;
    store.edit('תיקון שורה', (P) => {
      const l = P.lines.find((x) => x.id === sel)!;
      const [a, b] = lineSpan(l);
      const nw = v.split(/\s+/);
      // Same word count: keep existing timings; otherwise redistribute across the line span.
      l.words =
        nw.length === l.words.length
          ? l.words.map((w, j) => (w.w === nw[j] ? w : { ...w, w: nw[j], manual: true, conf: 'high' }))
          : makeWords(a, b, v).map((w) => ({ ...w, manual: true }));
      l.manual = true;
    });
    setSelW(null);
  };
  const commitWord = () => {
    const v = wordDraft.trim();
    if (!selWd || !v || v === selWd.w) return;
    if (/\s/.test(v)) return setWordDraft(selWd.w);
    editWord('תיקון מילה', (w) => {
      w.w = v;
      w.conf = 'high';
    });
    setLineDraft(lineText({ words: selL!.words.map((w) => (w.id === selW ? { ...w, w: v } : w)) }));
  };

  const jumpLow = () => {
    const l = lines.find((l) => l.words.some((w) => w.conf === 'low' && !w.manual));
    const w = l?.words.find((w) => w.conf === 'low' && !w.manual);
    if (!l || !w) return;
    setSel(l.id);
    setSelW(w.id);
    setLineDraft(lineText(l));
    setWordDraft(w.w);
    playback.seek(lineSpan(l)[0] - 0.3);
  };

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 20, alignItems: 'stretch', flex: 1, minHeight: 0 }}>
      <div style={{ flex: '1 1 560px', minWidth: 0, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <h1 style={{ margin: 0, fontSize: compact ? 24 : 32, fontWeight: 500 }}>בדיקה ותיקון מהיר</h1>
          {lowCount > 0 && (
            <button className="tag tag-accent" onClick={jumpLow} style={{ border: 0, cursor: 'pointer' }}>
              <i className="ph ph-warning" />
              {lowCount} מילים בביטחון מודל נמוך · מומלץ לבדוק
            </button>
          )}
        </div>
        <div style={{ height: compact ? 240 : 470, minHeight: 220 }}>
          <KaraokePreview />
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button className="btn btn-primary" onClick={() => ui.setStep('result')}><i className="ph ph-download-simple" />המשך להורדה</button>
          <button className="btn btn-secondary" onClick={() => openAdvanced()}><i className="ph ph-sliders-horizontal" />ערוך בסטודיו המקצועי</button>
          <button className="btn btn-ghost" onClick={() => store.undo()} disabled={!canUndo}><i className="ph ph-arrow-counter-clockwise" />בטל פעולה</button>
          <button className="btn btn-ghost" onClick={() => store.redo()} disabled={!canRedo}><i className="ph ph-arrow-clockwise" />בצע שוב</button>
        </div>
      </div>
      <div className="panel" style={{ flex: '1 1 380px', minWidth: 0, maxWidth: compact ? 'none' : 520, display: 'flex', flexDirection: 'column', gap: 14, padding: 16, maxHeight: compact ? 'none' : 720, overflow: 'auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 15 }}>מילים</span>
          <span className="dim" style={{ fontSize: 12 }}>לחץ על שורה כדי לקפוץ אליה ולערוך</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {lines.map((l) => {
            const sp = lineSpan(l);
            const isSel = l.id === sel;
            const act = t >= sp[0] - 0.2 && t <= sp[1] + 0.2;
            return (
              <div key={l.id} style={{ display: 'flex', flexDirection: 'column', gap: 8, padding: '8px 10px', borderRadius: 'var(--radius-md)', background: isSel ? 'var(--color-neutral-800)' : act ? 'color-mix(in srgb, var(--color-accent-900) 60%, transparent)' : 'transparent', boxShadow: isSel ? 'inset 0 0 0 1px var(--color-accent-800)' : 'none' }}>
                <button
                  onClick={() => {
                    setSel(isSel ? null : l.id);
                    setSelW(null);
                    setLineDraft(lineText(l));
                    if (!isSel) playback.seek(sp[0] - 0.3);
                  }}
                  aria-expanded={isSel}
                  style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', border: 0, background: 'none', font: 'inherit', color: 'inherit', padding: 0, textAlign: 'start' }}
                >
                  <span dir="ltr" className="num dim" style={{ flex: 'none', width: 44, fontSize: 11.5 }}>{fmtShort(sp[0])}</span>
                  <span dir="auto" style={{ flex: 1, minWidth: 0, fontSize: 14, color: act ? 'var(--color-accent-100)' : 'var(--color-text)' }}>{lineText(l)}</span>
                  {l.words.some((w) => w.conf === 'low' && !w.manual) && <i className="ph ph-warning" style={{ color: 'var(--color-accent-300)' }} title="ביטחון מודל נמוך" />}
                  {l.manual && <i className="ph ph-pencil-simple" style={{ color: 'var(--color-neutral-400)' }} title="תוקן ידנית" />}
                </button>
                {isSel && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: '4px 0 6px' }}>
                    <input
                      className="input"
                      dir="auto"
                      aria-label="טקסט השורה"
                      value={lineDraft}
                      onChange={(e) => setLineDraft(e.target.value)}
                      onBlur={commitLine}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
                        if (e.key === 'Escape') setLineDraft(lineText(l));
                      }}
                      style={{ fontSize: 14 }}
                    />
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                      <span className="muted" style={{ fontSize: 12 }}>תזמון שורה</span>
                      <button className="btn btn-ghost" onClick={() => shiftLine(-0.1)} style={{ height: 28, fontSize: 12 }}>−0.1 שנ׳</button>
                      <button className="btn btn-ghost" onClick={() => shiftLine(0.1)} style={{ height: 28, fontSize: 12 }}>+0.1 שנ׳</button>
                      <span dir="ltr" className="num" style={{ fontSize: 12, color: 'var(--color-neutral-300)' }}>{fmt(sp[0])} – {fmt(sp[1])}</span>
                    </div>
                    <WordChips
                      words={l.words}
                      selected={selW}
                      onPick={(w) => {
                        setSelW(w.id);
                        setWordDraft(w.w);
                        playback.seek(w.s - 0.15);
                      }}
                    />
                    {selWd && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', padding: 8, borderRadius: 'var(--radius-md)', background: 'var(--color-bg)' }}>
                        <input
                          className="input"
                          dir="auto"
                          aria-label="מילה"
                          value={wordDraft}
                          onChange={(e) => setWordDraft(e.target.value)}
                          onBlur={commitWord}
                          onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
                          style={{ width: 130, fontSize: 14 }}
                        />
                        <Nudge label="התחלה" v={selWd.s} minus={() => nudge('s', -0.05)} plus={() => nudge('s', 0.05)} />
                        <Nudge label="סיום" v={selWd.e} minus={() => nudge('e', -0.05)} plus={() => nudge('e', 0.05)} />
                        <button className="btn btn-ghost" onClick={() => playback.seek(selWd.s - 0.3, true)} style={{ height: 28, fontSize: 12 }}><i className="ph ph-play" />השמע</button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
        <QuickStyle />
      </div>
    </div>
  );
}

function Nudge({ label, v, minus, plus }: { label: string; v: number; minus: () => void; plus: () => void }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, whiteSpace: 'nowrap' }}>
      <span className="muted" style={{ fontSize: 12 }}>{label}</span>
      <button className="btn btn-ghost btn-icon" onClick={minus} aria-label={label + ' מוקדם'} style={{ height: 28, width: 28 }}>−</button>
      <span dir="ltr" className="num" style={{ fontSize: 12 }}>{v.toFixed(2)}</span>
      <button className="btn btn-ghost btn-icon" onClick={plus} aria-label={label + ' מאוחר'} style={{ height: 28, width: 28 }}>+</button>
    </span>
  );
}

const Row = ({ k, children }: { k: string; children: React.ReactNode }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
    <span className="muted" style={{ width: 70, fontSize: 12.5 }}>{k}</span>
    {children}
  </div>
);

function QuickStyle() {
  const { project: p } = useStore();
  if (!p) return null;
  const st = p.style;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, paddingTop: 6 }}>
      <span style={{ fontSize: 15 }}>עיצוב כתוביות</span>
      <Row k="תבנית"><Seg value={st.template} onChange={(v) => setStyle('template', v)} options={[['classic', 'קלאסי'], ['bold', 'מודגש'], ['minimal', 'מינימלי']]} /></Row>
      <Row k="מיקום"><Seg value={st.position} onChange={(v) => setStyle('position', v)} options={[['top', 'למעלה'], ['middle', 'אמצע'], ['bottom', 'למטה']]} /></Row>
      <Row k="רקע"><Seg value={st.background} onChange={(v) => setStyle('background', v)} options={[['video', 'וידאו מקורי'], ['blur', 'מעבר צבע'], ['solid', 'אחיד']]} /></Row>
      <Row k="יחס"><Seg value={st.aspect} onChange={(v) => setStyle('aspect', v)} options={[['16:9', '16:9'], ['9:16', '9:16'], ['1:1', '1:1']]} /></Row>
      <Row k="צבע"><ColorSwatches value={st.color} onChange={(c) => setStyle('color', c)} /></Row>
      <Row k="גודל"><SizeSlider value={st.size} onPreview={(v) => previewStyle('size', v)} onCommit={(v) => setStyle('size', v)} /></Row>
    </div>
  );
}

/** Opens the SAME project in the advanced studio: no duplicate, no re-processing. */
export function openAdvanced() {
  store.setMode('advanced');
  ui.go('subs-word');
  ui.toast('אותו פרויקט נפתח בסטודיו המקצועי · ללא עיבוד מחדש');
}

/* ---------------- 4. Result ---------------- */

function ResultStep({ compact }: { compact: boolean }) {
  const { project: p } = useStore();
  const [msg, setMsg] = useState('');
  if (!p) return null;
  const dl = (k: ExportKind) => {
    exportFile(k);
    setMsg('הורד ' + k.toUpperCase() + ' מהנתונים העדכניים של הפרויקט (' + p.lines.length + ' שורות).');
  };
  const downloads: [string, string, ExportKind | 'sim'][] = [
    ['ph ph-music-notes', 'הורד מוזיקה בלבד (Instrumental)', 'sim'],
    ['ph ph-microphone', 'הורד שירה בלבד (Vocals)', 'sim'],
    ['ph ph-subtitles', 'הורד כתוביות SRT', 'srt'],
    ['ph ph-text-align-right', 'הורד מילות שיר TXT', 'txt'],
    ['ph ph-brackets-curly', 'הורד פרויקט JSON', 'json'],
  ];
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 24, alignItems: 'flex-start' }}>
      <div style={{ flex: '1 1 520px', minWidth: 0, display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <span className="kicker">מוכן</span>
          <h1 style={{ margin: 0, fontSize: compact ? 24 : 32, fontWeight: 500 }}>{p.title}</h1>
          <span className="muted" style={{ fontSize: 13.5 }}>
            {(p.source.kind === 'youtube' ? 'YouTube' : 'קובץ מקומי') + ' · ' + p.lines.length + ' שורות · ' + p.stems.length + ' ערוצים · MP4 ' + p.export.res + ' · ' + (p.history.length - 1) + ' עריכות'}
          </span>
        </div>
        <div style={{ height: compact ? 220 : 420, minHeight: 200 }}>
          <KaraokePreview />
        </div>
      </div>
      <div style={{ flex: '1 1 360px', minWidth: 0, maxWidth: 460, display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div className="panel" style={{ display: 'flex', flexDirection: 'column', gap: 8, padding: 16 }}>
          <button
            className="btn btn-primary"
            onClick={() => { exportMp4(); setMsg('סימולציה: רינדור MP4 לא מתבצע באב־טיפוס. בגרסה המחוברת לשרת הקובץ יירד מכאן.'); }}
            style={{ height: 52, fontSize: 16, justifyContent: 'center' }}
          >
            <i className="ph ph-film-strip" />
            הורד סרטון קריוקי MP4
          </button>
          <span className="muted" style={{ fontSize: 12, textWrap: 'pretty' }}>רינדור MP4 דורש את שרת העיבוד. באב־טיפוס הזה הוא מדומה ולא תתבצע הורדה.</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2, padding: 8, borderRadius: 'var(--radius-lg)', background: 'var(--color-surface)' }}>
          {downloads.map(([i, l, k]) => (
            <button
              key={l}
              className="row-btn"
              onClick={() => (k === 'sim' ? setMsg('סימולציה: ערוצי אודיו נוצרים בשרת העיבוד. אין קובץ להורדה באב־טיפוס.') : dl(k))}
            >
              <i className={i} style={{ fontSize: 18, color: 'var(--color-accent-300)' }} />
              <span style={{ flex: 1, whiteSpace: 'normal' }}>{l}</span>
              <span className={k === 'sim' ? 'tag tag-neutral' : 'tag tag-outline'} style={{ fontSize: 11 }}>
                {k === 'sim' ? 'דורש שרת' : 'קובץ אמיתי'}
              </span>
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button className="btn btn-ghost" onClick={() => ui.setStep('preview')}><i className="ph ph-play-circle" />תצוגה מקדימה</button>
          <button className="btn btn-ghost" onClick={() => ui.setStep('preview')}><i className="ph ph-pencil-simple" />ערוך מילים</button>
          <button className="btn btn-ghost" onClick={openAdvanced}><i className="ph ph-sliders-horizontal" />פתח בסטודיו המקצועי</button>
          <button className="btn btn-ghost" onClick={() => setMsg('סימולציה: נוסף לתור ייצוא. SRT, TXT ו־JSON מופקים מחדש בכל הורדה מהנתונים העדכניים.')}><i className="ph ph-arrows-clockwise" />ייצא שוב</button>
          <button
            className="btn btn-ghost"
            onClick={() => {
              store.removeJob();
              store.setMedia(null);
              ui.setStep('input');
            }}
          >
            <i className="ph ph-plus" />
            צור קריוקי נוסף
          </button>
        </div>
        {msg && (
          <div role="status" style={{ padding: '10px 12px', borderRadius: 'var(--radius-md)', background: 'var(--color-neutral-800)', fontSize: 13 }}>
            {msg}
          </div>
        )}
      </div>
    </div>
  );
}
