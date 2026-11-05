import { useEffect, useRef } from 'react';
import { ModalHost } from './modals/Modals';
import { KaraokeStudio } from './screens/advanced/KaraokeStudio';
import { LyricsStudio } from './screens/advanced/LyricsStudio';
import { ProcessingOptions } from './screens/advanced/ProcessingOptions';
import { Queue } from './screens/advanced/Queue';
import { Dashboard, ExportScreen, Library, Outputs, Team } from './screens/advanced/Pages';
import { AudioStudio, ProjectDetails, VideoEditor } from './screens/advanced/ProjectScreens';
import { AcceptInvite, Forgot, Login, Settings } from './screens/advanced/SettingsAuth';
import { Sidebar } from './screens/advanced/Sidebar';
import { StateView } from './screens/advanced/SystemStates';
import { saveCheckpoint } from './lib/actions';
import { useWS, ws } from './store/workspace';
import { SimpleMode, openAdvanced } from './screens/simple/SimpleMode';
import { playback } from './store/playback';
import { jobPct, store, useStore } from './store/store';
import type { SaveInfo } from './store/types';
import { ui, useUI, type Route, type StateKind } from './store/ui';

const FLOW: [Route, string, string][] = [
  ['options', 'עיבוד', 'ph ph-sliders'],
  ['queue', 'תור', 'ph ph-queue'],
  ['subs-word', 'מילים', 'ph ph-text-aa'],
  ['karaoke', 'עיצוב', 'ph ph-paint-brush'],
  ['export', 'ייצוא', 'ph ph-export'],
];

function savePill(save: SaveInfo, remote: boolean): [string, string, string, string] {
  const where = remote ? 'בשרת' : 'בדפדפן';
  switch (save.state) {
    case 'saving':
      return ['ph ph-spinner-gap spin', 'שומר ' + where + '…', 'transparent', 'var(--color-neutral-300)'];
    case 'saved':
      return ['ph ph-check-circle', 'נשמר ' + where + ' · ' + (save.at || ''), 'transparent', 'var(--color-neutral-300)'];
    case 'local':
      return ['ph ph-hard-drives', 'נשמר בדפדפן · ' + save.pending + ' ממתינים לסנכרון', 'var(--color-neutral-800)', 'var(--color-accent-200)'];
    case 'failed':
      return ['ph ph-warning-circle', 'השמירה נכשלה · נסה שוב', 'var(--color-accent-800)', 'var(--color-accent-100)'];
    case 'conflict':
      return ['ph ph-git-merge', 'התנגשות גרסאות · פתור', 'var(--color-accent-800)', 'var(--color-accent-100)'];
    default:
      return ['ph ph-hard-drives', 'אין שינויים', 'transparent', 'var(--color-neutral-400)'];
  }
}

export function App() {
  const S = useStore();
  const { route, toast } = useUI();
  const { job, sim, save, project } = S;
  const isSimple = route === 'simple';
  const { session } = useWS();

  // One-time "restored" notice, global shortcuts.
  const restoredShown = useRef(false);
  useEffect(() => {
    if (S.restored && project && !restoredShown.current) {
      restoredShown.current = true;
      ui.toast('הפרויקט שוחזר מהדפדפן · ' + project.lines.length + ' שורות');
    }
    if (S.bgCatchUp) setTimeout(() => ui.toast('הממשק היה סגור · העבודה המשיכה ברקע: ' + S.bgCatchUp!.from + '% ← ' + S.bgCatchUp!.to + '%'), 1200);
    const onKey = (e: KeyboardEvent) => {
      const typing = /INPUT|TEXTAREA|SELECT/.test((e.target as HTMLElement).tagName);
      if (e.key === 'Escape' && ui.get().modal) ui.close();
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        saveCheckpoint();
        ui.toast('נשמר · נוצרה נקודת שמירה');
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'f' && store.get().project && ui.get().route !== 'simple') {
        e.preventDefault();
        ui.open('find');
      }
      if (e.code === 'Space' && !typing && !(e.target as HTMLElement).closest('button') && store.get().project) {
        e.preventDefault();
        playback.toggle();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Job status notifications.
  const lastStatus = useRef(job?.status);
  useEffect(() => {
    const st = job?.status;
    if (st !== lastStatus.current) {
      if (st === 'failed') ui.toast('העיבוד קרס בשלב ' + ['הייבוא', 'ההפרדה', 'התמלול', 'הסנכרון', 'הרינדור'][job!.stage] + ' · דורש התאוששות');
      if (st === 'done' && lastStatus.current === 'running') ui.toast(job?.kind === 'render' ? 'סרטון הקריוקי מוכן להורדה' : job?.serverId ? 'העיבוד בשרת הושלם' : 'העיבוד הושלם (סימולציה)');
      lastStatus.current = st;
    }
  }, [job?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  const remoteProject = S.backend.mode === 'server' && project?.serverRev != null;
  const [sIcon, sLabel, sBg, sFg] = savePill(save, remoteProject);
  const fi = FLOW.findIndex(([k]) => k === route || (k === 'subs-word' && (route === 'subs' || route === 'subs-lyrics')));
  const jobLbl = !job ? '' : job.status === 'running' ? ' · ' + jobPct(job) + '%' : job.status === 'failed' ? ' · קרס' : job.status === 'cancelled' ? ' · בוטל' : ' · הושלם';
  let nextOk = fi >= 0 && fi < FLOW.length - 1;
  let hint = '';
  if (route === 'queue' && job?.status !== 'done') {
    nextOk = false;
    hint = !job ? 'אין עבודה · התחל מבחירת עיבוד' : job.status === 'failed' ? 'בחר המשך / פרופיל מהיר / ביטול' : job.status === 'cancelled' ? 'הפעל מחדש כדי להמשיך' : 'ממתין לסיום העיבוד';
  }
  if (route === 'options') nextOk = false;

  const sims: [string, string, boolean, () => void, string][] = [
    ['קריסה בתמלול', 'ph ph-lightning-slash', sim.crash, () => { store.setSim('crash', !sim.crash); ui.toast(!sim.crash ? 'העבודה הבאה תקרוס בשלב התמלול' : 'קריסה מדומה כובתה'); }, 'הניסיון הראשון של העבודה הבאה ייעצר ב־38% של התמלול'],
    ['שרת לא זמין', 'ph ph-plugs', sim.offline, () => { store.setSim('offline', !sim.offline); ui.toast(!sim.offline ? 'החיבור לשרת נותק (מדומה)' : 'החיבור חזר · מסנכרן שינויים מקומיים'); }, 'ניתוק מדומה: עריכות נשמרות בדפדפן'],
    ['כשל בשמירה הבאה', 'ph ph-warning', sim.failNext, () => store.setSim('failNext', !sim.failNext), 'השמירה הבאה תיכשל'],
    ['התנגשות בשמירה הבאה', 'ph ph-git-merge', sim.conflictNext, () => store.setSim('conflictNext', !sim.conflictNext), 'השמירה הבאה תתנגש בגרסה של משתמש אחר'],
    ['איפוס נתוני דמו', 'ph ph-trash', false, () => { if (confirm('למחוק את הפרויקט השמור בדפדפן ולהתחיל מחדש?')) { store.reset(); ws.reset(); ui.resetKeys(); location.hash = ''; location.reload(); } }, 'מוחק רק את נתוני Lyra בדפדפן הזה'],
  ];

  return (
    <div className="shell" dir="rtl">
      <header className="topbar no-scrollbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <i className="ph ph-microphone-stage" style={{ color: 'var(--color-accent)', fontSize: 19 }} />
          <span style={{ fontSize: 14.5, fontWeight: 500 }}>Lyra Studio</span>
        </div>
        <div className="pill-group" role="group" aria-label="מצב עבודה">
          <button className="pill" aria-pressed={isSimple} onClick={() => ui.go('simple')}>
            <i className="ph ph-magic-wand" />
            מצב פשוט
          </button>
          <button className="pill" aria-pressed={!isSimple} onClick={() => (isSimple && project ? openAdvanced() : ui.go('dashboard'))}>
            <i className="ph ph-sliders-horizontal" />
            סטודיו מקצועי
          </button>
        </div>
        <button
          onClick={() => (save.state === 'failed' ? store.retrySave() : save.state === 'conflict' ? ui.open('conflict') : store.saveNow())}
          title={remoteProject ? 'נשמר בשרת Lyra עם בדיקת התנגשויות. ⌘S לשמירה ידנית ולנקודת שמירה.' : 'נשמר ב־localStorage של הדפדפן. אין שרת מחובר. ⌘S לשמירה ידנית.'}
          aria-live="polite"
          style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '5px 11px', border: 0, borderRadius: 999, font: 'inherit', fontSize: 12.5, cursor: 'pointer', background: sBg, color: sFg }}
        >
          <i className={sIcon} />
          {sLabel}
        </button>
        <div style={{ marginInlineStart: 'auto', display: 'flex', alignItems: 'center', gap: 10 }}>
          <span className={S.backend.mode === 'server' && S.backend.reachable ? 'tag tag-accent' : 'tag tag-neutral'} title={S.backend.gpu?.name ? S.backend.gpu.name + ' · ' + S.backend.gpu.vramGB + 'GB VRAM' : undefined}>
            <i className={S.backend.mode === 'server' ? (S.backend.reachable ? 'ph ph-plugs-connected' : 'ph ph-plugs') : 'ph ph-flask'} />{' '}
            {S.backend.mode === 'server' ? (S.backend.reachable ? 'שרת Lyra' + (S.backend.engine === 'mock' ? ' · מנוע בדיקה' : '') : 'שרת Lyra לא זמין') : 'מצב הדגמה'}
          </span>
          <div className="pill-group" role="group" aria-label="חיבור" style={{ borderRadius: 'var(--radius-md)', padding: 2 }}>
            {([['local', 'המחשב הזה', 'ph ph-desktop-tower'], ['remote', 'שרת מרוחק', 'ph ph-globe-simple']] as const).map(([k, l, i]) => (
              <button key={k} className="pill sm" aria-pressed={sim.conn === k} style={{ borderRadius: 6 }} onClick={() => { store.setSim('conn', k); ui.toast(k === 'local' ? 'מחובר למחשב הזה' : 'מחובר לשרת מרוחק lyra.home'); }}>
                <i className={i} />
                {l}
              </button>
            ))}
          </div>
        </div>
      </header>
      <div className="subbar no-scrollbar">
        {isSimple ? (
          <>
            <span style={{ fontSize: 12.5, color: 'var(--color-neutral-400)' }}>מצב פשוט: הדבק קישור ← צור קריוקי ← בדוק ← הורד</span>
            {job?.status === 'running' && <span className="tag tag-neutral">בתור: {job.name} · {jobPct(job)}%</span>}
          </>
        ) : (
          <>
            <nav aria-label="זרימה מקצועית" style={{ display: 'flex', gap: 6 }}>
              {FLOW.map(([k, l, i], n) => (
                <button key={k} className={'pill sm step ' + (fi > n ? 'done' : fi === n ? '' : 'todo')} aria-current={fi === n ? 'step' : undefined} onClick={() => ui.go(k)}>
                  <i className={fi > n ? 'ph-bold ph-check' : i} />
                  {n + 1} · {l}
                  {k === 'queue' ? jobLbl : ''}
                </button>
              ))}
            </nav>
            <JumpSelect route={route} />
            {fi >= 0 && fi < FLOW.length - 1 && (
              <button className="btn btn-primary" onClick={() => nextOk && ui.go(FLOW[fi + 1][0])} disabled={!nextOk} style={{ height: 30, fontSize: 12.5 }}>
                הבא
                <i className="ph ph-arrow-left" />
              </button>
            )}
            {hint && <span className="dim" style={{ fontSize: 11.5 }}>{hint}</span>}
          </>
        )}
        <div style={{ marginInlineStart: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
          <span className="dim" style={{ fontSize: 11.5 }}>הדמיה</span>
          {(S.backend.mode === 'server' ? sims.filter(([l]) => l === 'קריסה בתמלול' && S.backend.engine === 'mock') : sims).map(([l, i, on, go, title]) => (
            <button key={l} onClick={go} aria-pressed={on} className={on ? 'tag tag-outline' : 'tag tag-neutral'} title={title} style={{ border: 0, cursor: 'pointer', padding: '5px 9px', fontSize: 12 }}>
              <i className={i} /> {l}
            </button>
          ))}
        </div>
      </div>
      <main className="stage">
        {isSimple ? (session.userId ? <SimpleMode /> : <Login />) : <Advanced route={route} />}
        {S.backend.mode === 'server' && !S.backend.reachable && (
          <div className="banner-offline" role="status">
            <i className="ph ph-plugs" style={{ color: 'var(--color-accent-200)' }} />
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>שרת Lyra לא זמין · עריכות נשמרות בדפדפן ויסונכרנו כשהחיבור יחזור</span>
          </div>
        )}
        {sim.offline && (
          <div className="banner-offline" role="status">
            <i className="ph ph-plugs" style={{ color: 'var(--color-accent-200)' }} />
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {sim.conn === 'local' ? 'שירות Lyra במחשב הזה לא מגיב · עריכות נשמרות בדפדפן · עיבוד מושהה' : 'השרת המרוחק lyra.home לא זמין · עריכות נשמרות בדפדפן · מצב העבודות לא ידוע'}
            </span>
          </div>
        )}
        <ModalHost />
        {toast && (
          <div className="toast" role="status">
            <i className="ph-fill ph-check-circle" style={{ color: 'var(--color-accent-300)' }} />
            {toast}
          </div>
        )}
      </main>
    </div>
  );
}

const GROUPS: [string, [Route, string][]][] = [
  ['ראשי', [['dashboard', 'לוח בקרה'], ['library', 'ספריית שירים'], ['queue', 'תור עיבוד'], ['outputs', 'ספריית תוצרים'], ['team', 'צוות והרשאות']]],
  ['זרימה', [['options', '1 · אפשרויות עיבוד'], ['queue', '2 · תור'], ['subs-word', '3 · תיקון מילים'], ['karaoke', '4 · עיצוב'], ['export', '5 · ייצוא']]],
  ['עורך מילים', [['subs', 'ציר זמן'], ['subs-lyrics', 'מילים תחילה'], ['subs-word', 'זום מילה']]],
  ['פרויקט', [['project', 'פרטי פרויקט'], ['project-versions', 'היסטוריית גרסאות'], ['audio', 'אודיו וסטמים'], ['video', 'עורך וידאו']]],
  ['הגדרות', [['settings', 'שרת ו־Workers'], ['settings-general', 'כללי'], ['settings-models', 'מודלי AI'], ['settings-storage', 'אחסון'], ['settings-api', 'API']]],
  ['כניסה', [['login', 'התחברות'], ['invite', 'קבלת הזמנה'], ['forgot', 'איפוס סיסמה']]],
  ['מצבי מערכת', [['state:background', 'ממשק נסגר · העבודה המשיכה'], ['state:offline', 'שרת לא זמין'], ['state:crash', 'קריסת עיבוד'], ['state:failed', 'כשל תוכן'], ['state:recovery', 'שחזור טיוטה'], ['state:emptyDashboard', 'ריק · משתמש חדש'], ['state:emptyLibrary', 'ריק · אין תוצאות'], ['state:loading', 'טעינה'], ['state:forbidden', 'אין הרשאה'], ['state:notfound', '404']]],
];

function JumpSelect({ route }: { route: Route }) {
  return (
    <select className="input" aria-label="קפיצה למסך" value={route} onChange={(e) => ui.go(e.target.value as Route)} style={{ height: 30, width: 170, padding: '0 8px', fontSize: 12.5, flex: 'none' }}>
      {GROUPS.map(([g, items]) => (
        <optgroup key={g} label={g}>
          {items.map(([k, l]) => <option key={g + k} value={k}>{l}</option>)}
        </optgroup>
      ))}
    </select>
  );
}

const AUTH: Route[] = ['login', 'invite', 'forgot'];
const FULL: Route[] = ['subs', 'subs-lyrics', 'subs-word', 'karaoke'];

function Advanced({ route }: { route: Route }) {
  const { project } = useStore();
  const { session } = useWS();
  const needsProject = FULL.includes(route);
  useEffect(() => {
    if (needsProject && !project) store.ensure();
  }, [needsProject, project]);
  if (AUTH.includes(route) || !session.userId) {
    return route === 'invite' ? <AcceptInvite /> : route === 'forgot' ? <Forgot /> : <Login />;
  }
  const screen = (() => {
    if (route.startsWith('state:')) return <StateView kind={route.slice(6) as StateKind} />;
    switch (route) {
      case 'options': return <ProcessingOptions />;
      case 'queue': return <Queue />;
      case 'subs': return <LyricsStudio mode="timeline" />;
      case 'subs-lyrics': return <LyricsStudio mode="lyrics" />;
      case 'subs-word': return <LyricsStudio mode="word" />;
      case 'karaoke': return <KaraokeStudio />;
      case 'dashboard': return <Dashboard />;
      case 'library': return <Library />;
      case 'outputs': return <Outputs />;
      case 'export': return <ExportScreen />;
      case 'team': return <Team />;
      case 'project': case 'project-versions': return <ProjectDetails />;
      case 'audio': return <AudioStudio />;
      case 'video': return <VideoEditor />;
      default:
        if (route.startsWith('settings')) return <Settings />;
        return <StateView kind="notfound" />;
    }
  })();
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {needsProject && (
        <div className="adv-narrow-note" role="note" style={{ gap: 8, alignItems: 'center', padding: '8px 16px', fontSize: 12.5, background: 'var(--color-surface)' }}>
          <i className="ph ph-desktop" />
          העריכה המלאה מותאמת למחשב. בנייד אפשר לתקן במצב הפשוט.
          <a onClick={() => { ui.setStep('preview'); ui.go('simple'); }}>למצב הפשוט</a>
        </div>
      )}
      <div style={{ flex: 1, minHeight: 0, display: 'flex' }}>
        {!needsProject && <Sidebar />}
        <div style={{ flex: 1, minWidth: 0, minHeight: 0 }}>{screen}</div>
      </div>
    </div>
  );
}
