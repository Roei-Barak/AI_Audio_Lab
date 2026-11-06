import { useStore } from '../../store/store';
import { ui, useUI, type Route } from '../../store/ui';
import { ROLE_L, useWS, ws } from '../../store/workspace';
import * as RW from '../../store/remoteWs';

const MAIN: [Route, string, string][] = [
  ['dashboard', 'לוח בקרה', 'squares-four'],
  ['library', 'ספריית שירים', 'music-notes'],
  ['queue', 'תור עיבוד AI', 'cpu'],
  ['outputs', 'ספריית תוצרים', 'film-strip'],
  ['team', 'צוות והרשאות', 'users-three'],
];
const PROJ: [Route, string, string][] = [
  ['project', 'פרטי פרויקט', 'folder-open'],
  ['audio', 'אודיו וסטמים', 'waveform'],
  ['subs-word', 'עורך מילים', 'closed-captioning'],
  ['karaoke', 'עיצוב קריוקי', 'paint-brush'],
  ['video', 'עורך וידאו', 'video-camera'],
  ['export', 'ייצוא', 'export'],
];

const isOn = (k: Route, r: Route) => k === r || (k === 'project' && r === 'project-versions') || (k === 'subs-word' && (r === 'subs' || r === 'subs-lyrics')) || (k === 'settings' && r.startsWith('settings'));

export function Sidebar() {
  const { route } = useUI();
  const { project, job, sim } = useStore();
  const { waiting } = useWS();
  const me = ws.me();
  const queued = (job && job.status === 'running' ? 1 : 0) + waiting.length;
  const Item = ({ k, l, i, badge }: { k: Route; l: string; i: string; badge?: number }) => (
    <a
      onClick={() => ui.go(k)}
      aria-current={isOn(k, route) ? 'page' : undefined}
      style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '7px 10px', borderRadius: 'var(--radius-md)', fontSize: 13.5, textDecoration: 'none', background: isOn(k, route) ? 'var(--color-accent-900)' : 'transparent', color: isOn(k, route) ? 'var(--color-accent-300)' : 'color-mix(in srgb, var(--color-text) 72%, transparent)' }}
    >
      <i className={'ph ph-' + i} style={{ fontSize: 17 }} />
      <span style={{ flex: 1 }}>{l}</span>
      {!!badge && <span className="tag tag-accent" style={{ padding: '1px 7px', fontSize: 10.5 }}>{badge}</span>}
    </a>
  );
  return (
    <aside className="hide-narrow" dir="rtl" style={{ width: 232, flex: 'none', height: '100%', overflow: 'auto', display: 'flex', flexDirection: 'column', gap: 18, padding: '16px 12px', background: 'var(--color-surface)', borderInlineEnd: '1px solid color-mix(in srgb, var(--color-text) 8%, transparent)', boxSizing: 'border-box' }}>
      <button className="btn btn-primary" onClick={() => { ui.setStep('input'); ui.go('simple'); }} style={{ width: '100%', justifyContent: 'center' }}>
        <i className="ph ph-plus" />
        פרויקט חדש
      </button>
      <nav style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {MAIN.map(([k, l, i]) => <Item key={k} k={k} l={l} i={i} badge={k === 'queue' ? queued : undefined} />)}
      </nav>
      {project && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <div className="dim" style={{ fontSize: 10.5, letterSpacing: '0.08em', padding: '0 10px 6px' }}>פרויקט פתוח</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px 10px' }}>
            <div style={{ width: 26, height: 26, borderRadius: 'var(--radius-sm)', background: 'var(--color-accent-900)', display: 'grid', placeItems: 'center', color: 'var(--color-accent-300)' }}><i className="ph ph-music-note" /></div>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontSize: 13 }}>{project.title}</div>
              <div className="dim" style={{ fontSize: 11 }}>{project.artist}</div>
            </div>
          </div>
          {PROJ.map(([k, l, i]) => <Item key={k} k={k} l={l} i={i} />)}
        </div>
      )}
      <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: 10 }}>
        <Item k="settings" l="הגדרות ושרת" i="gear-six" />
        <button onClick={() => ui.go('settings')} style={{ border: 0, font: 'inherit', color: 'inherit', cursor: 'pointer', textAlign: 'start', padding: 10, borderRadius: 'var(--radius-md)', background: 'transparent', boxShadow: 'var(--shadow-sm)', display: 'flex', alignItems: 'center', gap: 7, fontSize: 12 }}>
          <i className={sim.offline ? 'ph ph-plugs' : sim.conn === 'remote' ? 'ph ph-globe-simple' : 'ph ph-desktop'} style={{ color: sim.offline ? 'var(--color-neutral-500)' : 'var(--color-accent-400)' }} />
          {sim.offline ? (sim.conn === 'remote' ? 'שרת מרוחק · לא זמין' : 'המחשב הזה · לא מגיב') : sim.conn === 'remote' ? 'שרת מרוחק · מחובר' : 'המחשב הזה · מחובר'}
        </button>
        {me && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '4px 6px' }}>
            <div style={{ width: 30, height: 30, borderRadius: '50%', background: 'var(--color-accent-800)', color: 'var(--color-accent-100)', display: 'grid', placeItems: 'center', fontSize: 13 }}>{me.name[0]}</div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 13 }}>{me.name}</div>
              <div className="dim" style={{ fontSize: 11 }}>{ROLE_L[me.role]}</div>
            </div>
            <button className="btn btn-ghost btn-icon" title="התנתקות" aria-label="התנתקות" onClick={async () => { await RW.logout(); ui.go('login'); }} style={{ color: 'var(--color-neutral-400)' }}>
              <i className="ph ph-sign-out" />
            </button>
          </div>
        )}
      </div>
    </aside>
  );
}

/** Standard advanced page body: title row + scrollable content. */
export function Page({ title, sub, actions, children, max = 1180 }: { title: string; sub?: React.ReactNode; actions?: React.ReactNode; children: React.ReactNode; max?: number }) {
  return (
    <div dir="rtl" style={{ height: '100%', overflow: 'auto', background: 'var(--color-bg)' }}>
      <div className="page-pad" style={{ maxWidth: max, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 18, boxSizing: 'border-box' }}>
        <header style={{ display: 'flex', alignItems: 'flex-end', gap: 12, flexWrap: 'wrap' }}>
          <h2 style={{ fontSize: 28, margin: 0 }}>{title}</h2>
          {sub && <span className="dim" style={{ fontSize: 14, paddingBottom: 4 }}>{sub}</span>}
          {actions && <div style={{ marginInlineStart: 'auto', display: 'flex', gap: 8, flexWrap: 'wrap' }}>{actions}</div>}
        </header>
        {children}
      </div>
    </div>
  );
}

export function Tabs<T extends string>({ value, tabs, onChange }: { value: T; tabs: [T, string][]; onChange: (t: T) => void }) {
  return (
    <div role="tablist" style={{ display: 'flex', gap: 4, borderBottom: '1px solid var(--color-divider)', overflowX: 'auto' }} className="no-scrollbar">
      {tabs.map(([k, l]) => (
        <button key={k} role="tab" aria-selected={value === k} onClick={() => onChange(k)} style={{ border: 0, background: 'none', font: 'inherit', fontSize: 13.5, padding: '8px 12px', cursor: 'pointer', whiteSpace: 'nowrap', color: value === k ? 'var(--color-accent-200)' : 'var(--color-neutral-400)', boxShadow: value === k ? 'inset 0 -2px 0 var(--color-accent)' : 'none' }}>
          {l}
        </button>
      ))}
    </div>
  );
}

export const Card = ({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) => (
  <section className="panel" style={{ padding: 18, display: 'flex', flexDirection: 'column', gap: 12, ...style }}>{children}</section>
);

export const KV = ({ k, v }: { k: string; v: React.ReactNode }) => (
  <div style={{ display: 'flex', fontSize: 13, gap: 10 }}>
    <span className="muted">{k}</span>
    <span style={{ marginInlineStart: 'auto', textAlign: 'end' }}>{v}</span>
  </div>
);
