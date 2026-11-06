import { useRef, type ReactNode } from 'react';
import { bidiRuns } from '../lib/format';
import type { HiColor, Word } from '../store/types';

export function Seg<T extends string | number>({
  value,
  options,
  onChange,
  size = 12.5,
  label,
}: {
  value: T;
  options: [T, ReactNode][];
  onChange: (v: T) => void;
  size?: number;
  label?: string;
}) {
  return (
    <div className="seg" role="group" aria-label={label} style={{ alignSelf: 'flex-start' }}>
      {options.map(([v, l]) => (
        <button
          key={String(v)}
          type="button"
          className="seg-opt"
          aria-pressed={value === v}
          onClick={() => onChange(v)}
          style={{ fontSize: size }}
        >
          {l}
        </button>
      ))}
    </div>
  );
}

export function Switch({ on, onChange, label }: { on: boolean; onChange: (v: boolean) => void; label: string }) {
  return <button type="button" role="switch" className="switch" aria-checked={on} aria-label={label} onClick={() => onChange(!on)} />;
}

export const HI_COLORS: Record<HiColor, string> = {
  accent: 'var(--color-accent-300)',
  soft: 'var(--color-accent-100)',
  light: 'var(--color-neutral-100)',
  deep: 'var(--color-accent-500)',
};
const HI_LABELS: Record<HiColor, string> = { accent: 'סגול', soft: 'לבנדר', light: 'לבן', deep: 'סגול עמוק' };

export function ColorSwatches({ value, onChange, size = 26 }: { value: HiColor; onChange: (c: HiColor) => void; size?: number }) {
  return (
    <div style={{ display: 'flex', gap: 10 }} role="radiogroup" aria-label="צבע הדגשה">
      {(Object.keys(HI_COLORS) as HiColor[]).map((k) => (
        <button
          key={k}
          type="button"
          role="radio"
          aria-checked={value === k}
          aria-label={HI_LABELS[k]}
          title={HI_LABELS[k]}
          onClick={() => onChange(k)}
          style={{
            width: size,
            height: size,
            flex: 'none',
            borderRadius: '50%',
            border: 0,
            cursor: 'pointer',
            background: HI_COLORS[k],
            boxShadow: value === k ? '0 0 0 2px var(--color-bg), 0 0 0 4px var(--color-accent)' : 'inset 0 0 0 1px var(--color-divider)',
          }}
        />
      ))}
    </div>
  );
}

/** Per-word chips with correct mixed Hebrew/English/number order. */
export function WordChips({ words, selected, onPick }: { words: Word[]; selected: string | null; onPick: (w: Word) => void }) {
  return (
    <div dir="rtl" style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
      {bidiRuns(words).map((run, i) => (
        <div key={i} dir={run.ltr ? 'ltr' : 'rtl'} style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {run.items.map((w) => (
            <button
              key={w.id}
              type="button"
              dir="auto"
              className={'chip' + (w.conf === 'low' && !w.manual ? ' low' : '')}
              aria-pressed={w.id === selected}
              onClick={() => onPick(w)}
            >
              {w.w}
            </button>
          ))}
        </div>
      ))}
    </div>
  );
}

export function Icon({ name, style }: { name: string; style?: React.CSSProperties }) {
  return <i className={name.includes(' ') ? name : 'ph ph-' + name} style={style} aria-hidden="true" />;
}

/** Font-size slider: live preview while dragging, one undoable edit when released. */
export function SizeSlider({ value, onPreview, onCommit }: { value: number; onPreview: (v: number) => void; onCommit: (v: number) => void }) {
  const timer = useRef<ReturnType<typeof setTimeout>>();
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, flex: 1 }}>
      <input
        type="range"
        aria-label="גודל כתב"
        min={28}
        max={72}
        step={2}
        value={value}
        onChange={(e) => {
          const v = parseInt(e.target.value, 10);
          onPreview(v);
          clearTimeout(timer.current);
          timer.current = setTimeout(() => onCommit(v), 250);
        }}
        style={{ flex: 1, minWidth: 0, accentColor: 'var(--color-accent)' }}
      />
      <span className="num" style={{ width: 34, fontSize: 12.5 }}>
        {value}
      </span>
    </div>
  );
}
