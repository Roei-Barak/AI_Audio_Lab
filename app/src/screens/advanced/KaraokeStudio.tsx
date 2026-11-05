import { KaraokePreview } from '../../components/KaraokePreview';
import { ColorSwatches, Seg, SizeSlider } from '../../components/controls';
import { lineSpan, lineText } from '../../lib/format';
import { exportFile, exportMp4, serverProject } from '../../lib/actions';
import { renderFresh } from '../../store/server';
import { previewStyle, setStyle } from '../../lib/styleEdit';
import { usePlayback } from '../../store/playback';
import { store, useStore } from '../../store/store';
import type { Template } from '../../store/types';
import { ui } from '../../store/ui';

const TEMPLATES: [Template, string, number, string][] = [
  ['classic', 'קלאסי', 22, 'var(--color-accent-300)'],
  ['bold', 'מודגש', 27, 'var(--color-neutral-100)'],
  ['minimal', 'מינימלי', 19, 'var(--color-neutral-300)'],
];

const Section = ({ k, children }: { k: string; children: React.ReactNode }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
    <span className="kicker">{k}</span>
    {children}
  </div>
);

export function KaraokeStudio() {
  const { project: p, canUndo, canRedo } = useStore();
  const { t } = usePlayback();
  if (!p) return null;
  const st = p.style;
  const act = p.lines.find((l) => {
    const sp = lineSpan(l);
    return t >= sp[0] - 0.5 && t <= sp[1] + 0.5;
  });
  return (
    <div dir="rtl" data-screen-label="Karaoke Studio" style={{ height: '100%', display: 'flex', flexDirection: 'column', background: 'var(--color-bg)', overflow: 'hidden' }}>
      <header className="no-scrollbar" style={{ minHeight: 56, flex: 'none', display: 'flex', alignItems: 'center', gap: 12, padding: '0 20px', overflowX: 'auto', whiteSpace: 'nowrap', borderBottom: '1px solid color-mix(in srgb, var(--color-text) 7%, transparent)' }}>
        <a onClick={() => ui.go('subs-word')} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, textDecoration: 'none', color: 'var(--color-neutral-300)' }}>
          <i className="ph ph-arrow-right" />
          עורך מילים
        </a>
        <span style={{ fontSize: 17 }}>עיצוב קריוקי ותצוגה מקדימה</span>
        <span className="tag tag-neutral">{p.title}</span>
        <div style={{ marginInlineStart: 'auto', display: 'flex', gap: 8 }}>
          <button className="btn btn-ghost btn-icon" onClick={() => store.undo()} disabled={!canUndo} aria-label="Undo"><i className="ph ph-arrow-counter-clockwise" /></button>
          <button className="btn btn-ghost btn-icon" onClick={() => store.redo()} disabled={!canRedo} aria-label="Redo"><i className="ph ph-arrow-clockwise" /></button>
        </div>
      </header>
      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexWrap: 'wrap', overflow: 'auto' }}>
        <main style={{ flex: '1 1 560px', minWidth: 0, minHeight: 360, padding: '18px 20px', display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ flex: 1, minHeight: 300 }}>
            <KaraokePreview />
          </div>
          <div className="muted" style={{ display: 'flex', gap: 8, flexWrap: 'wrap', fontSize: 12.5 }}>
            <span>שורה פעילה:</span>
            <span dir="auto" style={{ color: 'var(--color-text)' }}>{act ? lineText(act) : '—'}</span>
          </div>
        </main>
        <aside style={{ flex: '0 1 400px', minWidth: 0, overflow: 'auto', padding: 16, display: 'flex', flexDirection: 'column', gap: 16, borderInlineStart: '1px solid color-mix(in srgb, var(--color-text) 7%, transparent)' }}>
          <Section k="תבנית">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 8 }}>
              {TEMPLATES.map(([k, l, fs, c]) => (
                <button
                  key={k}
                  aria-pressed={st.template === k}
                  onClick={() => setStyle('template', k)}
                  style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'center', padding: '12px 6px', border: 0, borderRadius: 'var(--radius-md)', font: 'inherit', cursor: 'pointer', background: 'var(--color-surface)', color: 'var(--color-text)', boxShadow: st.template === k ? 'inset 0 0 0 1px var(--color-accent)' : 'inset 0 0 0 1px var(--color-divider)' }}
                >
                  <span style={{ fontSize: fs, fontWeight: 600, color: c }}>אבג</span>
                  <span style={{ fontSize: 12 }}>{l}</span>
                </button>
              ))}
            </div>
          </Section>
          <Section k="מיקום"><Seg value={st.position} onChange={(v) => setStyle('position', v)} options={[['top', 'למעלה'], ['middle', 'אמצע'], ['bottom', 'למטה']]} /></Section>
          <Section k="רקע"><Seg value={st.background} onChange={(v) => setStyle('background', v)} options={[['video', 'וידאו מקורי'], ['blur', 'מעבר צבע'], ['solid', 'אחיד']]} /></Section>
          <Section k="יחס תמונה"><Seg value={st.aspect} onChange={(v) => setStyle('aspect', v)} options={[['16:9', '16:9'], ['9:16', '9:16'], ['1:1', '1:1']]} /></Section>
          <Section k="צבע הדגשה"><ColorSwatches value={st.color} onChange={(c) => setStyle('color', c)} size={30} /></Section>
          <Section k="גודל כתב"><SizeSlider value={st.size} onPreview={(v) => previewStyle('size', v)} onCommit={(v) => setStyle('size', v)} /></Section>
          <div style={{ display: 'flex', gap: 18 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, cursor: 'pointer' }}>
              <input type="checkbox" checked={st.outline} onChange={() => setStyle('outline', !st.outline)} style={{ accentColor: 'var(--color-accent)' }} />
              קו מתאר
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, cursor: 'pointer' }}>
              <input type="checkbox" checked={st.shadow} onChange={() => setStyle('shadow', !st.shadow)} style={{ accentColor: 'var(--color-accent)' }} />
              צל
            </label>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, padding: 12, borderRadius: 'var(--radius-md)', background: 'var(--color-surface)', fontSize: 12.5, color: 'var(--color-neutral-300)' }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}><i className="ph ph-speaker-high" style={{ color: 'var(--color-accent-300)' }} />פסקול: Instrumental · אודיו מקורי מושתק</div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}><i className="ph ph-text-aa" style={{ color: 'var(--color-accent-300)' }} />{p.timing === 'word' ? 'זמני מילים קיימים · הדגשה הדרגתית' : 'זמני שורות בלבד · כתוביות רגילות'}</div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, padding: 12, borderRadius: 'var(--radius-md)', background: 'var(--color-surface)' }}>
            <span className="kicker">ייצוא</span>
            <button className="btn btn-primary" style={{ justifyContent: 'center' }} onClick={() => void exportMp4()}>
              <i className="ph ph-film-strip" />
              {serverProject() ? (renderFresh(p) ? 'הורד MP4' : 'רנדר MP4 בשרת') : 'ייצוא MP4 (דורש שרת · מדומה)'}
            </button>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {(['srt', 'lrc', 'txt', 'json'] as const).map((k) => (
                <button key={k} className="btn btn-secondary" onClick={() => exportFile(k)}>
                  <i className="ph ph-download-simple" />
                  {k.toUpperCase()}
                </button>
              ))}
            </div>
            <span className="dim" style={{ fontSize: 11.5 }}>SRT, LRC, TXT ו־JSON הם קבצים אמיתיים שנוצרים עכשיו מנתוני הפרויקט הערוכים.</span>
          </div>
        </aside>
      </div>
    </div>
  );
}
