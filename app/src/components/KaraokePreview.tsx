import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { HEB, fmt, lineSpan, lineText } from '../lib/format';
import { playback, usePlayback } from '../store/playback';
import { useStore } from '../store/store';
import { HI_COLORS } from './controls';

/** Live karaoke preview rendered from the shared project data (never fixed demo text). */
export function KaraokePreview({ controls = true }: { controls?: boolean }) {
  const { project: p, media } = useStore();
  const { t, playing } = usePlayback();
  const boxRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [box, setBox] = useState({ w: 800, h: 450 });

  useLayoutEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setBox({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    playback.registerVideo(videoRef.current);
    return () => playback.registerVideo(null);
  }, [media]);

  if (!p) return null;
  const st = p.style;
  const ar = { '16:9': 16 / 9, '9:16': 9 / 16, '1:1': 1 }[st.aspect];
  let fw = box.w,
    fh = box.w / ar;
  if (fh > box.h) {
    fh = box.h;
    fw = box.h * ar;
  }
  const k = (fw / 1920) * (st.aspect === '16:9' ? 1 : 1.6);
  const hi = HI_COLORS[st.color];
  const base = st.template === 'minimal' ? 'var(--color-neutral-400)' : st.color === 'light' ? 'var(--color-neutral-500)' : 'var(--color-neutral-100)';

  const lines = p.lines;
  const spans = lines.map(lineSpan);
  let ci = -1;
  for (let i = 0; i < lines.length; i++) if (t >= spans[i][0] - 0.5 && t <= spans[i][1] + 0.5) ci = i;
  const ni = ci >= 0 ? ci + 1 : spans.findIndex((sp) => sp[0] > t);
  const showUpcoming = ci < 0 && ni >= 0 && spans[ni][0] - t < 4;
  const cur = ci >= 0 ? lines[ci] : showUpcoming ? lines[ni] : null;
  const wordLevel = p.timing === 'word';
  const next = ci >= 0 && lines[ci + 1] && spans[ci + 1][0] - t < 6 ? lineText(lines[ci + 1]) : '';

  const fs = Math.max(12, Math.round(st.size * k * (st.template === 'bold' ? 1.2 : 1)));
  const shadow =
    [
      st.outline ? '0 0 2px var(--color-neutral-900), 0 0 2px var(--color-neutral-900), 0 0 1px var(--color-neutral-900)' : '',
      st.shadow ? '0 3px 10px rgba(0,0,0,0.7)' : '',
    ]
      .filter(Boolean)
      .join(', ') || 'none';
  const isVid = !!media && /^video/.test(media.type);
  const showVideo = isVid && st.background === 'video';
  const bg = st.background === 'blur' ? 'linear-gradient(160deg, var(--color-accent-900), var(--color-neutral-900))' : 'var(--color-neutral-900)';
  const pos = st.position;
  const tagStyle = { background: 'color-mix(in srgb, var(--color-bg) 80%, transparent)' };
  const dur = playback.duration();

  return (
    <div dir="rtl" style={{ width: '100%', height: '100%', minHeight: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div ref={boxRef} style={{ flex: 1, minHeight: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div
          style={{ position: 'relative', width: Math.round(fw), height: Math.round(fh), borderRadius: 'var(--radius-lg)', overflow: 'hidden', background: bg, boxShadow: 'var(--shadow-md)' }}
          aria-label="תצוגת קריוקי"
        >
          {media && (
            <video
              ref={videoRef}
              src={media.url}
              playsInline
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', display: showVideo ? 'block' : 'none' }}
            />
          )}
          {!showVideo && st.background === 'video' && (
            <div
              style={{
                position: 'absolute',
                inset: 0,
                background:
                  'radial-gradient(120% 90% at 30% 20%, color-mix(in srgb, var(--color-accent-800) 70%, transparent), transparent 60%), radial-gradient(80% 70% at 80% 90%, var(--color-neutral-800), transparent 70%)',
              }}
            />
          )}
          <div style={{ position: 'absolute', top: 10, right: 10, left: 10, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            <span className="tag tag-neutral" style={tagStyle}>
              <i className="ph ph-speaker-high" /> {media ? 'ניגון מקומי של הקובץ שהועלה · ללא הפרדה בדפדפן' : 'Instrumental · השירה המקורית מושתקת'}
            </span>
            <span className="tag tag-neutral" style={tagStyle}>
              <i className="ph ph-monitor-play" />{' '}
              {st.background === 'video'
                ? (media ? 'וידאו מקומי · ' : 'וידאו מקור · תצוגה מדומה · ') + st.aspect
                : 'רקע ' + (st.background === 'blur' ? 'מעבר צבע' : 'אחיד') + ' · ' + st.aspect}
            </span>
            <span className="tag tag-neutral" style={tagStyle}>
              <i className="ph ph-text-aa" /> {wordLevel ? 'הדגשה ברמת מילה' : 'תזמון שורות בלבד'}
            </span>
          </div>
          <div
            style={{
              position: 'absolute',
              left: '6%',
              right: '6%',
              top: pos === 'top' ? '14%' : pos === 'middle' ? '38%' : 'auto',
              bottom: pos === 'bottom' ? '9%' : 'auto',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: Math.round(fs * 0.25),
              textAlign: 'center',
            }}
          >
            <div
              data-testid="karaoke-line"
              dir={cur && !HEB.test(lineText(cur)) ? 'ltr' : 'rtl'}
              style={{ fontSize: fs, fontWeight: 600, lineHeight: 1.3, textShadow: shadow }}
            >
              {cur?.words.map((wd) => {
                const on = ci >= 0 && (wordLevel ? t >= wd.s : true);
                const now = wordLevel && ci >= 0 && t >= wd.s && t < wd.e;
                return (
                  <span
                    key={wd.id}
                    style={{
                      color: ci < 0 ? 'var(--color-neutral-400)' : on ? hi : base,
                      borderBottom: '3px solid ' + (now && st.template !== 'minimal' ? hi : 'transparent'),
                      transition: 'color 80ms',
                      margin: '0 0.14em',
                    }}
                  >
                    {wd.w}
                  </span>
                );
              })}
            </div>
            <div dir="auto" style={{ fontSize: Math.round(fs * 0.62), fontWeight: 500, color: 'var(--color-neutral-300)', opacity: 0.8, textShadow: shadow }}>
              {next}
            </div>
          </div>
        </div>
      </div>
      {controls && (
        <div dir="ltr" style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '0 4px' }}>
          <button className="btn btn-secondary btn-icon" onClick={() => playback.toggle()} aria-label={playing ? 'Pause' : 'Play'}>
            <i className={playing ? 'ph-fill ph-pause' : 'ph-fill ph-play'} />
          </button>
          <span className="num" style={{ fontSize: 12.5, color: 'var(--color-neutral-300)', minWidth: 52 }}>
            {fmt(t)}
          </span>
          <input
            type="range"
            aria-label="Seek"
            min={0}
            max={dur}
            step={0.05}
            value={Math.min(t, dur)}
            onChange={(e) => playback.seek(parseFloat(e.target.value))}
            style={{ flex: 1, minWidth: 0, accentColor: 'var(--color-accent)' }}
          />
          <span className="num dim" style={{ fontSize: 12.5 }}>
            {fmt(dur)}
          </span>
          <span className="tag tag-outline hide-narrow" style={{ fontSize: 11 }}>
            {media ? 'ניגון מקומי אמיתי' : 'סימולציית ניגון'}
          </span>
        </div>
      )}
    </div>
  );
}
