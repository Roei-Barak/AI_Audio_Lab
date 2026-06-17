import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin, { type Region } from 'wavesurfer.js/dist/plugins/regions.js'
import { FileVideo, Save, Upload } from 'lucide-react'

interface Line { start: number; end: number; text: string }

function parseAss(text: string): Line[] {
  return text.split('\n').filter(l => l.startsWith('Dialogue:')).map(l => {
    const parts = l.split(',', 10)
    if (parts.length < 10) return null
    const parseT = (s: string) => {
      const [h, m, rest] = s.trim().split(':')
      const [sec, cs]    = rest.split('.')
      return +h * 3600 + +m * 60 + +sec + +cs / 100
    }
    return { start: parseT(parts[1]), end: parseT(parts[2]), text: parts[9].trim() }
  }).filter(Boolean) as Line[]
}

export default function Editor() {
  const waveRef   = useRef<HTMLDivElement>(null)
  const ws        = useRef<WaveSurfer | null>(null)
  const regPlugin = useRef<ReturnType<typeof RegionsPlugin.create> | null>(null)
  const [lines, setLines]     = useState<Line[]>([])
  const [assText, setAssText] = useState('')
  const [status, setStatus]   = useState('טען קובץ ASS + אודיו')

  useEffect(() => {
    const rp = RegionsPlugin.create()
    regPlugin.current = rp
    const surf = WaveSurfer.create({
      container: waveRef.current!, waveColor: '#0BC', progressColor: '#007ACC',
      cursorColor: '#F44336', height: 80, plugins: [rp],
    })
    ws.current = surf
    return () => surf.destroy()
  }, [])

  useEffect(() => {
    const rp = regPlugin.current; if (!rp) return
    rp.clearRegions()
    lines.forEach((l, i) => rp.addRegion({
      id: `r${i}`, start: l.start, end: l.end,
      content: l.text.length > 20 ? l.text.slice(0, 20) + '…' : l.text,
      color: 'rgba(0,122,204,0.35)', drag: true, resize: true,
    }))
  }, [lines])

  useEffect(() => {
    const rp = regPlugin.current; if (!rp) return
    const unsub = rp.on('region-updated', (r: Region) => {
      const idx = parseInt(r.id.slice(1))
      setLines(prev => prev.map((l, i) => i === idx ? { ...l, start: r.start, end: r.end } : l))
    })
    return unsub
  }, [])

  function loadAssFile() {
    const inp = document.createElement('input'); inp.type = 'file'; inp.accept = '.ass'
    inp.onchange = async () => {
      const f = inp.files?.[0]; if (!f) return
      const text = await f.text()
      setAssText(text); setLines(parseAss(text))
      setStatus(`נטען: ${parseAss(text).length} שורות`)
    }; inp.click()
  }

  function loadAudio() {
    const inp = document.createElement('input'); inp.type = 'file'; inp.accept = 'audio/*,video/*'
    inp.onchange = () => {
      const f = inp.files?.[0]; if (!f) return
      ws.current?.load(URL.createObjectURL(f))
    }; inp.click()
  }

  function exportAss() {
    if (!assText) return
    const header = assText.split('\n').filter(l => !l.startsWith('Dialogue:')).join('\n')
    const fmt = (t: number) => {
      const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60)
      const s = Math.floor(t % 60), cs = Math.round((t - Math.floor(t)) * 100)
      return `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${String(cs).padStart(2,'0')}`
    }
    const dlg = lines.map(l => `Dialogue: 0,${fmt(l.start)},${fmt(l.end)},Karaoke,,0,0,0,,${l.text}`).join('\n')
    const blob = new Blob([header + '\n' + dlg], { type: 'text/plain;charset=utf-8' })
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'edited.ass'; a.click()
  }

  return (
    <div className="space-y-4 h-full flex flex-col" dir="rtl">
      <div className="flex items-center gap-3">
        <FileVideo className="w-6 h-6 text-accent" />
        <h1 className="text-xl font-bold">עורך כתוביות</h1>
      </div>
      <div className="card flex flex-wrap gap-2 py-2">
        <button onClick={loadAssFile} className="btn-ghost flex items-center gap-1.5 text-xs"><Upload className="w-3.5 h-3.5" /> טען ASS</button>
        <button onClick={loadAudio}   className="btn-ghost flex items-center gap-1.5 text-xs"><Upload className="w-3.5 h-3.5" /> טען אודיו</button>
        <button onClick={() => ws.current?.playPause()} className="btn-ghost text-xs">▶/⏸ נגן</button>
        <button onClick={exportAss} disabled={!lines.length} className="btn-primary flex items-center gap-1.5 text-xs">
          <Save className="w-3.5 h-3.5" /> ייצא ASS
        </button>
        <span className="text-xs text-gray-500 self-center mr-auto">{status}</span>
      </div>
      <div className="card p-2"><div ref={waveRef} className="w-full" /></div>
      <div className="card flex-1 overflow-auto">
        <table className="w-full text-sm border-collapse">
          <thead><tr className="text-gray-400 text-xs border-b border-border">
            <th className="text-right py-2 px-3 w-24">התחלה</th>
            <th className="text-right py-2 px-3 w-24">סיום</th>
            <th className="text-right py-2 px-3">טקסט</th>
          </tr></thead>
          <tbody>{lines.map((l, i) => (
            <tr key={i} className="border-b border-border/50 hover:bg-nav/50">
              <td className="py-1.5 px-3 font-mono text-xs text-gray-400">{l.start.toFixed(2)}s</td>
              <td className="py-1.5 px-3 font-mono text-xs text-gray-400">{l.end.toFixed(2)}s</td>
              <td className="py-1.5 px-3">
                <input value={l.text}
                       onChange={e => setLines(prev => prev.map((x, j) => j === i ? { ...x, text: e.target.value } : x))}
                       className="bg-transparent border-none outline-none w-full text-white text-sm" />
              </td>
            </tr>
          ))}</tbody>
        </table>
        {!lines.length && <p className="text-center text-gray-500 py-12">טען קובץ ASS להתחלה</p>}
      </div>
    </div>
  )
}
