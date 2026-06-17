import { useState } from 'react'
import { Music4, Upload, Download } from 'lucide-react'

interface Line { start: number; end: number; text: string; alt: string }

function parseAss(src: string): Line[] {
  const parseT = (s: string) => {
    const [h, m, rest] = s.trim().split(':')
    const [sec, cs]    = rest.split('.')
    return +h * 3600 + +m * 60 + +sec + +cs / 100
  }
  return src.split('\n').filter(l => l.startsWith('Dialogue:')).map(l => {
    const p = l.split(',', 10)
    if (p.length < 10) return null
    return { start: parseT(p[1]), end: parseT(p[2]), text: p[9].trim(), alt: '' }
  }).filter(Boolean) as Line[]
}

export default function Parody() {
  const [lines, setLines]     = useState<Line[]>([])
  const [assText, setAssText] = useState('')
  const [status, setStatus]   = useState('טען קובץ ASS להתחלה')
  const [useAlt, setUseAlt]   = useState(true)

  function loadAss() {
    const inp = document.createElement('input'); inp.type = 'file'; inp.accept = '.ass'
    inp.onchange = async () => {
      const f = inp.files?.[0]; if (!f) return
      const text = await f.text()
      setAssText(text); setLines(parseAss(text))
      setStatus(`נטען: ${parseAss(text).length} שורות — מלא את עמודת "חלופי"`)
    }; inp.click()
  }

  function exportAss() {
    const header = assText.split('\n').filter(l => !l.startsWith('Dialogue:')).join('\n')
    const fmt = (t: number) => {
      const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60)
      const s = Math.floor(t % 60), cs = Math.round((t - Math.floor(t)) * 100)
      return `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${String(cs).padStart(2,'0')}`
    }
    const dlg = lines.map(l => {
      const txt = useAlt && l.alt.trim() ? l.alt : l.text
      return `Dialogue: 0,${fmt(l.start)},${fmt(l.end)},Karaoke,,0,0,0,,${txt}`
    }).join('\n')
    const blob = new Blob([header + '\n' + dlg], { type: 'text/plain;charset=utf-8' })
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'parody.ass'; a.click()
  }

  return (
    <div className="space-y-4 flex flex-col" dir="rtl">
      <div className="flex items-center gap-3">
        <Music4 className="w-6 h-6 text-accent" />
        <h1 className="text-xl font-bold">עורך פרודיה</h1>
      </div>
      <div className="card flex flex-wrap gap-2 py-2 items-center">
        <button onClick={loadAss} className="btn-ghost flex items-center gap-1.5 text-xs">
          <Upload className="w-3.5 h-3.5" /> טען ASS
        </button>
        <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
          <input type="checkbox" checked={useAlt} onChange={e => setUseAlt(e.target.checked)} className="accent-accent" />
          השתמש בחלופי
        </label>
        <button onClick={exportAss} disabled={!lines.length} className="btn-primary flex items-center gap-1.5 text-xs">
          <Download className="w-3.5 h-3.5" /> ייצא ASS
        </button>
        <span className="text-xs text-gray-500 mr-auto">{status}</span>
      </div>
      <div className="card overflow-auto flex-1">
        <table className="w-full text-sm border-collapse">
          <thead><tr className="text-gray-400 text-xs border-b border-border">
            <th className="text-right py-2 px-3 w-20">התחלה</th>
            <th className="text-right py-2 px-3 w-20">סיום</th>
            <th className="text-right py-2 px-3 w-1/2">מקור</th>
            <th className="text-right py-2 px-3 w-1/2 text-yellow-400">חלופי (פרודיה)</th>
          </tr></thead>
          <tbody>{lines.map((l, i) => (
            <tr key={i} className="border-b border-border/50 hover:bg-nav/30">
              <td className="py-1.5 px-3 font-mono text-xs text-gray-500">{l.start.toFixed(2)}</td>
              <td className="py-1.5 px-3 font-mono text-xs text-gray-500">{l.end.toFixed(2)}</td>
              <td className="py-1.5 px-3 text-gray-300">{l.text}</td>
              <td className="py-1.5 px-3">
                <input value={l.alt}
                       onChange={e => setLines(prev => prev.map((x, j) => j === i ? { ...x, alt: e.target.value } : x))}
                       placeholder="הכנס חלופי..."
                       className="bg-transparent border-b border-border/60 outline-none w-full text-yellow-300 text-sm placeholder-gray-600 focus:border-yellow-500" />
              </td>
            </tr>
          ))}</tbody>
        </table>
        {!lines.length && <p className="text-center text-gray-500 py-12">טען קובץ ASS להתחלה</p>}
      </div>
    </div>
  )
}
