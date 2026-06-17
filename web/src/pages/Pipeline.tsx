import { useRef, useState } from 'react'
import { sseStream } from '@/api/client'
import { Play, Square, Zap } from 'lucide-react'

interface SseEvent {
  type: 'progress' | 'log' | 'done'
  idx?: number; total?: number; text?: string
  success?: boolean; relative?: string
}

export default function Pipeline() {
  const [url, setUrl]           = useState('')
  const [lang, setLang]         = useState('he')
  const [save4, setSave4]       = useState(false)
  const [bidi, setBidi]         = useState(false)
  const [force, setForce]       = useState(false)
  const [running, setRunning]   = useState(false)
  const [progress, setProgress] = useState(0)
  const [progText, setProgText] = useState('')
  const [logs, setLogs]         = useState<string[]>([])
  const [output, setOutput]     = useState<string | null>(null)
  const abortRef                = useRef<AbortController | null>(null)
  const logsEndRef              = useRef<HTMLDivElement>(null)

  function appendLog(line: string) {
    setLogs(prev => {
      const next = [...prev, line]
      setTimeout(() => logsEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 50)
      return next
    })
  }

  async function start() {
    if (!url.trim()) return
    abortRef.current = new AbortController()
    setRunning(true); setProgress(0); setProgText(''); setLogs([]); setOutput(null)
    try {
      await sseStream(
        '/api/pipeline/stream',
        { url, lang, output_formats: ['ass', 'srt'], save_4_stems: save4, use_bidi: bidi, force },
        (raw) => {
          const ev = raw as SseEvent
          if (ev.type === 'progress') {
            const pct = ev.total ? Math.round(100 * (ev.idx ?? 0) / ev.total) : 0
            setProgress(pct)
            setProgText(`[${ev.idx}/${ev.total}] ${ev.text ?? ''}`)
            appendLog(`[${ev.idx}/${ev.total}] ${ev.text ?? ''}`)
          } else if (ev.type === 'log') {
            appendLog(ev.text ?? '')
          } else if (ev.type === 'done') {
            setProgress(100)
            appendLog(ev.success ? '✅ הסתיים בהצלחה' : '❌ נכשל')
            if (ev.success && ev.relative) setOutput(`/api/files/${ev.relative}`)
          }
        },
        abortRef.current.signal,
      )
    } catch (err) {
      if ((err as Error).name !== 'AbortError')
        appendLog(`❌ שגיאה: ${(err as Error).message}`)
    } finally { setRunning(false) }
  }

  return (
    <div className="max-w-2xl space-y-6" dir="rtl">
      <div className="flex items-center gap-3">
        <Zap className="w-6 h-6 text-accent" />
        <h1 className="text-xl font-bold">Pipeline — קריוקי אוטומטי</h1>
      </div>
      <div className="card space-y-4">
        <div>
          <label className="block text-sm text-gray-400 mb-1">URL / שם שיר ב-YouTube</label>
          <input className="input-dark" placeholder="https://youtu.be/..." value={url}
                 onChange={e => setUrl(e.target.value)}
                 onKeyDown={e => e.key === 'Enter' && !running && start()} dir="ltr" />
        </div>
        <div className="flex flex-wrap gap-6 items-center">
          <label className="text-sm text-gray-400 flex items-center gap-2">
            שפה:
            <select value={lang} onChange={e => setLang(e.target.value)}
                    className="bg-nav border border-border rounded px-2 py-1 text-sm text-white">
              <option value="he">עברית</option>
              <option value="en">אנגלית</option>
              <option value="ar">ערבית</option>
              <option value="auto">אוטומטי</option>
            </select>
          </label>
          {[{label:'4 stems',val:save4,set:setSave4},{label:'BiDi RTL',val:bidi,set:setBidi},{label:'Force',val:force,set:setForce}].map(({label,val,set})=>(
            <label key={label} className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
              <input type="checkbox" checked={val} onChange={e=>set(e.target.checked)} className="accent-accent"/>
              {label}
            </label>
          ))}
        </div>
        <div className="flex gap-2">
          <button onClick={start} disabled={running || !url} className="btn-primary flex items-center gap-2">
            <Play className="w-4 h-4" /> התחל
          </button>
          {running && (
            <button onClick={() => abortRef.current?.abort()} className="btn-ghost flex items-center gap-2 text-red-400">
              <Square className="w-4 h-4" /> עצור
            </button>
          )}
        </div>
      </div>
      {(running || progress > 0) && (
        <div className="card space-y-2">
          <div className="progress-bar"><div className="progress-fill" style={{ width: `${progress}%` }} /></div>
          <p className="text-xs text-gray-400 font-mono">{progText}</p>
        </div>
      )}
      {output && (
        <div className="card">
          <p className="text-sm text-gray-400 mb-2">הסרטון מוכן:</p>
          <a href={output} target="_blank" rel="noopener noreferrer"
             className="text-accent hover:underline text-sm break-all" dir="ltr">{output}</a>
          <video src={output} controls className="mt-3 w-full rounded-md max-h-80" />
        </div>
      )}
      {logs.length > 0 && (
        <div className="card bg-black/40">
          <p className="text-xs text-gray-500 mb-2">לוג</p>
          <div className="max-h-48 overflow-y-auto space-y-0.5">
            {logs.map((l, i) => <p key={i} className="text-xs font-mono text-green-400 leading-5">{l}</p>)}
            <div ref={logsEndRef} />
          </div>
        </div>
      )}
    </div>
  )
}
