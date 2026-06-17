import { useRef, useState } from 'react'
import { sseStream } from '@/api/client'
import { ListVideo, Plus, Trash2, Play, Square } from 'lucide-react'

type Status = 'pending' | 'running' | 'done' | 'failed'
interface Job { id: number; query: string; status: Status; progress: number; statusText: string }
let _id = 0

export default function Batch() {
  const [input, setInput]   = useState('')
  const [jobs, setJobs]     = useState<Job[]>([])
  const [lang, setLang]     = useState('he')
  const [save4, setSave4]   = useState(false)
  const [bidi, setBidi]     = useState(false)
  const [force, setForce]   = useState(false)
  const [running, setRunning] = useState(false)
  const abortRef              = useRef<AbortController | null>(null)

  function addJob() {
    const q = input.trim(); if (!q) return
    setJobs(prev => [...prev, { id: ++_id, query: q, status: 'pending', progress: 0, statusText: 'ממתין' }])
    setInput('')
  }

  function updateJob(id: number, patch: Partial<Job>) {
    setJobs(prev => prev.map(j => j.id === id ? { ...j, ...patch } : j))
  }

  async function startAll() {
    const pending = jobs.filter(j => j.status !== 'done')
    if (!pending.length) return
    abortRef.current = new AbortController()
    setRunning(true)
    for (const job of pending) {
      if (abortRef.current.signal.aborted) break
      updateJob(job.id, { status: 'running', statusText: 'מעבד...', progress: 0 })
      try {
        await sseStream(
          '/api/pipeline/stream',
          { url: job.query, lang, output_formats: ['ass', 'srt'], save_4_stems: save4, use_bidi: bidi, force },
          (raw) => {
            const ev = raw as { type: string; idx?: number; total?: number; text?: string; success?: boolean }
            if (ev.type === 'progress') {
              const pct = ev.total ? Math.round(100 * (ev.idx ?? 0) / ev.total) : 0
              updateJob(job.id, { progress: pct, statusText: ev.text?.slice(0, 50) ?? '' })
            } else if (ev.type === 'done') {
              updateJob(job.id, { status: ev.success ? 'done' : 'failed', statusText: ev.success ? '✅ הסתיים' : '❌ נכשל', progress: 100 })
            }
          },
          abortRef.current.signal,
        )
      } catch { updateJob(job.id, { status: 'failed', statusText: 'בוטל / שגיאה' }) }
    }
    setRunning(false)
  }

  const statusColor: Record<Status, string> = {
    pending: 'text-gray-400', running: 'text-blue-400', done: 'text-green-400', failed: 'text-red-400',
  }

  return (
    <div className="space-y-4" dir="rtl">
      <div className="flex items-center gap-3">
        <ListVideo className="w-6 h-6 text-accent" />
        <h1 className="text-xl font-bold">Batch — עיבוד מרובה</h1>
      </div>
      <div className="card flex gap-2 items-center">
        <input className="input-dark flex-1" placeholder="URL או שם שיר..." value={input}
               onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && addJob()} dir="ltr" />
        <button onClick={addJob} className="btn-primary flex items-center gap-1">
          <Plus className="w-4 h-4" /> הוסף
        </button>
      </div>
      <div className="card flex flex-wrap gap-4 items-center py-2">
        <label className="text-sm text-gray-400 flex items-center gap-2">
          שפה:
          <select value={lang} onChange={e => setLang(e.target.value)}
                  className="bg-nav border border-border rounded px-2 py-1 text-sm text-white">
            <option value="he">עברית</option><option value="en">אנגלית</option><option value="auto">אוטומטי</option>
          </select>
        </label>
        {[{label:'4 stems',v:save4,s:setSave4},{label:'BiDi',v:bidi,s:setBidi},{label:'Force',v:force,s:setForce}].map(({label,v,s})=>(
          <label key={label} className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
            <input type="checkbox" checked={v} onChange={e=>s(e.target.checked)} className="accent-accent"/> {label}
          </label>
        ))}
        <div className="mr-auto flex gap-2">
          <button onClick={startAll} disabled={running||!jobs.length} className="btn-primary flex items-center gap-1.5">
            <Play className="w-4 h-4" /> הפעל הכל
          </button>
          {running && <button onClick={()=>abortRef.current?.abort()} className="btn-ghost text-red-400 flex items-center gap-1.5">
            <Square className="w-4 h-4" /> עצור
          </button>}
          <button onClick={()=>!running&&setJobs([])} disabled={running} className="btn-ghost text-xs">נקה</button>
        </div>
      </div>
      <div className="card overflow-auto">
        {!jobs.length ? <p className="text-center text-gray-500 py-10">אין שירים בתור</p> : (
          <table className="w-full text-sm border-collapse">
            <thead><tr className="text-gray-400 text-xs border-b border-border">
              <th className="text-right py-2 px-3">שיר / URL</th>
              <th className="text-right py-2 px-3 w-28">סטטוס</th>
              <th className="text-right py-2 px-3 w-32">התקדמות</th>
              <th className="w-8"/>
            </tr></thead>
            <tbody>{jobs.map(j=>(
              <tr key={j.id} className="border-b border-border/40">
                <td className="py-2 px-3 font-mono text-xs text-gray-300 max-w-xs truncate" dir="ltr">{j.query}</td>
                <td className={`py-2 px-3 text-xs ${statusColor[j.status]}`}>{j.statusText}</td>
                <td className="py-2 px-3"><div className="progress-bar"><div className="progress-fill" style={{width:`${j.progress}%`}}/></div></td>
                <td className="py-2 px-1">
                  <button onClick={()=>setJobs(prev=>prev.filter(x=>x.id!==j.id))} disabled={running}
                          className="text-red-500 hover:text-red-400 disabled:opacity-30">
                    <Trash2 className="w-3.5 h-3.5"/>
                  </button>
                </td>
              </tr>
            ))}</tbody>
          </table>
        )}
      </div>
    </div>
  )
}
