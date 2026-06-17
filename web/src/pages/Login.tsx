import { FormEvent, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Mic2 } from 'lucide-react'
import { login } from '@/api/auth'

export default function Login() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError]       = useState('')
  const [loading, setLoading]   = useState(false)
  const navigate                = useNavigate()

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(''); setLoading(true)
    try {
      await login(username, password)
      navigate('/', { replace: true })
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'שגיאה')
    } finally { setLoading(false) }
  }

  return (
    <div className="min-h-screen bg-bg flex items-center justify-center p-4" dir="rtl">
      <div className="w-full max-w-sm">
        <div className="flex flex-col items-center mb-8">
          <div className="w-16 h-16 bg-accent rounded-2xl flex items-center justify-center mb-4 shadow-lg">
            <Mic2 className="w-9 h-9 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white">KaraokeStudio</h1>
          <p className="text-gray-400 text-sm mt-1">כניסה למערכת</p>
        </div>
        <form onSubmit={handleSubmit} className="card space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">שם משתמש</label>
            <input type="text" value={username} onChange={e => setUsername(e.target.value)}
                   className="input-dark" placeholder="admin" autoFocus required />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">סיסמה</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)}
                   className="input-dark" placeholder="••••••••" required />
          </div>
          {error && (
            <p className="text-red-400 text-sm bg-red-900/20 border border-red-800 rounded px-3 py-2">{error}</p>
          )}
          <button type="submit" disabled={loading} className="btn-primary w-full justify-center">
            {loading ? '⏳ מתחבר...' : 'כניסה'}
          </button>
        </form>
      </div>
    </div>
  )
}
