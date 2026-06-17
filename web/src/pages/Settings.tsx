import { useEffect, useState } from 'react'
import { Settings as SettingsIcon, Plus, Trash2, KeyRound, User } from 'lucide-react'
import { apiGet, apiPost, apiDelete } from '@/api/client'
import { useAuthStore } from '@/stores/auth'

interface UserRecord { id: number; username: string; role: 'admin' | 'user'; created_at: string }

export default function Settings() {
  const { role, username } = useAuthStore()
  const isAdmin = role === 'admin'
  const [users, setUsers]     = useState<UserRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [newUser, setNewUser] = useState({ username: '', password: '', role: 'user' as 'admin' | 'user' })
  const [addError, setAddError] = useState(''); const [addOk, setAddOk] = useState('')
  const [pwForm, setPwForm]   = useState({ current: '', next: '', confirm: '' })
  const [pwError, setPwError] = useState(''); const [pwOk, setPwOk] = useState('')

  useEffect(() => { if (isAdmin) fetchUsers() }, [isAdmin])

  async function fetchUsers() {
    setLoading(true)
    try { setUsers(await apiGet<UserRecord[]>('/api/auth/users')) }
    finally { setLoading(false) }
  }

  async function addUser(e: React.FormEvent) {
    e.preventDefault(); setAddError(''); setAddOk('')
    if (!newUser.username.trim() || !newUser.password.trim()) { setAddError('שם משתמש וסיסמה נדרשים'); return }
    try {
      await apiPost('/api/auth/users', newUser)
      setAddOk(`משתמש "${newUser.username}" נוצר`); setNewUser({ username: '', password: '', role: 'user' }); fetchUsers()
    } catch (err) { setAddError(err instanceof Error ? err.message : 'שגיאה') }
  }

  async function deleteUser(id: number, name: string) {
    if (!confirm(`למחוק "${name}"?`)) return
    try { await apiDelete(`/api/auth/users/${id}`); fetchUsers() }
    catch (err) { alert(err instanceof Error ? err.message : 'שגיאה') }
  }

  async function changePassword(e: React.FormEvent) {
    e.preventDefault(); setPwError(''); setPwOk('')
    if (pwForm.next !== pwForm.confirm) { setPwError('הסיסמאות אינן תואמות'); return }
    if (pwForm.next.length < 6) { setPwError('מינימום 6 תווים'); return }
    try {
      await apiPost('/api/auth/change-password', { current_password: pwForm.current, new_password: pwForm.next })
      setPwOk('הסיסמה שונתה'); setPwForm({ current: '', next: '', confirm: '' })
    } catch (err) { setPwError(err instanceof Error ? err.message : 'שגיאה') }
  }

  return (
    <div className="space-y-6 max-w-2xl" dir="rtl">
      <div className="flex items-center gap-3">
        <SettingsIcon className="w-6 h-6 text-accent" />
        <h1 className="text-xl font-bold">הגדרות</h1>
      </div>
      <section className="card space-y-4">
        <div className="flex items-center gap-2 text-sm font-semibold text-gray-300 border-b border-border pb-2">
          <KeyRound className="w-4 h-4 text-accent" /> שינוי סיסמה ({username})
        </div>
        <form onSubmit={changePassword} className="space-y-3">
          {(['current', 'next', 'confirm'] as const).map(k => (
            <div key={k}>
              <label className="block text-xs text-gray-400 mb-1">
                {k === 'current' ? 'סיסמה נוכחית' : k === 'next' ? 'סיסמה חדשה' : 'אימות סיסמה חדשה'}
              </label>
              <input type="password" value={pwForm[k]}
                     onChange={e => setPwForm(p => ({ ...p, [k]: e.target.value }))}
                     className="input-dark" required />
            </div>
          ))}
          {pwError && <p className="text-red-400 text-xs">{pwError}</p>}
          {pwOk    && <p className="text-green-400 text-xs">{pwOk}</p>}
          <button type="submit" className="btn-primary text-sm">שמור סיסמה</button>
        </form>
      </section>
      {isAdmin && <>
        <section className="card space-y-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-gray-300 border-b border-border pb-2">
            <User className="w-4 h-4 text-accent" /> משתמשים
          </div>
          {loading ? <p className="text-gray-500 text-sm">טוען...</p> : (
            <table className="w-full text-sm border-collapse">
              <thead><tr className="text-gray-400 text-xs border-b border-border">
                <th className="text-right py-1.5 px-2">שם משתמש</th>
                <th className="text-right py-1.5 px-2">תפקיד</th>
                <th className="text-right py-1.5 px-2">נוצר</th>
                <th className="w-8" />
              </tr></thead>
              <tbody>{users.map(u => (
                <tr key={u.id} className="border-b border-border/40">
                  <td className="py-1.5 px-2 text-gray-200">{u.username}</td>
                  <td className="py-1.5 px-2">
                    <span className={`text-xs px-1.5 py-0.5 rounded ${u.role === 'admin' ? 'bg-accent/20 text-accent' : 'bg-gray-700 text-gray-300'}`}>
                      {u.role === 'admin' ? 'מנהל' : 'משתמש'}
                    </span>
                  </td>
                  <td className="py-1.5 px-2 text-xs text-gray-500">{new Date(u.created_at).toLocaleDateString('he-IL')}</td>
                  <td className="py-1.5 px-1">
                    <button onClick={() => deleteUser(u.id, u.username)} disabled={u.username === username}
                            className="text-red-500 hover:text-red-400 disabled:opacity-20">
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </td>
                </tr>
              ))}</tbody>
            </table>
          )}
        </section>
        <section className="card space-y-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-gray-300 border-b border-border pb-2">
            <Plus className="w-4 h-4 text-accent" /> הוסף משתמש
          </div>
          <form onSubmit={addUser} className="flex flex-wrap gap-2 items-end">
            <div className="flex-1 min-w-32">
              <label className="block text-xs text-gray-400 mb-1">שם משתמש</label>
              <input className="input-dark" dir="ltr" value={newUser.username}
                     onChange={e => setNewUser(p => ({ ...p, username: e.target.value }))} />
            </div>
            <div className="flex-1 min-w-32">
              <label className="block text-xs text-gray-400 mb-1">סיסמה</label>
              <input type="password" className="input-dark" value={newUser.password}
                     onChange={e => setNewUser(p => ({ ...p, password: e.target.value }))} />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">תפקיד</label>
              <select value={newUser.role} onChange={e => setNewUser(p => ({ ...p, role: e.target.value as 'admin' | 'user' }))}
                      className="bg-nav border border-border rounded px-2 py-2 text-sm text-white h-[38px]">
                <option value="user">משתמש</option><option value="admin">מנהל</option>
              </select>
            </div>
            <button type="submit" className="btn-primary flex items-center gap-1.5 text-sm">
              <Plus className="w-4 h-4" /> הוסף
            </button>
          </form>
          {addError && <p className="text-red-400 text-xs">{addError}</p>}
          {addOk    && <p className="text-green-400 text-xs">{addOk}</p>}
        </section>
      </>}
    </div>
  )
}
