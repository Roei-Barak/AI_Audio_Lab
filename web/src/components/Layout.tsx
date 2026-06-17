import { NavLink } from 'react-router-dom'
import { Zap, FileVideo, Music4, ListVideo, Settings, LogOut } from 'lucide-react'
import { useAuthStore } from '@/stores/auth'
import { logout } from '@/api/auth'

const nav = [
  { to: '/pipeline', icon: Zap,       label: 'Pipeline' },
  { to: '/editor',   icon: FileVideo,  label: 'עורך'     },
  { to: '/parody',   icon: Music4,     label: 'פרודיה'   },
  { to: '/batch',    icon: ListVideo,  label: 'Batch'    },
  { to: '/settings', icon: Settings,   label: 'הגדרות'   },
]

export default function Layout({ children }: { children: React.ReactNode }) {
  const username = useAuthStore(s => s.username)

  return (
    <div className="flex h-screen overflow-hidden">
      <aside className="w-52 bg-nav border-l border-border flex flex-col shrink-0">
        <div className="px-4 py-4 border-b border-border">
          <span className="text-accent font-bold text-lg">🎤 KaraokeStudio</span>
        </div>
        <nav className="flex-1 py-2">
          {nav.map(({ to, icon: Icon, label }) => (
            <NavLink key={to} to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-2.5 text-sm transition-colors
                 ${isActive
                   ? 'bg-accent/20 text-accent border-r-2 border-accent'
                   : 'text-gray-400 hover:text-white hover:bg-surface'}`
              }>
              <Icon className="w-4 h-4" />{label}
            </NavLink>
          ))}
        </nav>
        <div className="px-4 py-3 border-t border-border flex items-center justify-between">
          <span className="text-xs text-gray-500 truncate">{username}</span>
          <button onClick={logout} className="text-gray-500 hover:text-red-400 transition-colors">
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </aside>
      <main className="flex-1 overflow-auto p-6 bg-bg">{children}</main>
    </div>
  )
}
