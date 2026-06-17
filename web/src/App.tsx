import { Navigate, Route, BrowserRouter as Router, Routes } from 'react-router-dom'
import { useAuthStore } from '@/stores/auth'
import Layout   from '@/components/Layout'
import Login    from '@/pages/Login'
import Pipeline from '@/pages/Pipeline'
import Editor   from '@/pages/Editor'
import Parody   from '@/pages/Parody'
import Batch    from '@/pages/Batch'
import Settings from '@/pages/Settings'

function AuthGuard({ children }: { children: React.ReactNode }) {
  const logged = useAuthStore(s => s.isLoggedIn())
  return logged ? <>{children}</> : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/*" element={
          <AuthGuard>
            <Layout>
              <Routes>
                <Route path="/"         element={<Navigate to="/pipeline" replace />} />
                <Route path="/pipeline" element={<Pipeline />} />
                <Route path="/editor"   element={<Editor />} />
                <Route path="/parody"   element={<Parody />} />
                <Route path="/batch"    element={<Batch />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </Layout>
          </AuthGuard>
        } />
      </Routes>
    </Router>
  )
}
