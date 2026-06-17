import { apiPost } from './client'
import { useAuthStore } from '@/stores/auth'

interface LoginResponse { access_token: string; token_type: string }
interface MeResponse    { id: number; username: string; role: string }

export async function login(username: string, password: string): Promise<void> {
  const res = await apiPost<LoginResponse>('/api/auth/login', { username, password })
  const me  = await fetch('/api/auth/me', {
    headers: { Authorization: `Bearer ${res.access_token}` },
  }).then(r => r.json() as Promise<MeResponse>)
  useAuthStore.getState().setAuth(res.access_token, me.username, me.role)
}

export function logout(): void {
  useAuthStore.getState().clearAuth()
  window.location.href = '/login'
}
