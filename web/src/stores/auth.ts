import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface AuthState {
  token:      string | null
  username:   string | null
  role:       string | null
  setAuth:    (token: string, username: string, role: string) => void
  clearAuth:  () => void
  isLoggedIn: () => boolean
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token:     null,
      username:  null,
      role:      null,
      setAuth:   (token, username, role) => set({ token, username, role }),
      clearAuth: () => set({ token: null, username: null, role: null }),
      isLoggedIn: () => !!get().token,
    }),
    { name: 'karaoke-auth' },
  ),
)
