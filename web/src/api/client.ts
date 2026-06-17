import { useAuthStore } from '@/stores/auth'

function authHeaders(): Record<string, string> {
  const token = useAuthStore.getState().token
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export async function apiGet<T>(path: string): Promise<T> {
  const r = await fetch(path, { headers: authHeaders() })
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json()
}

export async function apiPost<T = unknown>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body:    JSON.stringify(body),
  })
  if (!r.ok) {
    const text = await r.text()
    let detail = text
    try { detail = JSON.parse(text).detail ?? text } catch {}
    throw new Error(detail)
  }
  return r.json()
}

export async function apiDelete(path: string): Promise<void> {
  const r = await fetch(path, { method: 'DELETE', headers: authHeaders() })
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
}

export async function sseStream(
  path: string,
  body: unknown,
  onEvent: (ev: unknown) => void,
  signal?: AbortSignal,
): Promise<void> {
  const r = await fetch(path, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body:    JSON.stringify(body),
    signal,
  })
  if (!r.ok || !r.body) throw new Error(`${r.status}`)

  const reader  = r.body.getReader()
  const decoder = new TextDecoder()
  let   buf     = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const parts = buf.split('\n\n')
    buf = parts.pop() ?? ''
    for (const part of parts) {
      const line = part.trim()
      if (line.startsWith('data: ')) {
        try { onEvent(JSON.parse(line.slice(6))) } catch {}
      }
    }
  }
}
