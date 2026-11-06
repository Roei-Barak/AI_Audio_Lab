import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // `npm run dev` talks to a local Lyra server (cd server && uvicorn lyra_server.app:app --port 8443)
  server: { proxy: { '/api': 'http://localhost:8443' } },
  base: './',
  test: { environment: 'jsdom', globals: true },
});
