import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: './',
  server: { proxy: { '/api': `http://127.0.0.1:${process.env.CODING_AGENT_BRIDGE_PORT || 8787}` } },
  preview: { proxy: { '/api': `http://127.0.0.1:${process.env.CODING_AGENT_BRIDGE_PORT || 8787}` } },
})
