import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    allowedHosts: ['.loca.lt'], 
  },
})

// to host on localtunnel, run in a seperate terminal: lt --port 5173
// go to that website and click password on the bottom