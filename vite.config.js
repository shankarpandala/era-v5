import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Project pages are served from https://<user>.github.io/era-v5/, so every
// asset URL must be prefixed with the repo name. Forgetting this is the #1
// cause of a blank GitHub Pages deploy.
export default defineConfig({
  base: '/era-v5/',
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      output: {
        // Keep three.js in its own chunk so the 2D demos load fast and the
        // 3D bundle is fetched only when RingLift3D is lazy-mounted.
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (id.includes('three') || id.includes('@react-three')) return 'vendor-three'
            if (id.includes('katex')) return 'vendor-katex'
            if (id.includes('react')) return 'vendor-react'
            return 'vendor'
          }
        },
      },
    },
  },
})
