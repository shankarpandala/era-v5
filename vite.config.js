import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Project pages are served from https://<user>.github.io/era-v5/, so every
// asset URL must be prefixed with the repo name. Forgetting this is the #1
// cause of a blank GitHub Pages deploy.
// No manualChunks: hand-splitting the React ecosystem across vendor chunks
// caused a chunk-init-order crash ("Cannot read properties of undefined
// (reading 'useLayoutEffect')"). Rollup's default chunking handles shared React
// correctly, and three.js still loads lazily because RingLift3D is its only
// importer and is brought in via a dynamic import().
export default defineConfig({
  base: '/era-v5/',
  plugins: [react(), tailwindcss()],
  // Multi-page build: Assignment 1 at dist/index.html (served /era-v5/), and
  // Assignment 2 at dist/tokenizer/index.html (served /era-v5/tokenizer/).
  // Each is a real static HTML file, so no router / 404.html fallback is needed.
  build: {
    rollupOptions: {
      input: {
        main: 'index.html',
        tokenizer: 'tokenizer/index.html',
        dataCollection: 'data-collection/index.html',
        dataCleaning: 'data-cleaning/index.html',
        assignment7: 'assignment-7/index.html',
        assignment8: 'assignment-8/index.html',
      },
    },
  },
})
