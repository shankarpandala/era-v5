import { useEffect, useState } from 'react'

// The site drives dark mode via a `dark` class on <html> (see src/hooks/useTheme.js).
// Canvas visualizers cannot use Tailwind classes, so they watch that class.
export default function useIsDark() {
  const get = () => typeof document !== 'undefined' && document.documentElement.classList.contains('dark')
  const [dark, setDark] = useState(get)
  useEffect(() => {
    const obs = new MutationObserver(() => setDark(get()))
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    return () => obs.disconnect()
  }, [])
  return dark
}

// Shared palette for canvases (light / dark).
export function palette(dark) {
  return {
    bg: dark ? '#18181b' : '#ffffff',
    grid: dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
    skipped: dark ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.025)',
    text: dark ? '#e4e4e7' : '#27272a',
    muted: dark ? '#a1a1aa' : '#71717a',
    frame: dark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.15)',
    accent: '#3b82f6',
    accent2: '#8b5cf6',
    accent3: '#f59e0b',
    accent4: '#10b981',
    danger: '#ef4444',
    pink: '#d946ef',
  }
}
