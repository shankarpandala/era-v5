import { useEffect, useRef, useState } from 'react'
import useIsDark, { palette } from '../lib/useIsDark.js'

// Draws an n×m attention grid. `cell(i, j)` returns null (not computed / skipped)
// or { w: weight in [0,1], cat?: category key }. Categories tint the cell.
// Hovering a row reports it back via onHover.
const CAT_COLORS = {
  default: [59, 130, 246], // blue
  window: [245, 158, 11], // amber
  sink: [239, 68, 68], // red
  global: [16, 185, 129], // emerald
  selected: [217, 70, 239], // pink
  compressed: [139, 92, 246], // violet
  random: [6, 182, 212], // cyan
  cached: [113, 113, 122], // zinc
  dev0: [59, 130, 246],
  dev1: [16, 185, 129],
  dev2: [245, 158, 11],
  dev3: [239, 68, 68],
}

export default function MaskCanvas({ n, m = n, cell, size = 300, onHover, rowLabel = 'query', colLabel = 'key', ariaLabel }) {
  const ref = useRef(null)
  const dark = useIsDark()
  const [hover, setHover] = useState(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = size * dpr
    canvas.height = size * dpr
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const p = palette(dark)
    ctx.fillStyle = p.bg
    ctx.fillRect(0, 0, size, size)
    const cw = size / m
    const ch = size / n
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        const c = cell(i, j)
        const x = j * cw
        const y = i * ch
        if (!c) {
          ctx.fillStyle = p.skipped
          ctx.fillRect(x, y, cw, ch)
          continue
        }
        const rgb = CAT_COLORS[c.cat || 'default'] || CAT_COLORS.default
        const w = Math.max(0, Math.min(1, c.w ?? 0))
        // weight → alpha; keep a floor so allowed-but-tiny cells stay visible
        const a = 0.12 + 0.88 * Math.pow(w, 0.6)
        ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`
        ctx.fillRect(x, y, cw, ch)
      }
      if (hover === i) {
        ctx.fillStyle = dark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.08)'
        ctx.fillRect(0, i * ch, size, ch)
      }
    }
    // grid lines when cells are big enough
    if (cw >= 6) {
      ctx.strokeStyle = p.grid
      ctx.lineWidth = 1
      for (let j = 0; j <= m; j++) {
        ctx.beginPath()
        ctx.moveTo(j * cw + 0.5, 0)
        ctx.lineTo(j * cw + 0.5, size)
        ctx.stroke()
      }
      for (let i = 0; i <= n; i++) {
        ctx.beginPath()
        ctx.moveTo(0, i * ch + 0.5)
        ctx.lineTo(size, i * ch + 0.5)
        ctx.stroke()
      }
    }
    ctx.strokeStyle = p.frame
    ctx.strokeRect(0.5, 0.5, size - 1, size - 1)
  }, [n, m, cell, size, dark, hover])

  const onMove = (e) => {
    const r = ref.current.getBoundingClientRect()
    const i = Math.floor(((e.clientY - r.top) / r.height) * n)
    const row = i >= 0 && i < n ? i : null
    setHover(row)
    onHover?.(row)
  }
  const onLeave = () => {
    setHover(null)
    onHover?.(null)
  }

  return (
    <div className="inline-block">
      <div className="mb-1 flex items-center justify-between font-mono text-[10px] text-zinc-400">
        <span>{rowLabel} ↓</span>
        <span>{colLabel} →</span>
      </div>
      <canvas
        ref={ref}
        style={{ width: size, height: size, maxWidth: '100%' }}
        className="block rounded-md"
        role="img"
        aria-label={ariaLabel || 'attention pattern'}
        onMouseMove={onMove}
        onMouseLeave={onLeave}
      />
    </div>
  )
}

export function Legend({ items }) {
  return (
    <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-zinc-500 dark:text-zinc-400">
      {items.map(([cat, label]) => {
        const rgb = CAT_COLORS[cat] || CAT_COLORS.default
        return (
          <span key={cat} className="inline-flex items-center gap-1">
            <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: `rgb(${rgb.join(',')})` }} />
            {label}
          </span>
        )
      })}
    </div>
  )
}
