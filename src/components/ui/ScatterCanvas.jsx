import { useEffect, useRef } from 'react'
import { makeGrid, drawHeatmap, toPixel, toPixelY, CLASS0, CLASS1 } from '../../lib/plotting.js'

// Renders a 2D scatter of labeled points, optionally over a model's decision
// boundary heatmap. Pass a `model` (anything with predictProba) to draw the
// boundary; omit it for plain data. `redrawKey` (e.g. the trainer tick) forces
// a repaint; while `running` we use a coarse grid and refine when it settles.
export default function ScatterCanvas({
  points,
  domain,
  model,
  redrawKey,
  running = false,
  size = 320,
  pointRadius = 3,
  title,
}) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = size * dpr
    canvas.height = size * dpr
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, size, size)

    const { min, max } = domain

    // Background
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, size, size)

    // Decision-boundary heatmap
    if (model) {
      const grid = running ? 44 : 150
      const gx = makeGrid(min, max, grid)
      const proba = model.predictProba(gx)
      drawHeatmap(ctx, proba, grid, size, 0.5)
    }

    // Points
    for (const p of points) {
      const px = toPixel(p.x, min, max, size)
      const py = toPixelY(p.y, min, max, size)
      const c = p.label === 0 ? CLASS0 : CLASS1
      ctx.beginPath()
      ctx.arc(px, py, pointRadius, 0, 2 * Math.PI)
      ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`
      ctx.fill()
      ctx.lineWidth = 1
      ctx.strokeStyle = 'rgba(255,255,255,0.85)'
      ctx.stroke()
    }

    // Frame
    ctx.strokeStyle = 'rgba(0,0,0,0.15)'
    ctx.lineWidth = 1
    ctx.strokeRect(0.5, 0.5, size - 1, size - 1)
  }, [points, domain, model, redrawKey, running, size, pointRadius])

  return (
    <figure className="m-0">
      {title && (
        <figcaption className="mb-1 text-center text-xs font-medium text-zinc-600 dark:text-zinc-300">
          {title}
        </figcaption>
      )}
      <canvas
        ref={canvasRef}
        style={{ width: size, height: size }}
        className="mx-auto block rounded-lg ring-1 ring-zinc-200 dark:ring-zinc-700"
      />
    </figure>
  )
}
