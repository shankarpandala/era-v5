import { useMemo, useState } from 'react'
import { ERAS, NODES, fmtDate } from '../data.js'

// A one-line overview of the whole timeline: era bands as coloured spans, one
// dot per node (bigger for major nodes). Click a dot to jump to its card.
const T0 = Date.UTC(2014, 6, 1)
const T1 = Date.UTC(2026, 8, 1)

function x(iso, W) {
  const [y, m, d] = iso.split('-').map(Number)
  const t = Date.UTC(y, (m || 1) - 1, d || 1)
  return ((t - T0) / (T1 - T0)) * W
}

export default function TimelineStrip({ onJump, visibleIds }) {
  const W = 1000
  const H = 92
  const [hover, setHover] = useState(null)
  const years = useMemo(() => Array.from({ length: 13 }, (_, i) => 2014 + i), [])
  const visible = visibleIds ? new Set(visibleIds) : null

  return (
    <div className="panel overflow-x-auto p-3">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full min-w-[640px]" role="img" aria-label="timeline overview 2014 to 2026">
        {ERAS.map((e) => {
          const x0 = x(e.from, W)
          const x1 = x(e.to, W)
          return (
            <g key={e.id}>
              <rect x={x0} y={30} width={Math.max(2, x1 - x0)} height={30} fill={e.color} opacity={0.14} rx={3} />
              <text x={x0 + 3} y={e.id % 2 ? 16 : 26} fontSize={9} className="fill-zinc-500 font-mono">
                {e.short}
              </text>
            </g>
          )
        })}
        {years.map((y) => {
          const xx = x(`${y}-01-01`, W)
          return (
            <g key={y}>
              <line x1={xx} x2={xx} y1={30} y2={62} className="stroke-zinc-300 dark:stroke-zinc-700" strokeWidth={1} />
              <text x={xx + 2} y={74} fontSize={9} className="fill-zinc-500 font-mono">
                {y}
              </text>
            </g>
          )
        })}
        {NODES.map((m) => {
          const cx = x(m.date, W)
          const major = m.tier === 'major'
          const dim = visible && !visible.has(m.id)
          const era = ERAS.find((e) => e.id === m.era)
          return (
            <g key={m.id} className="cursor-pointer" onClick={() => onJump?.(m.id)} onMouseEnter={() => setHover(m)} onMouseLeave={() => setHover(null)}>
              <circle
                cx={cx}
                cy={major ? 41 : 51}
                r={major ? 4.2 : 2.6}
                fill={dim ? '#a1a1aa' : era?.color || '#3b82f6'}
                opacity={dim ? 0.3 : m.instructorList ? 1 : 0.7}
                stroke={m.instructorList ? (dim ? 'none' : 'currentColor') : 'none'}
                strokeWidth={m.instructorList ? 1 : 0}
                className="text-zinc-900 dark:text-zinc-50"
              />
              <title>{`${fmtDate(m.date, m.datePrecision)} — ${m.name}`}</title>
            </g>
          )
        })}
        {hover && (
          <text x={Math.min(x(hover.date, W), W - 260)} y={88} fontSize={10} className="fill-zinc-700 dark:fill-zinc-200 font-mono">
            {fmtDate(hover.date, hover.datePrecision)} · {hover.short}
          </text>
        )}
      </svg>
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-zinc-500 dark:text-zinc-400">
        <span>● big dot = major node · small dot = minor node · outlined = on the instructor's list · click a dot to jump</span>
      </div>
    </div>
  )
}
