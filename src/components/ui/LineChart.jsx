// Tiny dependency-free SVG line chart for loss curves and the gap-vs-size plot.
// series: [{ label, color, data: [{x, y}], dashed? }]
export default function LineChart({
  series,
  width = 340,
  height = 220,
  xLabel,
  yLabel,
  yMin,
  yMax,
  xMin,
  xMax,
  logX = false,
  yTicks = 4,
}) {
  const pad = { l: 44, r: 12, t: 12, b: 34 }
  const iw = width - pad.l - pad.r
  const ih = height - pad.t - pad.b

  const allX = series.flatMap((s) => s.data.map((d) => d.x))
  const allY = series.flatMap((s) => s.data.map((d) => d.y))
  const tx = (x) => (logX ? Math.log10(Math.max(x, 1e-9)) : x)

  const x0 = tx(xMin ?? Math.min(...allX, 0))
  const x1 = tx(xMax ?? Math.max(...allX, 1))
  const y0 = yMin ?? Math.min(...allY, 0)
  const y1 = yMax ?? Math.max(...allY, 1)

  const sx = (x) => pad.l + ((tx(x) - x0) / (x1 - x0 || 1)) * iw
  const sy = (y) => pad.t + ih - ((y - y0) / (y1 - y0 || 1)) * ih

  const yTickVals = Array.from({ length: yTicks + 1 }, (_, i) => y0 + ((y1 - y0) * i) / yTicks)

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full"
      role="img"
      aria-label={`${yLabel || 'value'} vs ${xLabel || 'x'}`}
    >
      {/* y grid + ticks */}
      {yTickVals.map((v, i) => (
        <g key={i}>
          <line
            x1={pad.l}
            x2={width - pad.r}
            y1={sy(v)}
            y2={sy(v)}
            className="stroke-zinc-200 dark:stroke-zinc-700"
            strokeWidth="1"
          />
          <text x={pad.l - 6} y={sy(v) + 3} textAnchor="end" className="fill-zinc-500 text-[9px]">
            {v.toFixed(2)}
          </text>
        </g>
      ))}

      {/* series */}
      {series.map((s, si) => {
        if (!s.data.length) return null
        const path = s.data.map((d, i) => `${i === 0 ? 'M' : 'L'}${sx(d.x).toFixed(1)},${sy(d.y).toFixed(1)}`).join(' ')
        return (
          <g key={si}>
            <path
              d={path}
              fill="none"
              stroke={s.color}
              strokeWidth="2"
              strokeDasharray={s.dashed ? '5 4' : undefined}
              strokeLinejoin="round"
            />
            {s.markers &&
              s.data.map((d, i) => <circle key={i} cx={sx(d.x)} cy={sy(d.y)} r="3" fill={s.color} />)}
          </g>
        )
      })}

      {/* axis labels */}
      {xLabel && (
        <text x={pad.l + iw / 2} y={height - 6} textAnchor="middle" className="fill-zinc-500 text-[10px]">
          {xLabel}
        </text>
      )}
      {yLabel && (
        <text
          x={12}
          y={pad.t + ih / 2}
          textAnchor="middle"
          transform={`rotate(-90 12 ${pad.t + ih / 2})`}
          className="fill-zinc-500 text-[10px]"
        >
          {yLabel}
        </text>
      )}

      {/* legend */}
      <g>
        {series.map((s, si) => (
          <g key={si} transform={`translate(${pad.l + 6 + si * 96}, ${pad.t + 4})`}>
            <line x1="0" x2="16" y1="0" y2="0" stroke={s.color} strokeWidth="2" strokeDasharray={s.dashed ? '4 3' : undefined} />
            <text x="20" y="3" className="fill-zinc-600 dark:fill-zinc-300 text-[10px]">
              {s.label}
            </text>
          </g>
        ))}
      </g>
    </svg>
  )
}
