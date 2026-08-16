import { useState } from 'react'
import { NODES, SCENARIOS, fmtDate } from '../data.js'
import { PICK_GLYPH, Rich } from './Bits.jsx'

// Rows = mechanisms, columns = the four scenarios; ✓ / △ / ✗ from each node's
// pickWhen. Hover a cell for the one-line reason. Same data as the cards.

export default function DecisionMatrix() {
  const [onlyList, setOnlyList] = useState(true)
  const rows = NODES.filter((m) => m.pickWhen && (!onlyList || m.instructorList))
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <button
          type="button"
          onClick={() => setOnlyList((v) => !v)}
          className="rounded-md border border-zinc-300 px-2 py-1 text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800"
        >
          {onlyList ? 'show every node' : 'show the instructor’s list only'}
        </button>
        <span className="text-zinc-500">
          {Object.entries(PICK_GLYPH)
            .filter(([k]) => k !== 'na')
            .map(([k, g]) => `${g.glyph} ${g.label}`)
            .join(' · ')}{' '}
          · hover a cell for the reason
        </span>
      </div>
      <div className="overflow-x-auto rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
        <table className="w-full text-xs" style={{ minWidth: 760 }}>
          <thead className="sticky top-0 bg-white dark:bg-zinc-900">
            <tr className="border-b border-zinc-200 text-left text-[10px] uppercase tracking-wide text-zinc-500 dark:border-zinc-700">
              <th className="px-3 py-2">mechanism</th>
              <th className="px-3 py-2">launched</th>
              {SCENARIOS.map((s) => (
                <th key={s.key} className="px-3 py-2 text-center">
                  {s.label}
                </th>
              ))}
              <th className="px-3 py-2">verdict</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => (
              <tr key={m.id} className="border-b border-zinc-100 align-top last:border-0 dark:border-zinc-800">
                <td className="px-3 py-2">
                  <a href={`#m-${m.id}`} className="font-medium text-zinc-800 hover:underline dark:text-zinc-100">
                    {m.short}
                  </a>
                </td>
                <td className="whitespace-nowrap px-3 py-2 font-mono text-zinc-500">{fmtDate(m.date, m.datePrecision)}</td>
                {SCENARIOS.map((s) => {
                  const cell = m.pickWhen[s.key] || { v: 'na' }
                  const g = PICK_GLYPH[cell.v] || PICK_GLYPH.na
                  return (
                    <td key={s.key} className="px-3 py-2 text-center" title={cell.why || g.label}>
                      <span
                        className={`inline-block rounded-md px-2 py-0.5 font-mono font-bold ${g.cls}`}
                        role="img"
                        aria-label={`${s.label}: ${g.label}${cell.why ? ' — ' + cell.why : ''}`}
                      >
                        {g.glyph}
                      </span>
                    </td>
                  )
                })}
                <td className="max-w-md px-3 py-2 text-zinc-600 dark:text-zinc-300"><Rich text={m.pickWhen.verdict} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
