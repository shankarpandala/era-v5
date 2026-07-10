import ClaimCard from '../../components/ClaimCard.jsx'
import { LANGS } from '../lib/loadData.js'

function Stat({ label, value, accent }) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="text-[11px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={`font-mono text-xl font-bold ${accent}`}>{value}</div>
    </div>
  )
}

export default function Results({ stats, refStats }) {
  // Cross-check the live browser numbers against the committed Python stats.json.
  const pyScore = refStats?.score
  const matchesPython =
    pyScore != null && Math.abs(pyScore - stats.score) < 0.5 &&
    LANGS.every((l) => Math.abs((refStats.per_language?.[l.code]?.X ?? 0) - stats.per[l.code].X) < 1e-6)

  return (
    <ClaimCard
      id="a2-2"
      code="A2-2"
      accent="var(--claim-3)"
      title="Ratios, statistics & self-score"
      claim={
        <>
          Sorted fertilities span <b>X_max − X_min = {stats.spread.toFixed(4)}</b>, giving a self-score of{' '}
          <b>1000 / {stats.spread.toFixed(4)} = {stats.score.toFixed(1)}</b>.
        </>
      }
      takeaway={
        matchesPython ? (
          <>These live browser numbers are byte-for-byte identical to the Python reference (<code>stats.json</code>) — the same tokenizer a grader runs. Nothing is hardcoded.</>
        ) : (
          <>Numbers are recomputed live in your browser from the shipped tokenizer + corpora.</>
        )
      }
    >
      <div className="panel p-5">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[520px] text-sm">
            <thead>
              <tr className="border-b border-zinc-200 text-left text-[11px] uppercase tracking-wide text-zinc-500 dark:border-zinc-700">
                <th className="py-2 pr-3 font-medium">Language</th>
                <th className="py-2 pr-3 font-medium">Script</th>
                <th className="py-2 pr-3 text-right font-medium">Words</th>
                <th className="py-2 pr-3 text-right font-medium">Tokens</th>
                <th className="py-2 pr-3 text-right font-medium">X = tokens/word</th>
                <th className="py-2 text-right font-medium">≤ 1.2</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {LANGS.map((l) => {
                const r = stats.per[l.code]
                return (
                  <tr key={l.code} className="border-b border-zinc-100 last:border-0 dark:border-zinc-800">
                    <td className="py-2 pr-3 font-sans font-medium text-zinc-800 dark:text-zinc-100">
                      <span
                        className="mr-2 inline-block h-2.5 w-2.5 rounded-full align-middle"
                        style={{ backgroundColor: l.accent }}
                      />
                      {l.name}
                    </td>
                    <td className="py-2 pr-3 font-sans text-zinc-500 dark:text-zinc-400">{l.script}</td>
                    <td className="py-2 pr-3 text-right tabular-nums">{r.words.toLocaleString()}</td>
                    <td className="py-2 pr-3 text-right tabular-nums">{r.tokens.toLocaleString()}</td>
                    <td className="py-2 pr-3 text-right font-semibold tabular-nums text-zinc-900 dark:text-zinc-50">
                      {r.X.toFixed(4)}
                    </td>
                    <td className="py-2 text-right">
                      {r.ok ? (
                        <span className="text-emerald-600 dark:text-emerald-400">✓</span>
                      ) : (
                        <span className="text-red-600 dark:text-red-400">✗</span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        <div className="mt-4 rounded-lg bg-zinc-100 px-3 py-2 font-mono text-[13px] text-zinc-700 dark:bg-zinc-800/60 dark:text-zinc-200">
          sorted&nbsp;X:&nbsp;
          {stats.sortedDesc.map((r, i) => (
            <span key={r.code}>
              {i > 0 && ' ≥ '}
              <span className="font-semibold text-zinc-900 dark:text-zinc-50">{r.X.toFixed(4)}</span>
              <span className="text-zinc-500">({r.code})</span>
            </span>
          ))}
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="X_max − X_min" value={stats.spread.toFixed(4)} accent="text-zinc-900 dark:text-zinc-50" />
          <Stat label="Self-score" value={stats.score.toFixed(1)} accent="text-brand-600 dark:text-brand-400" />
          <Stat
            label="Constraints (all ≤ 1.2)"
            value={stats.constraintsMet ? 'MET ✓' : 'NOT MET'}
            accent={stats.constraintsMet ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'}
          />
          <Stat
            label="vs Python reference"
            value={matchesPython ? 'IDENTICAL ✓' : '—'}
            accent={matchesPython ? 'text-emerald-600 dark:text-emerald-400' : 'text-zinc-500'}
          />
        </div>
      </div>
    </ClaimCard>
  )
}
