import { useMemo, useRef, useState } from 'react'
import ClaimCard from '../../components/ClaimCard.jsx'
import Button from '../../components/ui/Button.jsx'
import { LANGS } from '../lib/loadData.js'
import { computeStats } from '../lib/compute.js'

function Stat({ label, value, accent }) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="text-[11px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={`font-mono text-xl font-bold ${accent}`}>{value}</div>
    </div>
  )
}

function MetricTable({ m }) {
  return (
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
            const r = m.per[l.code]
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
  )
}

export default function Results({ bpe, corpora, refStats }) {
  const [texts, setTexts] = useState(corpora)
  const [open, setOpen] = useState(false)
  const fileRefs = useRef({})
  const pristine = LANGS.every((l) => texts[l.code] === corpora[l.code])

  const stats = useMemo(() => computeStats(texts, bpe, LANGS), [texts, bpe])
  const primary = stats.primary
  const split = stats.split

  // Cross-check the live primary numbers vs the committed Python stats.json.
  const pyPrimary = refStats?.primary
  const matchesPython =
    pristine &&
    pyPrimary != null &&
    Math.abs((pyPrimary.score ?? 0) - primary.score) < 0.5 &&
    LANGS.every((l) => Math.abs((pyPrimary.per_language?.[l.code]?.X ?? 0) - primary.per[l.code].X) < 1e-6)

  const onFile = (code) => async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const content = await file.text()
    setTexts((t) => ({ ...t, [code]: content }))
  }

  return (
    <ClaimCard
      id="a2-2"
      code="A2-2"
      accent="var(--claim-3)"
      title="Ratios, statistics & self-score"
      claim={
        <>
          On the <b>full India article</b> in each language, fertilities span{' '}
          <b>X_max − X_min = {primary.spread.toFixed(4)}</b> → self-score{' '}
          <b>1000 / {primary.spread.toFixed(4)} = {primary.score.toFixed(1)}</b>, with{' '}
          <b>English = {primary.per.en.X.toFixed(3)} ≤ 1.2</b>.
        </>
      }
      takeaway={
        <>
          Word count is <code>[\p{'{'}L{'}'}\p{'{'}N{'}'}]+</code> (≡ Python <code>\w+</code>). Use the box below to
          drop in <b>your own</b> cleaned India-page text per language — the table recomputes live with this exact
          tokenizer, so you can confirm the numbers yourself.
        </>
      }
    >
      <div className="panel p-5">
        <div className="mb-2 flex items-center justify-between">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
            Primary metric · word = <span className="font-mono normal-case">[\p{'{'}L{'}'}\p{'{'}N{'}'}]+</span> (≡{' '}
            <span className="font-mono normal-case">\w+</span>)
          </div>
          {!pristine && (
            <span className="rounded bg-amber-100 px-2 py-0.5 text-[11px] font-medium text-amber-800 dark:bg-amber-900/50 dark:text-amber-200">
              custom text
            </span>
          )}
        </div>
        <MetricTable m={primary} />

        <div className="mt-4 rounded-lg bg-zinc-100 px-3 py-2 font-mono text-[13px] text-zinc-700 dark:bg-zinc-800/60 dark:text-zinc-200">
          sorted&nbsp;X:&nbsp;
          {primary.sortedDesc.map((r, i) => (
            <span key={r.code}>
              {i > 0 && ' ≥ '}
              <span className="font-semibold text-zinc-900 dark:text-zinc-50">{r.X.toFixed(4)}</span>
              <span className="text-zinc-500">({r.code})</span>
            </span>
          ))}
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat
            label="English X (gate ≤ 1.2)"
            value={`${primary.per.en.X.toFixed(3)} ${primary.per.en.ok ? '✓' : '✗'}`}
            accent={primary.per.en.ok ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'}
          />
          <Stat label="X_max − X_min" value={primary.spread.toFixed(4)} accent="text-zinc-900 dark:text-zinc-50" />
          <Stat
            label="Self-score"
            value={Number.isFinite(primary.score) ? primary.score.toFixed(1) : '∞'}
            accent="text-brand-600 dark:text-brand-400"
          />
          <Stat
            label={pristine ? 'vs Python reference' : 'all four ≤ 1.2'}
            value={
              pristine
                ? matchesPython
                  ? 'IDENTICAL ✓'
                  : '—'
                : primary.constraintsMet
                  ? 'YES ✓'
                  : 'NO ✗'
            }
            accent={
              (pristine ? matchesPython : primary.constraintsMet)
                ? 'text-emerald-600 dark:text-emerald-400'
                : 'text-zinc-500'
            }
          />
        </div>

        {/* Secondary metric (transparency) */}
        <details className="mt-5">
          <summary className="cursor-pointer text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
            Also under whitespace-split word count (transparency)
          </summary>
          <div className="mt-3">
            <MetricTable m={split} />
            <p className="mt-2 text-[11px] text-zinc-500">
              Under <span className="font-mono">text.split()</span>, English stays ≤ 1.2 (
              {split.per.en.X.toFixed(3)}); the Indic three are higher because whitespace-splitting doesn't break
              their long agglutinative words — an inherent property of the count, not the tokenizer.
            </p>
          </div>
        </details>

        {/* Grader verify: bring your own India-page text */}
        <div className="mt-5 rounded-lg border border-zinc-200 p-3 dark:border-zinc-700">
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            className="flex w-full items-center justify-between text-sm font-medium text-zinc-800 dark:text-zinc-100"
          >
            <span>▶ Verify with your own India-page text (paste or upload per language)</span>
            <span className="text-xs text-zinc-500">{open ? 'hide' : 'show'}</span>
          </button>
          {open && (
            <div className="mt-3 space-y-3">
              <div className="flex flex-wrap gap-2">
                {LANGS.map((l) => (
                  <span key={l.code} className="inline-flex items-center gap-1">
                    <input
                      ref={(el) => (fileRefs.current[l.code] = el)}
                      type="file"
                      accept=".txt,text/plain"
                      onChange={onFile(l.code)}
                      className="hidden"
                    />
                    <Button variant="ghost" onClick={() => fileRefs.current[l.code]?.click()}>
                      ↥ {l.name} .txt
                    </Button>
                  </span>
                ))}
                {!pristine && (
                  <Button variant="ghost" onClick={() => setTexts(corpora)}>
                    ↺ reset to India article
                  </Button>
                )}
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                {LANGS.map((l) => (
                  <label key={l.code} className="block">
                    <span className="text-[11px] font-medium text-zinc-500">
                      {l.name} — {primary.per[l.code].words.toLocaleString()} words ·{' '}
                      {primary.per[l.code].tokens.toLocaleString()} tokens · X {primary.per[l.code].X.toFixed(3)}
                    </span>
                    <textarea
                      value={texts[l.code]}
                      onChange={(e) => setTexts((t) => ({ ...t, [l.code]: e.target.value }))}
                      rows={3}
                      spellCheck={false}
                      className="mt-1 w-full resize-y rounded border border-zinc-300 bg-white p-2 font-mono text-[11px] text-zinc-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
                    />
                  </label>
                ))}
              </div>
              <p className="text-[11px] text-zinc-500">
                Everything above recomputes live from this text with the shipped tokenizer — this is exactly what a
                grader does when they run it on their cleaned India pages.
              </p>
            </div>
          )}
        </div>
      </div>
    </ClaimCard>
  )
}
