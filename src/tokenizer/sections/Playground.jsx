import { useMemo, useState } from 'react'
import ClaimCard from '../../components/ClaimCard.jsx'
import Button from '../../components/ui/Button.jsx'
import { LANGS } from '../lib/loadData.js'
import { faithfulUnits } from '../lib/hfbpe.js'

const CHIP_COLORS = [
  'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200',
  'bg-violet-100 text-violet-800 dark:bg-violet-900/50 dark:text-violet-200',
  'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-200',
  'bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-200',
]

export default function Playground({ tok, corpora, sample }) {
  const [text, setText] = useState(sample)

  const result = useMemo(() => {
    const chips = tok.encodeToTokens(text)
    const units = faithfulUnits(text.normalize('NFKC'))
    const decoded = tok.decode(chips.map((c) => c.id))
    return { chips, units, decoded }
  }, [tok, text])

  const fertility = result.units > 0 ? result.chips.length / result.units : 0
  const roundTrip = result.decoded === text.normalize('NFKC')

  return (
    <ClaimCard
      id="a2-3"
      code="A2-3"
      accent="var(--claim-2)"
      title="Tokenize anything (live)"
      claim="Paste any text — tokenized in your browser by the exact shipped tokenizer, with live fertility and a decode round-trip check."
      takeaway="▁ marks a space carried by the token (Metaspace). The decode must reproduce your input — that is the faithfulness requirement the grader enforces."
    >
      <div className="panel p-5">
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <span className="text-xs text-zinc-500">Load a sample:</span>
          <Button variant="ghost" onClick={() => setText(sample)}>
            grader sample
          </Button>
          {LANGS.map((l) => (
            <Button key={l.code} variant="ghost" onClick={() => setText(corpora[l.code].slice(0, 300))}>
              {l.name}
            </Button>
          ))}
        </div>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          spellCheck={false}
          className="w-full resize-y rounded-lg border border-zinc-300 bg-white p-3 font-mono text-sm text-zinc-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
        />

        <div className="mt-3 grid grid-cols-2 gap-2 text-center font-mono text-sm sm:grid-cols-4">
          <div className="rounded bg-zinc-100 py-2 dark:bg-zinc-800">
            units <span className="font-semibold text-zinc-900 dark:text-zinc-50">{result.units}</span>
          </div>
          <div className="rounded bg-zinc-100 py-2 dark:bg-zinc-800">
            tokens <span className="font-semibold text-zinc-900 dark:text-zinc-50">{result.chips.length}</span>
          </div>
          <div className="rounded bg-zinc-100 py-2 dark:bg-zinc-800">
            fertility{' '}
            <span className="font-semibold text-brand-600 dark:text-brand-400">
              {fertility ? fertility.toFixed(3) : '—'}
            </span>
          </div>
          <div className="rounded bg-zinc-100 py-2 dark:bg-zinc-800">
            round-trip{' '}
            <span
              className={`font-semibold ${
                roundTrip ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'
              }`}
            >
              {roundTrip ? '✓' : '✗'}
            </span>
          </div>
        </div>

        <div className="mt-4 max-h-64 overflow-y-auto rounded-lg border border-zinc-200 p-2 dark:border-zinc-700">
          <div className="flex flex-wrap gap-1">
            {result.chips.map((c, i) => (
              <span
                key={i}
                title={`id ${c.id}`}
                className={`whitespace-pre rounded px-1.5 py-0.5 font-mono text-xs ${CHIP_COLORS[i % CHIP_COLORS.length]}`}
              >
                {c.text}
              </span>
            ))}
            {result.chips.length === 0 && <span className="text-xs text-zinc-500">— type something —</span>}
          </div>
        </div>
        <p className="mt-2 text-[11px] text-zinc-500">
          Each chip is one token; hover for its id. <span className="font-mono">▁</span> is the Metaspace space
          marker; <span className="font-mono">[UNK]</span> appears only for characters never seen in the corpus.
        </p>
      </div>
    </ClaimCard>
  )
}
