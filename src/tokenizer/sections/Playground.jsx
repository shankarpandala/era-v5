import { useMemo, useState } from 'react'
import ClaimCard from '../../components/ClaimCard.jsx'
import Button from '../../components/ui/Button.jsx'
import { LANGS } from '../lib/loadData.js'
import { wordCount } from '../lib/bpe.js'

const CHIP_COLORS = [
  'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200',
  'bg-violet-100 text-violet-800 dark:bg-violet-900/50 dark:text-violet-200',
  'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-200',
  'bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-200',
]

const _dec = new TextDecoder('utf-8', { fatal: false })

function displayOf(bytes) {
  // Decode the token's raw bytes to text; make leading whitespace visible.
  return _dec
    .decode(new Uint8Array(bytes))
    .replace(/ /g, '·')
    .replace(/\n/g, '↵')
}

export default function Playground({ bpe, corpora }) {
  const [text, setText] = useState(
    'India, officially the Republic of India, is a country in South Asia. भारत एक देश है। భారతదేశం ఒక దేశం.',
  )

  const result = useMemo(() => {
    const tokens = bpe.encode(text)
    const words = wordCount(text)
    return {
      tokens,
      words,
      chips: tokens.map((id) => ({ id, display: displayOf(bpe.vocab[id]) })),
    }
  }, [bpe, text])

  const fertility = result.words > 0 ? result.tokens.length / result.words : 0

  return (
    <ClaimCard
      id="a2-3"
      code="A2-3"
      accent="var(--claim-2)"
      title="Tokenize anything (live)"
      claim="Paste any text — it is tokenized in your browser by the exact same tokenizer, and the token/word fertility is computed live."
      takeaway="This is the real encoder (identical to the Python reference). Try mixing scripts in one line to see the shared vocabulary at work."
    >
      <div className="panel p-5">
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <span className="text-xs text-zinc-500">Load a sample:</span>
          {LANGS.map((l) => (
            <Button
              key={l.code}
              variant="ghost"
              onClick={() => setText(corpora[l.code].split(/\s+/).slice(0, 40).join(' '))}
            >
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

        <div className="mt-3 grid grid-cols-3 gap-2 text-center font-mono text-sm">
          <div className="rounded bg-zinc-100 py-2 dark:bg-zinc-800">
            words <span className="font-semibold text-zinc-900 dark:text-zinc-50">{result.words}</span>
          </div>
          <div className="rounded bg-zinc-100 py-2 dark:bg-zinc-800">
            tokens{' '}
            <span className="font-semibold text-zinc-900 dark:text-zinc-50">{result.tokens.length}</span>
          </div>
          <div className="rounded bg-zinc-100 py-2 dark:bg-zinc-800">
            X = tokens/word{' '}
            <span className="font-semibold text-brand-600 dark:text-brand-400">
              {fertility ? fertility.toFixed(3) : '—'}
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
                {c.display}
              </span>
            ))}
            {result.chips.length === 0 && <span className="text-xs text-zinc-500">— type something —</span>}
          </div>
        </div>
        <p className="mt-2 text-[11px] text-zinc-500">
          Each chip is one token. <span className="font-mono">·</span> marks a leading space,{' '}
          <span className="font-mono">↵</span> a newline. Hover to see the token id.
        </p>
      </div>
    </ClaimCard>
  )
}
