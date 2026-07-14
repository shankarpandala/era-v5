import { useMemo } from 'react'
import ClaimCard from '../../components/ClaimCard.jsx'
import Button from '../../components/ui/Button.jsx'

async function downloadFrom(url, filename, type) {
  const res = await fetch(url)
  const content = await res.text()
  const blob = new Blob([content], { type })
  const href = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = href
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(href)
}

export default function Downloads({ tok, metrics }) {
  const base = import.meta.env.BASE_URL

  // Peek at the most word-like learned tokens (longest strings).
  const preview = useMemo(() => {
    const idx = tok.idToToken.map((t, id) => ({ id, t })).filter((x) => x.t && x.t !== '[UNK]')
    idx.sort((a, b) => b.t.length - a.t.length)
    return idx.slice(0, 60)
  }, [tok])

  return (
    <ClaimCard
      id="a2-4"
      code="A2-4"
      accent="var(--claim-4)"
      title="Download the tokenizer"
      claim={
        <>
          The shipped <b>tokenizer.json</b> is standard HuggingFace format —{' '}
          <code>tokenizers.Tokenizer.from_file("tokenizer.json")</code> gives you <code>.encode()</code> and{' '}
          <code>.decode()</code> directly. <b>vocab.txt</b> lists all {tok.vocabSize.toLocaleString()} tokens.
        </>
      }
      takeaway={
        <>
          Reproduce end-to-end: <code>python build_wiki_faithful_markdown.py && python train_tokenizer.py && python
          evaluate_tokenizer.py</code> — or run the instructor's published evaluator directly on these files.
        </>
      }
    >
      <div className="panel p-5">
        <div className="flex flex-wrap items-center gap-3">
          <Button
            onClick={() => downloadFrom(`${base}tokenizer/tokenizer.json`, 'tokenizer.json', 'application/json;charset=utf-8')}
          >
            ↓ tokenizer.json (HF format)
          </Button>
          <Button
            variant="ghost"
            onClick={() => downloadFrom(`${base}tokenizer/vocab.txt`, 'vocab.txt', 'text/plain;charset=utf-8')}
          >
            ↓ vocab.txt ({tok.vocabSize.toLocaleString()} tokens)
          </Button>
          <Button
            variant="ghost"
            onClick={() => downloadFrom(`${base}tokenizer/metrics.json`, 'metrics.json', 'application/json;charset=utf-8')}
          >
            ↓ metrics.json
          </Button>
          <span className="font-mono text-xs text-zinc-500">
            weights {Object.entries(metrics.weights).map(([k, v]) => `${k}=${v}`).join(' ')}
          </span>
        </div>

        <div className="mt-4">
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
            Longest learned tokens (a peek at the vocabulary)
          </div>
          <div className="flex flex-wrap gap-1">
            {preview.map((x) => (
              <span
                key={x.id}
                title={`id ${x.id}`}
                className="whitespace-pre rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-xs text-zinc-700 dark:bg-zinc-800 dark:text-zinc-200"
              >
                {x.t}
              </span>
            ))}
          </div>
        </div>
      </div>
    </ClaimCard>
  )
}
