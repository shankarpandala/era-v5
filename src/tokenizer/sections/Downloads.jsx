import { useMemo } from 'react'
import ClaimCard from '../../components/ClaimCard.jsx'
import Button from '../../components/ui/Button.jsx'

const _dec = new TextDecoder('utf-8', { fatal: false })

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

export default function Downloads({ tok, bpe }) {
  const base = import.meta.env.BASE_URL
  const nMerges = tok.merges.length

  // Preview the most "word-like" learned tokens (longest byte sequences).
  const preview = useMemo(() => {
    const ids = bpe.vocab.map((_, i) => i).filter((i) => i >= 256)
    ids.sort((a, b) => bpe.vocab[b].length - bpe.vocab[a].length)
    return ids.slice(0, 60).map((id) => ({
      id,
      display: _dec.decode(new Uint8Array(bpe.vocab[id])).replace(/ /g, '·').replace(/\n/g, '↵'),
    }))
  }, [bpe])

  return (
    <ClaimCard
      id="a2-4"
      code="A2-4"
      accent="var(--claim-4)"
      title="Download the tokenizer"
      claim={
        <>
          The full tokenizer is downloadable — a plain <b>vocab.txt</b> (all {tok.vocab_size.toLocaleString()}{' '}
          tokens) and the <b>tokenizer.json</b> (pattern + ordered merges) that the Python reference and this widget
          both load.
        </>
      }
      takeaway="Load tokenizer.json with the shipped Python BPETokenizer (or this widget's JS encoder) and you reproduce every number above exactly."
    >
      <div className="panel p-5">
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={() => downloadFrom(`${base}tokenizer/vocab.txt`, 'vocab.txt', 'text/plain;charset=utf-8')}>
            ↓ vocab.txt ({tok.vocab_size.toLocaleString()} tokens)
          </Button>
          <Button
            variant="ghost"
            onClick={() =>
              downloadFrom(`${base}tokenizer/tokenizer.json`, 'tokenizer.json', 'application/json;charset=utf-8')
            }
          >
            ↓ tokenizer.json
          </Button>
          <span className="font-mono text-xs text-zinc-500">
            256 base bytes + {nMerges.toLocaleString()} merges = {tok.vocab_size.toLocaleString()} tokens
          </span>
        </div>

        <div className="mt-4">
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
            Longest learned tokens (a peek at the vocabulary)
          </div>
          <div className="flex flex-wrap gap-1">
            {preview.map((t) => (
              <span
                key={t.id}
                title={`id ${t.id}`}
                className="whitespace-pre rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-xs text-zinc-700 dark:bg-zinc-800 dark:text-zinc-200"
              >
                {t.display}
              </span>
            ))}
          </div>
        </div>
      </div>
    </ClaimCard>
  )
}
