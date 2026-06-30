import { useCallback, useMemo, useState } from 'react'
import ClaimCard from '../components/ClaimCard.jsx'
import Slider from '../components/ui/Slider.jsx'
import Button from '../components/ui/Button.jsx'
import useTrainer from '../hooks/useTrainer.js'
import { makeRng } from '../lib/rng.js'
import { makeGrammarPairs, oneHot, GRAMMAR_TOKENS, CATEGORY_COLOR } from '../lib/datasets.js'
import { MLP, Adam } from '../lib/nn.js'
import { pca2d } from '../lib/pca.js'

const ACCENT = '#10b981'
const VOCAB = GRAMMAR_TOKENS.length

function cosine(a, b) {
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9)
}

// SVG scatter of the PCA-projected embeddings, one labeled dot per token.
function EmbeddingPlot({ coords }) {
  const size = 320
  const pad = 34
  if (!coords) return <div className="h-[320px] animate-pulse rounded-lg bg-zinc-100 dark:bg-zinc-800" />
  const xs = coords.map((c) => c[0])
  const ys = coords.map((c) => c[1])
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const sx = (x) => pad + ((x - minX) / (maxX - minX || 1)) * (size - 2 * pad)
  const sy = (y) => size - pad - ((y - minY) / (maxY - minY || 1)) * (size - 2 * pad)

  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="w-full rounded-lg ring-1 ring-zinc-200 dark:ring-zinc-700" style={{ background: 'transparent' }}>
      {GRAMMAR_TOKENS.map((t, i) => {
        const color = CATEGORY_COLOR[t.cat]
        return (
          <g key={t.token} transform={`translate(${sx(coords[i][0])}, ${sy(coords[i][1])})`}>
            <circle r="6" fill={color} fillOpacity="0.85" />
            <text x="9" y="4" className="fill-zinc-700 dark:fill-zinc-200 text-[11px] font-medium">
              {t.token}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

export default function S1Embeddings() {
  const [embDim, setEmbDim] = useState(8)
  const [lr, setLr] = useState(0.1)
  const [epochsTarget, setEpochsTarget] = useState(400)
  const [seed, setSeed] = useState(1)
  const [nPairs, setNPairs] = useState(1500)

  const build = useCallback(() => {
    const { xIdx, yIdx } = makeGrammarPairs(nPairs, makeRng(seed))
    const X = oneHot(xIdx, VOCAB)
    const model = new MLP({
      inDim: VOCAB,
      hidden: [embDim],
      activation: 'none', // embedding (linear) → softmax over vocab
      outDim: VOCAB,
      head: 'softmax',
      bias: false,
      rng: makeRng(seed * 100 + 5),
    })
    const opt = new Adam({ lr })
    return {
      model,
      step() {
        model.trainStep(X, yIdx, opt)
      },
      snapshot() {
        return { loss: model.loss(X, yIdx) }
      },
    }
  }, [embDim, lr, seed, nPairs])

  const { tick, running, snapshot, trainer, epoch, start, stop, reset } = useTrainer({
    build,
    epochs: epochsTarget,
    deps: [embDim, lr, seed, nPairs, epochsTarget],
  })

  const { coords, neighbors, purity } = useMemo(() => {
    const t = trainer.current
    if (!t) return { coords: null, neighbors: [], purity: null }
    const W = t.model.denseWeights()[0] // (VOCAB × embDim)
    const rows = []
    for (let i = 0; i < VOCAB; i++) {
      const r = []
      for (let j = 0; j < embDim; j++) r.push(W.d[i * embDim + j])
      rows.push(r)
    }
    const coords = pca2d(rows, makeRng(7))
    // Nearest neighbor (cosine, full dim) per token.
    let same = 0
    const neighbors = GRAMMAR_TOKENS.map((tok, i) => {
      let best = -Infinity
      let bestJ = -1
      for (let j = 0; j < VOCAB; j++) {
        if (j === i) continue
        const s = cosine(rows[i], rows[j])
        if (s > best) {
          best = s
          bestJ = j
        }
      }
      const isSame = GRAMMAR_TOKENS[bestJ].cat === tok.cat
      if (isSame) same++
      return { token: tok.token, cat: tok.cat, nn: GRAMMAR_TOKENS[bestJ].token, isSame }
    })
    return { coords, neighbors, purity: same / VOCAB }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, embDim])

  return (
    <ClaimCard
      id="s1-3"
      code="S1-3"
      accent={ACCENT}
      title="Embeddings learn similarity from nothing but next-token"
      claim="Trained only to predict the next token in a tiny synthetic grammar, the embedding table clusters related tokens — even though similarity was never supplied as a label."
      takeaway="Same-category tokens share next-token distributions, so the only way to predict well is to give them similar embeddings. Watch animals, fruits, and verbs drift into three separate clusters as training runs — and every nearest neighbor becomes same-category. Emergent structure, learned from prediction alone."
    >
      <div className="panel p-4">
        <div className="grid gap-6 lg:grid-cols-[1fr_280px]">
          <div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <div className="mb-1 text-center text-xs font-medium text-zinc-600 dark:text-zinc-300">
                  Learned embeddings (PCA → 2D)
                </div>
                <EmbeddingPlot coords={coords} />
                <div className="mt-2 flex justify-center gap-3 text-[11px]">
                  {Object.entries(CATEGORY_COLOR).map(([cat, color]) => (
                    <span key={cat} className="flex items-center gap-1">
                      <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: color }} />
                      {cat}s
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <div className="mb-1 text-center text-xs font-medium text-zinc-600 dark:text-zinc-300">
                  Nearest neighbor (cosine)
                </div>
                <div className="overflow-hidden rounded-lg ring-1 ring-zinc-200 dark:ring-zinc-700">
                  <table className="w-full text-xs">
                    <tbody>
                      {neighbors.map((row) => (
                        <tr key={row.token} className="border-b border-zinc-100 last:border-0 dark:border-zinc-800">
                          <td className="px-2 py-1.5 font-medium" style={{ color: CATEGORY_COLOR[row.cat] }}>
                            {row.token}
                          </td>
                          <td className="px-2 py-1.5 text-zinc-400">→</td>
                          <td className="px-2 py-1.5 font-medium text-zinc-700 dark:text-zinc-200">{row.nn}</td>
                          <td className="px-2 py-1.5 text-right">
                            {row.isSame ? (
                              <span className="text-emerald-600 dark:text-emerald-400">same ✓</span>
                            ) : (
                              <span className="text-red-500">diff</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="mt-2 text-center text-xs text-zinc-500">
                  same-category neighbors:{' '}
                  <span className="font-mono font-semibold text-zinc-800 dark:text-zinc-100">
                    {purity == null ? '—' : `${Math.round(purity * 100)}%`}
                  </span>
                </div>
              </div>
            </div>

            <div className="mt-4 flex flex-wrap items-center gap-3">
              <Button onClick={running ? stop : epoch >= epochsTarget ? () => reset(true) : start}>
                {running ? '⏸ Pause' : epoch >= epochsTarget ? '↻ Re-train' : '▶ Train'}
              </Button>
              <Button variant="ghost" onClick={() => reset(true)}>
                ↻ Restart
              </Button>
              <span className="font-mono text-xs text-zinc-500">
                epoch {epoch}/{epochsTarget}
                {snapshot?.loss != null && ` · loss ${snapshot.loss.toFixed(3)}`}
              </span>
            </div>
          </div>

          <div className="space-y-4">
            <Slider label="embedding dim" value={embDim} min={2} max={16} step={1} onChange={setEmbDim} />
            <Slider label="learning rate" value={lr} min={0.01} max={0.3} step={0.01} onChange={setLr} format={(v) => v.toFixed(2)} />
            <Slider label="epochs" value={epochsTarget} min={50} max={800} step={50} onChange={setEpochsTarget} />
            <Slider label="training pairs" value={nPairs} min={200} max={3000} step={100} onChange={setNPairs} />
            <Slider label="seed" value={seed} min={1} max={20} step={1} onChange={setSeed} />
            <p className="text-[11px] leading-relaxed text-zinc-500 dark:text-zinc-400">
              Grammar: <code>animal → verb → (fruit | animal)</code>, <code>fruit → animal</code>. The next-token
              distribution depends only on a token's category, so the model is forced to cluster categories — no
              similarity label is ever given.
            </p>
          </div>
        </div>
      </div>
    </ClaimCard>
  )
}
