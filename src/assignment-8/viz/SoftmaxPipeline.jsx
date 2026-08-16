import { useMemo, useState } from 'react'
import { BlockMath } from 'react-katex'
import Slider from '../../components/ui/Slider.jsx'
import Toggle from '../../components/ui/Toggle.jsx'
import MaskCanvas, { Legend } from './MaskCanvas.jsx'
import { makeQKV, scores, softmax, MASKS } from '../lib/attnMath.js'

const SENTENCE = 'the cat sat on the mat because it was tired after chasing the red ball all afternoon in the garden'.split(' ')

function Bars({ values, labels, color, max, title, highlight }) {
  const mx = max ?? Math.max(...values.map((v) => Math.abs(v)), 1e-9)
  return (
    <div>
      <div className="mb-1 font-mono text-[11px] text-zinc-500">{title}</div>
      <div className="flex h-24 items-end gap-[2px]">
        {values.map((v, i) => {
          const h = Math.max(1, (Math.abs(v) / mx) * 88)
          return (
            <div key={i} className="group relative flex flex-1 flex-col items-center justify-end" title={`${labels[i]}: ${v.toFixed(3)}`}>
              <div
                className="w-full rounded-t"
                style={{ height: h, backgroundColor: color, opacity: highlight === i ? 1 : 0.7 }}
              />
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function SoftmaxPipeline() {
  const [n, setN] = useState(10)
  const [d, setD] = useState(64)
  const [causal, setCausal] = useState(true)
  const [scale, setScale] = useState(true)
  const [row, setRow] = useState(6)

  const tokens = SENTENCE.slice(0, n)
  const { Q, K } = useMemo(() => makeQKV(n, d, 11), [n, d])
  const S = useMemo(() => scores(Q, K), [Q, K])
  const q = Math.min(row, n - 1)
  const allow = useMemo(() => (causal ? MASKS.causal() : MASKS.full()), [causal])

  const raw = S[q]
  const scaled = raw.map((v) => (scale ? v / Math.sqrt(d) : v))
  const masked = scaled.map((v, j) => (allow(q, j) ? v : -Infinity))
  const probs = softmax(masked)

  const weights = useMemo(
    () => S.map((r, i) => softmax(r.map((v, j) => (allow(i, j) ? (scale ? v / Math.sqrt(d) : v) : -Infinity)))),
    [S, allow, scale, d],
  )
  const cell = useMemo(
    () => (i, j) => {
      if (!allow(i, j)) return null
      const mx = Math.max(...weights[i])
      return { w: mx > 0 ? weights[i][j] / mx : 0, cat: i === q ? 'selected' : 'default' }
    },
    [allow, weights, q],
  )

  // entropy of the chosen row, to show what scaling does
  const H = -probs.reduce((a, p) => (p > 0 ? a + p * Math.log2(p) : a), 0)
  const maxP = Math.max(...probs)

  return (
    <div className="panel p-4">
      <div className="mb-3">
        <BlockMath math={'\\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}\\!\\left(\\frac{QK^{\\top}}{\\sqrt{d_k}}\\right)V'} />
      </div>
      <div className="grid gap-4 lg:grid-cols-[auto_1fr]">
        <div>
          <MaskCanvas n={n} cell={cell} size={260} rowLabel="query" colLabel="key" ariaLabel="causal attention weights" />
          <Legend items={[['default', 'softmax weight (darker = larger)'], ['selected', `query row ${q}: “${tokens[q]}”`]]} />
        </div>
        <div className="min-w-0 space-y-3">
          <div className="flex flex-wrap gap-1">
            {tokens.map((t, i) => (
              <button
                key={i}
                type="button"
                onClick={() => setRow(i)}
                className={`rounded-md border px-2 py-0.5 font-mono text-xs ${
                  i === q
                    ? 'border-pink-500 bg-pink-500/10 text-pink-600 dark:text-pink-300'
                    : 'border-zinc-200 text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800'
                }`}
              >
                {t}
              </button>
            ))}
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <Bars values={raw} labels={tokens} color="#71717a" title="1 · raw scores q·k" />
            <Bars values={scaled} labels={tokens} color="#3b82f6" title={scale ? `2 · ÷ √d = ÷ ${Math.sqrt(d).toFixed(1)}` : '2 · (scaling OFF)'} />
            <Bars values={probs} labels={tokens} color="#d946ef" title="3 · softmax → weights" max={1} />
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
            <div className="rounded-lg border border-zinc-200 px-2 py-1.5 dark:border-zinc-800">
              <div className="text-[10px] uppercase text-zinc-500">max weight</div>
              <div className="font-mono font-semibold">{maxP.toFixed(2)}</div>
            </div>
            <div className="rounded-lg border border-zinc-200 px-2 py-1.5 dark:border-zinc-800">
              <div className="text-[10px] uppercase text-zinc-500">row entropy (bits)</div>
              <div className="font-mono font-semibold">{H.toFixed(2)}</div>
            </div>
            <div className="rounded-lg border border-zinc-200 px-2 py-1.5 dark:border-zinc-800">
              <div className="text-[10px] uppercase text-zinc-500">scores / layer / head</div>
              <div className="font-mono font-semibold">{causal ? (n * (n + 1)) / 2 : n * n}</div>
            </div>
            <div className="rounded-lg border border-zinc-200 px-2 py-1.5 dark:border-zinc-800">
              <div className="text-[10px] uppercase text-zinc-500">KV cached / token / layer</div>
              <div className="font-mono font-semibold">2·h·d_head</div>
            </div>
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            <Slider label="tokens n" value={n} min={4} max={SENTENCE.length} onChange={setN} />
            <Slider label="head dim d_k" value={d} min={4} max={256} step={4} onChange={setD} />
            <Toggle label="causal mask" checked={causal} onChange={setCausal} hint="decoder: a query may not read the future" />
            <Toggle label="scale by 1/√d_k" checked={scale} onChange={setScale} hint="turn it off and raise d_k: softmax saturates, entropy collapses" />
          </div>
        </div>
      </div>
    </div>
  )
}
