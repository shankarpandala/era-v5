import { useMemo, useState } from 'react'
import Slider from '../../components/ui/Slider.jsx'
import MaskCanvas, { Legend } from './MaskCanvas.jsx'
import {
  makeQKV,
  scores,
  softmax,
  MASKS,
  topkMask,
  blockSelectMask,
  lshMask,
  routingMask,
} from '../lib/attnMath.js'

// One parameterised widget for every "which keys does a query read?" mechanism.
// `mode` selects the mask family; the sliders expose that family's knobs.
// The bill line underneath is the point: what fraction of the n² dense pattern
// is actually computed, and how the KV cache behaves.

const D = 16

function Stat({ label, value }) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white px-3 py-2 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className="font-mono text-sm font-semibold text-zinc-800 dark:text-zinc-100">{value}</div>
    </div>
  )
}

export default function AttentionMatrix({ mode = 'causal', params = {} }) {
  const [n, setN] = useState(params.n ?? 32)
  const [w, setW] = useState(params.w ?? 6)
  const [s, setS] = useState(params.s ?? 2)
  const [g, setG] = useState(params.g ?? 2)
  const [k, setK] = useState(params.k ?? 4)
  const [block, setBlock] = useState(params.block ?? 4)
  const [top, setTop] = useState(params.top ?? 2)
  const [stride, setStride] = useState(params.stride ?? 4)
  const [pattern, setPattern] = useState('strided')
  const [rounds, setRounds] = useState(2)
  const [clusters, setClusters] = useState(3)
  const [seg, setSeg] = useState(8)
  const [devices, setDevices] = useState(4)
  const [rank, setRank] = useState(8)
  const [hoverRow, setHoverRow] = useState(null)

  const { Q, K } = useMemo(() => makeQKV(n, D, 7), [n])
  const S = useMemo(() => scores(Q, K).map((r) => r.map((v) => v / Math.sqrt(D))), [Q, K])

  // Build the (mask, category) function for this mode.
  const { allow, cat, legend, describe, extraControls, m } = useMemo(() => {
    let allow
    let cat = () => 'default'
    let legend = [['default', 'computed (softmax weight)']]
    let describe = ''
    let extraControls = null
    let m = n
    switch (mode) {
      case 'cross': {
        allow = MASKS.full()
        m = n
        describe = 'Encoder–decoder (2014): every target step scores every source state. No causal mask, no efficiency question yet.'
        break
      }
      case 'cross-local': {
        allow = (i, j) => Math.abs(i - j) <= w
        cat = () => 'window'
        legend = [['window', `local-p window ±${w} (Luong 2015)`]]
        describe = 'Luong local attention: a Gaussian-tapered window around a predicted source position — per-step cost fixed at 2D+1.'
        extraControls = <Slider label="window D" value={w} min={1} max={16} onChange={setW} />
        break
      }
      case 'window': {
        allow = MASKS.window(w)
        cat = () => 'window'
        legend = [['window', `sliding window (w=${w})`]]
        describe = 'Each query reads only the previous w keys. Cost O(n·w); the KV cache is a rolling buffer of w entries.'
        extraControls = <Slider label="window w" value={w} min={1} max={32} onChange={setW} />
        break
      }
      case 'window-sinks': {
        allow = MASKS.windowSinks(w, s)
        cat = (i, j) => (j < s ? 'sink' : 'window')
        legend = [
          ['sink', `${s} sink token${s === 1 ? '' : 's'} (never evicted)`],
          ['window', `rolling window (w=${w})`],
        ]
        describe = 'StreamingLLM: keep the first tokens (where softmax dumps excess mass) plus a rolling window; perplexity stays flat while the cache stays bounded.'
        extraControls = (
          <>
            <Slider label="window w" value={w} min={1} max={32} onChange={setW} />
            <Slider label="sink tokens" value={s} min={0} max={8} onChange={setS} />
          </>
        )
        break
      }
      case 'window-global': {
        allow = MASKS.windowGlobal(w, g)
        cat = (i, j) => (j < g || i < g ? 'global' : 'window')
        legend = [
          ['window', `sliding window (w=${w})`],
          ['global', `${g} global token${g === 1 ? '' : 's'} (attend / attended everywhere)`],
        ]
        describe = 'Longformer: a window for everyone plus a few global tokens ([CLS], the question) that read and are read by all.'
        extraControls = (
          <>
            <Slider label="window w" value={w} min={1} max={32} onChange={setW} />
            <Slider label="global tokens" value={g} min={0} max={8} onChange={setG} />
          </>
        )
        break
      }
      case 'bigbird': {
        allow = MASKS.bigbird(w, g, 2)
        const win = MASKS.window(w)
        cat = (i, j) => (j < g ? 'global' : win(i, j) ? 'window' : 'random')
        legend = [
          ['window', `window (w=${w})`],
          ['global', `${g} global`],
          ['random', 'random blocks'],
        ]
        describe = 'BigBird: window + global + a few random keys per query — linear cost, provably as expressive as full attention.'
        extraControls = (
          <>
            <Slider label="window w" value={w} min={1} max={16} onChange={setW} />
            <Slider label="global tokens" value={g} min={0} max={6} onChange={setG} />
          </>
        )
        break
      }
      case 'strided': {
        allow = pattern === 'strided' ? MASKS.strided(stride) : MASKS.fixed(stride, 1)
        cat = (i, j) => (i - j < stride && pattern === 'strided' ? 'window' : pattern === 'fixed' && Math.floor(i / stride) === Math.floor(j / stride) ? 'window' : 'compressed')
        legend =
          pattern === 'strided'
            ? [
                ['window', `local (last ${stride})`],
                ['compressed', `every ${stride}-th column`],
              ]
            : [
                ['window', `own block of ${stride}`],
                ['compressed', 'summary column of each block'],
              ]
        describe =
          pattern === 'strided'
            ? 'Sparse Transformer "strided": a local band plus every ℓ-th column — any token reaches any other in two layers, O(n√n).'
            : 'Sparse Transformer "fixed": attend within your block plus to a summary column of every block.'
        extraControls = (
          <>
            <div className="flex gap-2 text-xs">
              {['strided', 'fixed'].map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => setPattern(p)}
                  className={`rounded-md border px-2 py-1 ${pattern === p ? 'border-brand-500 bg-brand-500 text-white' : 'border-zinc-300 text-zinc-600 dark:border-zinc-700 dark:text-zinc-300'}`}
                >
                  {p}
                </button>
              ))}
            </div>
            <Slider label="stride ℓ" value={stride} min={2} max={8} onChange={setStride} />
          </>
        )
        break
      }
      case 'topk': {
        allow = topkMask(S, k)
        cat = () => 'selected'
        legend = [['selected', `top-${k} keys per query`]]
        describe = 'Explicit Sparse Transformer: compute every score, keep the k largest, mask the rest to −∞. Sharper attention — but the dense matrix was still computed.'
        extraControls = <Slider label="k" value={k} min={1} max={16} onChange={setK} />
        break
      }
      case 'dsa': {
        allow = topkMask(S, k)
        cat = () => 'selected'
        legend = [['selected', `top-${k} tokens chosen by the lightning indexer`]]
        describe = 'DeepSeek Sparse Attention: a tiny FP8 indexer scores every key cheaply, the top-k=2048 latent entries are gathered, and MLA runs only over those — O(L·k).'
        extraControls = <Slider label="k (tokens kept)" value={k} min={1} max={16} onChange={setK} />
        break
      }
      case 'block-select': {
        allow = blockSelectMask(S, block, top)
        cat = (i, j) => (Math.floor(i / block) === Math.floor(j / block) ? 'window' : 'selected')
        legend = [
          ['window', `own block (${block})`],
          ['selected', `top-${top} other blocks by q·mean(K_block)`],
        ]
        describe = 'MoBA / Quest-style: mean-pool keys per block, gate each query onto its top blocks. Cheap to index, coarse in what it can find.'
        extraControls = (
          <>
            <Slider label="block size" value={block} min={2} max={8} onChange={setBlock} />
            <Slider label="top blocks" value={top} min={1} max={6} onChange={setTop} />
          </>
        )
        break
      }
      case 'nsa': {
        const sel = blockSelectMask(S, block, top)
        const win = MASKS.window(w)
        allow = (i, j) => j <= i && (win(i, j) || sel(i, j) || (i - j) % stride === 0)
        cat = (i, j) => (win(i, j) ? 'window' : sel(i, j) ? 'selected' : 'compressed')
        legend = [
          ['compressed', `compressed: every ${stride}-th (block summary)`],
          ['selected', `selected: top-${top} blocks of ${block}`],
          ['window', `sliding window (${w})`],
        ]
        describe = 'NSA: three branches per query — coarse compressed blocks, fine-grained selected blocks, and a local window — mixed by a learned gate; kernels group heads so the selection is shared.'
        extraControls = (
          <>
            <Slider label="compress stride" value={stride} min={2} max={8} onChange={setStride} />
            <Slider label="select: block size" value={block} min={2} max={8} onChange={setBlock} />
            <Slider label="select: top blocks" value={top} min={1} max={4} onChange={setTop} />
            <Slider label="window" value={w} min={1} max={12} onChange={setW} />
          </>
        )
        break
      }
      case 'csa': {
        // DeepSeek-V4 CSA: every `stride` tokens → one compressed entry; the indexer keeps the top-`top`
        // compressed entries per query; a small window of uncompressed recent tokens.
        const win = MASKS.window(w)
        const compressedOf = (j) => Math.floor(j / stride)
        const keep = []
        for (let i = 0; i < n; i++) {
          const nb = compressedOf(i) + 1
          const sc = []
          for (let b = 0; b < nb; b++) {
            let acc = 0
            let c = 0
            for (let j = b * stride; j < Math.min(n, (b + 1) * stride); j++) if (j <= i) { acc += S[i][j]; c++ }
            sc.push([c ? acc / c : -Infinity, b])
          }
          keep.push(new Set(sc.sort((a, b) => b[0] - a[0]).slice(0, top).map(([, b]) => b)))
        }
        const selected = (i, j) => keep[i].has(compressedOf(j)) && j % stride === stride - 1
        allow = (i, j) => j <= i && (win(i, j) || selected(i, j))
        cat = (i, j) => (win(i, j) ? 'window' : 'selected')
        legend = [
          ['selected', `top-${top} compressed entries (each = ${stride} tokens) chosen by the indexer`],
          ['window', `recent uncompressed window (${w})`],
        ]
        describe = 'DeepSeek-V4 Compressed Sparse Attention: compress every m tokens into one KV entry, let the lightning indexer pick the top-k compressed entries, keep a small window of raw recent tokens. Only one cell per compressed block is drawn.'
        extraControls = (
          <>
            <Slider label="compression m" value={stride} min={2} max={8} onChange={setStride} />
            <Slider label="top-k compressed entries" value={top} min={1} max={6} onChange={setTop} />
            <Slider label="window" value={w} min={1} max={12} onChange={setW} />
          </>
        )
        break
      }
      case 'lsh': {
        allow = lshMask(K, rounds)
        cat = () => 'selected'
        legend = [['selected', `same LSH bucket (${rounds} hash bits)`]]
        describe = 'Reformer: hash queries/keys with random rotations, attend only within a bucket. O(n log n), but hashing noise and shared-QK made it awkward.'
        extraControls = <Slider label="hash bits" value={rounds} min={1} max={4} onChange={setRounds} />
        break
      }
      case 'routing': {
        allow = routingMask(K, clusters)
        cat = () => 'selected'
        legend = [['selected', `same k-means cluster (${clusters} centroids)`]]
        describe = 'Routing Transformer: cluster keys with online k-means, attend within your cluster — content-based sparsity, 2020 tools.'
        extraControls = <Slider label="clusters" value={clusters} min={2} max={6} onChange={setClusters} />
        break
      }
      case 'segment-recurrence': {
        allow = MASKS.segment(seg)
        cat = (i, j) => (Math.floor(i / seg) === Math.floor(j / seg) ? 'default' : 'cached')
        legend = [
          ['default', `current segment (${seg})`],
          ['cached', 'previous segment (cached hidden states, no gradient)'],
        ]
        describe = 'Transformer-XL: attend to your segment plus the cached previous one; effective context grows with depth × segment length.'
        extraControls = <Slider label="segment length" value={seg} min={2} max={16} onChange={setSeg} />
        break
      }
      case 'ring': {
        allow = MASKS.causal()
        const per = Math.ceil(n / devices)
        cat = (i, j) => `dev${Math.floor(j / per) % 4}`
        legend = Array.from({ length: Math.min(devices, 4) }, (_, d) => [`dev${d}`, `keys held by device ${d}`])
        describe = 'Ring Attention: exact causal attention, but K/V blocks live on different devices and rotate around a ring while each device computes its query block.'
        extraControls = <Slider label="devices" value={devices} min={2} max={4} onChange={setDevices} />
        break
      }
      case 'lowrank': {
        m = rank
        allow = () => true
        cat = () => 'compressed'
        legend = [['compressed', `keys projected to k=${rank} rows`]]
        describe = 'Linformer: project the n keys/values down to k along the sequence axis; attention is n×k. Bidirectional only, length baked into the weights.'
        extraControls = <Slider label="rank k" value={rank} min={2} max={16} onChange={setRank} />
        break
      }
      case 'local-global': {
        allow = MASKS.window(w)
        cat = () => 'window'
        legend = [['window', `local layer window (${w})`]]
        describe = 'Local:global interleave (Gemma 2/3, gpt-oss, Command A): most layers use a window; every r-th layer is dense so information still crosses the whole context.'
        extraControls = <Slider label="local window" value={w} min={1} max={16} onChange={setW} />
        break
      }
      case 'causal':
      default: {
        allow = MASKS.causal()
        describe = 'Standard causal attention: every query reads every earlier key. n(n+1)/2 scores per head per layer.'
      }
    }
    return { allow, cat, legend, describe, extraControls, m }
  }, [mode, n, w, s, g, k, block, top, stride, pattern, rounds, clusters, seg, devices, rank, S, K])

  // Softmax weights over allowed keys per row (so colour = actual weight).
  const weights = useMemo(() => {
    if (mode === 'lowrank') {
      // n×k projected: fake by pooling scores into k groups
      return S.map((row) => {
        const per = Math.ceil(n / m)
        const pooled = Array.from({ length: m }, (_, b) => {
          let acc = 0
          let c = 0
          for (let j = b * per; j < Math.min(n, (b + 1) * per); j++) {
            acc += row[j]
            c++
          }
          return c ? acc / c : -Infinity
        })
        return softmax(pooled)
      })
    }
    return S.map((row, i) => {
      const masked = row.map((v, j) => (allow(i, j) ? v : -Infinity))
      return softmax(masked)
    })
  }, [S, allow, mode, n, m])

  const cell = useMemo(
    () => (i, j) => {
      if (mode === 'lowrank') return { w: weights[i][j] * m * 0.5, cat: 'compressed' }
      if (!allow(i, j)) return null
      const rowMax = Math.max(...weights[i])
      return { w: rowMax > 0 ? weights[i][j] / rowMax : 0, cat: cat(i, j) }
    },
    [allow, weights, cat, mode, m],
  )

  const total = mode === 'lowrank' ? n * m : (() => {
    let c = 0
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (allow(i, j)) c++
    return c
  })()
  const dense = (n * (n + 1)) / 2
  const pct = Math.min(100, (100 * total) / dense)
  const rowReads = hoverRow == null ? null : (() => {
    let c = 0
    for (let j = 0; j < n; j++) if (allow(hoverRow, j)) c++
    return c
  })()

  const cacheNote = (() => {
    switch (mode) {
      case 'window':
        return `KV cache bounded at ${w} tokens`
      case 'window-sinks':
        return `KV cache bounded at ${w + s} tokens`
      case 'local-global':
        return `local layers cache ${w}; global layers cache all n`
      case 'lowrank':
        return `keys stored: ${m} (fixed)`
      case 'topk':
        return 'full KV stored; dense scores computed then masked'
      case 'dsa':
        return `full latent KV stored; ${k} entries gathered per query`
      case 'csa':
        return `KV cache ÷ ${stride} (one entry per ${stride} tokens) + ${w}-token window`
      case 'block-select':
        return 'full KV stored; block means indexed'
      case 'segment-recurrence':
        return `hidden states of ${seg}-token previous segment cached per layer`
      case 'ring':
        return 'full KV, sharded across devices'
      default:
        return 'full KV cache, grows with n'
    }
  })()

  return (
    <div className="panel p-4">
      <div className="grid gap-4 md:grid-cols-[auto_1fr]">
        <div>
          <MaskCanvas n={n} m={m} cell={cell} size={280} onHover={setHoverRow} ariaLabel={`${mode} attention pattern`} />
          <Legend items={legend} />
        </div>
        <div className="min-w-0 space-y-3">
          <p className="text-sm text-zinc-700 dark:text-zinc-200">{describe}</p>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            <Stat label="scores computed" value={`${total.toLocaleString()} / ${dense.toLocaleString()}`} />
            <Stat label="share of the causal n² bill" value={`${pct.toFixed(0)}%`} />
            <Stat label="KV cache" value={cacheNote} />
          </div>
          <div className="text-xs text-zinc-500 dark:text-zinc-400">
            {hoverRow == null
              ? 'Hover a row to see how many keys that query reads.'
              : `query ${hoverRow} reads ${rowReads} of ${hoverRow + 1} available keys`}
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            <Slider label="sequence length n" value={n} min={8} max={64} step={1} onChange={setN} />
            {extraControls}
          </div>
        </div>
      </div>
    </div>
  )
}
