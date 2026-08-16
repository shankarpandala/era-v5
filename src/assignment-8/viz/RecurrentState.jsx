import { useEffect, useMemo, useRef, useState } from 'react'
import Slider from '../../components/ui/Slider.jsx'
import Toggle from '../../components/ui/Toggle.jsx'
import useIsDark, { palette } from '../lib/useIsDark.js'
import { makeRng } from '../../lib/rng.js'

// The linear / recurrent family replaces the KV cache with a fixed d×d state S.
// Write: S ← f(S) + v·kᵀ (or the delta-rule correction). Read: v̂ = S·q.
// The widget writes T (key, value) pairs and then asks the memory for every
// stored key back: the recall error per stored item is the honest picture of
// what a bounded state can and cannot remember, and how gating / the delta
// rule change it. State size never grows — that is the whole point.

function unit(v) {
  const n = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1
  return v.map((x) => x / n)
}

function outer(v, k) {
  return v.map((vi) => k.map((kj) => vi * kj))
}

function matVec(S, k) {
  return S.map((row) => row.reduce((a, x, j) => a + x * k[j], 0))
}

const RULES = {
  linear: {
    label: 'linear attention (sum rule)',
    formula: 'S_t = S_{t−1} + v_t k_tᵀ',
    step: (S, k, v) => S.map((row, i) => row.map((x, j) => x + v[i] * k[j])),
  },
  retention: {
    label: 'RetNet retention (fixed decay γ)',
    formula: 'S_t = γ S_{t−1} + v_t k_tᵀ',
    step: (S, k, v, { gamma }) => S.map((row, i) => row.map((x, j) => gamma * x + v[i] * k[j])),
  },
  gla: {
    label: 'gated linear attention (data-dependent decay)',
    formula: 'S_t = diag(α_t) S_{t−1} + v_t k_tᵀ',
    step: (S, k, v, { alpha }) => S.map((row, i) => row.map((x, j) => alpha[j] * x + v[i] * k[j])),
  },
  rwkv: {
    label: 'RWKV-style time-mix (per-channel decay)',
    formula: 'S_t = diag(w) S_{t−1} + v_t k_tᵀ',
    step: (S, k, v, { alpha }) => S.map((row, i) => row.map((x, j) => alpha[j] * x + v[i] * k[j])),
  },
  mamba: {
    label: 'selective SSM (input-dependent decay + gate)',
    formula: 'S_t = diag(α_t) S_{t−1} + Δ_t v_t k_tᵀ',
    step: (S, k, v, { alpha }) => S.map((row, i) => row.map((x, j) => alpha[j] * x + 0.9 * v[i] * k[j])),
  },
  delta: {
    label: 'delta rule (erase, then write)',
    formula: 'S_t = S_{t−1} − β (S_{t−1} k_t) k_tᵀ + β v_t k_tᵀ',
    step: (S, k, v, { beta }) => {
      const old = matVec(S, k)
      return S.map((row, i) => row.map((x, j) => x - beta * old[i] * k[j] + beta * v[i] * k[j]))
    },
  },
  'gated-delta': {
    label: 'gated delta rule (decay + erase + write)',
    formula: 'S_t = α_t (I − β k_t k_tᵀ) S_{t−1} + β v_t k_tᵀ',
    step: (S, k, v, { beta, gamma }) => {
      const old = matVec(S, k)
      return S.map((row, i) => row.map((x, j) => gamma * (x - beta * old[i] * k[j]) + beta * v[i] * k[j]))
    },
  },
}

function StateHeat({ S, size = 150 }) {
  const ref = useRef(null)
  const dark = useIsDark()
  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = size * dpr
    canvas.height = size * dpr
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const p = palette(dark)
    ctx.fillStyle = p.bg
    ctx.fillRect(0, 0, size, size)
    const n = S.length
    const m = S[0].length
    const cw = size / m
    const ch = size / n
    const mx = Math.max(...S.flat().map(Math.abs), 1e-6)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < m; j++) {
        const v = S[i][j] / mx
        const a = Math.min(1, Math.abs(v))
        ctx.fillStyle = v >= 0 ? `rgba(59,130,246,${0.08 + 0.9 * a})` : `rgba(239,68,68,${0.08 + 0.9 * a})`
        ctx.fillRect(j * cw, i * ch, cw + 0.5, ch + 0.5)
      }
    ctx.strokeStyle = p.frame
    ctx.strokeRect(0.5, 0.5, size - 1, size - 1)
  }, [S, size, dark])
  return <canvas ref={ref} style={{ width: size, height: size }} className="block rounded-md" role="img" aria-label="recurrent state matrix" />
}

export default function RecurrentState({ mode = 'linear' }) {
  const [T, setT] = useState(12)
  const [d, setD] = useState(8)
  const [beta, setBeta] = useState(1)
  const [gamma, setGamma] = useState(0.9)
  const [reuse, setReuse] = useState(true)
  const [step, setStep] = useState(12)
  const rule = RULES[mode] || RULES.linear

  const { keys, vals, alphas } = useMemo(() => {
    const rng = makeRng(17)
    const keys = Array.from({ length: T }, () => unit(Array.from({ length: d }, () => rng.gaussian())))
    const vals = Array.from({ length: T }, () => unit(Array.from({ length: d }, () => rng.gaussian())))
    const alphas = Array.from({ length: T }, () => Array.from({ length: d }, () => 0.75 + 0.24 * rng.next()))
    if (reuse && T >= 4) keys[T - 2] = keys[1].slice() // same key, new value: the overwrite test
    return { keys, vals, alphas }
  }, [T, d, reuse])

  const states = useMemo(() => {
    let S = Array.from({ length: d }, () => new Array(d).fill(0))
    const out = [S]
    for (let t = 0; t < T; t++) {
      S = rule.step(S, keys[t], vals[t], { beta, gamma, alpha: alphas[t] })
      out.push(S)
    }
    return out
  }, [rule, keys, vals, alphas, T, d, beta, gamma])

  const cur = Math.min(step, T)
  const S = states[cur]

  // recall: ask for every stored key; for a reused key the "right" answer is the latest value
  const recall = useMemo(() => {
    const out = []
    for (let t = 0; t < cur; t++) {
      const k = keys[t]
      const target = reuse && t === 1 && cur > T - 2 ? vals[T - 2] : vals[t]
      const got = matVec(S, k)
      const n = Math.sqrt(got.reduce((a, x) => a + x * x, 0)) || 1
      const gotU = got.map((x) => x / n)
      const cos = gotU.reduce((a, x, i) => a + x * target[i], 0)
      out.push({ t, err: Math.max(0, 1 - cos) })
    }
    return out
  }, [S, keys, vals, cur, reuse, T])

  const meanErr = recall.length ? recall.reduce((a, r) => a + r.err, 0) / recall.length : 0

  return (
    <div className="panel p-4">
      <div className="grid gap-4 md:grid-cols-[auto_1fr]">
        <div>
          <div className="mb-1 font-mono text-[11px] text-zinc-500">state S after {cur} tokens · {d}×{d}, fixed</div>
          <StateHeat S={S} />
          <div className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
            <div className="font-mono">{rule.formula}</div>
            <div className="mt-1">
              memory held: <b>{d * d}</b> numbers, no matter how long the sequence · a KV cache would hold <b>{2 * cur * d}</b> after {cur} tokens
            </div>
          </div>
        </div>
        <div className="min-w-0">
          <div className="mb-1 font-mono text-[11px] text-zinc-500">recall test: ask the state for each stored key → error (0 = exact)</div>
          <div className="flex h-28 items-end gap-[3px] rounded-md border border-zinc-200 p-2 dark:border-zinc-800">
            {recall.map(({ t, err }) => (
              <div key={t} className="flex h-full flex-1 flex-col items-center justify-end" title={`token ${t}: error ${err.toFixed(2)}`}>
                <div
                  className="w-full rounded-t"
                  style={{
                    height: `${Math.max(2, Math.min(84, err * 84))}px`,
                    backgroundColor: reuse && t === 1 ? '#d946ef' : err > 0.5 ? '#ef4444' : err > 0.2 ? '#f59e0b' : '#10b981',
                  }}
                />
                <span className="mt-0.5 font-mono text-[9px] text-zinc-400">{t}</span>
              </div>
            ))}
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 text-xs sm:grid-cols-3">
            <div className="rounded-lg border border-zinc-200 px-2 py-1.5 dark:border-zinc-800">
              <div className="text-[10px] uppercase text-zinc-500">mean recall error</div>
              <div className="font-mono font-semibold">{meanErr.toFixed(2)}</div>
            </div>
            <div className="rounded-lg border border-zinc-200 px-2 py-1.5 dark:border-zinc-800">
              <div className="text-[10px] uppercase text-zinc-500">items stored / capacity d</div>
              <div className="font-mono font-semibold">{cur} / {d}</div>
            </div>
            {reuse && cur > T - 2 && (
              <div className="rounded-lg border border-pink-300 px-2 py-1.5 dark:border-pink-800">
                <div className="text-[10px] uppercase text-pink-600 dark:text-pink-300">overwrite test (token 1 re-keyed at {T - 2})</div>
                <div className="font-mono font-semibold">error {recall[1]?.err.toFixed(2)}</div>
              </div>
            )}
          </div>
          <p className="mt-2 text-[11px] text-zinc-500 dark:text-zinc-400">
            {mode === 'linear' && 'The sum rule cannot overwrite: a re-used key returns the average of both values, and once more items than dimensions are stored everything blurs together — the recall weakness of plain linear attention.'}
            {(mode === 'retention' || mode === 'rwkv') && 'A fixed decay forgets old items smoothly — recent recall is sharp, old recall fades whether or not it mattered.'}
            {(mode === 'gla' || mode === 'mamba') && 'Data-dependent decay lets the model choose what to forget per channel; still additive, so re-used keys still average.'}
            {mode === 'delta' && 'Read what the state returns for the key, subtract it, write the new value: a re-used key returns the latest value exactly. Capacity is still bounded by d — the state cannot hold more than d orthogonal keys sharply.'}
            {mode === 'gated-delta' && 'Erase-then-write plus a decay: exact overwrites and graceful forgetting. This is the layer inside Qwen3-Next and Kimi Linear.'}
          </p>
        </div>
      </div>
      <div className="mt-3 grid gap-2 sm:grid-cols-4">
        <Slider label="tokens written T" value={T} min={4} max={24} onChange={(v) => { setT(v); setStep(v) }} />
        <Slider label="state dim d" value={d} min={4} max={16} onChange={setD} />
        <Slider label="show state after step" value={cur} min={0} max={T} onChange={setStep} />
        {(mode === 'delta' || mode === 'gated-delta') && <Slider label="write strength β" value={beta} min={0} max={1} step={0.05} onChange={setBeta} format={(v) => v.toFixed(2)} />}
        {(mode === 'retention' || mode === 'gated-delta') && <Slider label="decay γ" value={gamma} min={0.5} max={1} step={0.01} onChange={setGamma} format={(v) => v.toFixed(2)} />}
        <Toggle label="overwrite test" checked={reuse} onChange={setReuse} hint="re-use token 1's key near the end with a new value" />
      </div>
    </div>
  )
}
