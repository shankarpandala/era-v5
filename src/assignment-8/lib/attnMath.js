// Small numeric helpers shared by the visualizers. Everything is seeded through
// src/lib/rng.js so a given seed reproduces the same picture.

import { makeRng } from '../../lib/rng.js'

export function softmax(xs) {
  const m = Math.max(...xs.filter(Number.isFinite))
  const ex = xs.map((x) => (Number.isFinite(x) ? Math.exp(x - m) : 0))
  const s = ex.reduce((a, b) => a + b, 0) || 1
  return ex.map((e) => e / s)
}

export function dot(a, b) {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

export function randVec(rng, d, scale = 1) {
  return Array.from({ length: d }, () => rng.gaussian(0, scale))
}

// Seeded Q, K, V for n tokens, d dims. Adds a little "semantic" structure so
// the heatmap is not pure noise: token i shares a component with token i%3.
export function makeQKV(n, d, seed = 7) {
  const rng = makeRng(seed)
  const themes = [randVec(rng, d), randVec(rng, d), randVec(rng, d)]
  const Q = []
  const K = []
  const V = []
  for (let i = 0; i < n; i++) {
    const t = themes[i % 3]
    Q.push(t.map((x) => 0.8 * x + 0.6 * rng.gaussian()))
    K.push(t.map((x) => 0.8 * x + 0.6 * rng.gaussian()))
    V.push(randVec(rng, d))
  }
  return { Q, K, V }
}

// Raw score matrix S[i][j] = q_i·k_j (no scaling, no mask).
export function scores(Q, K) {
  return Q.map((q) => K.map((k) => dot(q, k)))
}

// ---- masks -----------------------------------------------------------------
// A mask function returns true if query i may read key j. Every "pattern"
// mechanism on the timeline is one of these.

export const MASKS = {
  full: () => () => true,
  causal: () => (i, j) => j <= i,
  window: (w) => (i, j) => j <= i && i - j < w,
  windowSinks: (w, s) => (i, j) => j <= i && (j < s || i - j < w),
  strided: (l) => (i, j) => j <= i && (i - j < l || (i - j) % l === 0),
  fixed: (l, c = 1) => (i, j) => j <= i && (Math.floor(i / l) === Math.floor(j / l) || j % l >= l - c),
  windowGlobal: (w, g) => (i, j) => j <= i && (i - j < w || j < g || i < g),
  bigbird: (w, g, rnd, seed = 3) => {
    const rng = makeRng(seed)
    const cache = new Map()
    return (i, j) => {
      if (j > i) return false
      if (i - j < w || j < g) return true
      const key = i
      if (!cache.has(key)) {
        const set = new Set()
        for (let r = 0; r < rnd; r++) set.add(rng.int(i + 1))
        cache.set(key, set)
      }
      return cache.get(key).has(j)
    }
  },
  segment: (seg) => (i, j) => j <= i && Math.floor(i / seg) - Math.floor(j / seg) <= 1,
  blocks: (b) => (i, j) => j <= i && Math.floor(i / b) === Math.floor(j / b),
}

// Content-based masks need the scores.
export function topkMask(S, k) {
  const n = S.length
  const keep = S.map((row, i) => {
    const idx = row.map((v, j) => [v, j]).filter(([, j]) => j <= i).sort((a, b) => b[0] - a[0]).slice(0, k)
    return new Set(idx.map(([, j]) => j))
  })
  return (i, j) => keep[i].has(j)
}

// Block-select (MoBA / NSA "selected" branch): mean-pool keys per block, keep the
// top-b blocks per query (always own block), plus optional window and sinks.
export function blockSelectMask(S, block, topBlocks, { window = 0, sinks = 0 } = {}) {
  const n = S.length
  const nb = Math.ceil(n / block)
  const keep = []
  for (let i = 0; i < n; i++) {
    const own = Math.floor(i / block)
    const blockScores = []
    for (let b = 0; b <= own; b++) {
      let s = 0
      let c = 0
      for (let j = b * block; j < Math.min(n, (b + 1) * block); j++) {
        if (j <= i) {
          s += S[i][j]
          c++
        }
      }
      blockScores.push([c ? s / c : -Infinity, b])
    }
    const chosen = new Set(blockScores.sort((a, b) => b[0] - a[0]).slice(0, topBlocks).map(([, b]) => b))
    chosen.add(own)
    keep.push(chosen)
  }
  return (i, j) => j <= i && (keep[i].has(Math.floor(j / block)) || (window && i - j < window) || j < sinks)
}

// LSH buckets: hash by sign of projection onto a few random directions (Reformer-style).
export function lshMask(K, rounds = 2, seed = 5) {
  const rng = makeRng(seed)
  const d = K[0].length
  const dirs = Array.from({ length: rounds }, () => randVec(rng, d))
  const bucket = K.map((k) => dirs.map((u) => (dot(k, u) >= 0 ? 1 : 0)).join(''))
  return (i, j) => j <= i && bucket[i] === bucket[j]
}

// Routing (k-means-ish): assign to nearest of c seeded centroids.
export function routingMask(K, c = 3, seed = 11) {
  const rng = makeRng(seed)
  const d = K[0].length
  const cents = Array.from({ length: c }, () => randVec(rng, d))
  const assign = K.map((k) => {
    let best = 0
    let bs = -Infinity
    cents.forEach((ct, idx) => {
      const s = dot(k, ct)
      if (s > bs) {
        bs = s
        best = idx
      }
    })
    return best
  })
  return (i, j) => j <= i && assign[i] === assign[j]
}

export function countAllowed(n, mask) {
  let c = 0
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (mask(i, j)) c++
  return c
}

// ---- positional encodings --------------------------------------------------

export function sinusoidal(pos, d, base = 10000) {
  const out = new Array(d)
  for (let i = 0; i < d / 2; i++) {
    const w = 1 / Math.pow(base, (2 * i) / d)
    out[2 * i] = Math.sin(pos * w)
    out[2 * i + 1] = Math.cos(pos * w)
  }
  return out
}

// RoPE: rotate pairs of dims of x by pos·θ_i. `freqScale(i)` lets PI/NTK/YaRN
// modify per-band frequency; base changes θ.
export function rope(x, pos, base = 10000, freqScale = () => 1) {
  const d = x.length
  const out = x.slice()
  for (let i = 0; i < d / 2; i++) {
    const theta = Math.pow(base, (-2 * i) / d) * freqScale(i, d)
    const a = pos * theta
    const c = Math.cos(a)
    const s = Math.sin(a)
    const x1 = x[2 * i]
    const x2 = x[2 * i + 1]
    out[2 * i] = x1 * c - x2 * s
    out[2 * i + 1] = x1 * s + x2 * c
  }
  return out
}

// Frequency modifiers for the scaling family (s = extension factor, d = dims).
export const FREQ = {
  none: () => () => 1,
  pi: (s) => () => 1 / s,
  // NTK-aware: base' = base·s^{d/(d-2)} ⇒ per-band scale = s^{-2i/(d-2)}
  ntk: (s) => (i, d) => Math.pow(s, (-2 * i) / (d - 2)),
  // YaRN / NTK-by-parts: high-freq bands untouched, low-freq interpolated, ramp between.
  yarn: (s, L = 2048, alpha = 1, beta = 32) => (i, d) => {
    const theta = Math.pow(10000, (-2 * i) / d)
    const wavelength = (2 * Math.PI) / theta
    const r = L / wavelength // number of rotations over the training length
    if (r >= beta) return 1 // high frequency: keep
    if (r <= alpha) return 1 / s // low frequency: interpolate
    const t = (r - alpha) / (beta - alpha)
    return (1 - t) / s + t
  },
}

// ALiBi slope for head h of H heads: 2^{-8h/H}
export function alibiSlope(h, H) {
  return Math.pow(2, -(8 * (h + 1)) / H)
}
