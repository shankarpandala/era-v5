// Unit tests for the assignment-8 formulas: the cost model's closed forms
// against brute-force sums, the mask generators' invariants (causality, k per
// row, own-block always attended), and the position-scaling identities.
//
//   node --test tests/assignment8.test.mjs      (also: npm run a8:test)

import test from 'node:test'
import assert from 'node:assert/strict'
import {
  denseFlops,
  windowFlops,
  topkFlops,
  linearFlops,
  kvBytesDense,
  kvBytesWindow,
  kvBytesMLA,
  kvBytesLinear,
  modelTotals,
  layoutComparison,
  fmtTokens,
  PRESETS,
} from '../src/assignment-8/lib/costModel.js'
import {
  softmax,
  makeQKV,
  scores,
  MASKS,
  topkMask,
  blockSelectMask,
  lshMask,
  routingMask,
  countAllowed,
  rope,
  FREQ,
  alibiSlope,
  sinusoidal,
} from '../src/assignment-8/lib/attnMath.js'

// ---- cost model ------------------------------------------------------------

test('denseFlops equals the brute-force causal sum 4·d·Σᵢ i (i = 1..n)', () => {
  for (const n of [1, 7, 64, 1000]) {
    const d = 4096
    let brute = 0
    for (let i = 1; i <= n; i++) brute += 4 * d * i
    assert.equal(denseFlops(n, d), brute)
  }
})

test('windowFlops equals brute force 4·d·Σᵢ min(i, w) and reduces to dense when w ≥ n', () => {
  const d = 512
  for (const [n, w] of [[10, 3], [100, 17], [50, 50], [50, 200]]) {
    let brute = 0
    for (let i = 1; i <= n; i++) brute += 4 * d * Math.min(i, w)
    assert.equal(windowFlops(n, d, w), brute)
  }
  assert.equal(windowFlops(40, d, 40), denseFlops(40, d))
})

test('topkFlops = read k keys per query + an indexer that scores every key', () => {
  const n = 100
  const d = 512
  let reads = 0
  for (let i = 1; i <= n; i++) reads += 4 * d * Math.min(i, 8)
  const indexer = 2 * 64 * 128 * ((n * (n + 1)) / 2)
  assert.equal(topkFlops(n, d, 8), reads + indexer)
})

test('linearFlops is linear in n and the delta rule costs 2× the sum rule', () => {
  assert.equal(linearFlops(200, 8, 64), 2 * linearFlops(100, 8, 64))
  assert.equal(linearFlops(100, 8, 64, { delta: true }), 2 * linearFlops(100, 8, 64))
})

test('KV bytes: MHA/GQA/MQA/MLA/window/linear formulas', () => {
  // Llama-2-7B: 32 layers × 2 × 32 heads × 128 × 2 B = 0.5 MB / token
  assert.equal(32 * kvBytesDense(1, 32, 128, 2), 524288)
  // GQA-8 is 4× smaller than MHA-32
  assert.equal(kvBytesDense(1000, 32, 128, 2) / kvBytesDense(1000, 8, 128, 2), 4)
  // MLA (512 + 64) vs MHA 128 heads × 128: 56.9×
  const ratio = kvBytesDense(1, 128, 128, 2) / kvBytesMLA(1, 512, 64, 2)
  assert.ok(ratio > 56.8 && ratio < 57, `got ${ratio}`)
  // window caps at w + sinks tokens
  assert.equal(kvBytesWindow(100000, 8, 128, 2, 4096, 4), kvBytesDense(4100, 8, 128, 2))
  assert.equal(kvBytesWindow(100, 8, 128, 2, 4096, 4), kvBytesDense(100, 8, 128, 2))
  // linear state is independent of n
  assert.equal(kvBytesLinear(32, 128, 2), 32 * 128 * 128 * 2)
})

test('presets: dense KV grows linearly, window KV saturates, linear-hybrid KV grows only through its attention layers', () => {
  const llama = PRESETS.find((p) => p.key === 'llama2-7b')
  assert.equal(modelTotals(llama, 2000, 2).kv, 2 * modelTotals(llama, 1000, 2).kv)
  const mistral = PRESETS.find((p) => p.key === 'mistral-7b')
  assert.equal(modelTotals(mistral, 8192, 2).kv, modelTotals(mistral, 65536, 2).kv)
  const qwen = PRESETS.find((p) => p.key === 'qwen3-next')
  const a = modelTotals(qwen, 1024, 2).kv
  const b = modelTotals(qwen, 2048, 2).kv
  // 12 attention layers × 2 × 2 KV heads × 256 × 2 B = 24,576 B / token
  assert.equal(b - a, 1024 * 12 * 2 * 2 * 256 * 2)
  for (const p of PRESETS) {
    const t = modelTotals(p, 1 << 20, 2)
    assert.ok(Number.isFinite(t.kv) && Number.isFinite(t.flops) && t.kv > 0 && t.flops > 0, p.key)
  }
})

test('layoutComparison keeps MHA ≥ GQA ≥ MQA and MLA between them at DeepSeek numbers', () => {
  const rows = Object.fromEntries(layoutComparison({ layers: 60, heads: 128, dHead: 128, groups: 8, latent: 512, ropeDim: 64 }, 4096).map((r) => [r.key, r]))
  assert.ok(rows.mha.total > rows.gqa.total && rows.gqa.total > rows.mqa.total)
  assert.ok(rows.mla.total < rows.gqa.total && rows.mla.total > rows.mqa.total)
})

test('fmtTokens uses binary units consistently', () => {
  assert.equal(fmtTokens(1048576), '1M')
  assert.equal(fmtTokens(524288), '512K')
  assert.equal(fmtTokens(2048), '2K')
  assert.equal(fmtTokens(128), '128')
})

// ---- masks -----------------------------------------------------------------

test('softmax sums to 1 and ignores −Infinity entries', () => {
  const p = softmax([1, 2, -Infinity, 3])
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12)
  assert.equal(p[2], 0)
})

test('every causal mask never lets a query read the future', () => {
  const n = 32
  const { K } = makeQKV(n, 16, 7)
  const S = scores(makeQKV(n, 16, 7).Q, K)
  const masks = [
    MASKS.causal(),
    MASKS.window(5),
    MASKS.windowSinks(5, 2),
    MASKS.strided(4),
    MASKS.fixed(4, 1),
    MASKS.windowGlobal(4, 2),
    MASKS.bigbird(3, 2, 2),
    MASKS.segment(8),
    MASKS.blocks(4),
    topkMask(S, 4),
    blockSelectMask(S, 4, 2, { window: 2, sinks: 1 }),
    lshMask(K, 2),
    routingMask(K, 3),
  ]
  for (const m of masks) for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) assert.equal(m(i, j), false)
})

test('window(w) reads exactly min(i+1, w) keys; sinks add the first s', () => {
  const w = 5
  const m = MASKS.window(w)
  for (let i = 0; i < 20; i++) {
    let c = 0
    for (let j = 0; j < 20; j++) if (m(i, j)) c++
    assert.equal(c, Math.min(i + 1, w))
  }
  const ms = MASKS.windowSinks(5, 2)
  let c = 0
  for (let j = 0; j < 20; j++) if (ms(19, j)) c++
  assert.equal(c, 7)
})

test('topkMask keeps at most k keys per row and every allowed key is causal', () => {
  const n = 24
  const { Q, K } = makeQKV(n, 16, 3)
  const S = scores(Q, K)
  const m = topkMask(S, 3)
  for (let i = 0; i < n; i++) {
    let c = 0
    for (let j = 0; j < n; j++) if (m(i, j)) c++
    assert.equal(c, Math.min(i + 1, 3))
  }
})

test('blockSelectMask always attends to the own block and at most top other blocks', () => {
  const n = 32
  const block = 4
  const top = 2
  const { Q, K } = makeQKV(n, 16, 5)
  const S = scores(Q, K)
  const m = blockSelectMask(S, block, top)
  for (let i = 0; i < n; i++) {
    const own = Math.floor(i / block)
    for (let j = own * block; j <= i; j++) assert.equal(m(i, j), true, `own block ${i},${j}`)
    const others = new Set()
    for (let j = 0; j < own * block; j++) if (m(i, j)) others.add(Math.floor(j / block))
    assert.ok(others.size <= top)
  }
})

test('countAllowed of a causal mask is n(n+1)/2', () => {
  assert.equal(countAllowed(16, MASKS.causal()), 136)
})

// ---- positions -------------------------------------------------------------

test('RoPE: rotating q and k by the same position leaves their dot product unchanged (relative-only)', () => {
  const q = [1, 0.5, -0.3, 0.8, 0.2, -0.7, 0.4, 0.1]
  const k = [0.3, -0.2, 0.9, 0.4, -0.5, 0.6, 0.1, 0.7]
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0)
  const d1 = dot(rope(q, 5), rope(k, 5))
  const d2 = dot(rope(q, 105), rope(k, 105))
  assert.ok(Math.abs(d1 - d2) < 1e-9)
  // and depends only on the offset
  const d3 = dot(rope(q, 12), rope(k, 7))
  const d4 = dot(rope(q, 1012), rope(k, 1007))
  assert.ok(Math.abs(d3 - d4) < 1e-9)
})

test('FREQ.pi scales every band by 1/s; FREQ.ntk equals a base change base·s^{d/(d−2)}', () => {
  const s = 4
  const d = 64
  for (let i = 0; i < d / 2; i++) {
    assert.equal(FREQ.pi(s)(i, d), 1 / s)
    const theta = Math.pow(10000, (-2 * i) / d) * FREQ.ntk(s)(i, d)
    const thetaBase = Math.pow(10000 * Math.pow(s, d / (d - 2)), (-2 * i) / d)
    assert.ok(Math.abs(theta - thetaBase) < 1e-15, `band ${i}`)
  }
})

test('FREQ.yarn keeps high-frequency bands, interpolates low-frequency bands, and honours the base', () => {
  const s = 8
  const d = 64
  const L = 2048
  const y = FREQ.yarn(s, L, 1, 32, 10000)
  assert.equal(y(0, d), 1) // ~326 rotations over L: untouched
  assert.equal(y(d / 2 - 1, d), 1 / s) // < 1 rotation: fully interpolated
  const mid = y(20, d)
  assert.ok(mid > 1 / s && mid < 1)
  // a larger base makes more bands "low-frequency" ⇒ scale ≤ the base-10000 scale for every band
  const yBig = FREQ.yarn(s, L, 1, 32, 1e6)
  for (let i = 0; i < d / 2; i++) assert.ok(yBig(i, d) <= y(i, d) + 1e-12)
})

test('alibiSlope is the geometric sequence 2^{-8(h+1)/H}', () => {
  assert.equal(alibiSlope(0, 8), 0.5)
  assert.equal(alibiSlope(7, 8), 2 ** -8)
})

test('sinusoidal PE has unit-norm pairs', () => {
  const pe = sinusoidal(37, 16)
  for (let i = 0; i < 8; i++) assert.ok(Math.abs(pe[2 * i] ** 2 + pe[2 * i + 1] ** 2 - 1) < 1e-12)
})
