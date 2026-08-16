import { useEffect, useMemo, useRef, useState } from 'react'
import Slider from '../../components/ui/Slider.jsx'
import LineChart from '../../components/ui/LineChart.jsx'
import useIsDark, { palette } from '../lib/useIsDark.js'
import { makeRng } from '../../lib/rng.js'
import { sinusoidal, alibiSlope, FREQ, dot } from '../lib/attnMath.js'

// The one question every positional scheme has to answer: what happens at a
// distance the model never saw in training? Two views:
//  • the frequency ladder — for RoPE-family schemes, how many full rotations
//    each 2-D band completes over the training length (bands that never
//    complete one are the ones that extrapolate into unseen angles), and how
//    PI / NTK / YaRN / base-scaling move those bars;
//  • score-vs-distance — q·k for a fixed pair as the offset grows past the
//    training length, per scheme.

const HEADS = 8

// RoFormer §3.4.3 long-term-decay bound: (1/(d/2)) Σⱼ |Sⱼ₊₁| with
// Sⱼ = Σᵢ<ⱼ e^{i·Δ·θᵢ}, normalised to 1 at Δ = 0. Deterministic and smooth,
// and it moves the right way under PI / NTK / YaRN / base changes.
function ropeKernel(dist, d, base, freqScale) {
  let re = 0
  let im = 0
  let acc = 0
  const half = d / 2
  for (let i = 0; i < half; i++) {
    const theta = Math.pow(base, (-2 * i) / d) * freqScale(i, d)
    re += Math.cos(dist * theta)
    im += Math.sin(dist * theta)
    acc += Math.sqrt(re * re + im * im)
  }
  const norm = (half * (half + 1)) / 2
  return acc / norm
}

function FrequencyLadder({ d, L, base, freqScale, height = 150 }) {
  const ref = useRef(null)
  const dark = useIsDark()
  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const width = canvas.clientWidth || 360
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = width * dpr
    canvas.height = height * dpr
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const p = palette(dark)
    ctx.fillStyle = p.bg
    ctx.fillRect(0, 0, width, height)
    const bands = d / 2
    const bw = width / bands
    const pad = 18
    // rotations over training length per band (log10 scale from 1e-3 to 1e4)
    const lo = -3
    const hi = 4
    const yFor = (rot) => {
      const v = Math.log10(Math.max(rot, 1e-9))
      const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo)))
      return height - pad - t * (height - 2 * pad)
    }
    // one-rotation line
    ctx.strokeStyle = p.danger
    ctx.setLineDash([4, 3])
    ctx.beginPath()
    ctx.moveTo(0, yFor(1))
    ctx.lineTo(width, yFor(1))
    ctx.stroke()
    ctx.setLineDash([])
    ctx.fillStyle = p.muted
    ctx.font = '10px JetBrains Mono, monospace'
    ctx.fillText('1 full rotation over L_train', 4, yFor(1) - 3)
    for (let i = 0; i < bands; i++) {
      const theta = Math.pow(base, (-2 * i) / d) * freqScale(i, d)
      const rot = (L * theta) / (2 * Math.PI)
      const y = yFor(rot)
      const color = rot >= 32 ? p.accent4 : rot >= 1 ? p.accent3 : p.danger
      ctx.fillStyle = color
      ctx.fillRect(i * bw + 1, y, Math.max(1, bw - 2), height - pad - y)
    }
    ctx.fillStyle = p.muted
    ctx.fillText('high-freq bands (i small) →→ low-freq bands', 4, height - 4)
  }, [d, L, base, freqScale, height, dark])
  return <canvas ref={ref} style={{ width: '100%', height }} className="block rounded-md" role="img" aria-label="RoPE frequency ladder" />
}

function Heat({ rows, cols, value, height = 150, greyFrom = Infinity, ariaLabel }) {
  const ref = useRef(null)
  const dark = useIsDark()
  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const width = canvas.clientWidth || 360
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = width * dpr
    canvas.height = height * dpr
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const p = palette(dark)
    ctx.fillStyle = p.bg
    ctx.fillRect(0, 0, width, height)
    const cw = width / cols
    const ch = height / rows
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (r >= greyFrom) {
          ctx.fillStyle = dark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)'
          ctx.fillRect(c * cw, r * ch, cw + 0.5, ch + 0.5)
          continue
        }
        const v = value(r, c) // in [-1, 1]
        const t = (v + 1) / 2
        const rr = Math.round(59 + (239 - 59) * t)
        const gg = Math.round(130 + (68 - 130) * t)
        const bb = Math.round(246 + (68 - 246) * t)
        ctx.fillStyle = `rgb(${rr},${gg},${bb})`
        ctx.fillRect(c * cw, r * ch, cw + 0.5, ch + 0.5)
      }
    }
    if (greyFrom < rows) {
      ctx.fillStyle = p.text
      ctx.font = '11px Inter, sans-serif'
      ctx.fillText('no row exists past the trained length', 8, greyFrom * ch + 16)
    }
  }, [rows, cols, value, height, greyFrom, dark])
  return <canvas ref={ref} style={{ width: '100%', height }} className="block rounded-md" role="img" aria-label={ariaLabel} />
}

const MODE_TEXT = {
  sinusoidal: 'Each dimension pair is a sinusoid at a geometrically spaced wavelength (2π … 10000·2π). Defined for any position — but the model has still never seen the far ones.',
  learned: 'A trainable row per position. Position L_train+1 does not exist: the table simply ends.',
  relative: 'A learned vector per relative offset, clipped at ±k: offsets beyond the clip are indistinguishable.',
  t5: 'A learned scalar per log-spaced bucket of the offset; every offset beyond 128 shares one bucket.',
  rope: 'Rotate each 2-D pair of q and k by pos·θᵢ; the dot product depends only on the offset. Bands that never complete a rotation over L_train meet unseen angles when you test longer.',
  alibi: 'No positional embedding. Subtract m·|i−j| from every score, with a fixed geometric slope per head. The far past is discounted, never undefined.',
  nope: 'Nothing is added. A causal decoder can still count how many tokens it sees; that implicit position is what the model learns to use.',
  pi: 'Position Interpolation: multiply every frequency by 1/s so s·L positions map into the trained range. All bands move — including the high-frequency ones that carry local order.',
  ntk: 'NTK-aware: raise the base instead. High-frequency bands barely move; low-frequency bands are stretched — the ones that were extrapolating anyway.',
  'dyn-ntk': 'Dynamic NTK: the same base change, but computed from the current sequence length — no change until you pass L_train.',
  yarn: 'YaRN / NTK-by-parts: bands with many rotations are left alone, bands with few are interpolated, a ramp in between — plus a temperature on the logits.',
  abf: 'Adjusted base frequency, trained in (Code Llama θ=1e6, Llama 3 θ=500K): the whole ladder shifts to slower rotation and the model is fine-tuned at the new length.',
  xpos: 'RoPE times an exponential decay ξ^{m−n}: rotation for relative position, decay for a locality prior.',
  rerope: 'ReRoPE / SelfExtend / DCA: exact positions inside a window, then clamp or floor-divide the offset so the model never sees a distance it was not trained on.',
  drope: 'DroPE: pretrain with RoPE, then remove it from every layer and recalibrate briefly at the ORIGINAL length. Nothing positional is left to be out-of-distribution at longer lengths.',
}

export default function PositionExplorer({ mode = 'rope' }) {
  const [L, setL] = useState(2048)
  const [s, setS] = useState(4)
  const [d, setD] = useState(64)
  const [base, setBase] = useState(10000)
  const isScaling = ['pi', 'ntk', 'dyn-ntk', 'yarn', 'abf', 'rerope'].includes(mode)
  const isRopeFamily = ['rope', 'xpos', 'drope', ...['pi', 'ntk', 'dyn-ntk', 'yarn', 'abf', 'rerope']].includes(mode)

  // effective frequency modifier for the mode
  const { freqScale, effBase, label } = useMemo(() => {
    switch (mode) {
      case 'pi':
        return { freqScale: FREQ.pi(s), effBase: base, label: `all frequencies ÷ ${s}` }
      case 'ntk':
      case 'dyn-ntk':
        return { freqScale: FREQ.ntk(s), effBase: base, label: `base' = base·${s}^(d/(d−2)) ≈ ${Math.round(base * Math.pow(s, d / (d - 2))).toLocaleString()}` }
      case 'yarn':
        return { freqScale: FREQ.yarn(s, L, 1, 32, base), effBase: base, label: `by-parts: keep ≥32 rotations, interpolate ≤1, ramp between; s=${s}` }
      case 'abf':
        return { freqScale: FREQ.none(), effBase: 1_000_000, label: 'base θ = 1,000,000 (Code Llama), fine-tuned at the new length' }
      default:
        return { freqScale: FREQ.none(), effBase: base, label: 'base θ = 10,000' }
    }
  }, [mode, s, base, d, L])

  // fixed random q, k pair with a shared component so the score is meaningful
  const { q, k } = useMemo(() => {
    const rng = makeRng(23)
    const t = Array.from({ length: d }, () => rng.gaussian())
    const q = t.map((x) => 0.9 * x + 0.3 * rng.gaussian())
    const k = t.map((x) => 0.9 * x + 0.3 * rng.gaussian())
    return { q, k }
  }, [d])

  // score vs distance curve out to s·L (or 4L for non-scaling modes)
  const curve = useMemo(() => {
    const maxDist = mode === 'alibi' ? 128 : Math.max(2, Math.round((isScaling ? s : 4) * L))
    const pts = 160
    const out = []
    for (let p = 0; p <= pts; p++) {
      const dist = Math.round((p / pts) * maxDist)
      let y = 0
      switch (mode) {
        case 'sinusoidal': {
          y = dot(sinusoidal(dist, d), sinusoidal(0, d)) / (d / 2)
          break
        }
        case 'learned': {
          if (dist > L) continue // the table ends: no point past L_train
          y = 0.5 + 0.3 * Math.sin(dist / 97)
          break
        }
        case 'alibi': {
          y = -alibiSlope(0, HEADS) * dist
          break
        }
        case 'nope':
        case 'drope': {
          y = dot(q, k) / d
          break
        }
        case 'relative': {
          y = dist <= 16 ? 0.6 - 0.02 * dist : 0.28
          break
        }
        case 't5': {
          y = 0.5 - 0.08 * Math.log2(1 + Math.min(dist, 128))
          break
        }
        case 'xpos': {
          y = ropeKernel(dist, d, effBase, freqScale) * Math.pow(0.9995, dist)
          break
        }
        case 'rerope': {
          const eff = Math.min(dist, Math.round(L / 2))
          y = ropeKernel(eff, d, effBase, freqScale)
          break
        }
        case 'dyn-ntk': {
          // scale grows only once past L: factor = max(1, dist/L)
          const f = Math.max(1, dist / L)
          y = ropeKernel(dist, d, effBase, FREQ.ntk(f))
          break
        }
        default: {
          y = ropeKernel(dist, d, effBase, freqScale)
        }
      }
      if (Number.isFinite(y)) out.push({ x: dist, y })
    }
    return out
  }, [mode, L, s, d, q, k, effBase, freqScale, isScaling])

  const trainedMark = [
    { x: L, y: -1.2 },
    { x: L, y: 1.2 },
  ]

  const showLadder = isRopeFamily && mode !== 'drope'
  const showHeat = ['sinusoidal', 'learned'].includes(mode)
  const rows = 64
  const cols = Math.min(d, 64)
  const rng = useMemo(() => makeRng(3), [])
  const learnedTable = useMemo(() => Array.from({ length: rows }, () => Array.from({ length: cols }, () => rng.uniform(-1, 1))), [rows, cols, rng])

  return (
    <div className="panel p-4">
      <p className="mb-3 text-sm text-zinc-700 dark:text-zinc-200">{MODE_TEXT[mode] ?? MODE_TEXT.rope}</p>
      <div className="grid gap-4 md:grid-cols-2">
        <div>
          {showLadder && (
            <>
              <div className="mb-1 font-mono text-[11px] text-zinc-500">
                rotations per band over L_train = {L.toLocaleString()} · {label}
              </div>
              <FrequencyLadder d={d} L={L} base={effBase} freqScale={freqScale} />
              <div className="mt-1 flex flex-wrap gap-x-3 text-[11px] text-zinc-500 dark:text-zinc-400">
                <span><span className="inline-block h-2.5 w-2.5 rounded-sm bg-emerald-500" /> ≥32 rotations: local order, safe</span>
                <span><span className="inline-block h-2.5 w-2.5 rounded-sm bg-amber-500" /> 1–32</span>
                <span><span className="inline-block h-2.5 w-2.5 rounded-sm bg-red-500" /> &lt;1: unseen angles when tested longer</span>
              </div>
            </>
          )}
          {showHeat && (
            <>
              <div className="mb-1 font-mono text-[11px] text-zinc-500">
                {mode === 'sinusoidal' ? `PE(pos, dim): first ${rows} positions × ${cols} dims` : `learned table: ${rows} rows × ${cols} dims (L_train = 48 here)`}
              </div>
              <Heat
                rows={rows}
                cols={cols}
                value={mode === 'sinusoidal' ? (r, c) => sinusoidal(r, cols)[c] : (r, c) => learnedTable[r][c]}
                greyFrom={mode === 'learned' ? 48 : Infinity}
                ariaLabel={mode === 'sinusoidal' ? 'sinusoidal encoding heatmap' : 'learned position table'}
              />
            </>
          )}
          {mode === 'alibi' && (
            <>
              <div className="mb-1 font-mono text-[11px] text-zinc-500">per-head penalty −m·|i−j|, m = 2^(−8h/H), H = {HEADS}</div>
              <LineChart
                width={360}
                height={160}
                xLabel="distance |i−j| (0–32)"
                yLabel="bias"
                yMin={-alibiSlope(0, HEADS) * 32}
                yMax={0}
                showLegend={false}
                series={Array.from({ length: HEADS }, (_, h) => ({
                  label: '',
                  color: `hsl(${(h * 40) % 360} 70% 50%)`,
                  data: Array.from({ length: 33 }, (_, i) => ({ x: i, y: -alibiSlope(h, HEADS) * i })),
                }))}
              />
            </>
          )}
          {mode === 'drope' && (
            <div className="rounded-lg border border-dashed border-zinc-300 p-3 text-xs text-zinc-600 dark:border-zinc-700 dark:text-zinc-300">
              <div className="font-semibold">Recipe</div>
              <ol className="mt-1 list-decimal space-y-1 pl-4">
                <li>Pretrain with RoPE (it accelerates convergence — NoPE from scratch has vanishing positional-bias gradients).</li>
                <li>Delete the rotation from every attention layer.</li>
                <li>Continue pretraining briefly at the <b>original</b> length (0.5–2% of the token budget) so induction heads re-form without explicit position.</li>
                <li>Test at 2×–8×: no positional signal is out-of-distribution, so it extrapolates zero-shot.</li>
              </ol>
            </div>
          )}
          {['nope', 'relative', 't5', 'rerope', 'xpos'].includes(mode) && !showLadder && (
            <div className="rounded-lg border border-dashed border-zinc-300 p-3 text-xs text-zinc-600 dark:border-zinc-700 dark:text-zinc-300">
              {mode === 'nope'
                ? 'The only positional information is the causal mask: token i can see i tokens. Probes recover absolute position from the hidden states anyway (Haviv 2022).'
                : 'The curve on the right is the positional term this scheme adds to the score, as a function of offset.'}
            </div>
          )}
        </div>
        <div>
          <div className="mb-1 font-mono text-[11px] text-zinc-500">
            positional term vs distance (RoPE family: RoFormer §3.4.3 decay bound, normalised) · trained length marked at {L.toLocaleString()}
          </div>
          <LineChart
            width={360}
            height={190}
            xLabel="relative distance"
            yLabel="score"
            yMin={mode === 'alibi' ? -alibiSlope(0, HEADS) * 128 : isRopeFamily && mode !== 'drope' ? 0 : -1.2}
            yMax={mode === 'alibi' ? 0 : 1.2}
            series={
              mode === 'alibi'
                ? [{ label: 'head 0 (steepest slope)', color: '#3b82f6', data: curve }]
                : [
                    { label: mode, color: '#3b82f6', data: curve },
                    { label: 'L_train', color: '#ef4444', dashed: true, data: trainedMark },
                  ]
            }
          />
          <p className="mt-1 text-[11px] text-zinc-500 dark:text-zinc-400">
            {mode === 'rope' && 'Inside L_train the curve is what the model trained on. Beyond it, the low-frequency bands rotate into angles it never saw — the curve is still defined, the model is not.'}
            {mode === 'pi' && 'Everything is squeezed: the far region now looks like the near region — at the cost of resolution between neighbours.'}
            {(mode === 'ntk' || mode === 'dyn-ntk') && 'The near-range structure is kept; the far range is stretched into the trained band. Some bands still extrapolate a little, which is why YaRN treats bands separately.'}
            {mode === 'yarn' && 'High-frequency detail untouched, low-frequency bands brought into range, temperature keeps the softmax as sharp as it was in training.'}
            {mode === 'alibi' && 'The bias is the same linear function at every length (shown for the steepest head over 0–128 tokens; the shallowest head is 256× flatter). Extrapolation in perplexity comes free — because far tokens are simply discounted.'}
            {(mode === 'nope' || mode === 'drope') && 'Flat: there is no positional term to go out of distribution.'}
            {mode === 'sinusoidal' && 'Defined at any distance and quasi-periodic — but the model has only trained on the left of the red line.'}
            {mode === 'learned' && 'The curve stops: there is no embedding for positions past the table.'}
            {mode === 'abf' && 'A slower ladder: at base 1e6 even the low bands stay in a sane range out to ~100K, once fine-tuned there.'}
            {mode === 'rerope' && 'The offset is clamped at the window: beyond it, every distance looks the same (exact locally, coarse far away).'}
            {mode === 'xpos' && 'RoPE with a locality decay baked in — a bridge to RetNet.'}
          </p>
        </div>
      </div>
      <div className="mt-3 grid gap-2 sm:grid-cols-3">
        <Slider label="training length L" value={L} min={256} max={8192} step={256} onChange={setL} format={(v) => v.toLocaleString()} />
        {isScaling && mode !== 'abf' && <Slider label="extension factor s" value={s} min={1} max={16} step={1} onChange={setS} format={(v) => `${v}×`} />}
        {isRopeFamily && <Slider label="RoPE dims d" value={d} min={16} max={128} step={16} onChange={setD} />}
        {['rope', 'pi', 'ntk', 'dyn-ntk', 'yarn'].includes(mode) && (
          <Slider label="base θ" value={base} min={1000} max={1000000} step={1000} onChange={setBase} format={(v) => v.toLocaleString()} />
        )}
      </div>
    </div>
  )
}
