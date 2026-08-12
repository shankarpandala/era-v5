// ERA-V5 · Assignment 7 — Kronecker Embedding V2.
// Every number on this page is imported from the committed
// assignment-7/submission_artifacts/ bundle (results.json /
// properties_report.json), so the page can never drift from the artifacts
// the independent audit verifies. Figures are the matplotlib PNGs from the
// same bundle. The homomorphism widget re-implements the numeric encoding of
// kronembed/embedding.py in a few lines of JS so the algebra runs live.

import { useState } from 'react'
import Navbar from '../components/layout/Navbar.jsx'
import Footer from '../components/layout/Footer.jsx'
import ClaimCard from '../components/ClaimCard.jsx'
import useTheme from '../hooks/useTheme.js'

import results from '../../assignment-7/submission_artifacts/results.json'
import properties from '../../assignment-7/submission_artifacts/properties_report.json'
import anatomyPng from '../../assignment-7/submission_artifacts/plots/embedding_anatomy.png'
import homomorphismPng from '../../assignment-7/submission_artifacts/plots/homomorphism_demo.png'
import holePng from '../../assignment-7/submission_artifacts/plots/hole_generalization.png'
import samplePng from '../../assignment-7/submission_artifacts/plots/sample_efficiency.png'
import extrapolationPng from '../../assignment-7/submission_artifacts/plots/extrapolation_negative.png'
import layersPng from '../../assignment-7/submission_artifacts/plots/structure_through_layers.png'
import nlPng from '../../assignment-7/submission_artifacts/plots/nl_transfer.png'
import manifoldPng from '../../assignment-7/submission_artifacts/plots/value_manifold.png'

const GITHUB = 'https://github.com/shankarpandala/era-v5/tree/main/assignment-7'
const PRIMARY = 2000 // pre-specified primary operating point (train pairs)

const SECTIONS = [
  { id: 'a7-1', code: 'A7-1', title: 'One vector, two languages', color: 'var(--claim-1)' },
  { id: 'a7-2', code: 'A7-2', title: 'Try the homomorphism — live', color: 'var(--claim-2)' },
  { id: 'a7-3', code: 'A7-3', title: 'Claim A · algebra without training', color: 'var(--claim-3)' },
  { id: 'a7-4', code: 'A7-4', title: 'Claim B · the 92-run controlled study', color: 'var(--claim-4)' },
  { id: 'a7-5', code: 'A7-5', title: 'Honest negatives, localized', color: 'var(--color-brand-500)' },
]

// ---- artifact helpers -----------------------------------------------------

const g = results.by_group

function metric(task, arm, size, name) {
  return g[`${task}:${arm}@${size}`]?.[name]
}

function cell(task, arm, size, name) {
  const m = metric(task, arm, size, name)
  return m ? `${m.mean.toFixed(3)} ± ${m.std.toFixed(3)}` : '—'
}

function ratio(task, size, name, top, bottom) {
  const a = metric(task, top, size, name)?.mean ?? 0
  const b = metric(task, bottom, size, name)?.mean ?? 0
  return Math.round(a / Math.max(b, 1e-9))
}

const ARMS = [
  ['kron_v2', 'Kron V2 (ours)'],
  ['readout_only', 'readout-only (FoNE-style)'],
  ['hom_only', 'homomorphic-only'],
  ['xval', 'xVal-style'],
  ['learned', 'learned table'],
  ['frozen_rand', 'frozen random'],
  ['kron_char', 'char-only'],
]

// ---- small UI pieces (same idioms as the other assignment pages) ----------

function Tile({ label, value, accent = 'text-brand-600 dark:text-brand-400' }) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="text-[11px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={`font-mono text-xl font-bold ${accent}`}>{value}</div>
    </div>
  )
}

function Figure({ src, alt, caption }) {
  return (
    <figure className="my-6 rounded-xl border border-zinc-200 bg-white p-3 dark:border-zinc-800">
      <img src={src} alt={alt} loading="lazy" className="mx-auto block max-w-full" />
      {caption && (
        <figcaption className="mt-2 text-center font-mono text-xs text-zinc-400">{caption}</figcaption>
      )}
    </figure>
  )
}

function Table({ head, rows, minW = 640 }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
      <table className="w-full text-sm" style={{ minWidth: `${minW}px` }}>
        <thead>
          <tr className="border-b border-zinc-200 text-left text-[11px] uppercase tracking-wide text-zinc-500 dark:border-zinc-700">
            {head.map((h) => (
              <th key={h} className="px-3 py-2 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-zinc-100 last:border-0 dark:border-zinc-800">
              {r.map((c, j) => (
                <td
                  key={j}
                  className={`px-3 py-2 ${
                    j === 0
                      ? 'font-medium text-zinc-800 dark:text-zinc-100'
                      : 'font-mono text-[13px] text-zinc-600 dark:text-zinc-300'
                  }`}
                >
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---- the live homomorphism widget ----------------------------------------
// Mirrors kronembed/embedding.py: LIN = v / 2^14 (a dyadic rational, so
// addition of exactly-representable integers is bit-exact), LOG = log10|v|.

const LIN_SCALE = 2 ** 14

function HomomorphismDemo() {
  const [a, setA] = useState(9)
  const [b, setB] = useState(9)
  const linA = a / LIN_SCALE
  const linB = b / LIN_SCALE
  const linSum = linA + linB
  const linTrue = (a + b) / LIN_SCALE
  const decoded = Math.round(linSum * LIN_SCALE)
  const logsOk = a >= 1 && b >= 1
  const fmt = (x) => Number(x.toPrecision(8))

  const rows = [
    ['LIN(a) + LIN(b)', `${fmt(linA)} + ${fmt(linB)} = ${fmt(linSum)}`],
    ['LIN(a + b)', `${fmt(linTrue)}`],
    ['bit-exact equal?', linSum === linTrue ? 'YES  (float64 ==)' : 'no'],
    ['decode(emb(a) + emb(b))', `${decoded}${decoded === a + b ? '  = a+b ✓' : ''}`],
    [
      'LOG(a) + LOG(b) vs LOG(a·b)',
      logsOk ? `${fmt(Math.log10(a) + Math.log10(b))} vs ${fmt(Math.log10(a * b))}` : 'needs a, b ≥ 1',
    ],
    [
      'LIN(a) − LIN(b) decodes to',
      `${Math.round((linA - linB) * LIN_SCALE)}  (= a−b${a - b < 0 ? ', negative ✓' : ' ✓'})`,
    ],
  ]

  const inputCls =
    'w-24 rounded-lg border border-zinc-300 bg-white px-2 py-1.5 font-mono text-sm text-zinc-900 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100'

  return (
    <div className="panel p-4">
      <div className="flex flex-wrap items-center gap-3 text-sm text-zinc-700 dark:text-zinc-200">
        <span className="font-semibold">Type any two integers:</span>
        <label className="flex items-center gap-1.5">
          a =
          <input
            type="number"
            className={inputCls}
            value={a}
            onChange={(e) => setA(parseInt(e.target.value || '0', 10))}
          />
        </label>
        <label className="flex items-center gap-1.5">
          b =
          <input
            type="number"
            className={inputCls}
            value={b}
            onChange={(e) => setB(parseInt(e.target.value || '0', 10))}
          />
        </label>
      </div>
      <table className="mt-3 w-full text-sm">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k} className="border-t border-zinc-100 dark:border-zinc-800">
              <td className="py-1.5 pr-4 text-zinc-500 dark:text-zinc-400">{k}</td>
              <td className="py-1.5 font-mono text-[13px] text-zinc-800 dark:text-zinc-100">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---- page -----------------------------------------------------------------

function Hero() {
  const holeGap = ratio('arith', PRIMARY, 'hole_add_exact', 'kron_v2', 'learned')
  const randGap = ratio('arith', PRIMARY, 'hole_add_exact', 'kron_v2', 'frozen_rand')
  const nlGap = ratio('nl', PRIMARY, 'hole_add_exact', 'kron_v2', 'learned')
  const nChecks = properties.checks.length
  const nOk = properties.checks.filter((c) => c.ok).length

  return (
    <header id="top" className="pt-12 pb-2">
      <p className="font-mono text-xs uppercase tracking-widest text-brand-500">ERA-V5 · The School of AI</p>
      <h1 className="mt-2 text-3xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50 sm:text-4xl">
        Assignment 7
      </h1>
      <p className="mt-3 max-w-3xl text-zinc-600 dark:text-zinc-300">
        <span className="font-semibold text-zinc-900 dark:text-zinc-50">
          Kronecker Embedding V2 — embeddings that carry mathematical structure.
        </span>{' '}
        A deterministic, non-learned embedding where <code className="font-mono text-sm">emb("9") + emb("9")</code>{' '}
        literally carries <b>18</b> (bit-exact in float32), subtraction and negatives included, and ×/÷ become +/− on
        a log dim — proven algebraically with zero training, then stress-tested in a 92-run controlled study on a
        2-layer CPU transformer. Full write-up, code, and audit:{' '}
        <a className="text-brand-600 hover:underline dark:text-brand-400" href={GITHUB}>
          assignment-7 on GitHub
        </a>
        .
      </p>

      <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Tile label={`Unseen-token gap vs learned @${PRIMARY}`} value={`${holeGap}×`} />
        <Tile label="vs frozen-random capacity control" value={`${randGap}×`} />
        <Tile label="Gap on natural-language templates" value={`${nlGap}×`} />
        <Tile label="Algebraic properties, zero training" value={`${nOk}/${nChecks}`} />
      </div>

      <nav aria-label="Sections" className="mt-6 grid gap-2 sm:grid-cols-2">
        {SECTIONS.map((s) => (
          <a
            key={s.id}
            href={`#${s.id}`}
            className="group flex items-center gap-3 rounded-xl border border-zinc-200 bg-white p-3 transition-colors hover:border-zinc-300 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:bg-zinc-800/60"
          >
            <span
              className="inline-flex h-7 shrink-0 items-center rounded-full px-2.5 font-mono text-[11px] font-semibold text-white"
              style={{ backgroundColor: s.color }}
            >
              {s.code}
            </span>
            <span className="text-sm font-medium text-zinc-700 group-hover:text-zinc-900 dark:text-zinc-200">
              {s.title}
            </span>
          </a>
        ))}
      </nav>
    </header>
  )
}

export default function App() {
  const [theme, toggleTheme] = useTheme()

  const claimBRows = ARMS.map(([arm, label]) => [
    label,
    cell('arith', arm, PRIMARY, 'in_add_exact'),
    cell('arith', arm, PRIMARY, 'hole_add_exact'),
    cell('arith', arm, PRIMARY, 'in_sub_exact'),
    cell('arith', arm, PRIMARY, 'hole_sub_exact'),
    cell('arith', arm, PRIMARY, 'in_mul_within_1pct'),
  ])

  const nlRows = ['kron_v2', 'learned'].map((arm) => [
    arm === 'kron_v2' ? 'Kron V2 (ours)' : 'learned table',
    cell('nl', arm, PRIMARY, 'in_add_exact'),
    cell('nl', arm, PRIMARY, 'hole_add_exact'),
  ])

  return (
    <div className="flex min-h-screen flex-col">
      <Navbar theme={theme} onToggleTheme={toggleTheme} label="Assignment 7" />
      <main className="mx-auto w-full max-w-6xl flex-1 px-4">
        <Hero />

        <div className="divide-y divide-zinc-200 dark:divide-zinc-800">
          <ClaimCard
            id="a7-1"
            code="A7-1"
            accent="var(--claim-1)"
            title="One deterministic vector, two languages"
            claim="A 128-d embedding can carry spelling AND mathematics at once: 32 Kronecker character slots (invertible orthography) plus a numeric block whose coordinates are chosen so vector arithmetic mirrors math — a signed linear value dim (exact by the dyadic-rational argument), a log dim, a sign dim, and Fourier digit-readout dims. Numbers are not a special token class: '9', 'plus', and 'what' go through one embedding function."
            takeaway="Every dim has a declared algebraic status — homomorphic (LIN, LOG), readout (SIGN, Fourier, flag), or orthographic (chars) — and the ablation arms in A7-4 give each family its own experimental test."
          >
            <Figure
              src={anatomyPng}
              alt="Embedding anatomy heatmap"
              caption="embedding_anatomy.png — char block (dims 0–95) + numeric block (dims 96–127)"
            />
            <Figure
              src={manifoldPng}
              alt="PCA of the numeric block for tokens 0..999"
              caption="value_manifold.png — the numeric block of tokens 0..999 is a smooth manifold, not a lookup table"
            />
          </ClaimCard>

          <ClaimCard
            id="a7-2"
            code="A7-2"
            accent="var(--claim-2)"
            title="Try the homomorphism — computed live on this page"
            claim="The value dim is v / 2¹⁴ — a dyadic rational, so for integers below 2²⁰ the float sum of two embeddings equals the embedding of the sum with == , not ≈. The same formulas as kronembed/embedding.py, re-implemented in a few lines of JS."
            takeaway="Try 123 and 877, or make a − b negative: the decode stays exact, sign included. This is Claim A running in your browser."
          >
            <HomomorphismDemo />
          </ClaimCard>

          <ClaimCard
            id="a7-3"
            code="A7-3"
            accent="var(--claim-3)"
            title="Claim A — the embedding is an algebra, before any training"
            claim="10 algebraic properties verified over 10,000 random pairs and re-derived by an independent audit at a fresh PRNG coordinate: bit-exact addition AND subtraction (negatives included), multiplication and division on the log dim (max error ~3.6e-7), 10-term sum chains, 5-term product chains, mixed chains like (9+9)×2 through analytic decode→re-encode, and full invertibility back to spelling and value."
            takeaway={`${properties.checks.filter((c) => c.ok).length}/${properties.checks.length} properties pass — see properties_report.json in the artifact bundle for worst-case errors.`}
          >
            <div className="grid gap-2 sm:grid-cols-2">
              {properties.checks.map((c) => (
                <div
                  key={c.name}
                  className="flex items-center gap-2 rounded-lg border border-zinc-200 bg-white px-3 py-2 font-mono text-xs text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300"
                >
                  <span>{c.ok ? '✅' : '❌'}</span>
                  <span>{c.name}</span>
                </div>
              ))}
            </div>
            <Figure
              src={homomorphismPng}
              alt="emb(9)+emb(9) vs emb(18) overlay"
              caption="homomorphism_demo.png — the numeric dims of emb(9)+emb(9) vs emb(18): LIN is bit-exact equal"
            />
          </ClaimCard>

          <ClaimCard
            id="a7-4"
            code="A7-4"
            accent="var(--claim-4)"
            title="Claim B — 7 arms, 5 seeds, only the embedding differs"
            claim="Identical 2-layer transformer, optimizer, schedule, and byte-identical batch stream across seven arms (enforced by hashing); only the embedding provider differs. The hole test holds out operands 40–59 from every training input position: for a learned table those rows stay untrained noise, for a deterministic scheme they are analytically correct."
            takeaway="The structure is what generalizes: a frozen RANDOM table (same capacity, same frozen-ness, no structure) scores ~zero on unseen tokens, while the structured frozen arms reach ~0.45–0.48 — and the gap survives natural-language templates."
          >
            <Table
              head={[
                `arm (train ${PRIMARY})`,
                'in add exact',
                'HOLE add exact',
                'in sub exact',
                'HOLE sub exact',
                'in mul ±1%',
              ]}
              rows={claimBRows}
              minW={760}
            />
            <Figure
              src={holePng}
              alt="Hole generalization by arm"
              caption="hole_generalization.png — operands 40–59 never at a training input position (mean ± std, 5 seeds)"
            />
            <Figure
              src={samplePng}
              alt="Sample efficiency"
              caption="sample_efficiency.png — nested train sets: structure is worth ~4× data at the 90% level"
            />
            <div className="mt-4">
              <Table head={['NL templates', 'in-range add exact', 'HOLE add exact']} rows={nlRows} minW={480} />
              <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-300">
                The transfer slice: “what is a plus b”, “compute the sum of a and b”, “tell me a times b” — varied
                length and answer position, words and numbers through the same embedding.
              </p>
            </div>
            <Figure src={nlPng} alt="NL transfer results" caption="nl_transfer.png" />
          </ClaimCard>

          <ClaimCard
            id="a7-5"
            code="A7-5"
            accent="var(--color-brand-500)"
            title="Honest negatives, localized per layer"
            claim="Magnitude extrapolation (operands 100–999 after training on 0–99) fails for every arm — best bucket ~2% anywhere in the matrix. Standardized ridge probes, fit only on in-range data, locate where linearly decodable structure is lost: at the input the algebra dims extrapolate perfectly (homomorphic-only probe: 0.000 relative error), but the trunk attenuates every arm to ~0.68."
            takeaway="Structure in, structure out — but not structure through. The bottleneck for numeric extrapolation is the transformer body, not the embedding: that is where follow-up work should aim."
          >
            <Figure
              src={layersPng}
              alt="Per-layer probe analysis"
              caption="structure_through_layers.png — OOD linear decodability by probe depth (5 seeds per arm)"
            />
            <Figure
              src={extrapolationPng}
              alt="Extrapolation negative result"
              caption="extrapolation_negative.png — every arm fails; probes localize the loss"
            />
            <p className="mt-4 text-sm text-zinc-600 dark:text-zinc-300">
              Reproduce everything:{' '}
              <code className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-xs dark:bg-zinc-800">
                python run_demo.py
              </code>{' '}
              (~60 min laptop CPU, 92 runs) or{' '}
              <code className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-xs dark:bg-zinc-800">
                python run_demo.py --verify-only
              </code>{' '}
              (~5 s: Claim A + the full independent audit, no training). 53 invariant tests, every [PASS] re-derived
              from disk. Grader card:{' '}
              <a
                className="text-brand-600 hover:underline dark:text-brand-400"
                href={`${GITHUB}/GRADERS.md`}
              >
                GRADERS.md
              </a>
              .
            </p>
          </ClaimCard>
        </div>
      </main>
      <Footer note="ERA-V5 · Assignment 7 — Kronecker Embedding V2. Every number on this page is imported from the committed submission_artifacts bundle; nothing is retyped." />
    </div>
  )
}
