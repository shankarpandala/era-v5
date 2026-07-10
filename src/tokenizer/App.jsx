import { useEffect, useMemo, useState } from 'react'
import Navbar from '../components/layout/Navbar.jsx'
import Footer from '../components/layout/Footer.jsx'
import useTheme from '../hooks/useTheme.js'
import { BPE } from './lib/bpe.js'
import { LANGS, loadTokenizerData } from './lib/loadData.js'
import { computeStats } from './lib/compute.js'
import Method from './sections/Method.jsx'
import Results from './sections/Results.jsx'
import Playground from './sections/Playground.jsx'
import Downloads from './sections/Downloads.jsx'

const SECTIONS = [
  { id: 'a2-1', code: 'A2-1', title: 'The method & the metric', color: 'var(--claim-1)' },
  { id: 'a2-2', code: 'A2-2', title: 'Ratios, statistics & self-score', color: 'var(--claim-3)' },
  { id: 'a2-3', code: 'A2-3', title: 'Tokenize anything (live)', color: 'var(--claim-2)' },
  { id: 'a2-4', code: 'A2-4', title: 'Download the tokenizer', color: 'var(--claim-4)' },
]

function Hero({ stats }) {
  return (
    <header id="top" className="pt-12 pb-2">
      <p className="font-mono text-xs uppercase tracking-widest text-brand-500">ERA-V5 · The School of AI</p>
      <h1 className="mt-2 text-3xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50 sm:text-4xl">
        Assignment 2 — Multilingual BPE Tokenizer
      </h1>
      <p className="mt-3 max-w-3xl text-sm leading-relaxed text-zinc-600 dark:text-zinc-300">
        One from-scratch, byte-level Byte-Pair-Encoding tokenizer with a single shared vocabulary of{' '}
        <span className="font-mono font-semibold text-zinc-900 dark:text-zinc-100">10,000 tokens</span> for
        English, Hindi, Telugu and Marathi — trained on India's Wikipedia page. Every number below is recomputed{' '}
        <span className="font-semibold text-zinc-900 dark:text-zinc-100">live in your browser</span> from the same
        tokenizer file you can download.
      </p>

      {stats && (
        <div className="mt-6 flex flex-wrap items-center gap-3">
          <div className="rounded-xl border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
            <div className="text-[11px] uppercase tracking-wide text-zinc-500">Self-score = 1000 / spread</div>
            <div className="font-mono text-2xl font-bold text-brand-600 dark:text-brand-400">
              {stats.score.toFixed(1)}
            </div>
          </div>
          <div className="rounded-xl border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
            <div className="text-[11px] uppercase tracking-wide text-zinc-500">All four ≤ 1.2 ?</div>
            <div
              className={`font-mono text-2xl font-bold ${
                stats.constraintsMet ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'
              }`}
            >
              {stats.constraintsMet ? 'YES ✓' : 'NO ✗'}
            </div>
          </div>
        </div>
      )}

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

function Skeleton() {
  return (
    <div className="mt-10 space-y-4">
      <div className="h-8 w-2/3 animate-pulse rounded bg-zinc-200 dark:bg-zinc-800" />
      <div className="panel h-48 animate-pulse" />
      <div className="panel h-64 animate-pulse" />
    </div>
  )
}

export default function App() {
  const [theme, toggleTheme] = useTheme()
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let alive = true
    loadTokenizerData()
      .then((d) => {
        if (alive) setData(d)
      })
      .catch((e) => {
        if (alive) setError(String(e))
      })
    return () => {
      alive = false
    }
  }, [])

  const bpe = useMemo(() => (data ? BPE.fromJSON(data.tok) : null), [data])
  const stats = useMemo(() => (bpe && data ? computeStats(data.corpora, bpe, LANGS) : null), [bpe, data])

  return (
    <div className="flex min-h-screen flex-col">
      <Navbar
        theme={theme}
        onToggleTheme={toggleTheme}
        label="Assignment 2"
        crossLink={{ href: `${import.meta.env.BASE_URL}`, text: '← Assignment 1' }}
      />
      <main className="mx-auto w-full max-w-6xl flex-1 px-4">
        <Hero stats={stats} />
        {error && (
          <div className="mt-8 rounded-xl border border-red-300 bg-red-50 p-4 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/40 dark:text-red-300">
            Failed to load tokenizer data: {error}
          </div>
        )}
        {!error && !stats && <Skeleton />}
        {stats && (
          <div className="divide-y divide-zinc-200 dark:divide-zinc-800">
            <Method stats={stats} tok={data.tok} />
            <Results stats={stats} refStats={data.stats} />
            <Playground bpe={bpe} corpora={data.corpora} />
            <Downloads tok={data.tok} bpe={bpe} />
          </div>
        )}
      </main>
      <Footer note="ERA-V5 · Assignment 2 — a from-scratch 10k-vocab BPE tokenizer for English, Hindi, Telugu & Marathi. Every ratio is recomputed live in your browser from the downloadable tokenizer; nothing is hardcoded." />
    </div>
  )
}
