import { useEffect, useMemo, useState } from 'react'
import Navbar from '../components/layout/Navbar.jsx'
import Footer from '../components/layout/Footer.jsx'
import useTheme from '../hooks/useTheme.js'
import { HFTokenizer } from './lib/hfbpe.js'
import { LANGS, loadTokenizerData } from './lib/loadData.js'
import { computeStats, checkFaithfulness } from './lib/compute.js'
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

const SAMPLE = "India's population is 1,428,627,663."

function Hero({ stats, faithful }) {
  const allFaithful = faithful && Object.values(faithful).every(Boolean)
  return (
    <header id="top" className="pt-12 pb-2">
      <p className="font-mono text-xs uppercase tracking-widest text-brand-500">ERA-V5 · The School of AI</p>
      <h1 className="mt-2 text-3xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50 sm:text-4xl">
        Assignment 2 — Multilingual BPE Tokenizer
      </h1>
      <p className="mt-3 max-w-3xl text-sm leading-relaxed text-zinc-600 dark:text-zinc-300">
        One shared <span className="font-mono font-semibold text-zinc-900 dark:text-zinc-100">10,000-token</span>{' '}
        tokenizer for the <b>wiki-faithful Markdown</b> India pages in English, Hindi, Telugu and Maithili — the
        exact corpus and scoring the course grader uses. The shipped <code>tokenizer.json</code> loads with{' '}
        <code>tokenizers.Tokenizer.from_file</code>, and every number below is recomputed{' '}
        <b>live in your browser</b> from the same files.
      </p>

      {stats && (
        <div className="mt-6 flex flex-wrap items-center gap-3">
          <div className="rounded-xl border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
            <div className="text-[11px] uppercase tracking-wide text-zinc-500">Self-score = 1000 / spread</div>
            <div className="font-mono text-2xl font-bold text-brand-600 dark:text-brand-400">
              {Number.isFinite(stats.score) ? stats.score.toFixed(1) : '∞'}
            </div>
          </div>
          <div className="rounded-xl border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
            <div className="text-[11px] uppercase tracking-wide text-zinc-500">Hindi penalty factor</div>
            <div
              className={`font-mono text-2xl font-bold ${
                stats.hindiPenalty <= 1 + 1e-9
                  ? 'text-emerald-600 dark:text-emerald-400'
                  : 'text-red-600 dark:text-red-400'
              }`}
            >
              ×{stats.hindiPenalty.toFixed(4)} {stats.hindiPenalty <= 1 + 1e-9 ? '✓' : ''}
            </div>
          </div>
          <div className="rounded-xl border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
            <div className="text-[11px] uppercase tracking-wide text-zinc-500">decode(encode(x)) faithful?</div>
            <div
              className={`font-mono text-2xl font-bold ${
                faithful == null
                  ? 'text-zinc-400'
                  : allFaithful
                    ? 'text-emerald-600 dark:text-emerald-400'
                    : 'text-red-600 dark:text-red-400'
              }`}
            >
              {faithful == null ? 'checking…' : allFaithful ? 'YES ✓ (all 4 pages)' : 'NO ✗'}
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
  const [faithful, setFaithful] = useState(null)

  useEffect(() => {
    let alive = true
    loadTokenizerData()
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(String(e)))
    return () => {
      alive = false
    }
  }, [])

  const tok = useMemo(() => (data ? HFTokenizer.fromJSON(data.tok) : null), [data])
  const stats = useMemo(() => (tok && data ? computeStats(data.corpora, tok, LANGS) : null), [tok, data])

  useEffect(() => {
    if (!tok || !data) return
    // Faithfulness gate over all four full corpora — run off the critical path.
    const t = setTimeout(() => setFaithful(checkFaithfulness(data.corpora, tok, LANGS)), 50)
    return () => clearTimeout(t)
  }, [tok, data])

  return (
    <div className="flex min-h-screen flex-col">
      <Navbar theme={theme} onToggleTheme={toggleTheme} label="Assignment 2" />
      <main className="mx-auto w-full max-w-6xl flex-1 px-4">
        <Hero stats={stats} faithful={faithful} />
        {error && (
          <div className="mt-8 rounded-xl border border-red-300 bg-red-50 p-4 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/40 dark:text-red-300">
            Failed to load tokenizer data: {error}
          </div>
        )}
        {!error && !stats && <Skeleton />}
        {stats && (
          <div className="divide-y divide-zinc-200 dark:divide-zinc-800">
            <Method stats={stats} metrics={data.metrics} tok={tok} sample={SAMPLE} />
            <Results tok={tok} corpora={data.corpora} refMetrics={data.metrics} />
            <Playground tok={tok} corpora={data.corpora} sample={SAMPLE} />
            <Downloads tok={tok} metrics={data.metrics} />
          </div>
        )}
      </main>
      <Footer note="ERA-V5 · Assignment 2 (resubmission) — a shared 10k tokenizer for the wiki-faithful Markdown India pages in English, Hindi, Telugu & Maithili. Loads with tokenizers.Tokenizer.from_file; every ratio recomputed live in your browser; decode(encode(x)) preserves visible text." />
    </div>
  )
}
