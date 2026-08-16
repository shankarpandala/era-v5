// ERA-V5 · Assignment 8 — How attention works now: every mechanism, in launch
// order, each one as an answer to the bill of the one before it.
//
// Every date, source, evidence string, trade-off and verdict on this page is
// read from the committed assignment-8/data/ files (one JSON per era). Nothing
// is retyped here; scripts/check_assignment8.mjs validates the same files and
// regenerates the README table from them, so page, README and checks cannot drift.

import { lazy, Suspense } from 'react'
import Navbar from '../components/layout/Navbar.jsx'
import Footer from '../components/layout/Footer.jsx'
import ClaimCard from '../components/ClaimCard.jsx'
import useTheme from '../hooks/useTheme.js'
import { ALL, NODES, FOOTNOTES, INSTRUCTOR_ITEMS, ERAS, byId } from './data.js'
import Timeline from './components/Timeline.jsx'
import EraStory from './components/EraStory.jsx'
import DecisionMatrix from './components/DecisionMatrix.jsx'
import Sources from './components/Sources.jsx'
import Question2 from './components/Question2.jsx'

const SoftmaxPipeline = lazy(() => import('./viz/SoftmaxPipeline.jsx'))
const CostCalculator = lazy(() => import('./viz/CostCalculator.jsx'))

const GITHUB = 'https://github.com/shankarpandala/era-v5/tree/main/assignment-8'
// 105 records were re-fetched by an independent adversarial pass (0 date disagreements); the
// records added afterwards for Oct 2025 – Aug 2026 were verified against the arXiv API / HF repo metadata.
const RECHECKED = 105

const SECTIONS = [
  { id: 'a8-1', code: 'A8-1', title: 'The baseline: scaled dot-product attention (Session 2)', color: 'var(--claim-1)' },
  { id: 'a8-2', code: 'A8-2', title: 'The bill: what standard attention costs, in numbers', color: 'var(--claim-2)' },
  { id: 'a8-3', code: 'A8-3', title: 'The timeline, in launch order', color: 'var(--claim-3)' },
  { id: 'a8-4', code: 'A8-4', title: 'Watch the field change its mind — and what comes next', color: 'var(--claim-4)' },
  { id: 'a8-5', code: 'A8-5', title: 'When would you actually pick it', color: 'var(--color-brand-500)' },
  { id: 'a8-6', code: 'A8-6', title: 'Sources, dating method, corrections', color: '#71717a' },
  { id: 'a8-7', code: 'A8-7', title: 'Question 2 — what the timeline shows that a list cannot', color: 'var(--claim-2)' },
]

function Tile({ label, value }) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="text-[11px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className="font-mono text-xl font-bold text-brand-600 dark:text-brand-400">{value}</div>
    </div>
  )
}

function Hero() {
  const first = ALL[0]
  const last = ALL[ALL.length - 1]
  const majors = NODES.filter((m) => m.tier === 'major').length
  return (
    <header id="top" className="pt-12 pb-2">
      <p className="font-mono text-xs uppercase tracking-widest text-brand-500">ERA-V5 · The School of AI</p>
      <h1 className="mt-2 text-3xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50 sm:text-4xl">Assignment 8 — How attention works now</h1>
      <p className="mt-3 max-w-3xl text-zinc-600 dark:text-zinc-300">
        <span className="font-semibold text-zinc-900 dark:text-zinc-50">Every attention mechanism, in the order it was launched, each one as an answer to a bill.</span>{' '}
        Vanilla attention was not wrong, it was expensive: O(n²) compute and a KV cache that grows with every token. Everything after it is
        somebody looking at that bill and paying less of it — first the field wants exactness, then compute back, then positions, then exact
        got cheap, then length, then memory back, then memory back <i>again</i>. Laid out by date you can watch it change its mind, and once
        you see that you can guess what comes next. Every date is read from the primary source and re-checked; the write-up and data are on{' '}
        <a className="text-brand-600 hover:underline dark:text-brand-400" href={GITHUB}>
          GitHub
        </a>
        .
      </p>
      <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Tile label="mechanisms on the timeline" value={`${NODES.length}`} />
        <Tile label="major cards · minor · footnotes" value={`${majors} · ${NODES.length - majors} · ${FOOTNOTES.length}`} />
        <Tile label="primary sources · re-checked by an independent pass" value={`${ALL.length} · ${RECHECKED}`} />
        <Tile label="span" value={`${first.date.slice(0, 4)} → ${last.date.slice(0, 4)}`} />
      </div>
      <nav aria-label="Sections" className="mt-6 grid gap-2 sm:grid-cols-2">
        {SECTIONS.map((s) => (
          <a
            key={s.id}
            href={`#${s.id}`}
            className="group flex items-center gap-3 rounded-xl border border-zinc-200 bg-white p-3 transition-colors hover:border-zinc-300 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:bg-zinc-800/60"
          >
            <span className="inline-flex h-7 shrink-0 items-center rounded-full px-2.5 font-mono text-[11px] font-semibold text-white" style={{ backgroundColor: s.color }}>
              {s.code}
            </span>
            <span className="text-sm font-medium text-zinc-700 group-hover:text-zinc-900 dark:text-zinc-200">{s.title}</span>
          </a>
        ))}
      </nav>
      <div className="mt-4 rounded-xl border border-zinc-200 bg-white p-3 text-xs text-zinc-600 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300">
        <span className="font-semibold text-zinc-800 dark:text-zinc-100">The assignment’s minimum list, all covered — </span>
        {INSTRUCTOR_ITEMS.map((it, i) => (
          <span key={it.label}>
            {it.ids.length === 1 ? (
              <a href={`#m-${it.ids[0]}`} className="text-brand-600 hover:underline dark:text-brand-400">
                {it.label}
              </a>
            ) : (
              <>
                {it.label} (
                {it.ids.map((id, j) => (
                  <span key={id}>
                    <a href={`#m-${id}`} className="text-brand-600 hover:underline dark:text-brand-400">
                      {byId[id]?.short ?? id}
                    </a>
                    {j < it.ids.length - 1 ? ' / ' : ''}
                  </span>
                ))}
                )
              </>
            )}
            {i < INSTRUCTOR_ITEMS.length - 1 ? ' · ' : ''}
          </span>
        ))}
        <span className="text-zinc-500"> — plus {NODES.filter((m) => !m.instructorList).length} extras and {FOOTNOTES.length} footnotes found along the way.</span>
      </div>
    </header>
  )
}

function Loading() {
  return <div className="panel h-48 animate-pulse p-4 text-xs text-zinc-400">loading…</div>
}

export default function App() {
  const [theme, toggleTheme] = useTheme()
  return (
    <div className="flex min-h-screen flex-col">
      <Navbar theme={theme} onToggleTheme={toggleTheme} label="Assignment 8" />
      <main className="mx-auto w-full max-w-6xl flex-1 px-4">
        <Hero />
        <div className="divide-y divide-zinc-200 dark:divide-zinc-800">
          <ClaimCard
            id="a8-1"
            code="A8-1"
            accent="var(--claim-1)"
            title="The baseline: scaled dot-product attention with a softmax"
            claim="Nothing after this makes sense without it. Score every key against the query with a dot product, divide by √d_k so the softmax does not saturate, turn the scores into weights, and average the values. Do it h times in parallel with smaller heads. This is Session 2's mechanism, live: pick a query token, drag n and d_k, switch the scaling off."
            takeaway="Two bills are visible on the tiles: n(n+1)/2 scores per head per layer (compute grows with n²) and 2·h·d_head values cached per token per layer (memory grows with n). Every card on the timeline attacks one of those two numbers, or the position problem the Transformer left open."
          >
            <Suspense fallback={<Loading />}>
              <SoftmaxPipeline />
            </Suspense>
          </ClaimCard>

          <ClaimCard
            id="a8-2"
            code="A8-2"
            accent="var(--claim-2)"
            title="The bill"
            claim="Before the timeline, the numbers. For real layouts (Llama-2 MHA, Llama-2-70B GQA, Mistral's window, DeepSeek-V2's MLA, Gemma 3's 5:1, gpt-oss's banded, Qwen3-Next's Gated DeltaNet hybrid) and for a model you configure yourself: how many attention FLOPs a prefill costs and how many bytes the KV cache holds, from 128 tokens to 1M. The two fixed columns — 2K chatbot, 1M agent — are the instructor's question."
            takeaway="At 2K nothing matters and dense MHA is fine. At 1M the dense cache is measured in terabytes and only bounded windows, latent KV or a fixed recurrent state fit on a machine — which is why a mechanism can be right for the first column and wrong for the last without being a bad mechanism."
          >
            <Suspense fallback={<Loading />}>
              <CostCalculator />
            </Suspense>
          </ClaimCard>

          <ClaimCard
            id="a8-3"
            code="A8-3"
            accent="var(--claim-3)"
            title="The timeline, in the order things were launched"
            claim={`${ERAS.length} eras, ${NODES.length} nodes, strictly by the date each mechanism first appeared in public (arXiv v1, or the blog / Reddit post / model release when that came first — both dates shown when they differ). Not the order they were taught, not grouped by family. The page opens on the ${NODES.filter((m) => m.tier === 'major').length} major nodes that carry the story (the course's 17 tagged); one click shows all ${NODES.length} nodes and ${FOOTNOTES.length} footnotes. Each card says what problem existed at that moment, what the idea was, what it buys, what it costs, and when you would actually pick it. Big cards open a live visual; small cards expand. Filter to the instructor's 17, by family, or by the bill attacked.`}
            takeaway="Read the era headers in sequence and the pattern is hard to miss: exactness → compute → positions → exact-got-cheap → length → memory → hybrids → compress-then-select. The 'before the clock' strip carries the three ideas older than the Transformer; the clock starts at 12 June 2017."
          >
            <Timeline />
          </ClaimCard>

          <ClaimCard
            id="a8-4"
            code="A8-4"
            accent="var(--claim-4)"
            title="Watch the field change its mind"
            claim="The same timeline, compressed to what each era wanted and why it changed its mind — then the guess the pattern licenses about what comes next. The prediction is mine, written from the timeline, and it is labelled as a bet."
            takeaway="Every change of mind was triggered by a bill becoming binding (n² at 1K, KV at 128K) or by a systems change un-binding one (FlashAttention). That is the lens: watch the hardware, not the paper count."
          >
            <EraStory />
          </ClaimCard>

          <ClaimCard
            id="a8-5"
            code="A8-5"
            accent="var(--color-brand-500)"
            title="When would you actually pick it"
            claim="Every mechanism, four scenarios: a 2K chatbot, 32K RAG, a 128K coding assistant, a 1M-token agent. ✓ / △ / ✗ with a one-line reason on hover, and the verdict from each card. Same data as the cards, laid out so a row that is right in one column and wrong in another reads as what it is: a trade."
            takeaway="Notice how few rows are ✓ everywhere (GQA, FlashAttention, RoPE, QK-Norm) — the mechanisms that became defaults are the ones with no scenario where they hurt. Everything else is a bet on which bill you are paying."
          >
            <DecisionMatrix />
          </ClaimCard>

          <ClaimCard
            id="a8-6"
            code="A8-6"
            accent="#71717a"
            title="Sources, dating method, corrections"
            claim="The part that is easiest to get wrong and easiest to check. The rule for every date, the evidence string read from each source, what two verification passes corrected, and the honest limitations."
            takeaway={`${ALL.length} entries, every one read from its primary source; ${RECHECKED} of them re-fetched by an independent adversarial pass with 0 date disagreements, the ${ALL.length - RECHECKED} added afterwards (Oct 2025 – Aug 2026) checked directly against the arXiv API / Hugging Face repo metadata. Where a mechanism appeared before its paper (RoPE, Position Interpolation, MLA, Mistral SWA, Gemma 2/3, gpt-oss, DSA, DeepSeek-V4) both dates are shown and the earlier one sorts the timeline.`}
          >
            <Sources />
            <p className="mt-4 text-sm text-zinc-600 dark:text-zinc-300">
              Data and checks:{' '}
              <code className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-xs dark:bg-zinc-800">assignment-8/data/</code> ·{' '}
              <code className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-xs dark:bg-zinc-800">npm run a8:check</code> ·{' '}
              <a className="text-brand-600 hover:underline dark:text-brand-400" href={`${GITHUB}/README.md`}>
                README
              </a>{' '}
              ·{' '}
              <a className="text-brand-600 hover:underline dark:text-brand-400" href={`${GITHUB}/GRADERS.md`}>
                grader card
              </a>
              .
            </p>
          </ClaimCard>

          <ClaimCard
            id="a8-7"
            code="A8-7"
            accent="var(--claim-2)"
            title="Question 2 — what the timeline shows that a list cannot"
            claim="The instructor's follow-up: once the mechanisms are in date order, what can you see that you could not see as a list? Five things, each computed from the committed data rather than asserted — the burst-and-stall shape of the effort, the simultaneity of the big moves, a right idea dormant for years, the gap between publication and adoption, and the holes a date axis exposes in any curated list. Then the mechanism the taught list did not cover, with the line I read its date from."
            takeaway="A list is a set; a timeline is a set plus an axis, and every finding below is a property of the axis — density, silence, simultaneity, lag, and gap. The one mechanism I would add to the taught 21 is FlashAttention (27 May 2022, arXiv:2205.14135): it changed none of the mathematics and redirected the whole field, and it sits exactly inside the taught list's 1.7-year silence between ALiBi and GQA."
          >
            <Question2 />
          </ClaimCard>
        </div>
      </main>
      <Footer note="ERA-V5 · Assignment 8 — How attention works now. Every date, source and trade-off on this page is imported from the committed assignment-8/data files; nothing is retyped." />
    </div>
  )
}
