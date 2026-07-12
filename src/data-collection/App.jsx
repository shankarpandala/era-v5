import Navbar from '../components/layout/Navbar.jsx'
import Footer from '../components/layout/Footer.jsx'
import Button from '../components/ui/Button.jsx'
import ClaimCard from '../components/ClaimCard.jsx'
import useTheme from '../hooks/useTheme.js'

const SECTIONS = [
  { id: 'a3-1', code: 'A3-1', title: 'Data — what, how much, why', color: 'var(--claim-1)' },
  { id: 'a3-2', code: 'A3-2', title: 'Cleaning for the objectives', color: 'var(--claim-2)' },
  { id: 'a3-3', code: 'A3-3', title: 'Testing against the objectives', color: 'var(--claim-3)' },
  { id: 'a3-4', code: 'A3-4', title: 'Fertility targets → tokenizer size', color: 'var(--claim-4)' },
]

function Tile({ label, value, accent = 'text-brand-600 dark:text-brand-400' }) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="text-[11px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={`font-mono text-xl font-bold ${accent}`}>{value}</div>
    </div>
  )
}

function Table({ head, rows }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] text-sm">
        <thead>
          <tr className="border-b border-zinc-200 text-left text-[11px] uppercase tracking-wide text-zinc-500 dark:border-zinc-700">
            {head.map((h) => (
              <th key={h} className="py-2 pr-4 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-zinc-100 align-top last:border-0 dark:border-zinc-800">
              {r.map((c, j) => (
                <td
                  key={j}
                  className={`py-2 pr-4 ${j === 0 ? 'font-medium text-zinc-800 dark:text-zinc-100' : 'text-zinc-600 dark:text-zinc-300'}`}
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

function Note({ children }) {
  return (
    <p className="mt-3 rounded-lg bg-zinc-100 px-3 py-2 text-[13px] leading-relaxed text-zinc-700 dark:bg-zinc-800/60 dark:text-zinc-200">
      {children}
    </p>
  )
}

async function downloadBrief() {
  const res = await fetch(`${import.meta.env.BASE_URL}data-collection/bharat-40b-brief.md`)
  const blob = new Blob([await res.text()], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'bharat-40b-design-brief.md'
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

export default function App() {
  const [theme, toggleTheme] = useTheme()

  return (
    <div className="flex min-h-screen flex-col">
      <Navbar
        theme={theme}
        onToggleTheme={toggleTheme}
        label="Assignment 3"
        crossLinks={[
          { href: `${import.meta.env.BASE_URL}`, text: 'Assignment 1' },
          { href: `${import.meta.env.BASE_URL}tokenizer/`, text: 'Assignment 2' },
        ]}
      />
      <main className="mx-auto w-full max-w-6xl flex-1 px-4">
        <header id="top" className="pt-12 pb-2">
          <p className="font-mono text-xs uppercase tracking-widest text-brand-500">ERA-V5 · The School of AI</p>
          <h1 className="mt-2 text-3xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50 sm:text-4xl">
            Assignment 3 — Bharat-40B: Data &amp; Design
          </h1>
          <p className="mt-3 max-w-3xl text-sm leading-relaxed text-zinc-600 dark:text-zinc-300">
            The complete design brief for a <b>40B-parameter, India-first</b> model that matches the latest Gemma
            class and excels at coding, agentic work, and Indic languages. Four decisions — data, cleaning,
            evaluation, tokenizer — each quantified and grounded in published training recipes and our own
            Assignment-2 tokenizer measurements.
          </p>

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <Tile label="Shape · budget" value="40B dense · 16T + distillation" />
            <Tile label="Tokenizer vocab" value="262,144 (2¹⁸)" />
            <Tile
              label="Cost-equity principle"
              value="Indic ≤ 1.4× English tokens"
              accent="text-emerald-600 dark:text-emerald-400"
            />
            <Button variant="ghost" onClick={downloadBrief}>
              ↓ brief.md
            </Button>
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

        <div className="divide-y divide-zinc-200 dark:divide-zinc-800">
          {/* ---------------- A3-1 DATA ---------------- */}
          <ClaimCard
            id="a3-1"
            code="A3-1"
            accent="var(--claim-1)"
            title="Data — what, how much, why"
            claim={
              <>
                <b>16T pre-training tokens in 3 stages plus knowledge distillation</b> — Gemma-3-27B needed 14T + KD
                for this class, so 8T-style budgets cannot reach it. India-first is a <b>pre-training</b> decision,
                not an RLHF patch.
              </>
            }
            takeaway="Signal-per-token beats raw scale: staged mixture, an honest real-vs-synthetic Indic ledger, agentic priors injected from pre-training onward, and native-authored (never translated) Indic post-training."
          >
            <div className="panel space-y-5 p-5 text-sm leading-relaxed text-zinc-700 dark:text-zinc-200">
              <div>
                <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
                  Stage A · breadth — 12.5T
                </div>
                <Table
                  head={['Bucket', '%', '~Tokens', 'Why']}
                  rows={[
                    ['High-quality English web', '38%', '4.75T', 'FineWeb-Edu-class filtering; general capability lives here'],
                    ['Code', '20%', '2.5T', 'Stack-v2-scale + PR/issue/commit threads — agentic priors start in pre-training'],
                    ['Math + science', '10%', '1.25T', 'OpenWebMath / arXiv / textbooks; structured reasoning'],
                    ['Indic (12 languages)', '17%', '2.1T', 'Honest ledger below'],
                    ['Other multilingual', '5%', '0.6T', 'Transfer + translation strength'],
                    ['Books / reference', '5%', '0.6T', 'Long-form coherence'],
                    ['Synthetic reasoning', '5%', '0.6T', 'Teacher-generated, verified-only'],
                  ]}
                />
                <Note>
                  <b>The Indic ledger (2.1T):</b> all cleaned native Indic text in existence is ≈275B tokens
                  (Sangraha 251B — itself 65% machine-translated, only 64B human-verified — plus IndicCorp v2 ~21B
                  and legal/govt corpora). So: 275B real × ~2.5 epochs + ~1.3T quality-gated synthetic (en→Indic
                  translation of educational web, transliteration pairs, native-script textbook generations) +{' '}
                  <b>15% romanized/code-mixed</b> — 52% of Hindi UGC online is romanized, so Hinglish is first-class
                  data, not noise. Language weights ∝ speakers × digital availability (hi ~35%; bn/te/mr/ta ~8–10%
                  each; then gu/ur/kn/ml/pa/or/as; a pinch of Sanskrit).
                </Note>
                <Note>
                  <b>India-first injections, upsampled 2–4×:</b> Constitution + BNS, SC/HC judgments, Parliament
                  debates, RBI/SEBI/NITI, NCERT/NPTEL, Census & data.gov.in, Indian newspapers, regional
                  literature, UPI/ONDC/DigiLocker documentation. Worldview is learned in pre-training.
                </Note>
              </div>

              <div>
                <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
                  Stage B · mid-train 3T &nbsp;·&nbsp; Stage C · long-context 0.5T
                </div>
                <p>
                  B: code→30% with repo-level context; math→20% with verified solutions; <b>~300B agentic-trace
                  tokens</b> (tool-call logs, terminal sessions, SWE trajectories synthesized from PR-issue-patch
                  triples); top-decile Indic (exams, judgments); anneal on the best data. C: context 8k→128k on full
                  repos, case law, multi-document sets.
                </p>
              </div>

              <div>
                <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
                  Post-training
                </div>
                <p>
                  <b>SFT ~1M curated:</b> 35% agentic/code (sandbox-verified trajectories — a few hundred verified
                  SWE trajectories measurably move a 30B-class model double digits on SWE-bench; verified tool-call
                  sets), 25% reasoning CoT, <b>25% Indic written natively by paid writers — never translated</b>{' '}
                  (translationese destroys cultural grounding), 15% India-domain (law/agri/health/GST/UPI) + safety.
                </p>
                <p className="mt-2">
                  <b>RL, verifiable-first:</b> GRPO against unit tests, math answers, tool-call schemas and ~5K
                  sandboxed SWE/tool environments (pure RL at 32B is proven to exceed 40% SWE-bench Verified). Then
                  preference RL with a <b>paid Indian annotator pool</b> balanced across language, region, gender,
                  caste — with no reward for verbosity. Alignment spec: an India-first constitution — Indian
                  constitutional values, plural, non-partisan; Indian defaults when locale is unspecified
                  (₹/lakh/crore, DD-MM-YYYY, BNS not US common law, Survey-of-India borders).
                </p>
              </div>
            </div>
          </ClaimCard>

          {/* ---------------- A3-2 CLEANING ---------------- */}
          <ClaimCard
            id="a3-2"
            code="A3-2"
            accent="var(--claim-2)"
            title="Cleaning for the objectives"
            claim="Most quality comes from what you remove. Every bucket gets its own pipeline; everything passes global dedup, India-aware PII scrubbing, and decontamination against the entire eval suite."
            takeaway="Two India-specific rules generic pipelines get wrong: preserve ZWJ/ZWNJ or Indic conjuncts shatter (our Assignment-2 lesson), and dedup at byte level — word-level MinHash fails agglutinative morphology."
          >
            <div className="panel p-5">
              <Table
                head={['Bucket', 'Pipeline']}
                rows={[
                  ['All data', 'Exact + MinHash-LSH dedup → PII scrub incl. Aadhaar/PAN/UPI formats → 13-gram decontamination vs the ENTIRE eval suite, Indic benchmarks included'],
                  ['English web', 'Educational-value classifier + AI-slop / content-farm detector'],
                  ['Indic', 'Triple language-ID (reject wrong-script) · Unicode NFC preserving ZWJ/ZWNJ · byte-level MinHash on char 5-grams · per-language perplexity gates · natively-built casteist/communal slur lexicons · romanized text kept and tagged · MT-quality gate on all synthetic'],
                  ['Code', 'License allowlist · secrets scrub · repo-level dedup · AST/lint/compile filters · PR/issue threads kept intact'],
                  ['Agentic traces', 'Replay in sandbox; keep only success or valid error-recovery'],
                  ['Math / science', 'LaTeX-preserving extraction; answer-verifiable subset tagged for RL reuse'],
                ]}
              />
            </div>
          </ClaimCard>

          {/* ---------------- A3-3 EVALUATION ---------------- */}
          <ClaimCard
            id="a3-3"
            code="A3-3"
            accent="var(--claim-3)"
            title="Testing against the objectives"
            claim="Each objective gets a numeric bar on benchmarks that resist contamination and gaming — plus private held-out agent environments, because public agentic benchmarks are exploitable."
            takeaway="The two India-first tests most suites lack: romanization robustness (native vs roman script, <5% gap) and refusal-balance parity across religion, caste, and region."
          >
            <div className="panel p-5">
              <Table
                head={['Objective', 'Suite → bar']}
                rows={[
                  ['General parity', 'MMLU-Pro, GPQA-D, IFEval, Arena-Hard → within ~2–3 pts of latest same-size Gemma'],
                  ['Coding', 'LiveCodeBench (contamination-resistant) + SWE-bench Verified ≥ 40% + RepoBench; HumanEval as smoke test only'],
                  ['Agentic', 'BFCL v3 ≥ 70%, tau-bench, terminal-bench, OSWorld subset — score success AND steps/cost; private held-out environments'],
                  ['Indic', 'MILU (11 languages) — must beat same-size Gemma; IndicGenBench; IN22 chrF++; romanization-robustness gap < 5%; Hinglish QA'],
                  ['India-first', '3-layer custom eval: (1) factuality — polity/schemes/GST/railways, UPSC-style; (2) default-perspective probes — which currency/law/examples does it assume when unspecified; (3) fairness — IndiBias + refusal-balance parity across religion/caste/region; quarterly Indian-rater human eval'],
                  ['Continuous', 'Per-domain loss dashboards; tokenizer-fertility regression in CI; decontamination audit before any reported number'],
                ]}
              />
            </div>
          </ClaimCard>

          {/* ---------------- A3-4 FERTILITY & TOKENIZER ---------------- */}
          <ClaimCard
            id="a3-4"
            code="A3-4"
            accent="var(--claim-4)"
            title="Fertility targets → tokenizer size"
            claim={
              <>
                Targets anchored to measured tokenizers (Sarvam-1: 1.4–2.1 across Indic; 200K Indic-weighted vocabs
                reach Hindi ≈1.2; Llama-class tokenizers cost Indic users 3–8× English). Principle:{' '}
                <b>an Indian-language user pays ≤ 1.4× English tokens for the same content.</b>
              </>
            }
            takeaway="Vocab = 262,144 (2^18) byte-fallback BPE: ~110K en+code+math+symbols, ~108K across 12 Indic scripts, ~15K romanized/code-mix, ~20K other. The vocabulary scaling law puts a 40B model's optimum at 200–300K; Gemma 3 ships exactly 262K; tied embeddings hold the cost to ~4% of parameters — and a larger vocab directly cuts Indic serving cost."
          >
            <div className="panel p-5">
              <Table
                head={['Domain / language', 'Fertility target', 'Note']}
                rows={[
                  ['English', '≤ 1.30 tokens/word', 'o200k-class reality; ≤1.05 claims are unattainable'],
                  ['Hindi, Marathi (Devanagari)', '≤ 1.45', '1.2 proven possible at 200K with Indic-heavy training'],
                  ['Bengali, Urdu', '≤ 1.55', ''],
                  ['Gujarati, Punjabi, Odia, Assamese', '≤ 1.70', ''],
                  ['Tamil, Telugu, Kannada, Malayalam', '≤ 1.85', 'agglutinative (sandhi); Sarvam-1 band'],
                  ['Romanized Hinglish', '≤ 1.35', 'first-class data — majority of Hindi UGC'],
                  ['Code', '≥ 3.3 chars/token', 'whole-token identifiers and keywords'],
                  ['Math', '≥ 3.0 chars/token', 'digits split 0–9: arithmetic accuracy > fertility'],
                ]}
              />
              <Note>
                Tokenizer training mix ≈ 35% en+code · 45% Indic (fertility-driven upsample) · 10% math · 10%
                other, with a ZWJ/ZWNJ-preserving pre-tokenizer; per-language fertility tracked in CI on every
                release — the exact machinery built and verified in Assignment 2.
              </Note>
            </div>
          </ClaimCard>
        </div>
      </main>
      <Footer note="ERA-V5 · Assignment 3 — Bharat-40B design brief: data, cleaning, evaluation, and a fertility-derived 262K tokenizer for a 40B India-first coding & agentic model." />
    </div>
  )
}
