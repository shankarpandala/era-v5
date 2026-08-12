import Navbar from '../components/layout/Navbar.jsx'
import Footer from '../components/layout/Footer.jsx'
import Button from '../components/ui/Button.jsx'
import ClaimCard from '../components/ClaimCard.jsx'
import useTheme from '../hooks/useTheme.js'

const SECTIONS = [
  { id: 'a3-1', code: 'A3-1', title: 'Data — what, how much, why', color: 'var(--claim-1)' },
  { id: 'a3-2', code: 'A3-2', title: 'India-first by construction', color: 'var(--claim-2)' },
  { id: 'a3-3', code: 'A3-3', title: 'Cleaning for the objectives', color: 'var(--claim-3)' },
  { id: 'a3-4', code: 'A3-4', title: 'Testing against the objectives', color: 'var(--claim-4)' },
  { id: 'a3-5', code: 'A3-5', title: 'Fertility targets → tokenizer size', color: 'var(--color-brand-500)' },
]

function Tile({ label, value, accent = 'text-brand-600 dark:text-brand-400' }) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="text-[11px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={`font-mono text-xl font-bold ${accent}`}>{value}</div>
    </div>
  )
}

function Table({ head, rows, minW = 560 }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm" style={{ minWidth: `${minW}px` }}>
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

function Label({ children }) {
  return <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">{children}</div>
}

async function downloadBrief() {
  const res = await fetch(`${import.meta.env.BASE_URL}data-collection/design-brief.md`)
  const blob = new Blob([await res.text()], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'design-brief.md'
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

export default function App() {
  const [theme, toggleTheme] = useTheme()

  return (
    <div className="flex min-h-screen flex-col">
      <Navbar theme={theme} onToggleTheme={toggleTheme} label="Assignment 3" />
      <main className="mx-auto w-full max-w-6xl flex-1 px-4">
        <header id="top" className="pt-12 pb-2">
          <p className="font-mono text-xs uppercase tracking-widest text-brand-500">ERA-V5 · The School of AI</p>
          <h1 className="mt-2 text-3xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50 sm:text-4xl">
            Assignment 3 — A 40B India-First Model: Data &amp; Design
          </h1>
          <p className="mt-3 max-w-3xl text-sm leading-relaxed text-zinc-600 dark:text-zinc-300">
            Design brief for a 40B-parameter model matching the latest Gemma class — top-tier coding, agentic work,
            Indic languages, and a world view that defaults to the Indian perspective. Five decisions, each a table.
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
              ↓ design-brief.md
            </Button>
          </div>

          {/* Model shape: dense vs MoE — a conditional decision */}
          <div className="panel mt-6 p-5">
            <Label>Model shape — why dense and not MoE (a conditional decision, not dogma)</Label>
            <Table
              minW={620}
              head={['At a FIXED 40B total', 'Dense 40B', 'MoE @ 40B total (~8B active)']}
              rows={[
                ['Quality ceiling', 'Highest — all params active (Gemma-4’s 31B dense outranks its own 26B MoE)', 'Capped near ~8B-active quality'],
                ['Training cost / token', '1×', '~0.25×'],
                ['Serving', 'More FLOPs/token; either way all 40B must sit in memory', 'Cheaper FLOPs, plus routing complexity'],
                ['RL / RLVR stability', 'Well-trodden', 'Router load-balancing fights RL'],
                ['Community finetuning', 'Easy (LoRA)', 'Harder'],
              ]}
            />
            <p className="mt-3 text-[13px] leading-relaxed text-zinc-600 dark:text-zinc-300">
              The assignment fixes <b>total</b> parameters at 40B and the bar is the best same-size dense Gemma — so
              dense maximizes quality inside the cap. <b>MoE becomes the right call</b> if the constraint were
              serving throughput, or if the total budget could grow (e.g. ~120B-total / 17B-active). Budget:{' '}
              <b>16T tokens (~400/param) + distillation from a stronger teacher</b> — Gemma-3-27B needed 14T + KD for
              this class; 8T does not reach it.
            </p>
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
            claim={<><b>16T pre-training tokens in 3 stages + knowledge distillation.</b> Signal-per-token beats raw scale.</>}
            takeaway="An honest real-vs-synthetic Indic ledger; agentic priors injected from pre-training onward; native-authored (never translated) Indic post-training."
          >
            <div className="panel space-y-5 p-5">
              <div>
                <Label>Stage A · breadth — 12.5T</Label>
                <Table
                  head={['Bucket', '%', '~Tokens', 'Why']}
                  rows={[
                    ['High-quality English web (Indian-authored upweighted)', '38%', '4.75T', 'general capability; the English distribution itself becomes India-centric'],
                    ['Code + PR/issue threads', '20%', '2.5T', 'SE patterns; agentic priors start here'],
                    ['Math + science', '10%', '1.25T', 'structured reasoning'],
                    ['Indic — 12 scheduled languages', '17%', '2.1T', 'ledger below'],
                    ['Other multilingual · books · verified synthetic reasoning', '15%', '1.9T', 'transfer · coherence · reasoning'],
                  ]}
                />
              </div>

              <div>
                <Label>The Indic ledger — 2.1T from a 275B-token reality</Label>
                <Table
                  head={['Component', 'Tokens', 'Note']}
                  rows={[
                    ['Real native Indic (all that exists, cleaned)', '≈275B × ~2.5 epochs', 'Sangraha 251B (itself 65% machine-translated; 64B human-verified) + IndicCorp v2 ~21B + legal/govt'],
                    ['Quality-gated synthetic', '~1.3T', 'en→Indic educational translation · transliteration pairs · native-script textbook-style'],
                    ['Romanized / code-mixed (Hinglish)', '~15% of bucket', '52% of Hindi UGC online is romanized — data, not noise'],
                    ['Language weights', '—', '∝ speakers × digital availability: hi ~35%; bn/te/mr/ta 8–10% each; gu/ur/kn/ml/pa/or/as rest'],
                  ]}
                />
              </div>

              <div>
                <Label>Stage B · mid-train 3T &nbsp;·&nbsp; Stage C · long-context 0.5T</Label>
                <Table
                  head={['Stage', 'Content']}
                  rows={[
                    ['B (3T)', 'code→30% repo-level · math→20% verified solutions · ~300B agentic-trace tokens (tool-call logs, terminal sessions, SWE trajectories from PR-issue-patch triples) · top-decile Indic · anneal on best data'],
                    ['C (0.5T)', 'context 8k→128k: full repos, case law, multi-document'],
                  ]}
                />
              </div>

              <div>
                <Label>Post-training</Label>
                <Table
                  head={['Component', 'Content']}
                  rows={[
                    ['SFT ~1M', '35% agentic/code (sandbox-verified trajectories + verified tool-calls) · 25% reasoning CoT · 25% Indic authored natively — never translated · 15% India-domain + safety'],
                    ['RL step 1 — verifiable', 'GRPO on unit tests, math answers, tool-call schemas, ~5K sandboxed SWE/tool environments (pure RL at 32B is proven >40% SWE-bench Verified)'],
                    ['RL step 2 — preference', 'Indian annotator pool (language/region/gender/caste-balanced); no reward for verbosity'],
                  ]}
                />
              </div>
            </div>
          </ClaimCard>

          {/* ---------------- A3-2 INDIA-FIRST ---------------- */}
          <ClaimCard
            id="a3-2"
            code="A3-2"
            accent="var(--claim-2)"
            title="India-first by construction"
            claim={<>"Views the world from the Indian perspective" is engineered at <b>every</b> layer of the stack — not patched in at RLHF.</>}
            takeaway="Each lever produces a measurable default; the eval row makes perspective testable, not vibes."
          >
            <div className="panel p-5">
              <Table
                minW={640}
                head={['Lever', 'Mechanism', 'Default it produces']}
                rows={[
                  ['Pre-training corpus', 'Indian-authored English upweighted (press, textbooks, .in domains); India injections ×2–4: Constitution + BNS, SC/HC judgments, Parliament debates, RBI/SEBI, NCERT/NPTEL, Census, UPI/ONDC docs; quality classifiers calibrated on Indian English so idiom isn’t filtered as “low quality”', '“the Constitution” → India’s'],
                  ['Knowledge frame', 'NCERT / BNS / RBI / Census treated as canonical for civics, law, finance, geography', 'tax → GST & IT Act, not IRS'],
                  ['SFT', 'Native-written Indian daily-life scenarios: UPI dispute, IRCTC booking, ration card, monsoon sowing, board exams', '₹/lakh/crore, Indian names & examples'],
                  ['RL reward', 'Indian annotators ARE the preference signal; constitution derived from Indian constitutional values (plural, non-partisan); judge models trained on Indian-perspective rubrics', 'refusal norms per Indian law; Survey-of-India borders'],
                  ['Inference', 'Locale-default system prompt', 'DD-MM-YYYY, IST, Indian units'],
                  ['Eval', 'Default-perspective probes — a question with unspecified locale must resolve to the Indian frame; refusal-balance parity across religion/caste/region; quarterly Indian-rater human eval', 'measurable, not vibes'],
                ]}
              />
            </div>
          </ClaimCard>

          {/* ---------------- A3-3 CLEANING ---------------- */}
          <ClaimCard
            id="a3-3"
            code="A3-3"
            accent="var(--claim-3)"
            title="Cleaning for the objectives"
            claim="Most quality comes from what you remove — each bucket gets its own pipeline."
            takeaway="Two India-specific rules generic pipelines get wrong: preserve ZWJ/ZWNJ (else conjuncts shatter — our Assignment-2 lesson) and dedup at byte level (word-level MinHash fails agglutination)."
          >
            <div className="panel p-5">
              <Table
                head={['Bucket', 'Pipeline']}
                rows={[
                  ['All data', 'exact + MinHash-LSH dedup → PII scrub incl. Aadhaar/PAN/UPI formats → 13-gram decontamination vs the ENTIRE eval suite'],
                  ['English web', 'educational-value classifier + AI-slop / content-farm detector'],
                  ['Indic', 'triple language-ID (reject wrong-script) · NFC preserving ZWJ/ZWNJ · byte-level MinHash char-5-grams · per-language perplexity gates · natively-built casteist/communal slur lexicons · romanized kept + tagged · MT-quality gate on synthetic'],
                  ['Code', 'license allowlist · secrets scrub · repo dedup · AST/lint/compile filters · PR/issue threads intact'],
                  ['Agentic traces', 'sandbox replay; keep success or valid error-recovery only'],
                  ['Math / science', 'LaTeX-preserving extraction; answer-verifiable subset tagged for RL'],
                ]}
              />
            </div>
          </ClaimCard>

          {/* ---------------- A3-4 EVALUATION ---------------- */}
          <ClaimCard
            id="a3-4"
            code="A3-4"
            accent="var(--claim-4)"
            title="Testing against the objectives"
            claim="Each objective gets a numeric bar on contamination-resistant benchmarks — plus private held-out agent environments, because public agentic benchmarks are gameable."
            takeaway="The two India-first tests most suites lack: romanization robustness (<5% gap) and refusal-balance parity across religion, caste, region."
          >
            <div className="panel p-5">
              <Table
                head={['Objective', 'Suite → bar']}
                rows={[
                  ['General parity', 'MMLU-Pro, GPQA-D, IFEval, Arena-Hard → within ~2–3 pts of same-size Gemma'],
                  ['Coding', 'LiveCodeBench + SWE-bench Verified ≥ 40% + RepoBench (HumanEval = smoke only)'],
                  ['Agentic', 'BFCL v3 ≥ 70%, tau-bench, terminal-bench, OSWorld subset → success AND steps/cost; private held-out environments'],
                  ['Indic', 'MILU (beat same-size Gemma) · IndicGenBench · IN22 chrF++ · romanization-robustness gap < 5% · Hinglish QA'],
                  ['India-first', 'factuality (polity/schemes/GST, UPSC-style) · default-perspective probes · IndiBias + refusal-balance parity · quarterly Indian-rater human eval'],
                  ['Continuous', 'per-domain loss dashboards · tokenizer-fertility regression in CI · decontamination audit before any reported number'],
                ]}
              />
            </div>
          </ClaimCard>

          {/* ---------------- A3-5 FERTILITY & TOKENIZER ---------------- */}
          <ClaimCard
            id="a3-5"
            code="A3-5"
            accent="var(--color-brand-500)"
            title="Fertility targets → tokenizer size"
            claim={<>Targets anchored to measured tokenizers (Sarvam-1: 1.4–2.1 across Indic; 200K Indic-weighted vocabs reach Hindi ≈1.2; Llama-class tokenizers cost Indic users 3–8× English). <b>Principle: an Indian-language user pays ≤ 1.4× English tokens for the same content.</b></>}
            takeaway="Vocab = 262,144 (2^18) byte-fallback BPE: ~110K en+code+math+symbols + ~108K across 12 Indic scripts + ~15K romanized/code-mix + ~20K other. Scaling law puts a 40B optimum at 200–300K; Gemma 3 ships exactly 262K; tied embeddings ≈ 4% of params; larger vocab = cheaper Indic serving."
          >
            <div className="panel p-5">
              <Table
                head={['Domain / language', 'Fertility target', 'Note']}
                rows={[
                  ['English', '≤ 1.30 tokens/word', '≤ 1.05 claims are unattainable (no production tokenizer is close)'],
                  ['Hindi, Marathi (Devanagari)', '≤ 1.45', '≈1.2 proven at 200K with Indic-heavy training'],
                  ['Bengali, Urdu', '≤ 1.55', ''],
                  ['Gujarati, Punjabi, Odia, Assamese', '≤ 1.70', ''],
                  ['Tamil, Telugu, Kannada, Malayalam', '≤ 1.85', 'agglutinative (sandhi)'],
                  ['Romanized Hinglish', '≤ 1.35', 'majority of Hindi UGC'],
                  ['Code / Math', '≥ 3.3 / ≥ 3.0 chars per token', 'digits split 0–9: arithmetic accuracy > fertility'],
                ]}
              />
              <p className="mt-3 rounded-lg bg-zinc-100 px-3 py-2 text-[13px] leading-relaxed text-zinc-700 dark:bg-zinc-800/60 dark:text-zinc-200">
                Tokenizer training mix ≈ 35% en+code · 45% Indic (fertility-driven upsample) · 10% math · 10% other,
                with a ZWJ/ZWNJ-preserving pre-tokenizer; per-language fertility tracked in CI on every release —
                the exact machinery built and verified in Assignment 2.
              </p>
            </div>
          </ClaimCard>
        </div>
      </main>
      <Footer note="ERA-V5 · Assignment 3 — design brief for a 40B India-first coding & agentic model: data, cleaning, evaluation, and a fertility-derived 262K tokenizer." />
    </div>
  )
}
