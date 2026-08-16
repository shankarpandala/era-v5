import Navbar from '../components/layout/Navbar.jsx'
import Footer from '../components/layout/Footer.jsx'
import Button from '../components/ui/Button.jsx'
import ClaimCard from '../components/ClaimCard.jsx'
import useTheme from '../hooks/useTheme.js'

const SECTIONS = [
  { id: 'a3-1', code: 'A3-1', title: 'Data — pre-training, post-training, RL/alignment', color: 'var(--claim-1)' },
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
            Design brief for a 40B-parameter model that matches <b>Gemma-4-31B</b> — the dense, same-size quality
            frontier of the Gemma 4 family — on general benchmarks and <i>beats</i> it on coding, agentic work, and
            Indic languages, defaulting to an Indian view of the world. Five decisions, each a table; every number
            source-grounded.
          </p>

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <Tile label="Shape · budget" value="40B dense · 16T + KD" />
            <Tile label="Tokenizer vocab" value="262,144 (2¹⁸)" />
            <Tile
              label="Cost-equity principle"
              value="Indic ≤ ~1.4× English tokens"
              accent="text-emerald-600 dark:text-emerald-400"
            />
            <Button variant="ghost" onClick={downloadBrief}>
              ↓ design-brief.md
            </Button>
          </div>

          {/* Model shape: dense vs MoE — settled in one paragraph by Gemma 4's own split */}
          <div className="panel mt-6 p-5">
            <Label>Model shape — dense, not MoE</Label>
            <p className="text-[13px] leading-relaxed text-zinc-600 dark:text-zinc-300">
              Gemma 4 itself ships both a <b>31B dense</b> (quality frontier) and a <b>26B MoE</b> (~3.8B active,
              throughput frontier). At a <b>fixed</b> 40B total, with the bar set by the best same-size dense, an
              all-active dense model maximizes quality inside the cap and keeps RLVR and community LoRA simple. MoE would
              win only if the constraint were serving throughput or the total budget could grow (e.g. ~120B-total /
              17B-active). Budget: <b>16T tokens (~400/param) + knowledge distillation</b> from a stronger teacher —
              Gemma-3-27B needed 14T + KD to reach this class; under-8T does not.
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
            title="Data — pre-training, post-training, RL/alignment"
            claim={<><b>16T pre-training tokens in 3 stages + KD, then SFT, then RL/alignment.</b> Signal-per-token beats raw scale.</>}
            takeaway="An honest native-vs-synthetic Indic ledger (native supply is the binding constraint, not token count); agentic priors from pre-training onward; RL/alignment is its own pillar with verifiable rewards and an Indian preference signal."
          >
            <div className="panel space-y-5 p-5">
              <div>
                <Label>Pre-training · Stage A breadth — 12.5T</Label>
                <Table
                  head={['Bucket', '%', '~Tokens', 'Why']}
                  rows={[
                    ['English web (Indian-authored upweighted)', '38%', '4.75T', 'general capability; the English distribution itself skews Indian'],
                    ['Code + PR/issue threads', '20%', '2.5T', 'SE patterns; agentic priors start here'],
                    ['Math + science', '10%', '1.25T', 'structured reasoning; LaTeX/units preserved'],
                    ['Indic — 12 languages', '17%', '2.1T', 'ledger below'],
                    ['Other multilingual · books · verified synthetic reasoning', '15%', '1.9T', 'transfer · coherence'],
                  ]}
                />
                <p className="mt-3 text-[13px] leading-relaxed text-zinc-600 dark:text-zinc-300">
                  <b>Which 12 languages, and why —</b> the highest-speaker, highest-digital-availability of the 22
                  scheduled languages, across both families: <b>Hindi, Bengali, Marathi, Telugu, Tamil, Gujarati, Urdu,
                  Kannada, Odia, Malayalam, Punjabi, Assamese</b> — together ~95% of Indian first-language speakers.
                  Weights ∝ speakers × digital text: hi ~35%; bn/te/mr/ta 8–10% each; gu/ur/kn/ml/pa/or/as the rest.
                </p>
              </div>

              <div>
                <Label>The Indic ledger — 2.1T from a ~100–150B native base</Label>
                <Table
                  head={['Component', 'Amount', 'Note']}
                  rows={[
                    ['Native cleaned Indic that exists (deduplicated)', '~100–150B', 'Sangraha native core 64B verified + 24B unverified; IndicCorp v2 ~21B, CulturaX-Indic ~45B, Varta ~9B — heavily overlapping crawls'],
                    ['Sangraha 251B headline', '65% synthetic', 'MT + transliteration, NOT native — the field’s own tell that native supply is the binding constraint'],
                    ['Bucket plan', '~120B ×~4 epochs (~0.5T) + ~1.3T synthetic', 'quality-gated en→Indic MT · transliteration pairs · native-script textbook-style'],
                    ['Romanized / code-mixed (Hinglish)', '~15% of bucket', '≈50–56% of Hindi social-media users prefer romanized — data, not noise'],
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
                <Label>Post-training (SFT ~1M) &nbsp;·&nbsp; RL &amp; alignment</Label>
                <Table
                  head={['Pillar', 'Content']}
                  rows={[
                    ['SFT ~1M', '35% agentic/code (sandbox-verified trajectories + verified tool-calls, both JSON-tool and code-action formats) · 25% reasoning CoT (math/science) · 25% Indic authored natively — never translated · 15% India-domain + safety'],
                    ['RL step 1 — RLVR (GRPO)', 'verifiable rewards: unit tests (code), exact-answer (math), tool-call schema/execution (agentic) over ~4.5K sandboxed SWE/tool envs (R2E-Gym-style). Precedent: DeepSWE lifted a 32B ~23% → 42.2% SWE-bench Verified with RL alone'],
                    ['RL step 2 — preference & constitutional', 'Indian annotator pool (language/region/gender/caste-balanced) IS the preference signal; constitution from Indian constitutional values (plural, non-partisan); refusals track Indian law; no verbosity reward'],
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
                  ['Pre-training corpus', 'Indian-authored English upweighted (press, textbooks, .in domains); India injections ×2–4: Constitution + BNS, SC/HC judgments, RBI/SEBI, NCERT/NPTEL, Census, UPI/ONDC docs; quality classifiers calibrated on Indian English so idiom isn’t filtered as “low quality”', '“the Constitution” → India’s'],
                  ['Knowledge frame', 'NCERT / BNS / RBI / Census treated as canonical for civics, law, finance, geography', 'tax → GST & IT Act, not IRS'],
                  ['SFT', 'Native-written Indian daily-life scenarios: UPI dispute, IRCTC booking, ration card, board exams', '₹/lakh/crore, Indian names & examples'],
                  ['Alignment', 'Indian annotators ARE the preference signal; constitution derived from Indian constitutional values (plural, non-partisan)', 'Survey-of-India borders; refusal norms per Indian law'],
                  ['Inference', 'Locale-default system prompt', 'DD-MM-YYYY, IST, Indian units'],
                  ['Eval', 'Unspecified-locale probes must resolve to the Indian frame; refusal-balance parity across religion/caste/region', 'measurable, not vibes'],
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
                  ['Indic', 'triple language-ID (reject wrong-script) · NFC preserving ZWJ/ZWNJ · byte-level MinHash · per-language perplexity gates · natively-built casteist/communal slur lexicons · MT-quality gate on synthetic · romanized kept + tagged'],
                  ['Code', 'license allowlist · secrets scrub · repo dedup · AST/lint/compile filters'],
                  ['Agentic traces', 'sandbox replay; keep success or valid error-recovery only'],
                  ['Math / science', 'LaTeX- and units-preserving extraction; answer-verifiable subset tagged for RLVR'],
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
                  ['General parity', 'MMLU-Pro, GPQA-Diamond, IFEval, Arena-Hard → within ~2–3 pts of Gemma-4-31B'],
                  ['Math / science', 'MATH-500, AIME, GPQA-Diamond → ≥ same-size Gemma'],
                  ['Coding', 'LiveCodeBench + SWE-bench Verified ≥ 42% (Pass@1) + RepoBench (HumanEval = smoke only)'],
                  ['Agentic', 'BFCL v3 ≥ 70%, τ²-bench, terminal-bench, OSWorld subset → success AND steps/cost; private held-out environments'],
                  ['Indic', 'MILU (beat same-size Gemma) · IndicGenBench · IN22 chrF++ · romanization-robustness gap < 5% · Hinglish QA'],
                  ['India-first', 'factuality (polity/schemes/GST, UPSC-style) · default-perspective probes · IndiBias + refusal-balance parity · quarterly Indian-rater human eval'],
                  ['Continuous', 'per-domain loss dashboards · per-language & per-domain fertility regression in CI · decontamination audit before any reported number'],
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
            claim={<>Targets anchored to measured tokenizers (Sarvam-1 avg ~2.0 tok/word over Indic at a 68K vocab; a 200K Indic-heavy vocab reaches Hindi ~1.2; Llama-3 costs Indic users ~2× on Hindi up to ~10–13× on Dravidian). <b>Principle: an Indian-language user pays ≤ ~1.4× English tokens for the same content.</b></>}
            takeaway="Vocab = 262,144 (2¹⁸) — deliberately ABOVE the ~170K compute-optimal for 40B, bought by heavy over-training + Indic/code/science/agentic coverage + Gemma-4 parity; tied embeddings cost only ~3.4–4% of params."
          >
            <div className="panel p-5">
              <Table
                head={['Domain / language', 'Fertility target', 'Note']}
                rows={[
                  ['English', '≤ 1.30 tokens/word', 'needs a vocab > Sarvam’s 68K — we have 262K'],
                  ['Hindi, Marathi (Devanagari)', '≤ 1.45', '~1.2 proven at 200K Indic-heavy'],
                  ['Bengali, Assamese, Urdu, Gujarati, Punjabi, Odia', '≤ 1.65', ''],
                  ['Tamil, Telugu, Kannada', '≤ 1.85', 'agglutination + sandhi'],
                  ['Malayalam', '≤ 2.05', 'most agglutinative — persistent outlier'],
                  ['Romanized Hinglish', '≤ 1.50', 'orthographic variance (measured ~1.46)'],
                  ['Code', '≥ 3.4 chars/token', 'merge whitespace/indent runs (~25% of code tokens)'],
                  ['Math', '≥ 3.0 chars/token (prose)', 'digits split 0–9 → numeric runs = 1.0 by design; RTL number formatting in prep'],
                  ['Science', '≥ 3.0 chars/token (prose)', 'single-token element symbols / SI units / Greek / operators; formulae & SMILES below target by design'],
                  ['Agentic / tool-call', '≥ 3.0 chars/token', '≥256 reserved special tokens + single-token tool boundaries + JSON-punct merges; code-action ≈ code fertility'],
                ]}
              />
              <p className="mt-3 rounded-lg bg-zinc-100 px-3 py-2 text-[13px] leading-relaxed text-zinc-700 dark:bg-zinc-800/60 dark:text-zinc-200">
                <b>Vocab = 262,144 (2¹⁸), byte-fallback BPE.</b> The vocabulary scaling law (Tao et al., NeurIPS 2024)
                puts a 40B <i>compute-optimal</i> vocab near ~170K; we go deliberately above it because (a) 16T tokens
                ≈ 400/param is heavily over-trained, which raises the optimum and cures rare-token undertraining;
                (b) 12 Indic scripts + code/math/science/agentic are exactly where extra vocab buys lower fertility and
                cheaper serving; (c) 262K matches Gemma 4 and is matmul-friendly. Budget ≈ 110K en+code+math+science+symbols
                · 108K across 12 Indic scripts · 15K romanized/code-mix · 256 reserved special (agentic) · remainder rare.
                ZWJ/ZWNJ-preserving pre-tokenizer; per-language fertility gated in CI — the Assignment-2 machinery.
              </p>
            </div>
          </ClaimCard>
        </div>
      </main>
      <Footer note="ERA-V5 · Assignment 3 — design brief for a 40B India-first coding & agentic model: data, cleaning, evaluation, and a fertility-derived 262K tokenizer." />
    </div>
  )
}
