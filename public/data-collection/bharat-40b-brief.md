# Bharat-40B — a 40B, India-first, coding + agentic model

**Shape:** 40B dense (single-node serving in India). **Budget: ~16T tokens (~400/param) + knowledge
distillation from a stronger teacher** — Gemma-3-27B needed 14T + distillation for this class; 8T
would not reach it. Tokenizer designed *before* data: fertility drives cost-equity for Indian users.

## 1. Data

**Pre-training — 16T in 3 stages.** Stage A, breadth (12.5T): high-quality English web 38% ·
code 20% (Stack-v2-scale + PR/issue threads — agentic priors start in pretraining) · math/science 10% ·
**Indic 17%** · other multilingual 5% · books/reference 5% · verified synthetic reasoning 5%.

*The Indic 17% (2.1T) is an honest ledger:* all cleaned native Indic data on earth is ~275B tokens
(Sangraha 251B — itself 65% machine-translated — + IndicCorp v2 ~21B + legal/govt). So: 275B real
× ~2.5 epochs + ~1.3T quality-gated synthetic (en→Indic translation of educational web,
transliteration pairs, native-script textbook-style) + **15% romanized/code-mixed** — 52% of Hindi
UGC online is romanized, so Hinglish is first-class data, not noise. 12 scheduled languages,
weighted by speakers × digital availability (hi ~35%; bn/te/mr/ta ~8–10% each; then gu/ur/kn/ml/pa/or/as).

*India-first is a pretraining decision, not an RLHF patch* — upsample 2–4×: Constitution + BNS,
SC/HC judgments, Parliament debates, RBI/SEBI/NITI, NCERT/NPTEL, Census/data.gov.in, newspapers,
regional literature, UPI/ONDC/DigiLocker documentation.

Stage B, mid-train (3T): code→30% with repo-level context, math→20% with verified solutions,
**~300B agentic-trace tokens** (tool-call logs, terminal sessions, synthesized SWE trajectories from
PR-issue-patch triples), top-decile Indic; anneal on the best data. Stage C (0.5T): context 8k→128k
(full repos, case law, multi-document).

**Post-training.** SFT ~1M curated: 35% agentic/code (sandbox-verified trajectories — a few hundred
verified SWE trajectories measurably move a 30B-class model double digits on SWE-bench; verified
tool-call sets), 25% reasoning CoT, **25% Indic written natively by paid writers — never translated**
(translationese destroys cultural grounding), 15% India-domain (law/agri/health/GST/UPI) + safety.

**RL:** verifiable rewards first — GRPO against unit tests, math answers, tool-call schemas, and
~5K sandboxed SWE/tool environments (pure RL at 32B is proven to reach >40% SWE-bench Verified).
Then preference RL with a **paid Indian annotator pool** balanced across language, region, gender,
caste; no reward for verbosity. Alignment spec: an India-first constitution — grounded in Indian
constitutional values, plural, non-partisan; Indian defaults when locale is unspecified
(₹/lakh/crore, DD-MM-YYYY, BNS not US common law, Survey-of-India borders).

## 2. Cleaning

| Bucket | Pipeline |
|---|---|
| All | exact + MinHash-LSH dedup → PII scrub incl. **Aadhaar/PAN/UPI formats** → 13-gram decontamination vs the **entire** eval suite, Indic benchmarks included |
| English web | educational-value classifier + AI-slop/content-farm detector |
| Indic | triple language-ID (reject wrong-script) · Unicode NFC **preserving ZWJ/ZWNJ** (else conjuncts shatter) · byte-level MinHash on char 5-grams (word-level fails agglutinative text) · per-language perplexity gates · **natively-built** casteist/communal slur lexicons · romanized text kept and tagged · MT-quality gate on all synthetic |
| Code | license allowlist · secrets scrub · repo-level dedup · AST/lint/compile filters · keep PR/issue threads intact |
| Agentic traces | replay in sandbox; keep only success or valid error-recovery |
| Math/science | LaTeX-preserving extraction; answer-verifiable subset tagged for RL reuse |

## 3. Evaluation

| Objective | Suite → bar |
|---|---|
| General parity | MMLU-Pro, GPQA-D, IFEval, Arena-Hard → within ~2–3 pts of latest same-size Gemma |
| Coding | LiveCodeBench (contamination-resistant) + **SWE-bench Verified ≥ 40%** + RepoBench; HumanEval = smoke test only |
| Agentic | BFCL v3 ≥ 70%, tau-bench, terminal-bench, OSWorld subset — score **success AND steps/cost**, plus private held-out environments (public agent benchmarks are gameable) |
| Indic | MILU (11 languages) — must beat same-size Gemma; IndicGenBench; IN22 chrF++; **romanization-robustness: native vs roman script score gap < 5%**; Hinglish QA |
| India-first | 3-layer custom eval: (1) factuality (polity/schemes/GST/railways, UPSC-style); (2) default-perspective probes — which currency/law/examples does it assume when unspecified; (3) fairness — IndiBias + **refusal-balance parity across religion/caste/region**; quarterly Indian-rater human eval |
| Continuous | per-domain loss dashboards; tokenizer-fertility regression in CI; decontamination audit before any reported number |

## 4. Fertility targets → tokenizer size

Targets anchored to measured tokenizers (Sarvam-1: 1.4–2.1 across Indic; 200K Indic-weighted
vocabs reach Hindi ~1.2; Llama-class tokenizers cost Indic users 3–8× English):

| Domain | Target (tokens/word) |
|---|---|
| English | ≤ 1.30 |
| Hindi, Marathi | ≤ 1.45 |
| Bengali, Urdu | ≤ 1.55 |
| Gujarati, Punjabi, Odia, Assamese | ≤ 1.70 |
| Tamil, Telugu, Kannada, Malayalam (agglutinative) | ≤ 1.85 |
| Romanized Hinglish | ≤ 1.35 |
| Code / Math | ≥ 3.3 / ≥ 3.0 chars per token; **digits split 0–9** (arithmetic accuracy > fertility) |

**Principle: an Indian-language user pays ≤ 1.4× English tokens for the same content.**

**Vocab = 262,144 (2^18), byte-fallback BPE.** Budget: ~110K English+code+math+symbols + 12 Indic
scripts × ~9K ≈ 108K + ~15K romanized/code-mix + ~20K other ≈ 253K → 262K. Cross-checks: vocabulary
scaling law puts a 40B model's optimum in the 200–300K band (larger models deserve larger vocabs);
Gemma 3 ships exactly 262K for multilingual balance; embedding cost at d=6144 is 4% of parameters
with **tied embeddings**; and a larger vocab directly cuts Indic serving cost — fewer tokens per
request. Trained on ~35% en+code / 45% Indic (fertility-driven upsample) / 10% math / 10% other,
with a ZWJ/ZWNJ-preserving pre-tokenizer; per-language fertility tracked in CI on every release.
