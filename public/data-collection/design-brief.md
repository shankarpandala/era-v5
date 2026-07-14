# A 40B India-first model — data, cleaning, evaluation, tokenizer

**Goal.** A 40B model that matches **Gemma-4-31B** — the dense, same-size quality frontier of the Gemma 4 family — on general benchmarks, and *beats* it on coding, agentic work, and Indic languages, while defaulting to an Indian view of the world.

**Shape & budget.** Dense, not MoE. Gemma 4 itself ships both a **31B dense** (quality frontier) and a **26B MoE** (~3.8B active, throughput frontier) — so at a *fixed* 40B total, with the bar set by the best same-size dense, all-active dense maximizes quality and keeps RLVR and community LoRA simple. MoE would win only if the constraint were serving throughput or the total budget could grow. Budget: **16T pre-training tokens (~400/param) + knowledge distillation** from a stronger teacher — Gemma-3-27B needed 14T + KD to reach this class; under-8T does not.

## 1. Data — three pillars

### Pre-training — 16T, 3 stages

Stage A · breadth · 12.5T:

| Bucket | % | ~Tok | Why |
|---|--:|--:|---|
| English web, Indian-authored upweighted | 38% | 4.75T | general capability; the English distribution itself skews Indian |
| Code + PR/issue threads | 20% | 2.5T | SE patterns; agentic priors begin in pre-training |
| Math + science | 10% | 1.25T | structured reasoning; LaTeX/units preserved |
| Indic — 12 languages | 17% | 2.1T | ledger below |
| Other multilingual · books · verified synthetic reasoning | 15% | 1.9T | transfer · coherence |

**Which 12 Indic languages, and why.** The highest-speaker, highest-digital-availability of the 22 scheduled languages, across both families: **Hindi, Bengali, Marathi, Telugu, Tamil, Gujarati, Urdu, Kannada, Odia, Malayalam, Punjabi, Assamese** — together ~95% of Indian first-language speakers. Weights ∝ speakers × digital text: hi ~35%; bn/te/mr/ta 8–10% each; gu/ur/kn/ml/pa/or/as the rest.

**Indic ledger (2.1T from a scarce base).** Genuinely-native, cleaned, *deduplicated* Indic web text that exists is only **~100–150B tokens** (Sangraha's native core = 64B verified + 24B unverified; IndicCorp v2 ~21B, CulturaX-Indic ~45B, Varta ~9B — heavily overlapping crawls). Sangraha's 251B headline is **65% synthetic** (machine-translation + transliteration), not native — the field's own tell that native supply is the binding constraint. Plan: **~120B unique native × ~4 epochs (≈0.5T)** + **~1.3T quality-gated synthetic** (en→Indic educational MT, transliteration pairs, native-script textbook-style) + **~15% romanized/code-mixed** Hinglish (≈50–56% of Hindi social-media users prefer it — data, not noise).

Stage B · mid-train 3T: code→30% (repo-level) · math→20% (verified solutions) · **~300B agentic-trace tokens** (tool-call logs, terminal sessions, SWE trajectories from PR-issue-patch triples) · top-decile Indic; anneal on the best data. Stage C · 0.5T: context 8k→128k (full repos, case law, multi-doc).

### Post-training — SFT ~1M

35% agentic/code (sandbox-verified trajectories + verified tool-calls, in **both** JSON-tool and code-action formats) · 25% reasoning CoT (math/science) · **25% Indic authored natively — never translated** · 15% India-domain + safety.

### RL & alignment — two stages

- **RLVR (verifiable rewards), GRPO:** unit tests (code), exact-answer (math), tool-call schema/execution (agentic), over **~4.5K sandboxed SWE/tool environments** (R2E-Gym-style). Precedent: DeepSWE lifted a 32B from ~23% → **42.2% SWE-bench Verified with RL alone**.
- **Preference & constitutional alignment:** an Indian annotator pool (language/region/gender/caste-balanced) *is* the preference signal; the constitution is derived from Indian constitutional values (plural, non-partisan); refusals track Indian law. No verbosity reward.

## 2. India-first by construction — engineered at every layer, not an RLHF patch

| Lever | Mechanism | Default it produces |
|---|---|---|
| Pre-training | Indian-authored English upweighted; India sources ×2–4 (Constitution + BNS, SC/HC judgments, RBI/SEBI, NCERT/NPTEL, Census, UPI/ONDC); quality classifiers calibrated on Indian English so idiom isn't filtered as "low quality" | "the Constitution" → India's |
| Knowledge frame | NCERT / BNS / RBI / Census canonical for civics, law, finance, geography | tax → GST & IT Act, not IRS |
| SFT | Native Indian daily-life scenarios (UPI dispute, IRCTC, ration card, board exams) | ₹ / lakh / crore, Indian names |
| Alignment | Indian annotators + Indian-values constitution | Survey-of-India borders; Indian-law refusals |
| Inference | Locale-default system prompt | DD-MM-YYYY, IST, Indian units |
| Eval | Unspecified-locale probes must resolve to the Indian frame | measurable, not vibes |

## 3. Cleaning — per objective

| Bucket | Pipeline |
|---|---|
| All | exact + MinHash-LSH dedup → PII scrub (incl. Aadhaar / PAN / UPI) → 13-gram decontamination vs the **entire** eval suite |
| English web | educational-value classifier + AI-slop / content-farm detector |
| Indic | triple language-ID (reject wrong-script) · NFC **preserving ZWJ/ZWNJ** (else conjuncts shatter) · **byte-level** MinHash (word-level fails agglutination) · per-language perplexity gates · natively-built casteist/communal slur lexicons · MT-quality gate on synthetic · romanized kept + tagged |
| Code | license allowlist · secrets scrub · repo dedup · AST/lint/compile filters |
| Agentic traces | sandbox replay; keep success or valid error-recovery only |
| Math / science | LaTeX- and units-preserving; answer-verifiable subset tagged for RLVR |

## 4. Evaluation — a numeric bar per objective

| Objective | Suite → bar |
|---|---|
| General parity | MMLU-Pro, GPQA-Diamond, IFEval, Arena-Hard → within ~2–3 pts of Gemma-4-31B |
| Math / science | MATH-500, AIME, GPQA-Diamond → ≥ same-size Gemma |
| Coding | LiveCodeBench + **SWE-bench Verified ≥ 42%** (Pass@1) + RepoBench (HumanEval = smoke only) |
| Agentic | BFCL v3 ≥ 70%, τ²-bench, terminal-bench, OSWorld subset → success **and** steps/cost; private held-out envs (public agent benchmarks are gameable) |
| Indic | MILU (beat same-size Gemma) · IndicGenBench · IN22 chrF++ · **romanization-robustness gap < 5%** · Hinglish QA |
| India-first | polity / schemes / GST factuality (UPSC-style) · default-perspective probes · IndiBias + **refusal-balance parity across religion / caste / region** · quarterly Indian-rater human eval |
| Continuous | per-domain loss dashboards · per-language & per-domain **fertility regression in CI** · decontamination audit before any reported number |

## 5. Fertility targets → tokenizer size

Anchored to measured tokenizers (Sarvam-1 avg ~2.0 tok/word over Indic at a 68K vocab; a 200K Indic-heavy vocab reaches Hindi ~1.2; Llama-3 costs Indic users ~2× on Hindi up to ~10–13× on Dravidian). **Principle: an Indian-language user pays ≤ ~1.4× English tokens for the same content** — Dravidian agglutinative scripts land ~1.4–1.6×, an accepted script-morphology floor.

| Domain / language | Target | Note |
|---|---|---|
| English | ≤ 1.30 tok/word | needs a vocab > Sarvam's 68K — we have 262K |
| Hindi, Marathi (Devanagari) | ≤ 1.45 | ~1.2 proven at 200K Indic-heavy |
| Bengali, Assamese, Urdu, Gujarati, Punjabi, Odia | ≤ 1.65 | |
| Tamil, Telugu, Kannada | ≤ 1.85 | agglutination + sandhi |
| Malayalam | ≤ 2.05 | most agglutinative — persistent outlier |
| Romanized Hinglish | ≤ 1.50 | orthographic variance (measured ~1.46) |
| Code | ≥ 3.4 chars/token | merge whitespace/indent runs (~25% of code tokens) |
| Math | ≥ 3.0 chars/token (prose) | digits split 0–9 → numeric runs = 1.0 *by design*; right-to-left number formatting in data prep |
| Science | ≥ 3.0 chars/token (prose) | single-token element symbols / SI units / Greek / operators; formulae & SMILES tokenize below target by design |
| Agentic / tool-call | ≥ 3.0 chars/token | ≥ 256 reserved special tokens + single-token tool boundaries + JSON-punctuation merges; in code-action mode ≈ code fertility |

**Vocab = 262,144 (2¹⁸), byte-fallback BPE.** The vocabulary scaling law (Tao et al., NeurIPS 2024) puts a 40B *compute-optimal* vocab near **~170K**; we go deliberately above it because (a) 16T tokens ≈ 400/param is heavily over-trained, which raises the optimum and cures rare-token undertraining; (b) 12 Indic scripts + code / math / science / agentic are exactly where extra vocab buys lower fertility and cheaper serving; (c) 262K matches Gemma 4 and is matmul-friendly (2¹⁸); (d) **tied embeddings cost only ~3.4–4% of params** at hidden dim ≈ 5120–6144. Budget: ~110K en+code+math+science+symbols · ~108K across 12 Indic scripts · ~15K romanized/code-mix · 256 reserved special (agentic) · remainder rare/other. ZWJ/ZWNJ-preserving pre-tokenizer; per-language fertility gated in CI — the Assignment-2 machinery.
