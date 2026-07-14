# A 40B India-first model — data, cleaning, evaluation, tokenizer

Goal: 40B params, latest-Gemma-class quality; top-tier coding, agentic work, Indic languages; views the world from an Indian perspective.

## Model shape: dense — a conditional decision, not dogma

| At a FIXED 40B total | Dense 40B | MoE @ 40B total (~8B active) |
|---|---|---|
| Quality ceiling | Highest — all params active (Gemma-4's 31B **dense** outranks its own 26B MoE) | Capped near ~8B-active quality |
| Training cost/token | 1× | ~0.25× |
| Serving | More FLOPs/token; either way all 40B must sit in memory | Cheaper FLOPs; + routing complexity |
| RL / RLVR stability | Well-trodden | Router load-balancing fights RL |
| Community finetuning | Easy (LoRA) | Harder |

The assignment fixes **total** params and the quality bar is the best same-size dense Gemma → dense wins inside the cap. **MoE becomes right** if the constraint were serving throughput or the total budget could grow (e.g. ~120B-total/17B-active). Budget: **16T tokens (~400/param) + distillation from a stronger teacher** — Gemma-3-27B needed 14T + KD for this class; 8T does not reach it.

## 1. Data

**Pre-training, 16T in 3 stages.** Stage A — breadth, 12.5T:

| Bucket | % | Why |
|---|--:|---|
| High-quality English web (Indian-authored upweighted) | 38% | general capability; the *English* distribution itself becomes India-centric |
| Code (+ PR/issue threads) | 20% | SE patterns; agentic priors start in pretraining |
| Math + science | 10% | structured reasoning |
| Indic, 12 scheduled languages | 17% | ledger below |
| Other multilingual / books / verified synthetic reasoning | 15% | transfer · coherence · reasoning |

**Indic ledger (2.1T):** all cleaned native Indic text in existence ≈ **275B tokens** (Sangraha 251B — itself 65% machine-translated — + IndicCorp v2 ~21B + legal/govt). Plan: 275B × ~2.5 epochs + ~1.3T quality-gated synthetic (en→Indic educational translation, transliteration pairs, native-script textbook-style) + **15% romanized/code-mixed** (52% of Hindi UGC is romanized — Hinglish is data, not noise). Weights ∝ speakers × digital availability: hi ~35%; bn/te/mr/ta 8–10% each; gu/ur/kn/ml/pa/or/as rest.

**Stage B — mid-train 3T:** code→30% (repo-level), math→20% (verified solutions), **~300B agentic-trace tokens** (tool-call logs, terminal sessions, SWE trajectories from PR-issue-patch triples), top-decile Indic; anneal on best data. **Stage C — 0.5T:** context 8k→128k (repos, case law, multi-doc).

**Post-training:**

| Component | Content |
|---|---|
| SFT ~1M | 35% agentic/code (sandbox-verified trajectories, verified tool-calls) · 25% reasoning CoT · **25% Indic authored natively — never translated** · 15% India-domain + safety |
| RL step 1 | GRPO on verifiable rewards: unit tests, math answers, tool-call schemas, ~5K sandboxed SWE/tool envs (pure RL at 32B is proven >40% SWE-bench Verified) |
| RL step 2 | Preference RL — Indian annotator pool (language/region/gender/caste-balanced); no verbosity reward |

## 2. India-first by construction (not an RLHF patch)

| Lever | Mechanism | Default it produces |
|---|---|---|
| Pretraining corpus | Indian-authored English upweighted; India injections ×2–4 (Constitution+BNS, SC/HC judgments, Parliament, RBI/SEBI, NCERT/NPTEL, Census, newspapers, UPI/ONDC docs); quality classifiers calibrated on Indian English so idiom isn't filtered as "low quality" | "the Constitution" → India's |
| Knowledge frame | NCERT / BNS / RBI / Census as canonical for civics, law, finance, geography | tax → GST & IT Act, not IRS |
| SFT | Native-written Indian daily-life scenarios: UPI dispute, IRCTC, ration card, monsoon sowing, board exams | ₹/lakh/crore, Indian names & examples |
| RL reward | Indian annotators ARE the preference signal; constitution from Indian constitutional values (plural, non-partisan); judge models on Indian-perspective rubrics | refusal norms per Indian law; Survey-of-India borders |
| Inference | Locale-default system prompt | DD-MM-YYYY, IST, Indian units |
| Eval | Perspective probes: unspecified locale must resolve to the Indian frame | measurable, not vibes |

## 3. Cleaning

| Bucket | Pipeline |
|---|---|
| All | exact + MinHash-LSH dedup → PII scrub incl. **Aadhaar/PAN/UPI formats** → 13-gram decontamination vs the **entire** eval suite |
| English web | edu-value classifier + AI-slop detector |
| Indic | triple language-ID (reject wrong-script) · NFC **preserving ZWJ/ZWNJ** (else conjuncts shatter) · **byte-level** MinHash char-5-grams (word-level fails agglutination) · per-language perplexity gates · natively-built casteist/communal slur lexicons · romanized kept + tagged · MT-quality gate on synthetic |
| Code | license allowlist · secrets scrub · repo dedup · AST/lint/compile filters · PR/issue threads intact |
| Agentic traces | sandbox replay; keep success or valid error-recovery only |
| Math/science | LaTeX-preserving; answer-verifiable subset tagged for RL |

## 4. Evaluation

| Objective | Suite → bar |
|---|---|
| General parity | MMLU-Pro, GPQA-D, IFEval, Arena-Hard → within ~2–3 pts of same-size Gemma |
| Coding | LiveCodeBench + **SWE-bench Verified ≥ 40%** + RepoBench (HumanEval = smoke only) |
| Agentic | BFCL v3 ≥ 70%, tau-bench, terminal-bench, OSWorld subset → success **and** steps/cost; private held-out envs (public agent benchmarks are gameable) |
| Indic | MILU (beat same-size Gemma), IndicGenBench, IN22 chrF++, **romanization-robustness gap < 5%**, Hinglish QA |
| India-first | factuality (polity/schemes/GST, UPSC-style) · default-perspective probes · IndiBias + **refusal-balance parity across religion/caste/region** · quarterly Indian-rater human eval |
| Continuous | per-domain loss dashboards; tokenizer-fertility regression in CI; decontamination audit before any reported number |

## 5. Fertility targets → tokenizer size

Anchored to measured tokenizers (Sarvam-1: 1.4–2.1 across Indic; 200K Indic-weighted vocabs reach Hindi ≈1.2; Llama-class tokenizers cost Indic users 3–8× English). **Principle: an Indian-language user pays ≤ 1.4× English tokens for the same content.**

| Domain / language | Target (tokens/word) |
|---|---|
| English | ≤ 1.30 |
| Hindi, Marathi | ≤ 1.45 |
| Bengali, Urdu | ≤ 1.55 |
| Gujarati, Punjabi, Odia, Assamese | ≤ 1.70 |
| Tamil, Telugu, Kannada, Malayalam (agglutinative) | ≤ 1.85 |
| Romanized Hinglish | ≤ 1.35 |
| Code / Math | ≥ 3.3 / ≥ 3.0 chars per token; digits split 0–9 (arithmetic > fertility) |

**Vocab = 262,144 (2^18) byte-fallback BPE:** ~110K en+code+math+symbols + 12 Indic scripts × ~9K ≈ 108K + ~15K romanized/code-mix + ~20K other. Cross-checks: vocab scaling law puts a 40B optimum at 200–300K; Gemma 3 ships exactly 262K; **tied embeddings ≈ 4% of params**; larger vocab = fewer tokens/request = cheaper Indic serving. Tokenizer mix ≈ 35% en+code / 45% Indic / 10% math / 10% other; ZWJ/ZWNJ-preserving pre-tokenizer; per-language fertility in CI (the Assignment-2 machinery).
