# Assignment 5 — Data Mixture & Curriculum

**16T tokens for the 40B dense India-first coding / agentic / reasoning model** (the
Assignment 3 design brief made concrete). This directory is the spec:

| File | What it is |
|---|---|
| `README.md` | The human-readable mixture + curriculum specification (this file) |
| [`mixture.json`](mixture.json) | Machine-readable mirror — every number in this README's tables lives there |
| [`validate_mixture.py`](validate_mixture.py) | Self-audit that re-derives every arithmetic claim from `mixture.json` and fails loudly on any inconsistency |

The assignment's practical half — actually feeding the mixture's most starved lane —
lives in [`../data-cleaning/pipeline/`](../data-cleaning/pipeline/): a fresh
**SWE-smith agentic-trajectory slice** run end-to-end through the Assignment 4
cleaning pipeline (§8).

```bash
python3 validate_mixture.py     # exit 0 = the plan adds up; currently 0 failures, 0 warnings
```

---

## 1. The ledger at a glance

16,000B tokens across eight lanes and four curriculum stages. This table is
*derived* (by `validate_mixture.py`) from the per-stage shares in §4 — it is an
output of the spec, not an input:

| Lane | Main run (B) | Anneal (B) | Total (B) | Share |
|---|---:|---:|---:|---:|
| General web | 5,400 | 40 | 5,440 | 34.0% |
| Code | 3,150 | 100 | 3,250 | 20.3% |
| Math + science | 1,830 | 60 | 1,890 | 11.8% |
| Indic languages | 2,208 | 60 | 2,268 | 14.2% |
| Other multilingual + books | 912 | 0 | 912 | 5.7% |
| Reasoning traces | 1,002 | 80 | 1,082 | 6.8% |
| Agentic trajectories | 510 | 48 | 558 | 3.5% |
| Long-context documents | 588 | 12 | 600 | 3.8% |
| **Total** | **15,600** | **400** | **16,000** | 100% |

Why 16T for 40B: we are training far past compute-optimal on purpose —
inference-optimal, the Llama 3 argument (8B/70B at 15T+). The binding constraint
at this over-training ratio is not compute, it is **honest data supply**, which is
why the spec starts from supply, not from wishes.

## 2. Eight lanes, each pinned to benchmarks

A lane earns tokens only if there is a benchmark that will notice them:

| Lane | Contents | Must move |
|---|---|---|
| `web` | General web English, Indian-authored sources upweighted | MMLU-Pro, GPQA-Diamond, Arena-Hard, IFEval |
| `code` | Repo files, PR/issue threads, notebooks | SWE-bench Verified, LiveCodeBench, RepoBench |
| `math_science` | Math + science text and worked solutions | MATH, AIME, GPQA-Diamond |
| `indic` | 12 scheduled Indian languages, 4 provenance tiers (§6) | MILU, IndicGenBench, IN22 chrF++, romanization-robustness gap |
| `multilingual_other` | Non-Indic non-English + books | transfer support; no direct benchmark |
| `reasoning` | Reasoning traces in R0–R4 length bands | GPQA-Diamond, AIME, LiveCodeBench (reasoning mode) |
| `agentic` | Tool-use / SWE trajectories with loss masks | SWE-bench Verified, BFCL v3, tau-bench, terminal-bench |
| `long_context` | Whole repos, books, legal, multi-doc | RULER, HELMET, repo-level SWE tasks |

## 3. Supply before demand — the honest inventory

The rule that makes this a plan instead of a wish:

> **main-run allocation ≤ (unique − reserved-for-anneal) × epoch cap + declared synthetic capacity**

Epoch caps follow the data-constrained scaling result (Muennighoff et al. 2023,
arXiv 2305.16264): up to ~4 epochs of repetition costs little vs. fresh data,
and returns decay rapidly beyond. High-value scarce lanes get their full cap;
abundant commodity lanes don't need it.

| Lane | Unique real (B) | Epoch cap | Synthetic cap (B) | Held for anneal (B) | Where the tokens come from |
|---|---:|---:|---:|---:|---|
| `web` | 5,500 | 4 | 0 | 100 | FineWeb-Edu 1.3T + DCLM-baseline 3.8T (dedup-overlap discounted) + Nemotron-CC HQ ~1.1T |
| `code` | 1,500 | 4 | 0 | 60 | Stack v2 dedup ~775B + issues/PRs/StackExchange/notebooks |
| `math_science` | 500 | 4 | 300 | 20 | MegaMath 371B, FineMath 34B, OpenWebMath 15B, peS2o, arXiv |
| `indic` | 385 | 4 | 1,520 | 15 | Sangraha (verified/unverified/synthetic splits), IndicCorp v2, MADLAD-400 Indic |
| `multilingual_other` | 2,000 | 2 | 0 | 0 | CulturaX / MADLAD non-Indic non-English, PG-19 + open books |
| `reasoning` | 60 | 3 | 950 | 20 | OpenThoughts3, AM-R1-Distilled, OpenR1-Math, Nemotron-PT, our cleaned Stratos 81.5M |
| `agentic` | 5 | 3 | 550 | 2 | SWE-smith trajectories (incl. our cleaned slice, §8), SWE-Gym, xlam/glaive/ToolACE FC |
| `long_context` | 500 | 2 | 100 | 10 | Repo-level concatenation (Stack v2 sources), PG-19, Indian case law, arXiv full text |

Implied epochs on real data, per the validator: web **1.0**, code **2.2**,
math **3.2**, indic **1.9**, multilingual **0.5**, reasoning **1.3**,
agentic **~0** (synthetic-carried), long-context **1.0** — every lane inside its
cap. The two structurally starved lanes are honest about it: **reasoning** and
**agentic** run mostly on *declared, verification-gated synthetic capacity*
(§7), not on silently recycled epochs.

## 4. The curriculum — four stages

Percent of each stage's tokens per lane (each column sums to 100):

| Lane | A · foundation<br>0 → 12.0T | B · rebalance<br>12.0 → 15.0T | C · long-context<br>15.0 → 15.6T | D · anneal<br>15.6 → 16.0T |
|---|---:|---:|---:|---:|
| `web` | 40 | 18 | 10 | 10 |
| `code` | 18 | 30 | 15 | 25 |
| `math_science` | 10 | 20 | 5 | 15 |
| `indic` | 16 | 8 | 8 | 15 |
| `multilingual_other` | 7 | 2 | 2 | 0 |
| `reasoning` | 5 | 12 | 7 | 20 |
| `agentic` | 2 | 8 | 5 | 12 |
| `long_context` | 2 | 2 | 48 | 3 |

**A — foundation (12T).** Web-heavy: broad world knowledge and linguistic
competence are the substrate everything else fine-tunes. Indic runs *hot* here
(16%) — scripts, morphology and code-mixing are learned early while the model is
most plastic, then defended by the floor (§5) instead of re-taught. Agentic sits
at its floor because its real supply (5B unique) cannot honestly fill more.

**B — capability rebalance (3T).** Code 30%, math 20%, reasoning 12%, agentic
8%. Upweighting domain data late in training buys benchmark capability far more
cheaply than spreading it uniformly (Blakeney et al. 2024, arXiv 2406.03476);
reasoning traces step up their length bands here (R0–R2 in stage A, R2–R4 from
stage B on).

**C — long-context extension (600B).** RoPE base is rescaled and the context
window extended; 48% of tokens are genuinely long documents (whole repos, books,
case law, multi-doc bundles) while every other lane keeps a maintenance share so
short-context skills don't regress — the Llama 3 recipe: a short dedicated
stage, late, rather than paying long-sequence attention cost for 16T.

**D — anneal (400B).** WSD-style: learning rate decays to zero over 400B tokens
of **never-before-seen, highest-quality data** — the held-back reserve (§6) plus
anneal-grade verified synthetic. Annealing over held-out high-quality data is
the consistent winner in MiniCPM's WSD ablations, Llama 3's final stage, and
OLMo 2's mid-training. Multilingual-other goes to 0 — transfer support has done
its job by 15.6T.

## 5. Protected floors — the always-on guarantee

The stage-B/C rebalancing pressure that funds code and reasoning is exactly the
pressure that historically crushes small-share lanes. Three lanes carry a floor
the mixture selector may never cross, in any stage:

| Lane | Floor | Stage-by-stage actual (A/B/C/D) |
|---|---:|---|
| `indic` | 8% | 16 / **8** / **8** / 15 — sovereignty is load-bearing, not decorative |
| `agentic` | 2% | **2** / 8 / 5 / 12 — the model must never forget how to act |
| `reasoning` | 3% | 5 / 12 / 7 / 20 — trace format stays warm from token 0 |

The validator checks the floor against every stage's share; bold entries are
stages that sit exactly on their floor.

## 6. The Indic lane — four provenance tiers

2,268B Indic tokens total (main run + anneal), split by how much we trust the
text, with per-tier epoch caps — trust buys repetition:

| Tier | Unique (B) | Epoch cap | Synthetic (B) | Total (B) | Sources |
|---|---:|---:|---:|---:|---|
| Verified | 75 | 4 | 0 | 300 | Sangraha verified, IndicCorp v2, NCERT/govt/legal curated |
| Unverified | 150 | 3 | 0 | 450 | Sangraha unverified, MADLAD-400 Indic clean, CulturaX Indic |
| Translated | 160 | 2 | 400 | 720 | Sangraha translated Wikimedia + quality-gated en→Indic MT of educational/technical text |
| Synthetic | 0 | 1 | 798 | 798 | Native-style textbook generation, transliteration/romanized pairs, code-mixed Hinglish UGC-style |

Tiers sum to the lane total exactly (validator check 5). Two policy caps ride on
top: **translated ≤ 35% of the lane** (720B ≤ 794B — translationese must not
become the model's idea of Hindi), and the synthetic tier targets **15%
romanized text**, because real Indian users type Indic languages in Latin
script and the romanization-robustness gap is a benchmark, not an afterthought.

The anneal reserve (§6.1) and the always-on floor (§5) mean Indic is present in
every stage *and* in the final, highest-influence 400B.

### 6.1 The anneal reserve — genuinely held back

"Anneal on held-out high-quality data" only works if the data is actually held
out. Each lane's `reserved_for_anneal` tokens are excluded from stages A–C by
the supply-honesty check itself, and the anneal pools must be backed by that
reserved real data plus anneal-grade synthetic:

| Lane | Pool (B) | = reserved real (B) | + anneal-grade synthetic (B) |
|---|---:|---:|---:|
| `web` | 40 | 100 | 0 |
| `code` | 100 | 60 | 40 |
| `math_science` | 60 | 20 | 40 |
| `indic` | 60 | 15 | 45 |
| `reasoning` | 80 | 20 | 60 |
| `agentic` | 48 | 2 | 46 |
| `long_context` | 12 | 10 | 2 |
| **Total** | **400** | | |

Stage D spends exactly the 400B the pools hold; the un-annealed remainder of
the reserved real data (e.g. web's other 60B) feeds SFT/RLVR in Sessions 17–18
— reserve once, spend twice, never double-count.

## 7. Synthetic data policy — capacity only where there is a verifier

Synthetic capacity is **0** wherever we cannot check the output:

| Lane | Capacity (B) | Verifier that gates generation |
|---|---:|---|
| `web` | 0 | none exists → none declared |
| `math_science` | 300 | solution checked against ground-truth answer |
| `indic` | 1,520 | MT quality gates; transliteration pairs are checked by round-trip |
| `reasoning` | 950 | self-distilled traces kept only when the final answer verifies |
| `agentic` | 550 | SWE-smith-style generation, kept only on sandbox resolution |
| `long_context` | 100 | multi-doc synthesis with citation-consistency checks |

This is the same principle the cleaning increment applies at document level
(§8): *sandbox resolution is the quality signal for trajectories, not a
web-prose classifier.*

## 8. Feeding the starved lane — the agentic cleaning increment

The ledger's weakest point is agentic: **5B unique real tokens** carrying a
558B allocation. So this assignment doesn't just budget the lane — it ships a
cleaned increment. A fresh slice of
[`SWE-bench/SWE-smith-trajectories`](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)
(MIT) went through the Assignment 4 pipeline (v1.1.0), with every stage
re-thought for terminal trajectories instead of web prose:

| | |
|---|---|
| Raw → kept | 3,287 → **790 trajectories** (24% survive) |
| Final tokens | **15.79M** (Qwen2.5-0.5B tokenizer) |
| Supervised tokens | **3.49M** (22.1%) — loss on assistant turns only, never on tool observations |
| Quality gate | **sandbox-resolved only** + final submit turn + ≥2 tool-calling turns + non-empty patch + degenerate-loop limit |
| Agentic texture | mean 23.3 assistant turns; **68.7%** of kept trajectories recover from a failing command to a verified fix |
| Dedup | keyed on task identity (task text + patch) — transcript shingling is blinded by shared scaffold boilerplate; 5 extra rollouts removed |
| Decontamination | **0** 13-gram hits vs SWE-bench Verified/Lite, MATH-500, GSM8K; repo-level audit vs the 12 SWE-bench eval repos: 0 (SWE-smith mines other repos by design, arXiv 2504.21798 — *verified on the slice, not trusted*); 3 canaries injected |
| PII | 226 emails + 396 IPs masked with typed placeholders; code/fixture exemptions from the A4 v1.1 precision rules |
| Fidelity | NFC / mojibake repair / whitespace collapse **measured but not applied** — tool observations are sandbox ground truth and patches are whitespace-sensitive; Trojan-Source bidi controls stripped |
| Determinism | content SHA-256 `e7d0f4ea…` reproduced bit-identically by `--verify` |

Two deliberate inversions of the web-prose pipeline: the **edu classifier is
not consulted** (scoring terminal logs with a web-prose model is the A4
filter-bias trap — sandbox resolution is this corpus's quality signal), and the
**loss mask is part of the canonical format** — training on tool observations
teaches a model to invent tool results instead of calling tools.

```bash
cd ../data-cleaning/pipeline
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
export LID_PATH=$PWD/lid.176.bin
python3 agentic_slice.py            # full 8-stage run -> out/stats_agentic.json, out/manifest_agentic.json
python3 agentic_slice.py --verify   # determinism proof (identical content hash)
```

## 9. Self-audit — "it adds up" is a checked property

`validate_mixture.py` re-derives every arithmetic claim above from
`mixture.json` and exits non-zero on any failure:

1. Every stage's shares sum to 100%.
2. Stage budgets sum to the 16,000B total.
3. **Supply honesty** — each lane's main-run allocation fits inside
   (unique − reserved) × epoch cap + declared synthetic; implied epochs printed.
4. Protected floors hold in every stage.
5. Indic tiers sum to the lane total, respect per-tier epoch caps, translated
   stays under its 35% ceiling.
6. The anneal reserve is genuinely held back: stage-D spend ≤ per-lane pools,
   pools backed by reserved real + anneal-grade synthetic, 400B total.

Current output: **all checks pass — 0 failures, 0 warnings.**

## References

- Muennighoff et al., *Scaling Data-Constrained Language Models*, arXiv 2305.16264 — epoch caps.
- Blakeney et al., *Does your data spark joy? Performance gains from domain upsampling at the end of training*, arXiv 2406.03476 — stage-B rebalance.
- Hu et al., *MiniCPM*, arXiv 2404.06395 — WSD anneal on held-out high-quality data.
- Grattafiori et al., *The Llama 3 Herd of Models*, arXiv 2407.21783 — over-training, late long-context stage, anneal.
- OLMo team, *2 OLMo 2 Furious*, arXiv 2501.00656 — mid-training on held-out quality data.
- Penedo et al., *FineWeb*, arXiv 2406.17557; Li et al., *DataComp-LM*, arXiv 2406.11794 — web supply.
- Yang et al., *SWE-smith*, arXiv 2504.21798 — agentic trajectory generation + the repo-disjointness claim §8 verifies.
