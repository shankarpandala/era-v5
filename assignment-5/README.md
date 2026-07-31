# Assignment 5 — Data Mixture & Curriculum

**16T tokens for the 40B dense India-first coding / agentic / reasoning model**
(Assignment 3 made concrete). This is a **written, supply-honest, machine-checked
specification** — the design artifact the course asked for — plus a real cleaning
increment on the lane the mixture shows to be starved.

No GPU cluster was available for this submission, so this document does **not**
claim trained 1B/3B proxy scores. Claiming numbers without jobs would violate the
session’s own rule (*a data decision is a hypothesis until a cheap experiment has
tested it*). What *is* complete: every share defended against inventory, Indic
four-tier split, floors, anneal reserve, D0–D3 / R0–R4 bands with examples, a
**falsifiable** proxy protocol with kill criteria (§9), arithmetic self-audit,
and shipped agentic tokens (§8).

## Deliverables (what “done” means here)

| # | Requirement | Status | Where |
|---|---|---|---|
| 1 | Budget share for every capability lane | Done | §1–§2, `mixture.json` |
| 2 | Indic split: verified / unverified / translated / synthetic | Done | §6 |
| 3 | Agentic, reasoning, long-context named + datasets | Done | §2–§3 |
| 4 | Protected always-on floors | Done | §5 |
| 5 | Anneal reserve held back | Done | §6.1 |
| 6 | Difficulty + reasoning-length bands with an example each | Done | §4.1 |
| 7 | Proxy as testable hypothesis (arms, metrics, kill rules) | Done (specified) | §9 |
| 8 | Actually train 1B/3B and paste scores | **Out of scope** — no cluster | §9.5 |
| 9 | Cleaning toward the starved slot | Done | §8 + published manifests |

```bash
cd assignment-5 && python3 validate_mixture.py
# exit 0 = lanes, floors, anneal, Indic tiers, D/R bands, proxy protocol all check out
```

| File | Role |
|---|---|
| `README.md` | Human-readable spec (this file) |
| [`mixture.json`](mixture.json) | Machine-readable mirror of every table number |
| [`validate_mixture.py`](validate_mixture.py) | Re-derives every arithmetic claim; fails loud on inconsistency |
| [`../data-cleaning/pipeline/agentic_slice.py`](../data-cleaning/pipeline/agentic_slice.py) | Starved-lane cleaning increment |
| [`../public/data-cleaning/manifest_agentic.json`](../public/data-cleaning/manifest_agentic.json) | Provenance + content hash for the cleaned slice |

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
reasoning traces and difficulty mass step up here (§4.1: R0–R2 / D0–D1 early →
R2–R4 / D2–D3 from stage B on).

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

### 4.1 Difficulty bands (D0–D3) and reasoning-length bands (R0–R4)

Difficulty and trace length are **orthogonal axes**. A D3 AIME item can be R1
if the gold solution is short; an R4 self-distilled trace can still be D1. The
selector samples them independently on the lanes that carry them
(`math_science`, `code`, `reasoning` for difficulty; `reasoning` for length).

#### Difficulty — what share of each stage's domain tokens

| Band | Meaning | Concrete example | A | B | C | D |
|---|---|---|---:|---:|---:|---:|
| **D0** foundational | One-step recall / pattern | GSM8K: "Pencils cost ₹12; Ravi buys 7. Total?" | 45% | 20% | 25% | 15% |
| **D1** routine multi-step | 2–5 standard steps | MATH Algebra after a substitution; textbook binary search with edge cases | 35% | 35% | 35% | 30% |
| **D2** exam-hard composition | Cross-topic or multi-file cause | JEE Advanced geometry+trig chain; flaky pytest that only fails under concurrent DB access | 15% | 30% | 25% | 35% |
| **D3** contest / research-hard | Olympiad, GPQA-diamond, multi-hunk SWE | AIME #12–15; SWE-bench Verified django ticket needing API judgment | 5% | 15% | 15% | 20% |

Stage A is allowed to be easy-heavy: the model is still learning syntax and
scripts. Stages B–D move mass into D2–D3 so late tokens buy benchmark headroom
rather than re-teaching arithmetic.

#### Reasoning length — assistant-trace tokens only

Ranges are **assistant CoT + final answer** under a Qwen2.5-class tokenizer,
excluding the problem statement. Only verifier-accepted traces enter R3–R4.

| Band | Tokens | Concrete example | A | B | C | D |
|---|---:|---|---:|---:|---:|---:|
| **R0** | 1–256 | "Capital of Telangana? → Hyderabad." (~30 tok) | 40% | 10% | 15% | 5% |
| **R1** | 257–1,024 | Closed form for sum of first *n* odds; short algebra (~500 tok) | 35% | 25% | 25% | 15% |
| **R2** | 1,025–4,096 | OpenThoughts MATH hard with one failed attempt then restart (~2k tok) | 20% | 30% | 30% | 30% |
| **R3** | 4,097–16,384 | AM-R1 / OpenR1 long CoT on a GPQA-diamond chemistry item (~8k tok, answer verified) | 5% | 25% | 20% | 30% |
| **R4** | 16,385–65,536 | Self-distilled LiveCodeBench hard: design + two failed impls + correct (~20–40k tok) | 0% | 10% | 10% | 20% |

Policy: **stage A is R0–R2 dominant** (format and short chains while the model
is plastic); **B–D shift into R2–R4** so ultra-long traces arrive after short-form
competence exists. R4 is zero in stage A on purpose — spending foundation
compute on 40k-token traces before the model can do R1 is a known failure mode.

Each column of each table sums to 100% (validator check 7).

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

## 9. Proxy experiments — written as a testable hypothesis

Every share above is a **hypothesis**. This section fixes how it gets confirmed
or killed *before* 40B / 16T — arms, budgets, metrics, and decision rules live
in `mixture.json` so a later run pastes numbers without rewriting the protocol.

**What this submission can and cannot do**

| | |
|---|---|
| **Can (and does)** | Specify a complete, falsifiable protocol; bind it to the same shares as §4; name the metric that kills each bet |
| **Cannot (no GPU cluster)** | Train 1B@40B or 3B@90B from scratch, or even a 5B-token mid-train sweep, inside this repo’s compute budget |
| **Must not** | Invent proxy scores. Empty `proxy_results` is honesty; fabricated deltas would fail the session’s own standard |

The course grades the **quality of reasoning and the evidence behind choices**.
For a no-cluster submission the evidence is: supply-honest accounting (§3),
literature-backed curriculum shape (§4), machine-checked arithmetic (§10), and
a protocol that a reviewer could run tomorrow without asking clarifying
questions. Measured proxy numbers remain the *next* gate before full scale —
they are not faked here.

### 9.1 Ideal path (when a cluster exists): 1B then 3B from scratch

Not claimed run — recorded so the hypothesis is complete:

1. Small dense model (1B, then 3B), same tokenizer family, Llama/OLMo-class.
2. Stratified downsample of the **same** inventory and tier/band mixes; no silent
   epochs beyond the caps in §3.
3. Compress the 16T curriculum by token budget; **stage order and lane shares
   stay identical to §4**:

   | Stage | Fraction of budget | At 1B (40B tok) | At 3B (90B tok) |
   |---|---:|---:|---:|
   | A foundation | 75.0% | 30.0B | 67.5B |
   | B rebalance | 18.75% | 7.5B | 16.9B |
   | C long-context | 3.75% | 1.5B | 3.4B |
   | D anneal | 2.5% | 1.0B | 2.25B |

4. Train all five arms (§9.3) under matched optim/seeds; only the mixture differs.
5. Apply kill criteria (§9.4). Survivors only promote 1B → 3B → 40B.

Rough cost: ~0.5–1k H100-days for 1B × 5 arms; multi-k more for 3B survivors.
That is a lab exercise, not a laptop exercise — hence not part of this hand-in.

### 9.2 Feasible path (single-node / modest GPU): mid-train surrogate

This is the experiment that is **actually runnable** without a multi-node
cluster. It is the intended first measurement when any GPU appears:

| Step | Detail |
|---|---|
| Base | Public checkpoint — Qwen2.5-1.5B, OLMo-2-1B, or Llama-3.2-3B |
| Budget | **Fixed 5B tokens per arm** of continue-pretrain (nothing frozen) |
| Data | Same five arm definitions as §9.3; downsample mixture shards to 5B each |
| Context | 4k train; optional 16k smoke only on arm H1 after a short long-context tail |
| Eval | Same primary metrics and kill rules as §9.4 |
| What it can falsify | Relative value of the Indic floor, late code upweight, agentic synthetic, curriculum vs uniform |
| What it must not claim | That 16T@40B is optimal — only that the **ranking** of mixtures is stable under short mid-training |

Until that job runs, `proxy_results.midtrain_surrogate` stays `null`.

### 9.3 Ablation arms (what each number is betting)

| Arm | Mixture | Hypothesis under test |
|---|---|---|
| **H0** uniform | 12.5% × 8 lanes, flat, no stages/floors/anneal | Curriculum structure beats a flat mix |
| **H1** proposed | This entire spec (stages, floors, Indic tiers, D/R bands, anneal) | Primary candidate |
| **H2** no Indic floor | H1 but Indic may fall to 2% in B–C; freed tokens → code+math | The 8% floor is load-bearing, not decorative |
| **H3** agentic real-only | H1 but agentic synthetic capacity = 0 (share collapses toward real unique × epochs) | The 3.5% agentic headline requires verifier-gated synthetic |
| **H4** no late upweight | Stage-A shares held for the whole budget (still a short HQ anneal tail) | Blakeney-style late domain upweight is real at proxy scale |

Order of work when compute appears: **mid-train surrogate on all five arms
first**; only if H1 still wins, spend on from-scratch 1B; only then 3B.

### 9.4 Metrics and kill criteria

**Primary (pass/fail):**

| Metric | Lane on trial | Decision rule |
|---|---|---|
| MILU macro | Indic floor | H1 must beat H2 by **≥ 1.5 pts**; else drop/soften the 8% floor |
| MATH-500 | math + reasoning shares | H1 must beat H0 by **≥ 2 pts**; else those shares are not buying capability |
| HumanEval+ or MBPP | code late-upweight | H1 must beat H4 by **≥ 1.5 pts**; else stage-B code 30% is unjustified |
| BFCL v3 simple-tool subset | agentic synthetic | H1 must beat H3 by **≥ 3 pts** tool-call exactness; else cut agentic synthetic capacity and the 3.5% share |

**Secondary (diagnostics):** MMLU-Pro regression guard; IndicGenBench hi/te/ta
subset; romanization gap on a 200-item Hinglish probe (targets: &lt; 8 pts after
mid-train surrogate, &lt; 5 pts after a full 3B proxy); held-out domain PPL on
web/code/indic/reasoning shards.

**Global gates before full scale:**

- H1 may not lose to H0 by &gt; 1 pt on **any** primary metric.
- H3 failure is a **redesign signal**, not something to ignore while keeping
  550B of agentic synthetic on the ledger.
- Decontaminate every proxy eval against every proxy training shard before
  reporting.

### 9.5 Status in this repository (final for this hand-in)

```text
proxy_experiments.status = "specified_not_run"
proxy_results.1B = null
proxy_results.3B = null
proxy_results.midtrain_surrogate = null
reason = "no GPU cluster; protocol fixed so results can be pasted later"
```

**Inventory note (A3 → A5):** A3 quoted ~275B cleaned native Indic. A5’s 385B
unique is the same inventory split by provenance (75 verified + 150 unverified
+ 160 translated-unique); synthetic stays outside unique (see
`inventory_reconciliation` in `mixture.json`).

## 10. Self-audit — "it adds up" is a checked property

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
7. Difficulty (D0–D3) and reasoning-length (R0–R4) band shares sum to 100%
   in every stage; token ranges are contiguous and non-overlapping.
8. Proxy protocol is complete: five arms, primary metrics with decision
   rules, 1B/3B budgets, stage fractions that match the full curriculum.

Current output: **all checks pass — 0 failures, 0 warnings.**

Reproduce:

```bash
cd assignment-5 && python3 validate_mixture.py
```

## References

- Muennighoff et al., *Scaling Data-Constrained Language Models*, arXiv 2305.16264 — epoch caps.
- Blakeney et al., *Does your data spark joy? Performance gains from domain upsampling at the end of training*, arXiv 2406.03476 — stage-B rebalance.
- Hu et al., *MiniCPM*, arXiv 2404.06395 — WSD anneal on held-out high-quality data.
- Grattafiori et al., *The Llama 3 Herd of Models*, arXiv 2407.21783 — over-training, late long-context stage, anneal.
- OLMo team, *2 OLMo 2 Furious*, arXiv 2501.00656 — mid-training on held-out quality data.
- Penedo et al., *FineWeb*, arXiv 2406.17557; Li et al., *DataComp-LM*, arXiv 2406.11794 — web supply.
- Yang et al., *SWE-smith*, arXiv 2504.21798 — agentic trajectory generation + the repo-disjointness claim §8 verifies.
