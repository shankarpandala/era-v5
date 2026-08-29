# ERA-V5

My assignments for the ERA-V5 course (building an LLM from scratch). All served
under `/era-v5/`:

| Assignment | Live | What it is |
|---|---|---|
| **1** | [/era-v5/](https://www.pandala.in/era-v5/) | Interactive, in-browser proofs of why activations, depth, embeddings and data matter. |
| **2** | [/era-v5/tokenizer/](https://www.pandala.in/era-v5/tokenizer/) | A shared 10k-token tokenizer for the wiki-faithful Markdown India pages (en/hi/te/mai) — grader-compatible HuggingFace format, live fertility ratios, self-score 81,400, downloadable. |
| **3** | [/era-v5/data-collection/](https://www.pandala.in/era-v5/data-collection/) | Design brief for a 40B India-first coding & agentic model: data, cleaning, evaluation, and a fertility-derived 262K tokenizer. |
| **4** | [/era-v5/data-cleaning/](https://www.pandala.in/era-v5/data-cleaning/) | Session 4's 8 cleaning strategies applied end-to-end to Bespoke-Stratos-17k (≈85M tokens) **and** a Sangraha Telugu slice — ghost-tag restructuring, MinHash dedup, benchmark decontamination, PII, manifests, determinism proven. Pipeline: [`data-cleaning/pipeline/`](data-cleaning/pipeline/). |
| **6** | [`assignment-6/`](assignment-6/README.md) | Training Data Execution System: documents → shards → manifests → mixture → packing → batches → training → ledgers → checkpoint → **crash** → resume → replay → fork → audit. One command (`python run_demo.py`) regenerates the whole `submission_artifacts/` bundle; 12/12 audit checks and 65 invariant tests pass, and the consumption ledger is byte-identical across runs. |
| **7** | [/era-v5/assignment-7/](https://www.pandala.in/era-v5/assignment-7/) · [`assignment-7/`](assignment-7/README.md) | **Kronecker Embedding V2 — embeddings that carry mathematical structure.** A deterministic 128-d embedding where `emb("9")+emb("9")` carries **exactly 18** (bit-exact float32, subtraction and negatives included) and a log dim makes ×/÷ additive, while 32 char slots keep spelling invertible — 10 algebraic properties proven with zero training. Across 7 arms differing only in the embedding, a 2-layer CPU transformer scores **36×** a learned table on number tokens never seen at a training input position (**377×** vs a frozen-random capacity control; the gap survives natural-language templates at **43×**); magnitude extrapolation is an honest, per-layer-localized negative. 92-run matrix, 53 tests, adversarially reviewed twice, independent audit, [grader card](assignment-7/GRADERS.md), interactive [`report.html`](assignment-7/submission_artifacts/report.html). |
| **8** | [/era-v5/assignment-8/](https://www.pandala.in/era-v5/assignment-8/) · [`assignment-8/`](assignment-8/README.md) | **How attention works now — every attention mechanism, in launch order, each as an answer to a bill.** From Bahdanau (1 Sep 2014) to DeepSeek-V4's Compressed Sparse Attention (24 Apr 2026), MiniMax M3, Kimi K3 and LongCat's indexer (Aug 2026): **88 timeline nodes + 47 footnotes** (the assignment's 17 minimum items → 21 tagged nodes, plus 67 extras), strictly by first public appearance, each with the problem at that moment, the idea, a live visual, honest buys / costs, and "when you'd actually pick it" for a 2K chatbot · 32K RAG · 128K coding · 1M agent. Nine eras show the field changing its mind (exactness → compute → positions → exact-got-cheap → length → memory → hybrids → compress-then-select) and a labelled prediction of what comes next. **Every date read from the primary source; 105 of 135 independently re-checked to the day (0 disagreements), the 2025–26 additions checked against the arXiv/HF APIs; trade-off text tightened by a 71-finding adversarial review**; corrections listed in the README, data machine-checked by [`check_assignment8.mjs`](scripts/check_assignment8.mjs), [grader card](assignment-8/GRADERS.md). |
| **9** | [`assignment-9/`](assignment-9/README.md) | **The Last Inch — silent failures between the model output and the scalar** (v3 — twice adversarially reviewed, every surviving finding rebuilt in). One Colab-ready notebook ([open in Colab](https://colab.research.google.com/github/shankarpandala/era-v5/blob/main/assignment-9/loss_harness.ipynb), committed fully executed, zero installs/downloads) makes the three lines between `model(tokens)` and the loss scalar correct **and observable**: all three shift alignments audited in printed **strings** before training (correct rate 1.0000); padding masks 508→186 contributing plus the wrong-`.mean()` reduction that silently reads 0.838 instead of 2.289 (gradient ×0.366) and a completions-only SFT variant; a packed seam dissected per-token (learnable 0.39 vs impossible 6.87 nats) with a full isolation ablation ending in doc2 ≡ doc2-alone; untrained PPL 261.3 vs V=259 — **and the check catches a real architectural leak: tied head + no-shift scores PPL 143.5 at init, before any training** (untied: 265); tied-vs-untied Δ = exactly V·d = 33,152; **three gradient-equivalent cross-entropies** (plain / row-chunked / online-softmax that never materializes the vocab dimension) shrinking the largest logits buffer **785→24.5→32 MiB** at GPT-2 vocab (measured ≈2,379→181→170 MiB here, ~13×; the analytic column is the audited, machine-independent claim). A t+2 head (both heads untied, full-stream sweeps) measures the MTP gap as conditional entropy (held +0.40; train plateaus +0.31 — memorization asymmetry, audited; a same-head two-shift estimator brackets it from above at +1.67), and the packing contract runs inside the training loop, not just the demo. The bug zoo quantifies the warning — bugs train to 0.0102/0.0000 vs 2.3793 — and **copy-circuit accuracies name each bug's mechanism** (reversed→prev 0.999 after a ~140-step attention-circuit delay; no-shift→self 1.000 instantly: its wire ships with the init). 40 tests exec the notebook's own cells; [`audit.py`](assignment-9/audit.py) re-derives all 45 README numbers from disk and fails on drift; [grader card](assignment-9/GRADERS.md). |
| **10** | [`assignment-10/`](assignment-10/README.md) | **Make the Loop Tell the Truth — one training step, instrumented end to end.** One Colab-ready notebook ([open in Colab](https://colab.research.google.com/github/shankarpandala/era-v5/blob/main/assignment-10/training_loop.ipynb), committed fully executed, zero downloads) measures all six Session-10 tasks on a real 493K-param byte-level transformer: every tensor in one step traced with each dimension named (31 tensors + grads + Adam's 2/weight state); one gradient verified by hand to **4.2e-10** relative (toy chain lands on the session's 64 and 16.064 exactly; a directional derivative certifies all 493,568 grads at once; the fp32 probe drowns while fp32 backward stays good — the stick, not the gradient); gradient accumulation **broken on purpose** — average-of-averages reads 3.0000 for a 2.6000 batch (15.4%) and, trained on a corpus where the bug cannot hide, converges to 0.0714 vs 0.0424 across 3 seeds (both optima computed by hand first; the buggy arm believes 92% of documents end after 2 bytes when the truth is 50%, *wins* its own doc-weighted metric, and the equal-count control collapses the gap to 0.00000) — plus a found-in-the-wild ratio bias in the "correct" per-step normalization (analytic p*=0.6681, measured); the grad norm reading **16.3×** its median *before* the update while the dashboard's EMA moved +3.9% (placebo-validated detectors, data-chosen caps — tight vs loose disclosed — realized clip factor ×0.119, the session's own number); MFU computed honestly at **24.5%** of measured attainable GEMM (6N shown to be +0.34% only by accidental cancellation, ±11–28% at other T; distance to 40% = matmuls own 45% of step time, d 64→512 lifts the loop to 50.2%); and **0.1 written out by hand in fp32/bf16/fp8-E4M3** bits via an exact-Fraction encoder matched bit-for-bit to torch (ties-to-even proven on purpose-built probes; fp16's flush-to-zero boundary measured at 2.9e-8; choice: bf16). 16 bytes/weight measured exactly. Five adversarial design reviews ran before a line was written; 63 audit checks re-derive every claim from disk ([`audit.py`](assignment-10/audit.py)), 36 tests exec the notebook's own cells, [grader card](assignment-10/GRADERS.md). |
| **5** | [`assignment-5/`](assignment-5/README.md) | Complete written mixture + curriculum for 40B / 16T: supply-honest 8-lane ledger, Indic 4-tier split, floors, anneal reserve, D0–D3 / R0–R4 bands with examples, falsifiable proxy protocol (5 arms + kill criteria; mid-train surrogate specified; **no fabricated GPU scores**), machine-checked by [`validate_mixture.py`](assignment-5/validate_mixture.py), plus cleaned SWE-smith agentic slice (3,287→790, 15.8M tokens). |

Single Vite multi-page build; each assignment is its own static page. Everything
runs client-side.

---

## Assignment 1 — Interactive Proofs

Interactive, in-browser proofs for the four claims of ERA-V5 Assignment 1. Every
model trains **live in your browser** on a tiny hand-written neural network
(no TensorFlow.js, no precomputed results) — turn the knobs and watch each claim
hold or break.

🔗 **Live:** https://www.pandala.in/era-v5/

### The four claims

| # | Claim | Money shot |
|---|-------|-----------|
| **S1-1** | Activations exist for a reason | Linear model is stuck at ~55% with a straight cut on two concentric rings; one ReLU hidden layer wraps the ring to ~99%. Plus a 3D feature-lift showing the rings becoming plane-separable. |
| **S1-2** | Depth without nonlinearity is a lie | 1 linear layer and 5 stacked linear layers draw the *same* line; the five weight matrices multiply out to a single 2×1 matrix. ReLU between the same layers breaks the tie. |
| **S1-3** | Embeddings learn similarity from next-token | Trained only to predict the next token in a toy grammar, the embedding table sorts itself into animal / fruit / verb clusters (PCA → 2D). Every nearest neighbour is same-category. |
| **S1-4** | Memorization vs generalization | An over-parameterized net memorizes 20 noisy points (train→0, test high), but the generalization gap collapses as the dataset grows to 2000. Data closes the gap. |

### Tech

- **React + Vite + Tailwind v4** — single page, all four demos.
- **`src/lib/nn.js`** — a small, dependency-free MLP with hand-written backprop (Dense, ReLU, sigmoid/softmax heads, Adam & SGD).
- **`src/lib/rng.js`** — seeded PRNG so every run is reproducible (seed is a knob).
- **react-three-fiber** — the S1-1 3D feature-lift (lazy-loaded).
- Decision boundaries are drawn on raw `<canvas>` by pushing a pixel grid through the model in one batched forward pass.

---

## Assignment 2 — Multilingual BPE Tokenizer (resubmission)

One shared **10,000-token HuggingFace BPE** tokenizer (NFKC + Metaspace) for the
**wiki-faithful Markdown** India pages in English, Hindi, Telugu and Maithili — the exact
corpus and scoring pipeline of the published course reference. `tokenizer.json` loads with
`tokenizers.Tokenizer.from_file`, `decode(encode(x))` preserves all visible text, and the
instructor's published evaluator reproduces our numbers drop-in.

🔗 **Live:** https://www.pandala.in/era-v5/tokenizer/

| Language | Fertility (tokens / faithful units) |
|---|---:|
| English | 0.605182 |
| Hindi | 0.615614 |
| Telugu | 0.603329 |
| Maithili | 0.610709 |

`spread = 0.012285` → **self-score = 81,399.95** (reference solution: 6,502.56); Hindi
penalty ×1.0. Full method, reproduce commands, and parity proof: [`tokenizer/README.md`](tokenizer/README.md).

```bash
npm run parity                        # JS == HuggingFace tokenizers (ids, decode, units)
cd tokenizer && pip install -r requirements.txt && python evaluate_tokenizer.py
```

## Assignment 8 — How attention works now (the timeline)

Every attention mechanism from Bahdanau (2014) to DeepSeek-V4 and Kimi K3 (2026), **in the order each
one was launched**, each explained as an answer to the bill of the one before it — with
honest pros / cons and "when you would actually pick it" for four scenarios, live visuals,
a cost calculator, and a labelled prediction of what comes next.

🔗 **Live:** https://www.pandala.in/era-v5/assignment-8/ · write-up [`assignment-8/README.md`](assignment-8/README.md) · [grader card](assignment-8/GRADERS.md)

Every date is read from the primary source (arXiv v1 line, blog / Reddit / model-card date)
and independently re-checked to the day; the sources and the dating method are in the
README, and the data is machine-checked:

```bash
npm run a8:check              # schema (incl. a reason in every scenario cell) · all 17 instructor items · README table in sync
npm run a8:check -- --links   # additionally fetch every source URL
npm run a8:test               # unit tests for the cost model, masks and RoPE scaling
```

## Develop

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # production build → dist/
npm run preview  # preview the production build
```

## Deploy

Pushing to `main` (or the assignment branch) runs `.github/workflows/deploy.yml`,
which builds and publishes `dist/` to the `gh-pages` branch via
`peaceiris/actions-gh-pages`. **One-time setup:** in the repo's
**Settings → Pages**, set the source to the **`gh-pages` branch**.
