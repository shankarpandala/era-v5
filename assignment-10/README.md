# Make the Loop Tell the Truth — one training step, instrumented end to end

**Assignment 10 — the training loop.** *"Take a small model and a real loop, and make it
tell you the truth about itself. […] Print things and check things. Every serious
training bug is silent, and the loss curve is not going to be the one that tells you."*

**One-page card for graders: [GRADERS.md](GRADERS.md). The notebook:
[`training_loop.ipynb`](training_loop.ipynb) — committed fully executed, runs top to
bottom on CPU with zero downloads.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shankarpandala/era-v5/blob/main/assignment-10/training_loop.ipynb)

Session 9 ended with one scalar. This assignment is about the step that scalar triggers
— forward → loss → backward → update → wipe — and about the instruments that tell you
whether the step is doing what you think it is. All six required tasks are *measured*
on a real (tiny, 493,568-parameter) byte-level transformer and a real loop, and every
claimed number below is re-derived from the committed artifacts by an independent audit
([`audit.py`](audit.py)) that never executes the notebook:

> **Task 2 in one line:** the session's toy chain lands on **64** three ways (hand chain
> rule, central nudge, autograd — and the forward nudge reproduces the session's
> **16.064**, whose 0.064 excess is the *law* err = 64·ε, verified to 1%); on the real
> model, nudging `blocks.0.mlp.fc1.weight[347,104]` reproduces `backward()`'s
> **+0.054479585399** to relative error **4.2 × 10⁻¹⁰** (~10.6 decimals), a directional
> derivative certifies **all 493,568 gradients at once** to 1.3 × 10⁻⁸ — and the same
> probe in fp32 tops out at 4.6 × 10⁻⁴ with five ε values giving *exactly zero* loss
> difference: the measuring stick drowns, not the gradient (fp32 `backward()` matches
> the fp64 reference to 3.9 × 10⁻⁷).
>
> **Task 3 in one line:** average-of-averages with unequal micro-batches mis-reads the
> session's batch as **3.0000** instead of **2.6000** (**15.4%**); trained for real on a
> corpus where the mistake cannot hide, the buggy arm converges to held-out
> **0.0714** vs **0.0424** (analytic asymptotes 0.0714 / floor 0.0385) in **all three
> seeds**, because it comes to believe **p(eos|prefix) = 0.916** when the truth is 0.5
> (probe: correct arm reads **0.499**) — and it *wins* the doc-weighted metric
> (0.051 vs 0.137): the bug is a silently substituted objective, not noise. The
> equal-count control collapses the gap to **0.00000**. Bonus finding, caught while
> building the section: normalizing by *this step's* token count is itself a ratio
> bias — analytically p* = **0.6681**, measured 0.638/0.740.
>
> **Task 4 in one line:** at step 150 the grad norm read **16.3×** its settled median
> *before* `optimizer.step()` was committed; every loss that could reveal the *damage*
> existed only afterwards (clean-probe +**0.654** nats, 17 steps to recover), the
> dashboard's EMA loss moved a **+3.9%** wiggle, and a cap chosen from the observed
> norm distribution cut the damage to +**0.116** (loose, guard-only cap: +**0.107**,
> binding on just 4 of 110 settled steps) — the realized scale factor at the spike,
> **×0.119**, is coincidentally the session's own worked number.
>
> **Task 5 in one line:** this container does **39,074 tokens/s** at d = 128 → achieved
> 115.3 GFLOP/s against a measured 471 GFLOP/s attainable-GEMM peak = **24.5%
> utilization** (a *flattering* denominator, and the README says so); the famous 6N is
> within **+0.34%** of the exact 2,951,424 FLOPs/token *only by accidental
> cancellation* (at T = 32 it is +11.5% off, at T = 512, −28.3%); and the distance to
> 40% is not mystery overhead — matmuls own only **45%** of step time, and widening the
> model to d = 512 lifts the same loop to **50.2%**.
>
> **Task 6 in one line:** 0.1 by hand is `0|01111011|10011001100110011001101` (fp32,
> rel err 1.5 × 10⁻⁸), `0|01111011|1001101` (bf16, 9.8 × 10⁻⁴), `0|0011|101` (fp8
> E4M3, 1.6 × 10⁻²) — every bit string re-derived by an exact rational-arithmetic
> encoder, matched **bit-for-bit against torch**, ties-to-even exercised in both
> directions on purpose-built probes (0.1 itself never ties) — and the training choice
> is **bf16**, because fp16 sends every gradient below 2.98 × 10⁻⁸ to *exactly zero*
> (measured at the boundary: 3.0 × 10⁻⁸ survives as the smallest subnormal, 2.9 × 10⁻⁸
> dies) and loss scaling ×1024 only moves that cliff to ~3 × 10⁻¹¹ while parking values
> in the ~7-bit subnormal range.

```bash
cd assignment-10
pip install -r requirements.txt
python run_demo.py --verify-only   # ~5 s: independent audit of the committed artifacts
python run_demo.py --fast          # ~1 min smoke run, reduced budgets, same pipeline
python run_demo.py                 # ~3.5 min CPU: full re-run, rewrites every artifact
python -m pytest tests -q          # 36 invariant tests (they exec the notebook's cells)
```

**Definition of done** (all three green): `pytest` passes, `--verify-only` prints
`verdict: PASS` (63 checks), and every deterministic number in this README is
re-derived from [`submission_artifacts/results.json`](submission_artifacts/results.json)
by the audit — machine-specific *timings* (tokens/s, GFLOP/s, MFU) are quoted from the
committed run and audited only for arithmetic consistency, never verbatim (the
Session-9 convention for measured-vs-analytic claims).

---

## Requirement checklist

| # | requirement | where | the number |
|---|---|---|---|
| 1 | every tensor shape in one step, one line per dimension | notebook §3 | **31** tensors traced fwd+loss, every parameter's grad (same shape), Adam's 2×493,568 state values |
| 2 | verify one gradient by hand | §4 | toy chain **64** three ways; real model fd vs `backward()` rel err **4.2e-10**; whole gradient via directional derivative **1.3e-8** |
| 3 | break gradient accumulation on purpose, plot both curves | §5 | mirror **2.6000** vs **3.0000** (15.4%); trained arms **0.0424** vs **0.0714**, 3 seeds, both curves + analytic asymptotes on one plot |
| 4 | grad norm every step; one step where it moved before the loss | §6 | spike **16.3×** at step 150, known pre-update; clean-probe damage (+0.654) exists only post-update; EMA wiggle +3.9% |
| 5 | my own MFU, honestly, and the distance to 40% | §7 | **24.5%** of measured attainable GEMM (bias direction stated); matmul time share 45%; d-sweep 11.4 → 50.2% |
| 6 | 0.1 in fp32/bf16/fp8 E4M3 bits, by hand; pick a precision | §8 | three bit strings above + fp16 for contrast; torch agrees bit-for-bit; choice: **bf16**, from the measured cliff |
| + | the session's 16 bytes/weight, measured | §9 | **7,897,088** bytes = exactly 16 × 493,568; 2B → 29.8 GiB … 120B → 1,788 GiB; 80 GB ⇒ ~5.4B ceiling |

The notebook is the primary artifact and carries the same story cell by cell. Five
adversarial design reviews ran *before* a line of it was written (§10) — two of its
best sections exist because a reviewer or a measurement destroyed the first design.

## 1. One step, every tensor (§3)

The traced step prints 31 tensors in creation order — `tokens (8, 128)` [batch
sequences × positions] through `block0.attn.scores (8, 4, 128, 128)` [… query position
× key position] to `logits (8, 128, 259)` [one score per vocabulary id], the shifted
pair `(8, 127, 259)`/`(8, 127)`, and the flattened views `(1016, 259)`/`(1016,)` where
mis-sliced tensors go to hide — then every parameter with its gradient (grad shape ==
weight shape: one number *per weight*), then what Adam keeps between steps (`exp_avg`,
`exp_avg_sq`: 987,136 state values = 2 per weight — the session's widget-#4 bars), then
the wipe, asserted. All shape relations are audited against the config.

## 2. One gradient, verified by hand (§4)

**The toy chain teaches the method.** x=2, w1=3, w2=4, t=20 → h=6, y=24, loss=16; chain
rule one link at a time gives ∂L/∂y=8, ∂L/∂w2=48, ∂L/∂h=32, **∂L/∂w1=64**. The
session's forward nudge reproduces 16.064064 → 64.064: the 0.064 is not noise but
**exactly 64·ε** (the f″/2 Taylor term — verified at every ε over five decades). The
*central* difference cancels it, and because this loss is quadratic in w1 the
cancellation is complete — so the ε-tuning story is deliberately *not* told on the toy
(it would be theater) but on the real model, where the loss is genuinely non-quadratic.
`torch.autograd.gradcheck` cross-validates.

**The real model, protocol pinned before the numbers:** float64 deepcopy (`.eval()`,
one BLAS thread, the live model untouched), deterministic weight selection (argmax
|grad| of a named untied tensor — because the sharpest trap is real: **224 of 259**
embedding rows have *exactly zero* gradient on the fixed batch, and a finite difference
there "confirms" 0 = 0 while verifying nothing), save/assign/restore with bit-exact
restoration asserted, realized step (w⁺−w⁻)/2 in the denominator. The ε sweep shows the
U-curve: truncation error falling as ε² from the left, cancellation noise rising as 1/ε
on the right, plateau at **2.3 × 10⁻¹¹ absolute** (rel **4.2 × 10⁻¹⁰**) at ε = 10⁻⁵.
One element per parameter type (embedding row, attention qkv, LayerNorm gain, MLP,
head) all agree < 10⁻⁵; a random-unit-vector directional derivative certifies the
*entire* gradient in one scalar (rel err **1.3 × 10⁻⁸**).

**The honest fp32 lesson, precisely phrased:** the fp32 *probe* tops out at 4.6 × 10⁻⁴
and returns exactly 0.0 at five of ten ε values (complete cancellation: the loss
difference falls under one fp32 ulp) — while fp32 `backward()` itself matches the fp64
reference to **3.9 × 10⁻⁷**. The precision limit is in the measuring stick, not the
gradient. Both sweeps share one figure (`plots/fd_ucurve.png`).

## 3. Gradient accumulation, broken on purpose (§5)

**5a.** Gradients accumulate — asserted *bitwise* (`torch.equal(g₂, 2·g₁)`), not
`allclose`. The session's arithmetic is realized as actual logits whose per-token CE is
exactly 2.0/2.0/5.0 over 4/4/2 tokens (per-token losses committed; the audit re-combines
them from disk): token-weighted **2.6000**, average-of-averages **3.0000**, error
**15.4%** — and identically zero whenever counts match, which is why casual testing
never caught it. (That 15.4% is a *measurement* error on one batch; the training effect
below is a different, separately measured number.)

**5b.** The identity that makes accumulation legitimate: correct token-weighted
accumulation over micro-batches of 3/3/3/33 target slots equals one big batch —
forward logits bit-identical (max diff **0.0**), gradients rel-L2 **9.4 × 10⁻⁸** in
fp32 and **1.6 × 10⁻¹⁶** on a float64 copy (machine-clean). The buggy combine through
the *identical* harness lands at **0.57** — same code path, different denominator.

**5c.** The two training curves. On ordinary text a transformer *infers its way around*
a mis-weighted mixture (that negative result is §5d, measured, not hidden), so the
corpus is built so the conflicted decision sits at positions whose prefixes are
identical across document types: documents are `<bos>aa<eos>` or `<bos>` + 32 a's +
`<eos>`, 50/50, **one document per micro-batch** (the composition under which
per-micro-batch averaging actually happens in the wild: per-sequence loss averaging in
SFT). Both optima are computable by hand before training: correct → p(eos|`<bos>aa`) =
1/2, eval floor ln2/18 = **0.0385**; buggy → p = 11/12 ≈ **0.9167**, eval asymptote
**0.0714**. Measured (identical inits and checksummed identical streams, no clipping,
3 seeds): correct **0.0424 / 0.0449 / 0.0425**, buggy **0.0714 / 0.0715 / 0.0715** —
the buggy arm parks *on* its analytic asymptote; probes read **0.499/0.452/0.499** vs
**0.916/0.915/0.916**. The buggy model believes ~92% of documents end after two bytes;
the truth is 50%. And on the doc-weighted metric the sign **flips** (buggy 0.051 beats
correct 0.137): the bug is not damage, it is a *substituted objective* — which is
exactly why no exception fires. Both curves, the per-seed gap, and the belief probe
share one figure (`plots/accum_gap.png`) with the analytic lines drawn in.

**5c″ — found while building this, kept because it is real:** the first version of 5c
drew each step's documents 50/50 and the *correct* arm converged to p ≈ 0.67, not 0.5.
That is not a bug in the model — normalizing by *this step's* token count makes the
objective a ratio estimator, and a short document's hazard slot systematically lands in
smaller-N steps: enumerating the 8 compositions gives p* = **0.6681**, and the
measured probes (600 steps, 2 seeds) read **0.638 / 0.740**. The sharp version of
"normalize by tokens": divide by a *constant* (planned tokens per batch), not by
whatever this batch happened to contain, whenever count varies with content. 5c
therefore fixes the composition at 2 short + 2 long (N = 72, constant) — and the bias
vanishes (probes 0.499).

**5c′.** Negative control: micro-batches built to hold exactly 33 target slots each
(eleven short documents, or one long). Equal counts ⇒ 1/K *is* n/N: step-0 gradients
coincide to rel-L2 **0.0** and the curves' max divergence over training is **0.00000**
against the 0.0290 gap — unequal counts isolated as the sole cause.

**5d.** The honest negative: on prose + telemetry (the intuitive "two registers"
corpus, where average-of-averages overweights short telemetry ~7.6×), the same bug
moves the mixed held-out loss by **+1.6% relative** (+0.042 on 2.668) vs the hazard
corpus's **+68%** — the model infers the register from context and fits both
conditionally. Two curves 1.6% apart look like one line on a training chart
(`plots/accum_negative.png`), which is how this bug shipped in every major framework
until 2024.

## 4. The grad norm, and the step where it spoke first (§6)

What is honestly claimable: the per-step train loss and the grad norm come from the
*same* forward on the *same* batch — when the anomalous batch arrives they spike
**together**, and any "the raw train loss lagged" claim is bookkeeping theater. What
actually lags is the **model**: the damage exists only in losses computed on clean data
*after* the update, and it persists (momentum re-applies the bad gradient for
~1/(1−β₁) steps). So three series are logged per arm per step: train loss, pre-clip
norm (`clip_grad_norm_`'s return value, `max_norm=inf` in unclipped arms — same
estimator everywhere), and a clean probe (full-stream sweep of held-out prose,
post-update, position stated).

A 300-step CPU run cannot wait for an organic incident, so one is planted and framed as
exactly that — instrument validation on a known event: a shard decoded with the wrong
encoding (bytes uniform over the byte range) at step k = 150, from a dedicated
generator (the clean stream is checksummed identical across arms; A and B are asserted
*bitwise* equal pre-k). SGD + momentum on purpose — the update is proportional to the
gradient, so the mechanism clipping guards is undistorted; the AdamW version is run too
and reported as the honest note (its per-coordinate normalization absorbs most of a
one-step spike: damage +0.146 vs SGD's +0.654).

| arm | intervention | probe damage | recovery |
|---|---|---|---|
| A placebo | none | detector silent | — |
| B incident, unclipped | inject at k | **+0.654** | 17 steps |
| C tight cap (2×median = 0.365), every step | inject at k | **+0.116** | — |
| C2 loose cap (1.25×max settled = 1.096), every step | inject at k | **+0.107** | — |
| D oracle: clip at k only (labeled counterfactual) | inject at k | +0.018 | — |
| E AdamW, unclipped | inject at k | +0.146 | — |

Pre-registered detectors (norm: > 5× trailing median after burn-in; probe: > trailing
median + 0.02 for 2 consecutive steps) localize the norm spike at **exactly 150** and
the probe damage at **150** (post-update), and stay **silent on the placebo** — both
audited. The norm read **16.3×** its settled median (3.07 vs 0.188) *before*
`optimizer.step()` was committed — the one moment the update could still have been
refused; the dashboard's EMA loss moved **+3.9%**, a wiggle, and even that wiggle is
the weird batch's own loss, not yet the model's damage. The realized clip factor at the
spike, cap 0.365 / norm 3.07 = **×0.119**, happens to be the session's own worked
number (8.4 → cap 1.0 → ×0.119, asserted separately).

Two caps on purpose, because *how often a cap binds on ordinary steps decides what it
is*: the tight cap binds on **110/110** settled steps — a gradient normalizer wearing a
guard's uniform (disclosed, and it *helps* here: C's pre-incident probe 2.879 beats
B's 3.043) — while the loose cap binds on **4/110** settled steps and still contains
the incident: the session's safety valve. Both also clip the init transient (step-0
gradients are every run's first anomaly). A gentler incident (corpus bytes with order
shuffled) is run too: norm ratio 1.07×, damage +0.007 — incidents come in sizes, and
the norm ranks them before any loss can. Figure: `plots/gradnorm_lead.png` (norm / 
dashboard view / clean probe, all arms).

## 5. MFU, computed honestly (§7)

**The numerator.** Exact matmul FLOPs for this architecture:
6·(12·L·d² + V·d) + 12·L·T·d = **2,951,424** per token (fwd+bwd), cross-checked
against `torch.profiler`'s own per-op 2mnk count on a real forward pass (984,448 vs
983,808 per token, 0.07%). Against it, four 6N conventions: 6·N_total **+0.34%** — but
*only by accidental cancellation* (13.4% of parameters — position table + LayerNorms —
do zero matmul FLOPs, offsetting the ignored attention term at this T; at T=32 the same
convention is **+11.5%** off, at T=512 **−28.3%**); 6·(N−tok_emb) −6.4%;
6·(N−both embeddings) −13.1%; 6·N_matmul −13.3%. Pick a convention silently and two
people "measuring MFU" on the same run disagree by a quarter.

**The measurement.** tokens/s = total processed tokens (B·T = 1,024/step; 1,016
scored — both printed) over the wall-clock of a post-warmup window: **39,074 tok/s**
(median step 25.5 ms, IQR 1.4 ms). Peak = best sustained fp32 GEMM this container will
do (n ∈ {1024, 2048, 4096}, best-of-repeats, preallocated out, measured before *and*
after the training window): **471 GFLOP/s**. Achieved 115.3 GFLOP/s →

**utilization vs attainable GEMM = 24.5%** (6N convention: 24.6%)

— named that way because the denominator *flatters*: a good GEMM reaches only ~70–90%
of silicon peak, so true MFU against the industry's theoretical-peak convention is
lower still; the bias direction is stated wherever the number appears. The session's
worked example is reproduced with every assumption pinned: 6 × 9e9 × 12,000 = 648
TFLOP/s over 8 × 989 TFLOP/s (H100 SXM **dense** bf16) = **8.1901% ≈ 8.2%** — and
quoting NVIDIA's 2:4-sparsity 1,979 figure instead would read 4.1%: denominator choice
is half of any MFU claim.

**The distance to 40%, measured rather than folklored.** The GPU explanations (kernel
launches, fusion) mostly do not apply on CPU; the op-level profile does: FLOP-counted
matmul ops (`aten::mm` 38.4%, `aten::bmm` 6.4%) own **45%** of step time — softmax,
LayerNorm, GELU, AdamW (4.3%) and glue own the rest — so 45% × (small-matmul
efficiency) bounds the utilization before a single FLOP is "wasted"; the measured
24.5% implies the matmuls themselves run at ~54% of the 4096² GEMM rate. And the
diagnosis is confirmed by turning one knob: same loop, same T, same B, d = 64 → 512
gives **11.4% → 24.5% → 32.5% → 50.2%**, straight into the session's healthy band. The
distance to 40% is not mystery overhead; it is matrices that are too small
(`plots/mfu.png`). The four session traces — loss, grad norm, tokens/s, MFU — are
logged per-step on one instrumented run (`plots/four_traces.png`); instrumentation
itself costs ~5% throughput (37,216 vs 39,074 tok/s), which is why the headline came
from the uninstrumented loop.

## 6. 0.1, written out by hand (§8)

0.1 in binary is infinite: `0.0001100110011…`, the pattern `1100` forever — every
format must cut it, and where it cuts is the story. The notebook derives each encoding
from first principles in exact rational arithmetic (`Fraction`), prints the by-hand
steps (expand → normalize to 1.10011…×2⁻⁴ → exponent field = bias−4 → round the
mantissa to nearest, **ties to even**), reconstructs the stored value exactly, and
matches **torch bit-for-bit**:

| format | bits (s\|exp\|mantissa) | stores | abs err | rel err | rounded |
|---|---|---|---|---|---|
| fp32 | `0\|01111011\|10011001100110011001101` | 0.100000001490116… | 1.49e-09 | **1.5e-08** | up |
| bf16 | `0\|01111011\|1001101` | 0.10009765625 | 9.77e-05 | **9.8e-04** | up |
| fp16 | `0\|01011\|1001100110` | 0.0999755859375 | 2.44e-05 | 2.4e-04 | **down** |
| fp8 E4M3 | `0\|0011\|101` | 0.1015625 | 1.56e-03 | **1.6e-02** | up |

Honesty details a grader would probe: `Fraction(0.1)` is *not* 1/10 (it is the fp64
impostor with a 2⁵⁵ denominator — asserted, and all errors measure against
`Fraction(1,10)`); 0.1 never hits a half-ULP tie in any format (the expansion repeats),
so four dedicated tie probes exercise round-to-nearest-**even** in both directions
(17/256 → 0.0625 down-to-even and 19/256 → 0.078125 up-to-even in E4M3; 1+1/256 → 1.0
and 1+3/256 → 1.015625 in bf16), all matching torch; E4M3 is the OCP variant, not IEEE
(no infinities, NaN = S.1111.111, and torch *saturates*: fp8(500.0) = 448); only fp16
rounds 0.1 *down* — its guard bit lands on a 0 of the pattern.

**The choice — a recommendation the demos support, not something this fp32 CPU
notebook trained end-to-end: bf16.** The measured cliff: fp16's smallest subnormal is
2⁻²⁴ ≈ 5.96e-8 and under round-to-nearest everything below 2⁻²⁵ ≈ 2.98e-8 becomes
*exactly zero* (measured at the boundary: 4e-8 → 5.96e-8, 3.0e-8 → 5.96e-8, 2.9e-8 →
0.0, 1e-8 → 0.0) — a weight whose gradient lives there stops learning, silently, and
precisely where the signal is faintest. Loss scaling ×1024 rescues *representability*,
not precision (1e-8 → 1.0252e-5, a subnormal carrying ~7 effective bits, rel err
1.2e-3), and the cliff only moves: 1e-11 still dies. bf16 keeps fp32's full exponent
(1e-8 → 1.0012e-8) and needs none of the machinery, at the price the table shows —
digits the training state does not rely on, because the fp32 master copy in §9's
16 bytes holds the precision. fp8 E4M3 at 1.6e-2 is the 2026 production recipe *with*
per-block scaling and higher-precision attention — the next step, not the safe default.
(Subnormal claims are CPU-measured; accelerators that flush subnormals make fp16's
cliff worse, not better.)

## 7. Sixteen bytes per weight, measured (§9)

The five residents of a mixed-precision step are materialized for the real model and
byte-counted: bf16 weights (2) + bf16 gradients (2) + fp32 master (4) + Adam's two
moments (8) = **7,897,088 bytes = exactly 16 × 493,568**. Extrapolated:
2B → **29.8 GiB**, 9B → **134.1**, 20B → **298.0**, 120B → **1,788.1** — and an 80 GB
accelerator holds the training state of a **~5.4B** model with nothing left for
activations (which is why activation checkpointing trades ~30% compute for memory,
and why Sessions 12–13 exist).

## 8. What the pre-build adversarial review changed

Five design reviewers (finite differences, accumulation, grad-norm, MFU, float bits)
attacked the plan *before implementation*, running their own measurements. Every
section above contains their fingerprints; the load-bearing changes:

1. **The original Task-3 corpus produced an invisible gap.** The intuitive
   two-register design measured a ~0.01-nat separation — a transformer infers the
   register and fits both conditionally. The hazard corpus (identical prefixes,
   analytic optima, belief probe) replaced it as the headline; the register corpus
   became §5d's honest negative instead of being silently discarded.
2. **"The train loss lagged the norm" was rejected as bookkeeping theater** — loss and
   norm spike together at k, by construction. The section was rebuilt around the
   defensible claims: pre-update actionability, clean-probe damage, persistence,
   placebo-validated detectors — and AdamW's spike absorption became a taught result
   instead of a silent optimizer swap.
3. **6N was exposed as accidental cancellation** (+0.34% at T=128 masking ±11–28%
   elsewhere); "expect low single digits of MFU on CPU" was deleted as empirically
   false *before* being committed (measured: 24.5%, rising to 50.2% at d=512), and the
   gap explanation was rebuilt from the op-level profile instead of GPU folklore.
4. **Two wrong numbers in the planned float table** (fp32's rel/abs confusion, fp16's
   error attributed to bf16) were corrected pre-commit; the fp16 flush-to-zero
   boundary was moved from the folklore 5.96e-8 to the correct 2.98e-8 *and measured*;
   `Fraction(0.1)`'s impostor status became a taught pitfall.
5. **Found during the build, kept because it is real:** the per-step-normalization
   ratio bias (§5c″, p* = 0.6681 analytic, measured) — the "correct" combine has its
   own subtle failure mode when the denominator varies with content, and this
   notebook's first version stepped on it exactly as a real training run would.

## 9. What's in the box

```
assignment-10/
├── training_loop.ipynb       # THE deliverable: all harness code inline, executed
├── run_demo.py               # execute top-to-bottom (nbclient) + audit -> run.log
├── audit.py                  # independent: re-derives every claim from disk
├── requirements.txt
├── tests/                    # 36 tests; conftest exec's the notebook's export cells,
│   ├── conftest.py           #   so tests share the notebook's code — nothing retyped
│   ├── test_loss_and_accumulation.py
│   ├── test_gradient_check.py
│   ├── test_gradnorm_and_mfu.py
│   ├── test_float_bits.py
│   └── test_artifacts_and_audit.py
└── submission_artifacts/
    ├── results.json          # every number in this README, machine-checked
    ├── curves.json           # per-step series: all arms, all experiments
    ├── run_config.json  ·  run.log  ·  plots/*.png (6 figures)
```

## 10. Reproduce

`python run_demo.py` re-executes the notebook headlessly and re-audits (~3.5 min CPU;
the committed run: 210 s on a 4-thread container). `--fast` shrinks every budget
(training-outcome assertions downgrade to loud warnings; the audit is calibrated for
full budgets). `--verify-only` audits the committed artifacts in ~5 s. In Colab: open
the badge, `Runtime → Run all`. Every *deterministic* number reproduces bit-identically
on CPU at seed 1337; the timing numbers (tokens/s, GFLOP/s, MFU) are machine-specific
by nature, quoted from the committed container run, and audited for arithmetic
consistency only.

## 11. Limitations, honestly

1. **Toy scale.** 493,568 parameters, byte-level V = 259, KB-scale corpora. Every
   mechanism measured (combine rules, ratio bias, norm-lead anatomy, FLOP accounting,
   float behavior, bytes/weight) is scale-independent arithmetic or scale-independent
   semantics; no claim is about model quality.
2. **The incident is planted.** §6 validates instruments on a known event (and says
   so); at-scale runs show the phenomenon organically over thousands of steps, this
   demonstration compresses it to one. The placebo arm guards the detectors against
   firing on noise.
3. **"MFU" here divides by attainable GEMM**, not theoretical silicon peak — flattering
   by ~1.1–1.4×, stated at every use. On a shared container the theoretical peak is
   unknowable (turbo, quota); the committed numbers are this container's.
4. **Single seed for §6, three for §5c.** The audited claims are orderings, identities
   and detector verdicts with margins, not exact loss values; determinism at seed 1337
   is pinned by test.
5. **bf16-vs-fp8 is a recommendation** supported by measured demos, not an end-to-end
   mixed-precision training comparison — that is Session 11+ territory (and V5's own
   open question).
6. **Notebook markdown numbers** are prose, hand-synced to the committed run; only
   this README and `results.json` are machine-cross-checked.
