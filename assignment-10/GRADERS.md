# Assignment 10 — one-page grader card

**The deliverable:** [`training_loop.ipynb`](training_loop.ipynb) — one notebook, one
instrumented training loop, committed fully executed; runs top to bottom on CPU with
zero downloads ([open in Colab](https://colab.research.google.com/github/shankarpandala/era-v5/blob/main/assignment-10/training_loop.ipynb)).
Five adversarial design reviews ran *before* implementation; their findings are built
in and listed in README §8.

**The six tasks, each measured (every number re-derived from disk by `audit.py` — 63
checks, verdict PASS):**

| # | task | the number |
|---|---|---|
| 1 | every tensor shape, one line per dimension | 31 tensors traced through one full step; every grad shape == weight shape; Adam holds 2×493,568 state values; all relations audited |
| 2 | one gradient by hand | toy chain: **64** by chain rule, central nudge and autograd (forward nudge = session's 16.064, its 0.064 excess proven = 64·ε); real model: fd vs `backward()` rel err **4.2e-10** (~10.6 decimals); directional derivative certifies all 493,568 grads at once (1.3e-8); fp32 probe drowns (5 exact-zero ε's) while fp32 backward stays good (3.9e-7) — the stick, not the gradient |
| 3 | break accumulation on purpose | session mirror **2.6000** vs **3.0000** (15.4%, per-token losses committed); trained arms on the hazard corpus: **0.0424** vs **0.0714** held-out (analytic floor 0.0385 / asymptote 0.0714 — the buggy arm parks ON its optimum), 3 seeds, both curves plotted with analytic lines; belief probe 0.499 vs **0.916** (truth 0.5); doc-weighted metric FLIPS (buggy wins 0.051 vs 0.137): a substituted objective, not noise; equal-count control gap **0.00000**; honest negative: on prose+telemetry the same bug moves eval only +1.6% relative |
| 4 | grad norm moved before the loss | spike **16.3×** settled median at step 150, known BEFORE the update; clean-probe damage **+0.654** (17-step recovery) exists only after; placebo-validated detectors localize both, audited; tight cap (2×median, from data) contains to +0.116 but binds 110/110 settled steps (a normalizer — disclosed); loose cap (1.25×max) binds 4/110 and contains to +0.107; realized clip factor **×0.119** = the session's own number; AdamW absorbs to +0.146 (taught, not hidden) |
| 5 | my own MFU, honestly | **24.5%** of measured attainable GEMM (471 GFLOP/s; denominator flatters vs theoretical peak — stated); exact FLOPs/token 2,951,424 with profiler cross-check (0.07%); 6N is +0.34% *by accidental cancellation* (+11.5% at T=32, −28.3% at T=512); distance to 40% measured: matmuls own 45% of step time, and d 64→512 lifts the same loop 11.4%→**50.2%** — matrices too small, not mystery overhead; session example reproduced: 648/7,912 = **8.1901%** |
| 6 | 0.1 in bits, by hand | fp32 `0\|01111011\|10011001100110011001101` (rel 1.5e-8) · bf16 `0\|01111011\|1001101` (9.8e-4) · fp8 E4M3 `0\|0011\|101` (1.6e-2) · fp16 for contrast (rounds DOWN, 2.4e-4) — exact-Fraction encoder, bit-for-bit vs torch, ties-to-even proven on dedicated probes (0.1 never ties); choice: **bf16** — fp16 zeroes every gradient below 2.98e-8 (boundary measured: 3.0e-8 lives, 2.9e-8 dies), loss scaling ×1024 only moves the cliff to ~3e-11 |
| + | 16 bytes/weight | measured exactly: 7,897,088 bytes = 16 × 493,568; 2B→29.8 GiB … 120B→1,788 GiB; 80 GB card ⇒ ~5.4B ceiling |

**A finding of its own (§5c″):** normalizing the loss by *this step's* token count is a
ratio bias when composition varies — analytic p* = **0.6681** (vs truth 0.5), measured
0.638/0.740. "Normalize by tokens" sharpened: divide by a constant, not by what this
batch happened to contain.

**Honest negatives, reported:** the incident is planted (instrument validation, placebo
arm guards the detectors); "MFU" divides by attainable GEMM, flattering vs the industry
convention; bf16 is a recommendation from measured demos, not an end-to-end
mixed-precision run; toy scale throughout — the claims are mechanisms and arithmetic,
not model quality.

**One figure:** `submission_artifacts/plots/gradnorm_lead.png`

**Re-run:**

```bash
cd assignment-10 && pip install -r requirements.txt
python run_demo.py --verify-only   # ~5 s: audit the committed artifacts (63 checks)
python run_demo.py                 # ~3.5 min CPU: full re-run of everything
python -m pytest tests -q          # 36 tests, exec'd from the notebook's own cells
```

Every deterministic number is audited verbatim from `submission_artifacts/results.json`;
timings are audited for arithmetic consistency (the Session-9 convention). Audit
verdict: **PASS**.
