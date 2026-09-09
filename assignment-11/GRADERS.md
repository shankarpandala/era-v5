# Assignment 11 — one-page grader card

**The deliverable:** [`optimizers.ipynb`](optimizers.ipynb) — one notebook, all six tasks
measured, committed fully executed on this machine's CPU (float64 checks) and GPU (Apple
MPS, every sweep); runs top to bottom with no downloads locally and one sha256-verified
1.3 MB fetch of the repo's own corpus in Colab
([open in Colab](https://colab.research.google.com/github/shankarpandala/era-v5/blob/main/assignment-11/optimizers.ipynb)).
Seven adversarial design reviews ran *before* implementation; README §8 lists what they overturned.

**The six tasks, each measured (every number below is re-derived from disk by `audit.py` — 122 checks, verdict PASS with one acknowledged pre-registered failure, README §3):**

| # | task | the number |
|---|---|---|
| 1 | Adam by hand, checked against PyTorch | five gradients on one weight; m, v, m̂, v̂, u, w agree with `torch.optim.Adam` to **3 × 10⁻¹⁶** (fp64) and **6 decimals** (fp32), checked non-circularly (torch never stores m̂/v̂: the realized step and a β₁ = 0 run read them off); step 1 = ±η to 3 × 10⁻⁸ and would be **3.162 η** uncorrected; scale ladder shows invariance breaking at \|g\| ~ ε (\|u₁\| = \|g\|/(\|g\|+ε)); Keras ε̂ vs optax `eps_root` vs torch; **L2-in-Adam delivers 2.5× MORE decay than AdamW** at equal λ (√v̂ = 0.20 < 1 amplifies λw), halving ‖W‖ in 20 steps on the real model |
| 2 | bias correction off, first 20 steps both ways, when it stops mattering | same-state ratio = r(t) = (1−β₁ᵗ)/√(1−β₂ᵗ) to 1.2 × 10⁻²: **6.57× at t=12, within 10% only after step 1,751** (β₂ = 0.999) vs **0.45× at t=1, within 10% from step 6** (β₂ = 0.95; crossover β₂ = 0.99); trained: at 0.999 the paired held-out gap is **+0.67 at step 600**, plateaued at 0.65–0.68 from step 200, ‖θ‖ doubled — never stops mattering (a 30-step "hot" counterfactual reproduces it: +0.002 paired); at 0.95 with warmup the difference stops mattering at **step 50** (\|gap\| < 0.02 at every later eval, sign mixed across seeds), and *without* warmup the uncorrected arm is behind for six steps and then **ahead** from step 7 (by 0.14–0.34 from step 15: its r(t) < 1 start is a warmup) |
| 3 | update-to-weight ratio of every layer; the step warmup stops changing it | ρ of all 21 tensors every step, decomposed ρ = η · coherence / scale (ρ(1)·σ/η(1) = 0.9945…1.0050 asserted); **the step is W (100 for the W = 100 arm)** — S_η ≈ 1/W (0.0105 measured at W = 100) ends there by construction — and here the kink is visible in the block matrices (S_ρ/S_η seed means 1.48 / 1.64 / 2.27; 7/8 above 0.5 in every seed) with per-tensor peaks at 102/106/103 (W = 100, 3 seeds), 223 (W = 200); a counterfactual at width 128, η = 10⁻³ shows the cancellation regime (0.09) — the decomposition transfers, the raw ratio does not; the pre-registered global-ratio detector fails at W = 100 (177–241) and the per-tensor one misses its 25% band once (+26%): both reported as failures |
| 4 | cosine vs WSD, 300 planned, stopped at 200 | **cosine 1.5983 vs WSD 1.6666** (3 paired seeds, cosine lower by 0.068 in each); **keep cosine** — 0.024 better after equal 30-step cooldowns, 0.008 after 60; the WSD trunk's extension to 400 ≡ a planned-400 run and beats the re-warmed cosine by 0.0095 (unanimous), so the answer flips only if the horizon may grow; the step-200 sign flips with peak η (+0.033 at η/4 → −0.133 at 4η): a regime rule, not a property; each schedule tuned to its own best peak at the stop (cosine 2⁻⁹, WSD 2⁻¹⁰) cosine still leads by 0.054 (at η/2 itself the gap is 0.017) |
| 5 | lr sweep at 256/512/1024, the three minima, the value at 4,096 | vertices 2⁻⁸·⁸⁰ / 2⁻⁹·⁸⁶ / 2⁻¹⁰·⁴¹ (2.2e-3 / 1.1e-3 / 7.3e-4), three seeds + half-octave points, 71 runs, nothing diverged; slope **−0.79 ± 0.08** (CI [−0.99, −0.60]: between the −1 and −½ exponents and excluding both — partial alignment); 4,096 forecast **2.31e-4, ×1.75 band [1.3, 4.1]e-4** (3 widths); pre-registered 2048 test at half-octave resolution (−11.29 ± 0.68 → measured -11.32) **holds**; 4-width refit 2.28e-4 (×1.57); **use ≈ 2e-4, inside [1.8, 2.3]e-4** (the low-side rule rests on the 1024 asymmetry ÷2 +4.4% / ×2 +9.9%, which reversed at 2048: +7.7% / +3.5%); optimum moves −0.28 octaves per horizon doubling |
| 6 | tune both sides | own sweeps + 3 fresh seeds: **AdamW 1.4774, SGD 1.5906, Lion 1.5505** (AdamW wins by 0.113 / 0.073, pooled range 0.032); naive A (challenger at AdamW's η) exaggerates the deficits 12× / 6×; naive B (tuned challenger vs AdamW at the 3e-4 default) manufactures **"Lion beats Adam by 0.23, SGD by 0.19"** — the requirement's direction; clip 1.0 binds on 46–60% of steps; removing it at the clipped optima costs AdamW +0.035, Lion +0.119, SGD +1.38 (same seed), and SGD re-tuned without the clip reaches 1.7950 at 2^-4.5 (vs 1.5988 clipped, same seed: the clip is worth +0.196); decay matched as η·λ (λ = 0.1 as a decoupled shrink in SGD's units: +0.26); `fair_compare()` refuses on edge / config / noise (exercised, audited) |

**Honest negatives, reported:** at β₂ = 0.999 the bias-correction gap never closes inside the run (the plan expected a merge); the pre-registered blind detector on the *global* update ratio fails at W = 100 (177–241) and the per-tensor one misses its 25% band once (+26%); the warmup kink's visibility in hidden matrices is configuration-dependent (1.48 / 1.64 / 2.27 across seeds here, 0.09 at width 128, η = 10⁻³); tuned Lion does not match AdamW (the plan's hypothesis); SGD vs Lion, expected to be a refusal, is a verdict (Lion by 0.040); the 4,096 value is an extrapolation of two doublings, tested at one; "GPU" means Apple MPS, no CUDA run was made.

**One figure:** `submission_artifacts/plots/lr_sweep.png`

**Re-run:**

```bash
cd assignment-11 && pip install -r requirements.txt
python run_demo.py --verify-only   # ~2 s: audit the committed artifacts (122 checks)
python run_demo.py                 # full re-run (CPU + MPS/CUDA), --resume reuses cached sweep runs
python -m pytest tests -q          # 27 tests, exec'd from the notebook's own cells
```

Every deterministic number is audited from `submission_artifacts/results.json`; training
outcomes are audited as orderings with margins above the measured replicate and seed
spreads. Audit verdict: **PASS**.
