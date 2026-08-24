# Assignment 9 — one-page grader card

**The deliverable:** [`loss_harness.ipynb`](loss_harness.ipynb) — one notebook, one loss
harness, committed fully executed; runs top to bottom on CPU or GPU with zero installs,
zero downloads ([open in Colab](https://colab.research.google.com/github/shankarpandala/era-v5/blob/main/assignment-9/loss_harness.ipynb)).
This is v2: adversarially reviewed, and every review finding that survived verification
is fixed and regression-audited (README §8 lists them).

**Claim A (the seven numbers, each re-derived from disk by `audit.py`):**

| # | requirement | the number |
|---|---|---|
| 1 | shapes, every dimension named | `logits (8, 128, 259)` + 7 more tensors, one line each |
| 2 | shift verified in **strings** | all 3 alignments audited pre-training; correct rate 1.0000 exactly |
| 3 | padding masked, count changes | 508 → 186 slots; 10.3600 → 2.2887; wrong-mean bug reads 0.8380 (gradient ×0.366); SFT variant 64 → 16 |
| 4 | two docs packed, boundary masked | 2.5564 → 2.5363; seam anatomy 0.3934 (learnable) vs 6.8680 (impossible); isolation ablation ends in doc2 ≡ doc2-alone |
| 5 | untrained PPL ≈ V | 261.3 vs V = 259 — and the check catches a real leak: tied no-shift scores PPL 143.5 **at init** (untied: 265.0) |
| 6 | tied vs untied params | 495,488 vs 528,640 — Δ exactly V·d = 33,152 |
| 7 | peak memory, three CE implementations | 2,378.9 / 181.8 (row-chunked, 13.1×) / 170.3 MiB (online-softmax, 14.0×) at V = 50,257; all gradient-equivalent |

**Part 2 (both heads untied — the tied/untied confound was caught in review):** full-stream
sweeps, step 0 logged pre-update. Train L1 2.1711 / L2 2.4816 / sum 4.6526; held-out
L1 2.5650 / L2 2.9629 / sum 5.5278. The gap starts at zero, opens with training, keeps
growing held-out (+0.40) and plateaus lower on the memorizable train split (+0.31):
conditional entropy, measured, with the asymmetry audited.

**Claim B/C (the warning, quantified, and the circuits named):** identical budgets;
reversed ends at 0.0102 and no-shift at 0.0000 vs 2.3793 correct — the bugs *outshine*
the truth on the curve. The copy-accuracy instrument names each circuit from behavior:
reversed → copy-prev 0.999 (an attention circuit, formed after ~140 steps of tracking
the correct arm); no-shift → copy-self 1.000 within ~20 steps, because its wire exists
at init (the row-5 leak); correct → next-token 0.288 with copies at noise. Generation
prints the `<bos> <bos> …` runs as tokens: train/inference mismatch made audible.

**Honest negatives, reported:** toy scale, single seed, memorizable corpus; CPU memory
process-level (macOS fallback coarser, CUDA exact, analytic column is the portable
claim); the entropy-gap magnitude is a two-head estimate, direction is the claim.

**One figure:** `submission_artifacts/plots/copy_circuits.png`

**Re-run:**

```bash
cd assignment-9 && pip install -r requirements.txt
python run_demo.py --verify-only   # ~5 s: audit the committed artifacts
python run_demo.py                 # ~2 min CPU: full re-run of everything
python -m pytest tests -q          # 39 tests, exec'd from the notebook's own cells
```

Every number in README.md is quoted verbatim from `submission_artifacts/results.json`
(the audit fails on drift); `audit.py` re-derives each claim from committed per-token
losses and curves without executing the notebook. Audit verdict: **PASS**.
