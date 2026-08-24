# Assignment 9 — one-page grader card

**The deliverable:** [`loss_harness.ipynb`](loss_harness.ipynb) — one notebook, one loss
harness, committed fully executed; runs top to bottom on CPU or GPU with zero installs,
zero downloads ([open in Colab](https://colab.research.google.com/github/shankarpandala/era-v5/blob/main/assignment-9/loss_harness.ipynb)).

**Claim A (the seven numbers, each re-derived from disk by `audit.py`):**

| # | requirement | the number |
|---|---|---|
| 1 | shapes, every dimension named | `logits (8, 128, 259)` + 7 more tensors, one line each |
| 2 | shift verified in **strings** | correct(+1) match rate 1.0000; `targets == tokens[:, 1:]` exact |
| 3 | padding masked, count changes | 508 slots → 186 contributing; loss 10.4337 → 2.3146 |
| 4 | two docs packed, boundary masked | loss 2.5550 → 2.5357; seam per-token 0.15 / 7.03 |
| 5 | untrained PPL ≈ V | 262.2 vs V = 259 (CE 5.5692 vs ln V 5.5568); std=1.0 counterexample: CE 33.1 |
| 6 | tied vs untied params | 495,488 vs 528,640 — Δ exactly V·d = 33,152 |
| 7 | peak memory, plain vs chunked CE | 2,378.9 MiB vs 180.8 MiB = **13.2×** at V = 50,257 |

**Part 2:** t+2 head on the same trunk — held-out L1 2.5924, L2 2.9745, sum 5.5668. The
gap starts at zero (both heads ignorant), opens with training, and keeps growing held-out
(final +0.38 nats) while plateauing near +0.29 on the memorizable train split — that
train-vs-held asymmetry is the memorization signature: conditional entropy, measured.

**Claim B (the warning, quantified):** under identical budgets both classic shift bugs
produce *lower, smoother* curves than the correct objective — reversed 0.0045 and
no-shift 0.0000 vs correct 2.3386 — while emitting `<bos> <bos> <bos> …` forever (4-gram
rate 0.000 vs 0.284). The string audit names each bug from the data alone: CORRECT /
REVERSED / IDENTITY at match rate 1.0000 each. Loss ordering and quality ordering are
anti-correlated; a beautiful curve is not evidence.

**Honest negatives, reported:** CPU memory numbers are process-level (CUDA path is
byte-exact; ratio's order of magnitude robust, second digit not); the corpus is
deliberately memorizable, so train-split numbers describe memorization; training curves
are single-seed (invariants are audited, loss values would wobble).

**One figure:** `submission_artifacts/plots/bug_zoo_curves.png`

**Re-run:**

```bash
cd assignment-9 && pip install -r requirements.txt
python run_demo.py --verify-only   # ~5 s: audit the committed artifacts
python run_demo.py                 # ~2 min CPU: full re-run of everything
python -m pytest tests -q          # 30 tests, exec'd from the notebook's own cells
```

Every number in README.md is quoted verbatim from `submission_artifacts/results.json`
(the audit fails on drift); `audit.py` re-derives each claim from committed per-token
losses without executing the notebook. Audit verdict: **PASS**.
