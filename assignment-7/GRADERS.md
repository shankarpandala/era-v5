# Assignment 7 — one-page grader card

**Problem chosen:** #1 — *embeddings that store mathematical structure ("9 + 9
in embedding space is 18")*.

**Claim A (algebra, zero training):** a deterministic 128-d embedding whose
value dim makes vector addition literally integer addition —
`emb("9") + emb("9")` carries **exactly 18**, bit-exact float32 `==` — with
subtraction and negatives (`emb(9) − emb(4)` → 5, SIGN dim), multiplication
and division on a log dim (× → +, ÷ → −, max err ~1e-6), multi-step chains
via analytic decode→re-encode (`(9+9)×2 = 36`), and full invertibility
(spelling + value decode back exactly). **10/10 properties verified over
10,000 random pairs** and re-derived by an independent audit at a fresh PRNG
coordinate.

**Claim B (training, CPU-only):** across seven arms that differ **only in the
embedding** (identical architecture, optimizer, and byte-identical batch
stream — enforced by hashing), a 2-layer transformer with the deterministic
embedding beats a learned embedding table:

| what (train size 2000, mean over 5 seeds) | kron_v2 | learned | control |
|---|---|---|---|
| unseen number tokens (hole 40–59), exact add | **0.454** | 0.013 (**36×**) | frozen-random table: 0.001 (**377×**) |
| natural-language templates, hole add (3 seeds) | **0.518** | 0.012 (**43×**) | — |
| in-range exact add | **0.714** | 0.095 (**7.5×**) | — |
| subtraction (negative answers), hole exact | **0.494** | 0.017 | — |

(Full tables with std: [README §3](README.md#3-results) and
[`submission_artifacts/results.json`](submission_artifacts/results.json).)

**Honest negatives, reported:** magnitude extrapolation (operands 100–999)
fails for *every* arm; per-layer ridge probes locate where linearly decodable
structure is lost (the trunk, not the embedding). The FoNE-style readout
ablation matches the full scheme in trained models — the homomorphic dims'
unique value is the exact algebra and the invertibility, which no readout
dim can provide.

**One figure:** `submission_artifacts/plots/hole_generalization.png`
**Interactive report:** open `submission_artifacts/report.html` — includes a
live widget where you type any two numbers and watch the value dims add
bit-exactly.

**Re-run:**

```bash
cd assignment-7 && pip install -r requirements.txt
python run_demo.py --verify-only   # ~5 s: Claim A + full audit, no training
python run_demo.py --fast          # ~5 min: reduced matrix, same pipeline
python run_demo.py                 # ~60 min: the full 92-run matrix
python -m pytest tests/ -q         # 53 invariant tests
```

Every `[PASS]` in `submission_artifacts/run.log` is a derived result;
`kronembed/audit.py` re-derives every claim from disk, sharing no state with
the code that produced it. Audit verdict: **PASS**.
