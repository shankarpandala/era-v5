# Assignment 13 — grader card

Four executed notebooks, run in order; each embeds the whole implementation so **Run all** on a
Colab GPU needs no clone and no installs. Defaults are the assignment's full scale (50M tokens per
arm); the committed outputs are the same protocol executed on a 4-core CPU at 5,005,312 tokens
per arm (README §1 says why, and nothing mixes the two budgets).

| Requirement | Where to look |
|---|---|
| 20M LLM, fixed batch you can run | [`01_baseline_fixed_batch.ipynb`](01_baseline_fixed_batch.ipynb): 20,989,440 parameters, batch 32 × 256; val loss **2.6964**, **1,869 tok/s**, peak RSS **4.15 GiB**, 2,579 MiB of activations kept per step. |
| Train again with reversibility; which variant worked | [`02_reversible_fixed_batch.ipynb`](02_reversible_fixed_batch.ipynb): inverse exact to 3–8e-7 and gradients equal to stored-activation autograd to 1e-6 for `euler`, `midpoint`, `momentum`; exact activation bytes flat in depth (70 MiB at 2…12 layers vs 118→373 MiB); four-arm screening → **`momentum` (γ = 0.9)** wins (4.057 vs 4.544 / 4.752 / 4.894 at 1M tokens, none diverged); run 2 at the baseline's batch: val loss **2.6332**, **1,345 tok/s**, peak RSS **2.67 GiB**, 560 MiB kept. |
| Reversibility at the maximum batch | [`03_reversible_max_batch.ipynb`](03_reversible_max_batch.ipynb): search by real training steps within 75% of free RAM → baseline **96**, reversible **312** (3.25×); run 3 at batch 312: val loss **4.9749** after 63 steps, **986 tok/s**, peak RSS **10.95 GiB**. |
| Final loss, speed, memory, findings | [`04_report.ipynb`](04_report.ipynb) and README §1, §5–§7; every quoted number is re-derived from `submission_artifacts/` by [`audit.py`](audit.py). |
| Detailed README, repo with the notebooks | [`README.md`](README.md); [shankarpandala/era-v5](https://github.com/shankarpandala/era-v5) `assignment-13/`. |

```bash
cd assignment-13
python -m pip install -r requirements.txt
python run_demo.py --verify-only   # read-only audit of the committed evidence (PASS)
python -m pytest tests -q          # 29 tests
python run_demo.py                 # re-execute the four notebooks at the pilot budget, then audit
python run_demo.py --tokens 50000000 --screen-tokens 5000000   # full scale (GPU advised)
```

**Interpretation.** Real fp32 training on TinyStories with an 8,192-token BPE; a custom
`autograd.Function` that reconstructs activations in backward (parameter count identical to the
baseline); tokens/s measured inside optimiser steps; peak memory is the CUDA allocator on a GPU and
sampled RSS on CPU; the maximum batch is found by real steps (OOM bisection on CUDA, RSS-gated
doubling verified by a real step on CPU). Reversible training cost 28% throughput at the same
batch, saved 36% of peak RSS (78% of what autograd keeps) and no loss; the 3.25× batch it enables
costs 2.3 nats at a fixed 5M-token budget because 63 steps are too few.
