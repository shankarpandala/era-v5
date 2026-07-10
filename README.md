# ERA-V5

My assignments for the ERA-V5 course (building an LLM from scratch). All served
under `/era-v5/`:

| Assignment | Live | What it is |
|---|---|---|
| **1** | [/era-v5/](https://www.pandala.in/era-v5/) | Interactive, in-browser proofs of why activations, depth, embeddings and data matter. |
| **2** | [/era-v5/tokenizer/](https://www.pandala.in/era-v5/tokenizer/) | A from-scratch 10k-vocab multilingual BPE tokenizer with live fertility ratios, self-score, and a downloadable tokenizer. |

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

## Assignment 2 — Multilingual BPE Tokenizer

A single **byte-level BPE** tokenizer, built from scratch, with a shared vocabulary of
**10,000 tokens** for **English, Hindi, Telugu, and Marathi** (India's Wikipedia page in
each). For each language, fertility `X = tokens / words` must be `≤ 1.2`; the score is
`1000 / (X_max − X_min)`.

🔗 **Live:** https://www.pandala.in/era-v5/tokenizer/

**Result** (equal 2,100-word slice per language, equal training weights):

| Language | X = tokens/word | ≤ 1.2 |
|---|---:|:--:|
| English | 1.1367 | ✓ |
| Hindi | 1.0567 | ✓ |
| Telugu | 1.1757 | ✓ |
| Marathi | 1.1124 | ✓ |

`spread = 0.1190` → **self-score ≈ 8,400**, all four ≤ 1.2.

- **`tokenizer/`** — the from-scratch Python pipeline (trainer, encoder, evaluator). See
  [`tokenizer/README.md`](tokenizer/README.md) for the method and exact reproduce commands.
- **`src/tokenizer/`** — the React widget: live per-language ratios, self-score, a paste-and-tokenize
  playground, and tokenizer download. It re-implements the encoder in JS and **recomputes every
  number live in the browser** from the shipped tokenizer + corpora — nothing is hardcoded.
- **Trust:** `npm run parity` proves the JS encoder produces the *identical* token stream to the
  Python reference on every corpus, so the widget's numbers equal the numbers a grader gets
  running our Python tokenizer.

```bash
npm run parity                       # JS == Python token streams (uses committed artifacts)
cd tokenizer && pip install -r requirements.txt && python evaluate.py   # reproduce the table
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
