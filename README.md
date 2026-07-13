# ERA-V5

My assignments for the ERA-V5 course (building an LLM from scratch). All served
under `/era-v5/`:

| Assignment | Live | What it is |
|---|---|---|
| **1** | [/era-v5/](https://www.pandala.in/era-v5/) | Interactive, in-browser proofs of why activations, depth, embeddings and data matter. |
| **2** | [/era-v5/tokenizer/](https://www.pandala.in/era-v5/tokenizer/) | A shared 10k-token tokenizer for the wiki-faithful Markdown India pages (en/hi/te/mr) — grader-compatible HuggingFace format, live fertility ratios, self-score 10,817, downloadable. |
| **3** | [/era-v5/data-collection/](https://www.pandala.in/era-v5/data-collection/) | Design brief for a 40B India-first coding & agentic model: data, cleaning, evaluation, and a fertility-derived 262K tokenizer. |

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
**wiki-faithful Markdown** India pages in English, Hindi, Telugu and Marathi — the exact
corpus and scoring pipeline of the published course reference. `tokenizer.json` loads with
`tokenizers.Tokenizer.from_file`, `decode(encode(x))` preserves all visible text, and the
instructor's published evaluator reproduces our numbers drop-in.

🔗 **Live:** https://www.pandala.in/era-v5/tokenizer/

| Language | Fertility (tokens / faithful units) |
|---|---:|
| English | 0.677459 |
| Hindi | 0.585011 |
| Telugu | 0.586713 |
| Marathi | 0.663341 |

`spread = 0.092448` → **self-score = 10,816.91** (reference solution: 6,502.56);
Hindi penalty ×1.0; the instructor's published evaluator (hardcoded en/hi/te/mai) also runs
unchanged on these artifacts and scores 10,816.91. Full method, reproduce commands, and parity proof: [`tokenizer/README.md`](tokenizer/README.md).

```bash
npm run parity                        # JS == HuggingFace tokenizers (ids, decode, units)
cd tokenizer && pip install -r requirements.txt && python evaluate_tokenizer.py
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
