# ERA-V5 — Assignment 1: Interactive Proofs

Interactive, in-browser proofs for the four claims of ERA-V5 Assignment 1. Every
model trains **live in your browser** on a tiny hand-written neural network
(no TensorFlow.js, no precomputed results) — turn the knobs and watch each claim
hold or break.

🔗 **Live:** https://shankarpandala.github.io/era-v5/

## The four claims

| # | Claim | Money shot |
|---|-------|-----------|
| **S1-1** | Activations exist for a reason | Linear model is stuck at ~55% with a straight cut on two concentric rings; one ReLU hidden layer wraps the ring to ~99%. Plus a 3D feature-lift showing the rings becoming plane-separable. |
| **S1-2** | Depth without nonlinearity is a lie | 1 linear layer and 5 stacked linear layers draw the *same* line; the five weight matrices multiply out to a single 2×1 matrix. ReLU between the same layers breaks the tie. |
| **S1-3** | Embeddings learn similarity from next-token | Trained only to predict the next token in a toy grammar, the embedding table sorts itself into animal / fruit / verb clusters (PCA → 2D). Every nearest neighbour is same-category. |
| **S1-4** | Memorization vs generalization | An over-parameterized net memorizes 20 noisy points (train→0, test high), but the generalization gap collapses as the dataset grows to 2000. Data closes the gap. |

## Tech

- **React + Vite + Tailwind v4** — single page, all four demos.
- **`src/lib/nn.js`** — a small, dependency-free MLP with hand-written backprop (Dense, ReLU, sigmoid/softmax heads, Adam & SGD).
- **`src/lib/rng.js`** — seeded PRNG so every run is reproducible (seed is a knob).
- **react-three-fiber** — the S1-1 3D feature-lift (lazy-loaded).
- Decision boundaries are drawn on raw `<canvas>` by pushing a pixel grid through the model in one batched forward pass.

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
