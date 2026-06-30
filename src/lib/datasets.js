// Synthetic datasets for the four claims. Everything is driven by a seeded RNG
// (see lib/rng.js) so a given seed reproduces the exact same points.

import { mat } from './nn.js'
import { makeRng } from './rng.js'

function makeSeed(s) {
  return makeRng(s >>> 0 || 1)
}

// Two concentric noisy rings — inner = class 0, outer = class 1.
// Not linearly separable: the whole point of S1-1 and S1-2.
export function makeRings(n, noise, rng) {
  const points = []
  const X = mat(n, 2)
  const y = new Int32Array(n)
  const rInner = 1.0
  const rOuter = 2.3
  for (let i = 0; i < n; i++) {
    const label = i % 2 // alternate so both classes are balanced
    const base = label === 0 ? rInner : rOuter
    const r = base + rng.gaussian(0, noise)
    const theta = rng.uniform(0, 2 * Math.PI)
    const x0 = r * Math.cos(theta)
    const x1 = r * Math.sin(theta)
    X.d[i * 2] = x0
    X.d[i * 2 + 1] = x1
    y[i] = label
    points.push({ x: x0, y: x1, label })
  }
  return { X, y, points, domain: { min: -3.4, max: 3.4 } }
}

// A learnable but noisy binary classification with a curved boundary, used for
// the memorization-vs-generalization demo (S1-4). The true boundary is a sine
// wave; a fraction of labels are flipped so a high-capacity net can memorize
// noise on small samples but not on large ones.
function sampleClassification(count, noiseFlip, rng) {
  const points = []
  const X = mat(count, 2)
  const y = new Int32Array(count)
  for (let i = 0; i < count; i++) {
    const x0 = rng.uniform(-3, 3)
    const x1 = rng.uniform(-3, 3)
    let label = x1 > 1.1 * Math.sin(1.3 * x0) ? 1 : 0
    if (rng.next() < noiseFlip) label = 1 - label // label noise
    X.d[i * 2] = x0
    X.d[i * 2 + 1] = x1
    y[i] = label
    points.push({ x: x0, y: x1, label })
  }
  return { X, y, points, domain: { min: -3.3, max: 3.3 } }
}

// Build the S1-4 experiment: one shared held-out test set plus a train set of
// the requested size, all drawn i.i.d. from the same distribution.
export function makeGapExperiment(trainSize, seed, { noiseFlip = 0.18, testSize = 1500 } = {}) {
  // Separate RNGs keep the test set identical across train sizes while the
  // train draws differ — exactly the comparison the gap plot needs.
  const testRng = makeSeed(seed * 7 + 101)
  const trainRng = makeSeed(seed * 13 + trainSize)
  return {
    train: sampleClassification(trainSize, noiseFlip, trainRng),
    test: sampleClassification(testSize, noiseFlip, testRng),
  }
}

// --- toy grammar (S1-3) ---------------------------------------------------

export const GRAMMAR_TOKENS = [
  { token: 'cat', cat: 'animal' },
  { token: 'dog', cat: 'animal' },
  { token: 'cow', cat: 'animal' },
  { token: 'apple', cat: 'fruit' },
  { token: 'mango', cat: 'fruit' },
  { token: 'eat', cat: 'verb' },
  { token: 'chase', cat: 'verb' },
  { token: 'see', cat: 'verb' },
]

export const CATEGORY_COLOR = {
  animal: '#3b82f6', // blue
  fruit: '#10b981', // emerald
  verb: '#f59e0b', // amber
}

// Generate next-token training pairs from a category-level Markov chain:
//   animal → verb → (fruit | animal),  fruit → animal.
// Because the next-token distribution depends ONLY on the current token's
// category, all tokens in a category must produce the same output distribution
// — so the embedding→softmax model is forced to give them similar embeddings.
// Similarity is never supplied; it emerges. That is the whole claim.
export function makeGrammarPairs(nPairs, rng) {
  const idx = {}
  GRAMMAR_TOKENS.forEach((t, i) => (idx[t.cat] = idx[t.cat] || []).push(i))
  const pick = (cat) => idx[cat][rng.int(idx[cat].length)]

  const xIdx = []
  const yIdx = []
  let cat = 'animal'
  let cur = pick(cat)
  for (let i = 0; i < nPairs; i++) {
    let nextCat
    if (cat === 'animal') nextCat = 'verb'
    else if (cat === 'verb') nextCat = rng.next() < 0.5 ? 'fruit' : 'animal'
    else nextCat = 'animal' // fruit → animal
    const next = pick(nextCat)
    xIdx.push(cur)
    yIdx.push(next)
    cur = next
    cat = nextCat
  }
  return { xIdx, yIdx, vocab: GRAMMAR_TOKENS }
}

// One-hot encode token indices into an (n × vocabSize) matrix so the first
// Dense layer acts as the embedding table (its weight rows are the embeddings).
export function oneHot(indices, vocabSize) {
  const X = mat(indices.length, vocabSize)
  for (let i = 0; i < indices.length; i++) X.d[i * vocabSize + indices[i]] = 1
  return X
}
