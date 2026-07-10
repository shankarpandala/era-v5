// Live fertility computation — the numbers the widget SHOWS are produced here,
// in the browser, by running the JS encoder over the shipped corpora. Mirrors
// tokenizer/evaluate.py::compute_stats so it cross-checks against stats.json.
import { wordCount } from './bpe.js'

export const CONSTRAINT = 1.2

export function computeStats(corpora, bpe, langs) {
  const per = {}
  for (const l of langs) {
    const text = corpora[l.code]
    const words = wordCount(text)
    const tokens = bpe.encode(text).length
    const X = tokens / words
    per[l.code] = { ...l, words, tokens, X, ok: X <= CONSTRAINT }
  }
  const xs = langs.map((l) => per[l.code].X)
  const xMax = Math.max(...xs)
  const xMin = Math.min(...xs)
  const spread = xMax - xMin
  const score = spread > 0 ? 1000 / spread : Infinity
  const sortedDesc = langs.map((l) => per[l.code]).sort((a, b) => b.X - a.X)
  return {
    per,
    sortedDesc,
    xMax,
    xMin,
    spread,
    score,
    constraint: CONSTRAINT,
    constraintsMet: langs.every((l) => per[l.code].ok),
  }
}
