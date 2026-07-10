// Live fertility computation — the numbers the widget SHOWS are produced here,
// in the browser, by running the JS encoder over the given texts. Mirrors
// tokenizer/evaluate.py::compute_stats so it cross-checks against stats.json.
//
// Two word counts per the metric:
//   primary = whitespace runs (text.split()) — standard fertility. Equals the
//             word-faithful [\p{L}\p{N}\p{M}]+ count within 1-2% on the corpora.
//   wplus   = [\p{L}\p{N}]+ runs (== Python \w+) — the common classroom idiom.
//             CAVEAT: \w drops combining marks, so it splits Indic words at
//             matras/viramas (2-3x word-count inflation); shown for
//             comparability, not as a true word count.
// The assignment's hard gate is English <= 1.2 — met under BOTH counts.
import { wordCountSplit } from './bpe.js'

export const CONSTRAINT = 1.2

function metric(perTokens, texts, langs, wordFn) {
  const per = {}
  for (const l of langs) {
    const words = wordFn(texts[l.code])
    const tokens = perTokens[l.code]
    const X = words > 0 ? tokens / words : 0
    per[l.code] = { ...l, words, tokens, X, ok: X <= CONSTRAINT }
  }
  const xs = langs.map((l) => per[l.code].X)
  const xMax = Math.max(...xs)
  const xMin = Math.min(...xs)
  const spread = xMax - xMin
  return {
    per,
    sortedDesc: langs.map((l) => per[l.code]).sort((a, b) => b.X - a.X),
    xMax,
    xMin,
    spread,
    score: spread > 0 ? 1000 / spread : Infinity,
    constraintsMet: langs.every((l) => per[l.code].ok),
    englishOk: per.en ? per.en.ok : true,
  }
}

export function computeStats(texts, bpe, langs) {
  const perTokens = {}
  for (const l of langs) perTokens[l.code] = bpe.encode(texts[l.code]).length
  const primary = metric(perTokens, texts, langs, wordCountSplit)
  const wplus = metric(perTokens, texts, langs, (t) => bpe.countWords(t))
  return { primary, wplus, constraint: CONSTRAINT }
}
