// Live computation of the grader's exact metric, in the browser.
// Mirrors tokenizer/evaluate_tokenizer.py:
//   faithful unit = one [\p{L}\p{M}\p{N}]+ run OR one visible punctuation char
//   ratio(lang)   = tokens / faithful units
//   score         = 1000 / (max ratio − min ratio)
//   hindi penalty = exp(max(0, hi/1.2 − 1)); adjusted = score / penalty
import { faithfulUnits, roundTripFaithful } from './hfbpe.js'

export const CONSTRAINT = 1.2

export function computeStats(texts, tok, langs) {
  const per = {}
  for (const l of langs) {
    const text = texts[l.code]
    const units = faithfulUnits(text)
    const tokens = tok.encode(text).length
    const ratio = units > 0 ? tokens / units : 0
    per[l.code] = { ...l, units, tokens, ratio, ok: ratio <= CONSTRAINT }
  }
  const ratios = langs.map((l) => per[l.code].ratio)
  const spread = Math.max(...ratios) - Math.min(...ratios)
  const score = spread > 0 ? 1000 / spread : Infinity
  const hindiPenalty = Math.exp(Math.max(0, (per.hi?.ratio ?? 0) / CONSTRAINT - 1))
  return {
    per,
    sortedDesc: langs.map((l) => per[l.code]).sort((a, b) => b.ratio - a.ratio),
    spread,
    score,
    hindiPenalty,
    adjustedScore: score / hindiPenalty,
    constraintsMet: langs.every((l) => per[l.code].ok),
  }
}

/** Faithfulness gate across all corpora (can be slow on MB-scale text; run once). */
export function checkFaithfulness(texts, tok, langs) {
  const out = {}
  for (const l of langs) out[l.code] = roundTripFaithful(tok, texts[l.code])
  return out
}
