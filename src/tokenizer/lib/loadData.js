// Base-aware loading of the shipped tokenizer artifacts + faithful-Markdown
// corpus. Everything lives under public/tokenizer/ and is served at
// `${BASE_URL}tokenizer/` — the widget recomputes all numbers from the exact
// files the grader downloads and runs.

// Submission set — the 4th language (Marathi) is our choice per the assignment.
// The corpus dir also ships Maithili so the instructor's published evaluator
// (which hardcodes mai) runs on these artifacts unchanged.
export const LANGS = [
  { code: 'en', name: 'English', script: 'Latin', accent: 'var(--claim-1)' },
  { code: 'hi', name: 'Hindi', script: 'Devanagari', accent: 'var(--claim-2)' },
  { code: 'te', name: 'Telugu', script: 'Telugu', accent: 'var(--claim-3)' },
  { code: 'mr', name: 'Marathi', script: 'Devanagari', accent: 'var(--claim-4)' },
]

export async function loadTokenizerData() {
  const base = import.meta.env.BASE_URL
  const getJSON = async (p) => {
    const r = await fetch(base + p)
    if (!r.ok) throw new Error(`fetch ${p} -> ${r.status}`)
    return r.json()
  }
  const getText = async (p) => {
    const r = await fetch(base + p)
    if (!r.ok) throw new Error(`fetch ${p} -> ${r.status}`)
    return r.text()
  }

  const [tok, metrics] = await Promise.all([
    getJSON('tokenizer/tokenizer.json'),
    getJSON('tokenizer/metrics.json'),
  ])
  const corpora = {}
  await Promise.all(
    LANGS.map(async (l) => {
      corpora[l.code] = await getText(`tokenizer/corpus/${l.code}.faithful.txt`)
    }),
  )
  return { tok, metrics, corpora }
}
