// Base-aware loading of the shipped tokenizer artifacts + faithful-Markdown
// corpus. Everything lives under public/tokenizer/ and is served at
// `${BASE_URL}tokenizer/` — the widget recomputes all numbers from the exact
// files the grader downloads and runs.

export const LANGS = [
  { code: 'en', name: 'English', script: 'Latin', accent: 'var(--claim-1)' },
  { code: 'hi', name: 'Hindi', script: 'Devanagari', accent: 'var(--claim-2)' },
  { code: 'te', name: 'Telugu', script: 'Telugu', accent: 'var(--claim-3)' },
  { code: 'mai', name: 'Maithili', script: 'Devanagari', accent: 'var(--claim-4)' },
]

export async function loadTokenizerData() {
  const base = import.meta.env.BASE_URL
  // tokenizer.json / metrics.json / corpus files have fixed (non-hashed) names,
  // so a new deploy reuses the same URLs. Force a revalidation (conditional
  // request, cheap 304 when unchanged) so a browser can never serve a stale
  // copy from an earlier deploy — the class of bug that showed a language as
  // "0 corpus" after a corpus/language change.
  const getJSON = async (p) => {
    const r = await fetch(base + p, { cache: 'no-cache' })
    if (!r.ok) throw new Error(`fetch ${p} -> ${r.status}`)
    return r.json()
  }
  const getText = async (p) => {
    const r = await fetch(base + p, { cache: 'no-cache' })
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
