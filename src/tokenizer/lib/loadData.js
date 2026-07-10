// Base-aware loading of the shipped tokenizer artifacts + eval corpora.
// Everything lives under public/tokenizer/ and is served at `${BASE_URL}tokenizer/`
// (BASE_URL is '/era-v5/' in prod, '/' in dev). The widget recomputes all
// numbers from these exact files — the same ones a grader can download.

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

  const [tok, stats] = await Promise.all([
    getJSON('tokenizer/tokenizer.json'),
    getJSON('tokenizer/stats.json'),
  ])
  const corpora = {}
  await Promise.all(
    LANGS.map(async (l) => {
      corpora[l.code] = await getText(`tokenizer/corpora/${l.code}.txt`)
    }),
  )
  return { tok, stats, corpora }
}
