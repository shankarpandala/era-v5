// Parity proof: the JS encoder (src/tokenizer/lib/bpe.js) must produce the
// EXACT token stream the Python reference produced (public/tokenizer/parity_golden.json)
// on every frozen eval corpus. Nonzero exit on any mismatch.
//
//   node scripts/check_parity.mjs
//
// This is what backs the claim "the widget's live numbers equal the numbers a
// grader gets running our Python tokenizer".

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { BPE } from '../src/tokenizer/lib/bpe.js'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const PUB = path.join(HERE, '..', 'public', 'tokenizer')

const tok = JSON.parse(fs.readFileSync(path.join(PUB, 'tokenizer.json'), 'utf8'))
const golden = JSON.parse(fs.readFileSync(path.join(PUB, 'parity_golden.json'), 'utf8'))
const bpe = BPE.fromJSON(tok)

const LANGS = ['en', 'hi', 'te', 'mr']
let ok = true
for (const lang of LANGS) {
  const text = fs.readFileSync(path.join(PUB, 'corpora', `${lang}.txt`), 'utf8')
  const ids = bpe.encode(text)
  const g = golden[lang]
  const equal = ids.length === g.length && ids.every((v, i) => v === g[i])
  console.log(`${lang}: ${equal ? 'OK ' : 'MISMATCH'}  js=${ids.length} py=${g.length}`)
  if (!equal) {
    ok = false
    const n = Math.max(ids.length, g.length)
    for (let i = 0; i < n; i++) {
      if (ids[i] !== g[i]) {
        console.log(`   first diff @ ${i}: js=${ids[i]} py=${g[i]}`)
        break
      }
    }
  }
}
console.log(ok ? '\nPARITY OK — JS == Python on all corpora' : '\nPARITY FAILED')
process.exit(ok ? 0 : 1)
