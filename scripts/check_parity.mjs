// Parity proof for the resubmission pipeline: the JS reimplementation
// (src/tokenizer/lib/hfbpe.js) must produce the EXACT token-id stream and the
// EXACT decoded text (SHA-256) that the HuggingFace `tokenizers` library
// produces (public/tokenizer/parity_golden.json), on every corpus file.
// Also re-checks the grader's faithfulness sample. Nonzero exit on mismatch.
//
//   node scripts/check_parity.mjs

import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import { fileURLToPath } from 'node:url'
import { HFTokenizer, faithfulUnits } from '../src/tokenizer/lib/hfbpe.js'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const PUB = path.join(HERE, '..', 'public', 'tokenizer')

const tok = HFTokenizer.fromJSON(JSON.parse(fs.readFileSync(path.join(PUB, 'tokenizer.json'), 'utf8')))
const golden = JSON.parse(fs.readFileSync(path.join(PUB, 'parity_golden.json'), 'utf8'))
const metrics = JSON.parse(fs.readFileSync(path.join(PUB, 'metrics.json'), 'utf8'))

const LANGS = ['en', 'hi', 'te', 'mai']
let ok = true
for (const lang of LANGS) {
  const text = fs.readFileSync(path.join(PUB, 'corpus', `${lang}.faithful.txt`), 'utf8')
  const ids = tok.encode(text)
  const g = golden[lang]
  const idsEq = ids.length === g.ids.length && ids.every((v, i) => v === g.ids[i])
  const decHash = crypto.createHash('sha256').update(tok.decode(ids), 'utf8').digest('hex')
  const decEq = decHash === g.decode_sha256
  const units = faithfulUnits(text)
  const unitsEq = units === metrics.faithful_units[lang]
  console.log(
    `${lang}: ids ${idsEq ? 'OK ' : 'MISMATCH'} (js=${ids.length} py=${g.ids.length})` +
      `  decode ${decEq ? 'OK ' : 'MISMATCH'}  units ${unitsEq ? 'OK ' : 'MISMATCH'} (js=${units} py=${metrics.faithful_units[lang]})`,
  )
  if (!idsEq) {
    ok = false
    for (let i = 0; i < Math.max(ids.length, g.ids.length); i++) {
      if (ids[i] !== g.ids[i]) {
        console.log(`   first id diff @ ${i}: js=${ids[i]} py=${g.ids[i]}`)
        break
      }
    }
  }
  if (!decEq || !unitsEq) ok = false
}

// Grader's faithfulness sample must round-trip exactly.
const SAMPLE = "India's population is 1,428,627,663."
const dec = tok.decode(tok.encode(SAMPLE))
console.log(`sample round-trip exact: ${dec === SAMPLE}  ("${dec}")`)
if (dec !== SAMPLE) ok = false

console.log(ok ? '\nPARITY OK — JS == HuggingFace tokenizers on all corpora' : '\nPARITY FAILED')
process.exit(ok ? 0 : 1)
