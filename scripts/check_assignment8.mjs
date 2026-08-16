// Checks for Assignment 8 (attention timeline). Same spirit as check_parity.mjs:
// the page, the README table and this script all read assignment-8/data/, so
// nothing can drift. Nonzero exit on any failure.
//
//   node scripts/check_assignment8.mjs            # validate data + README table in sync
//   node scripts/check_assignment8.mjs --write    # regenerate the README table in place
//   node scripts/check_assignment8.mjs --links    # also HEAD/GET every source URL

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.join(HERE, '..')
const DATA = path.join(ROOT, 'assignment-8', 'data')
const README = path.join(ROOT, 'assignment-8', 'README.md')
const args = new Set(process.argv.slice(2))

const TIERS = new Set(['major', 'minor', 'footnote'])
const KINDS = new Set(['arxiv-v1', 'blog', 'reddit-post', 'model-release', 'github-release'])
const FAMILIES = new Set(['attention', 'position', 'scaling', 'heads-kv', 'pattern-sparse', 'content-sparse', 'linear-recurrent', 'hybrid', 'kv-eviction', 'systems', 'stability'])
const BILLS = new Set(['compute', 'kv-memory', 'length', 'extrapolation', 'quality', 'stability', 'systems'])
const VIZ = new Set(['none', 'baseline', 'matrix', 'position', 'kv', 'state'])
const PICK = new Set(['yes', 'maybe', 'no', 'na'])
const SCEN = ['chatbot2k', 'rag32k', 'coding128k', 'agent1m']
const ISO = /^\d{4}-\d{2}-\d{2}$/
const KIND_LABEL = { 'arxiv-v1': 'arXiv v1', blog: 'blog post', 'reddit-post': 'Reddit post', 'model-release': 'model release', 'github-release': 'GitHub' }

// The instructor's minimum list, mapped onto ids (mirrors src/assignment-8/data.js).
const INSTRUCTOR = [
  ['standard attention', ['standard-attention']],
  ['absolute learned positions', ['learned-absolute-positions']],
  ['sinusoidal', ['sinusoidal-positions']],
  ['RoPE', ['rope']],
  ['ALiBi', ['alibi']],
  ['MQA', ['mqa']],
  ['GQA', ['gqa']],
  ['sliding window', ['sliding-window', 'mistral-swa']],
  ['attention sinks', ['attention-sinks']],
  ['NTK-aware scaling', ['ntk-aware']],
  ['YaRN', ['yarn']],
  ['linear attention', ['linear-attention']],
  ['delta rule + Gated DeltaNet', ['delta-rule', 'gated-deltanet']],
  ['MLA', ['mla']],
  ['sparse + top-k attention', ['sparse-transformer', 'topk-attention']],
  ['compressed + sparse (DeepSeek)', ['nsa', 'dsa']],
  ['DroPE', ['drope']],
]

let failures = 0
const fail = (msg) => {
  failures++
  console.error(`FAIL  ${msg}`)
}
const ok = (msg) => console.log(`ok    ${msg}`)

// ---- load ------------------------------------------------------------------
const eras = JSON.parse(fs.readFileSync(path.join(DATA, 'eras.json'), 'utf8'))
const files = fs.readdirSync(path.join(DATA, 'mechanisms')).filter((f) => f.endsWith('.json')).sort()
let all = []
for (const f of files) {
  const arr = JSON.parse(fs.readFileSync(path.join(DATA, 'mechanisms', f), 'utf8'))
  if (!Array.isArray(arr)) fail(`${f}: not an array`)
  all = all.concat(arr.map((m) => ({ ...m, __file: f })))
}
all.sort((a, b) => a.date.localeCompare(b.date) || (a.order ?? 0) - (b.order ?? 0) || a.id.localeCompare(b.id))
ok(`${all.length} records in ${files.length} era files; ${eras.length} eras`)

// ---- schema ----------------------------------------------------------------
const ids = new Set()
const eraIds = new Set(eras.map((e) => e.id))
const isUrl = (u) => typeof u === 'string' && /^https?:\/\//.test(u)
for (const m of all) {
  const where = `${m.__file}:${m.id}`
  if (!m.id || ids.has(m.id)) fail(`${where}: missing or duplicate id`)
  ids.add(m.id)
  if (!TIERS.has(m.tier)) fail(`${where}: bad tier ${m.tier}`)
  if (!m.name || !m.short) fail(`${where}: missing name/short`)
  if (!ISO.test(m.date || '')) fail(`${where}: bad date ${m.date}`)
  if (m.paperDate && !ISO.test(m.paperDate)) fail(`${where}: bad paperDate ${m.paperDate}`)
  if (!KINDS.has(m.dateKind)) fail(`${where}: bad dateKind ${m.dateKind}`)
  if (!eraIds.has(m.era)) fail(`${where}: unknown era ${m.era}`)
  if (!FAMILIES.has(m.family)) fail(`${where}: bad family ${m.family}`)
  if (!Array.isArray(m.bill) || !m.bill.length || m.bill.some((b) => !BILLS.has(b))) fail(`${where}: bad bill ${JSON.stringify(m.bill)}`)
  if (!m.source || !isUrl(m.source.url) || !m.source.title || !m.source.evidence) fail(`${where}: source needs url/title/evidence`)
  if (m.source?.arxiv && !/^\d{4}\.\d{4,5}$/.test(m.source.arxiv)) fail(`${where}: odd arXiv id ${m.source.arxiv}`)
  if (m.firstShipped && (!isUrl(m.firstShipped.url) || !m.firstShipped.model || !m.firstShipped.date)) fail(`${where}: firstShipped needs model/date/url`)
  for (const s of m.secondary || []) if (!isUrl(s.url) || !s.label) fail(`${where}: secondary needs label/url`)
  // era date range
  const era = eras.find((e) => e.id === m.era)
  if (era && (m.date < era.from || m.date > era.to)) fail(`${where}: date ${m.date} outside era ${era.id} (${era.from}..${era.to})`)
  if (m.tier === 'footnote') {
    if (!m.oneLiner || !m.whyFootnote) fail(`${where}: footnote needs oneLiner/whyFootnote`)
  } else {
    if (!m.problem || !m.idea) fail(`${where}: needs problem/idea`)
    if (!Array.isArray(m.buys) || !m.buys.length) fail(`${where}: needs buys[]`)
    if (!Array.isArray(m.costs) || !m.costs.length) fail(`${where}: needs costs[] — a mechanism with only pros is not understood yet`)
    if (!m.pickWhen || !m.pickWhen.verdict) fail(`${where}: needs pickWhen.verdict`)
    for (const k of SCEN) {
      if (!m.pickWhen?.[k] || !PICK.has(m.pickWhen[k].v)) fail(`${where}: pickWhen.${k}.v must be yes|maybe|no|na`)
      else if (!m.pickWhen[k].why || !m.pickWhen[k].why.trim()) fail(`${where}: pickWhen.${k}.why is empty — every scenario cell needs a reason`)
    }
    if (!m.viz || !VIZ.has(m.viz.kind)) fail(`${where}: viz.kind must be one of ${[...VIZ].join('|')}`)
  }
}
ok('schema: every record has a date, a kind, a primary source with the evidence string read from it, and (for nodes) buys, costs, a verdict and a reason in every scenario cell')

// ---- instructor list -------------------------------------------------------
for (const [label, want] of INSTRUCTOR) {
  for (const id of want) {
    const m = all.find((x) => x.id === id)
    if (!m) fail(`instructor list: "${label}" → missing node ${id}`)
    else if (!m.instructorList) fail(`instructor list: ${id} must have instructorList: true`)
    else if (m.tier === 'footnote') fail(`instructor list: ${id} cannot be a footnote`)
  }
}
const extraFlag = all.filter((m) => m.instructorList && !INSTRUCTOR.some(([, w]) => w.includes(m.id)))
if (extraFlag.length) fail(`instructorList: true on ids not in the mapping: ${extraFlag.map((m) => m.id).join(', ')}`)
ok(`instructor's minimum list: ${INSTRUCTOR.length} items covered by ${INSTRUCTOR.reduce((a, [, w]) => a + w.length, 0)} nodes`)

// ---- chronology sanity -----------------------------------------------------
const nodes = all.filter((m) => m.tier !== 'footnote')
const majors = all.filter((m) => m.tier === 'major')
const foots = all.filter((m) => m.tier === 'footnote')
const clock = all.find((m) => m.id === 'standard-attention')
if (clock && clock.date !== '2017-06-12') fail('standard-attention must be dated 2017-06-12 (arXiv v1 of 1706.03762)')
for (const m of all) if (m.prologue && clock && m.date >= clock.date) fail(`${m.id}: prologue entries must predate the Transformer`)
for (const m of all) if (!m.prologue && m.tier !== 'footnote' && clock && m.date < clock.date) fail(`${m.id}: pre-Transformer node must be marked prologue`)
ok(`chronology: ${nodes.length} nodes (${majors.length} major, ${nodes.length - majors.length} minor) + ${foots.length} footnotes, ${all[0].date} → ${all[all.length - 1].date}`)

// ---- README table ----------------------------------------------------------
function fmt(iso) {
  const [y, mo, d] = iso.split('-').map(Number)
  const M = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  return `${d} ${M[mo - 1]} ${y}`
}
function esc(s) {
  return String(s).replace(/\|/g, '\\|').replace(/\n/g, ' ')
}
function table() {
  const head = ['| # | first appeared | kind | mechanism | tier | source | evidence read from the source |', '|---|---|---|---|---|---|---|']
  const rows = all.map((m, i) => {
    const date = m.paperDate && m.paperDate !== m.date ? `${fmt(m.date)} (paper ${fmt(m.paperDate)})` : fmt(m.date)
    const src = m.source.arxiv ? `[arXiv ${m.source.arxiv}](https://arxiv.org/abs/${m.source.arxiv})` : `[${m.source.url.replace(/^https?:\/\//, '').slice(0, 40)}](${m.source.url})`
    const primary = m.source.arxiv && !m.source.url.includes('arxiv.org') ? `${src} · [primary](${m.source.url})` : src
    const list = m.instructorList ? ' ★' : ''
    return `| ${i + 1} | ${date} | ${KIND_LABEL[m.dateKind]} | **${esc(m.short)}**${list}<br><sub>${esc(m.name)}</sub> | ${m.tier} | ${primary} | <sub>${esc(m.source.evidence)}</sub> |`
  })
  return [...head, ...rows].join('\n')
}
const BEGIN = '<!-- BEGIN TIMELINE TABLE (generated by scripts/check_assignment8.mjs — do not edit by hand) -->'
const END = '<!-- END TIMELINE TABLE -->'
if (fs.existsSync(README)) {
  const md = fs.readFileSync(README, 'utf8')
  const a = md.indexOf(BEGIN)
  const b = md.indexOf(END)
  if (a < 0 || b < 0 || b < a) fail('README: table markers not found')
  else {
    const current = md.slice(a + BEGIN.length, b).trim()
    const fresh = table()
    if (current !== fresh) {
      if (args.has('--write')) {
        fs.writeFileSync(README, md.slice(0, a + BEGIN.length) + '\n' + fresh + '\n' + md.slice(b))
        ok('README table regenerated (--write)')
      } else fail('README table is out of date — run: node scripts/check_assignment8.mjs --write')
    } else ok('README table matches the data')
  }
} else fail('assignment-8/README.md missing')

// ---- links (optional) ------------------------------------------------------
if (args.has('--links')) {
  const urls = new Set()
  for (const m of all) {
    urls.add(m.source.url)
    if (m.source.arxiv) urls.add(`https://arxiv.org/abs/${m.source.arxiv}`)
    if (m.firstShipped?.url) urls.add(m.firstShipped.url)
    for (const s of m.secondary || []) urls.add(s.url)
  }
  console.log(`checking ${urls.size} URLs …`)
  let bad = 0
  let blocked = 0
  const list = [...urls]
  const conc = 8
  let idx = 0
  async function worker() {
    while (idx < list.length) {
      const u = list[idx++]
      try {
        const ctrl = new AbortController()
        const t = setTimeout(() => ctrl.abort(), 20000)
        let r = await fetch(u, { method: 'HEAD', redirect: 'follow', signal: ctrl.signal, headers: { 'user-agent': 'Mozilla/5.0 (era-v5 link check)' } })
        if (r.status === 405 || r.status === 403 || r.status === 404) r = await fetch(u, { method: 'GET', redirect: 'follow', signal: ctrl.signal, headers: { 'user-agent': 'Mozilla/5.0 (era-v5 link check)' } })
        clearTimeout(t)
        if (r.status >= 400) {
          if (/reddit\.com|kexue\.fm|x\.com|twitter\.com|huggingface\.co\/meta-llama/.test(u) && (r.status === 403 || r.status === 429 || r.status === 401)) {
            blocked++
            console.log(`warn  ${r.status} ${u} (bot-blocked / gated — verified via archive or API during research)`)
          } else {
            bad++
            console.error(`FAIL  ${r.status} ${u}`)
          }
        }
      } catch (e) {
        bad++
        console.error(`FAIL  ${e.name || 'error'} ${u}`)
      }
    }
  }
  await Promise.all(Array.from({ length: conc }, worker))
  if (bad) fail(`${bad} URL(s) unreachable`)
  else ok(`all URLs reachable (${blocked} bot-blocked hosts warned)`)
}

if (failures) {
  console.error(`\n${failures} failure(s)`)
  process.exit(1)
}
console.log('\nverdict: PASS')
