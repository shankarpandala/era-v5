// ERA-V5 · Assignment 8 — data access.
// The single source of truth is assignment-8/data/mechanisms/*.json (one file
// per era) plus assignment-8/data/eras.json. Nothing on the page is retyped:
// every date, source, evidence string and trade-off is read from those files,
// and scripts/check_assignment8.mjs validates the same files and regenerates
// the README table from them.

import erasJson from '../../assignment-8/data/eras.json'

const files = import.meta.glob('../../assignment-8/data/mechanisms/*.json', { eager: true })

const raw = Object.keys(files)
  .sort()
  .flatMap((k) => files[k].default)

// Chronological: by first-appearance date, then an explicit tie-break order.
export const ALL = [...raw].sort(
  (a, b) => a.date.localeCompare(b.date) || (a.order ?? 0) - (b.order ?? 0) || a.id.localeCompare(b.id),
)

export const ERAS = erasJson
export const NODES = ALL.filter((m) => m.tier !== 'footnote')
export const FOOTNOTES = ALL.filter((m) => m.tier === 'footnote')
export const MAJOR = ALL.filter((m) => m.tier === 'major')
export const INSTRUCTOR = ALL.filter((m) => m.instructorList)
export const byId = Object.fromEntries(ALL.map((m) => [m.id, m]))

export function eraOf(m) {
  return ERAS.find((e) => e.id === m.era) ?? ERAS[0]
}

export function nodesInEra(eraId) {
  return NODES.filter((m) => m.era === eraId)
}

// The instructor's minimum list, mapped onto node ids (some items are two nodes).
export const INSTRUCTOR_ITEMS = [
  { label: 'standard attention', ids: ['standard-attention'] },
  { label: 'absolute learned positions', ids: ['learned-absolute-positions'] },
  { label: 'sinusoidal', ids: ['sinusoidal-positions'] },
  { label: 'RoPE', ids: ['rope'] },
  { label: 'ALiBi', ids: ['alibi'] },
  { label: 'MQA', ids: ['mqa'] },
  { label: 'GQA', ids: ['gqa'] },
  { label: 'sliding window', ids: ['sliding-window', 'mistral-swa'] },
  { label: 'attention sinks', ids: ['attention-sinks'] },
  { label: 'NTK-aware scaling', ids: ['ntk-aware'] },
  { label: 'YaRN', ids: ['yarn'] },
  { label: 'linear attention', ids: ['linear-attention'] },
  { label: 'the delta rule and Gated DeltaNet', ids: ['delta-rule', 'gated-deltanet'] },
  { label: 'MLA', ids: ['mla'] },
  { label: 'sparse and top-k attention', ids: ['sparse-transformer', 'topk-attention'] },
  { label: 'compressed and sparse attention as DeepSeek does it', ids: ['nsa', 'dsa'] },
  { label: 'DroPE', ids: ['drope'] },
]

export const FAMILIES = {
  attention: { label: 'attention core', color: '#3b82f6' },
  position: { label: 'positions', color: '#10b981' },
  scaling: { label: 'RoPE scaling', color: '#84cc16' },
  'heads-kv': { label: 'heads / KV layout', color: '#f59e0b' },
  'pattern-sparse': { label: 'sparse by pattern', color: '#8b5cf6' },
  'content-sparse': { label: 'sparse by content', color: '#d946ef' },
  'linear-recurrent': { label: 'linear / recurrent', color: '#ef4444' },
  hybrid: { label: 'hybrid layouts', color: '#06b6d4' },
  'kv-eviction': { label: 'KV eviction', color: '#f97316' },
  systems: { label: 'systems', color: '#71717a' },
  stability: { label: 'stability', color: '#a1a1aa' },
}

export const BILLS = {
  compute: { label: 'compute (n²)', glyph: '⚙' },
  'kv-memory': { label: 'KV memory', glyph: '▤' },
  length: { label: 'context length', glyph: '↔' },
  extrapolation: { label: 'extrapolation', glyph: '⤳' },
  quality: { label: 'quality', glyph: '★' },
  stability: { label: 'stability', glyph: '⚖' },
  systems: { label: 'systems', glyph: '⛭' },
}

export const SCENARIOS = [
  { key: 'chatbot2k', label: '2K chatbot' },
  { key: 'rag32k', label: '32K RAG' },
  { key: 'coding128k', label: '128K coding' },
  { key: 'agent1m', label: '1M agent' },
]

export const DATE_KIND_LABEL = {
  'arxiv-v1': 'arXiv v1',
  blog: 'blog post',
  'reddit-post': 'Reddit post',
  'model-release': 'model release',
  'github-release': 'GitHub',
}

export function fmtDate(iso, precision = 'day') {
  if (!iso) return ''
  const [y, m, d] = iso.split('-').map(Number)
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  if (!m) return String(y)
  if (precision === 'month' || !d) return `${months[m - 1]} ${y}`
  return `${d} ${months[m - 1]} ${y}`
}
