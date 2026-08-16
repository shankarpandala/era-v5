import { useEffect, useMemo, useState } from 'react'
import { ERAS, NODES, FOOTNOTES, FAMILIES, BILLS, fmtDate } from '../data.js'
import { MajorCard, MinorCard } from './MechanismCard.jsx'
import TimelineStrip from './TimelineStrip.jsx'
import { ExtLink, Rich } from './Bits.jsx'

// The timeline proper: era bands in launch order, each with its narrative and
// its nodes (major cards + minor cards), a filter bar, and the footnote table.

// Black or white text depending on the chip colour's luminance (WCAG contrast).
function readableOn(hex) {
  const n = parseInt(hex.replace('#', ''), 16)
  const r = (n >> 16) & 255
  const g = (n >> 8) & 255
  const b = n & 255
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
  return lum > 0.6 ? '#18181b' : '#ffffff'
}

function Chip({ active, onClick, children, color }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={!!active}
      className={`rounded-full border px-2.5 py-0.5 text-[11px] font-medium transition-colors ${
        active
          ? 'border-transparent bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900'
          : 'border-zinc-300 text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800'
      }`}
      style={active && color ? { backgroundColor: color, color: readableOn(color) } : undefined}
    >
      {children}
    </button>
  )
}

// Default view: the 32 major nodes carry the whole story; "everything" adds the
// 56 minor nodes and 47 footnotes. Remembered per browser; ?view=all overrides.
function initialView() {
  if (typeof window === 'undefined') return 'story'
  const q = new URLSearchParams(window.location.search).get('view')
  if (q === 'all' || q === 'story') return q
  const saved = window.localStorage.getItem('a8-view')
  return saved === 'all' ? 'all' : 'story'
}

export default function Timeline() {
  const [view, setView] = useState(initialView)
  useEffect(() => {
    try {
      window.localStorage.setItem('a8-view', view)
    } catch {}
  }, [view])
  const [onlyList, setOnlyList] = useState(false)
  const [families, setFamilies] = useState(new Set())
  const [bills, setBills] = useState(new Set())
  const [q, setQ] = useState('')
  const [showFootnotes, setShowFootnotes] = useState(() => initialView() === 'all')
  const [vizMode, setVizMode] = useState('list') // which major cards start with their visual open

  const toggle = (set, setter, key) => {
    const s = new Set(set)
    if (s.has(key)) s.delete(key)
    else s.add(key)
    setter(s)
  }

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    return NODES.filter((m) => {
      if (view === 'story' && m.tier !== 'major') return false
      if (onlyList && !m.instructorList) return false
      if (families.size && !families.has(m.family)) return false
      if (bills.size && !m.bill.some((b) => bills.has(b))) return false
      if (needle) {
        const hay = `${m.name} ${m.short} ${m.aka || ''} ${m.problem} ${m.idea} ${m.source.title} ${m.source.authors || ''}`.toLowerCase()
        if (!hay.includes(needle)) return false
      }
      return true
    })
  }, [view, onlyList, families, bills, q])
  const visibleIds = filtered.map((m) => m.id)
  const majors = NODES.filter((m) => m.tier === 'major').length

  const jump = (id) => {
    const el = document.getElementById(`m-${id}`)
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  const anyFilter = onlyList || families.size || bills.size || q

  return (
    <div className="space-y-6">
      <div
        className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border p-4"
        style={{ borderColor: 'var(--claim-3)', backgroundColor: 'color-mix(in oklab, var(--claim-3) 10%, transparent)' }}
        role="region"
        aria-label="start here"
      >
        <div className="text-sm text-zinc-800 dark:text-zinc-100">
          {view === 'story' ? (
            <>
              <span className="font-semibold">Start here.</span> Showing the <b>{majors} major nodes</b> that carry the whole story — the course’s 17
              are tagged <span className="rounded-md bg-brand-500/15 px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide text-brand-600 dark:text-brand-300">on the list</span>.
              The other {NODES.length - majors} nodes and {FOOTNOTES.length} footnotes are one click away.
            </>
          ) : (
            <>
              <span className="font-semibold">Showing everything:</span> {NODES.length} nodes + {FOOTNOTES.length} footnotes. The {majors} major nodes are the
              short version.
            </>
          )}
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => {
              setView((v) => (v === 'story' ? 'all' : 'story'))
              if (view === 'story') setShowFootnotes(true)
            }}
            className="rounded-lg bg-brand-500 px-3 py-1.5 text-sm font-medium text-white hover:bg-brand-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
          >
            {view === 'story' ? `Show everything (${NODES.length} + ${FOOTNOTES.length})` : `Back to the ${majors} that carry the story`}
          </button>
          <button
            type="button"
            onClick={() => setOnlyList((v) => !v)}
            aria-pressed={onlyList}
            className="rounded-lg border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800"
          >
            {onlyList ? 'Showing the course’s 17 only' : 'Just the course’s 17'}
          </button>
        </div>
      </div>

      <TimelineStrip onJump={jump} visibleIds={anyFilter || view === 'story' ? visibleIds : null} />

      <div className="panel sticky top-16 z-30 space-y-2 p-3">
        <div className="flex flex-wrap items-center gap-2">
          <Chip active={onlyList} onClick={() => setOnlyList((v) => !v)}>
            {onlyList ? 'showing the instructor’s 17 only' : 'instructor’s 17 only'}
          </Chip>
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="search names, authors, ideas…"
            className="min-w-[180px] flex-1 rounded-md border border-zinc-300 bg-white px-2 py-1 text-xs text-zinc-800 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
            aria-label="search the timeline"
          />
          <span className="font-mono text-[11px] text-zinc-500">
            {filtered.length} / {NODES.length} nodes
          </span>
          <span className="ml-auto inline-flex items-center gap-1 text-[10px] uppercase tracking-wide text-zinc-500">
            visuals
            {[
              ['list', 'on the 17'],
              ['all', 'all open'],
              ['none', 'all closed'],
            ].map(([k, label]) => (
              <Chip key={k} active={vizMode === k} onClick={() => setVizMode(k)}>
                {label}
              </Chip>
            ))}
          </span>
          {anyFilter && (
            <button
              type="button"
              className="text-[11px] text-brand-600 hover:underline dark:text-brand-400"
              onClick={() => {
                setOnlyList(false)
                setFamilies(new Set())
                setBills(new Set())
                setQ('')
              }}
            >
              clear
            </button>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[10px] uppercase tracking-wide text-zinc-500">family</span>
          {Object.entries(FAMILIES).map(([k, f]) => (
            <Chip key={k} active={families.has(k)} color={f.color} onClick={() => toggle(families, setFamilies, k)}>
              {f.label}
            </Chip>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[10px] uppercase tracking-wide text-zinc-500">bill attacked</span>
          {Object.entries(BILLS).map(([k, b]) => (
            <Chip key={k} active={bills.has(k)} onClick={() => toggle(bills, setBills, k)}>
              {b.glyph} {b.label}
            </Chip>
          ))}
        </div>
      </div>

      {ERAS.map((era) => {
        const nodes = filtered.filter((m) => m.era === era.id)
        if (!nodes.length) return null
        return (
          <section key={era.id} id={`era-${era.id}`} className="relative scroll-mt-24">
            <div className="rounded-2xl p-4 sm:p-5" style={{ backgroundColor: `${era.color}14`, borderLeft: `4px solid ${era.color}` }}>
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h3 className="text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
                  <span className="mr-2 font-mono text-xs uppercase tracking-widest" style={{ color: era.color }}>
                    era {era.id}
                  </span>
                  {era.title}
                </h3>
                <span className="font-mono text-xs text-zinc-500">{era.span} · wants: {era.wanted}</span>
              </div>
              <p className="mt-2 max-w-4xl text-sm leading-relaxed text-zinc-700 dark:text-zinc-200"><Rich text={era.narrative} /></p>
            </div>
            <div className="mt-3 space-y-3 border-l-2 pl-3 sm:pl-5" style={{ borderColor: `${era.color}55` }}>
              {nodes.map((m) => (m.tier === 'major' ? <MajorCard key={m.id} m={m} defaultOpen={m.instructorList} vizMode={vizMode} /> : <MinorCard key={m.id} m={m} />))}
            </div>
          </section>
        )
      })}

      {!filtered.length && <div className="panel p-6 text-center text-sm text-zinc-500">No node matches those filters.</div>}

      <section id="footnotes" className={`scroll-mt-24 ${view === 'story' && !showFootnotes ? 'opacity-70' : ''}`}>
        <button
          type="button"
          onClick={() => setShowFootnotes((v) => !v)}
          className="mb-2 inline-flex items-center gap-1.5 text-sm font-semibold text-zinc-800 dark:text-zinc-100"
          aria-expanded={showFootnotes}
        >
          <span aria-hidden="true">{showFootnotes ? '▾' : '▸'}</span>
          Footnotes — {FOOTNOTES.length} more dated entries that did not earn a card
        </button>
        <p className="mb-2 text-xs text-zinc-500 dark:text-zinc-400">
          Benchmarks, systems papers, pure-quality tweaks, models that first shipped a mechanism, and variants of a node above. Every one was date-checked the same way; the last column says why it is not a node.
        </p>
        {showFootnotes && (
          <div className="overflow-x-auto rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
            <table className="w-full text-xs" style={{ minWidth: 720 }}>
              <thead>
                <tr className="border-b border-zinc-200 text-left text-[10px] uppercase tracking-wide text-zinc-500 dark:border-zinc-700">
                  <th className="px-3 py-2">date</th>
                  <th className="px-3 py-2">entry</th>
                  <th className="px-3 py-2">what it is</th>
                  <th className="px-3 py-2">why a footnote</th>
                </tr>
              </thead>
              <tbody>
                {FOOTNOTES.map((f) => (
                  <tr key={f.id} id={`m-${f.id}`} className="border-b border-zinc-100 align-top last:border-0 dark:border-zinc-800">
                    <td className="whitespace-nowrap px-3 py-2 font-mono text-zinc-700 dark:text-zinc-200">{fmtDate(f.date, f.datePrecision)}</td>
                    <td className="px-3 py-2">
                      <ExtLink href={f.source.url} className="font-medium">
                        {f.name}
                      </ExtLink>
                      <div className="font-mono text-[10px] text-zinc-400">{f.source.evidence}</div>
                    </td>
                    <td className="px-3 py-2 text-zinc-600 dark:text-zinc-300">{f.oneLiner}</td>
                    <td className="px-3 py-2 text-zinc-500 dark:text-zinc-400">{f.whyFootnote}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
