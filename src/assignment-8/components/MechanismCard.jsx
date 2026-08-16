import { useEffect, useState } from 'react'
import Viz from '../viz/Viz.jsx'
import { DateBadge, FamilyChip, BillChips, ListTag, PickPill, ExtLink, Rich } from './Bits.jsx'
import { SCENARIOS, fmtDate, eraOf } from '../data.js'

function Sources({ m }) {
  return (
    <div className="mt-4 rounded-lg border border-dashed border-zinc-300 p-3 text-[11px] text-zinc-600 dark:border-zinc-700 dark:text-zinc-300">
      <div className="font-semibold uppercase tracking-wide text-zinc-500">Source of the date</div>
      <div className="mt-1">
        <ExtLink href={m.source.url}>{m.source.title}</ExtLink>
        {m.source.authors && <span className="text-zinc-500"> — {m.source.authors}</span>}
        {m.source.arxiv && (
          <>
            {' '}
            · <ExtLink href={`https://arxiv.org/abs/${m.source.arxiv}`}>arXiv {m.source.arxiv}</ExtLink>
          </>
        )}
      </div>
      <div className="mt-1 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">read from the source: “{m.source.evidence}”</div>
      {m.firstShipped && (
        <div className="mt-2">
          <span className="font-semibold text-zinc-500">First shipped at scale:</span>{' '}
          <ExtLink href={m.firstShipped.url}>{m.firstShipped.model}</ExtLink> <span className="font-mono">({fmtDate(m.firstShipped.date, m.firstShipped.date?.length === 7 ? 'month' : 'day')})</span>
        </div>
      )}
      {m.secondary?.length > 0 && (
        <ul className="mt-1 list-disc space-y-0.5 pl-4">
          {m.secondary.map((s, i) => (
            <li key={i}>
              <ExtLink href={s.url}>{s.label}</ExtLink> <span className="font-mono text-zinc-400">{fmtDate(s.date, s.date?.length <= 7 ? 'month' : 'day')}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function Tradeoffs({ m }) {
  return (
    <div className="mt-4 grid gap-3 md:grid-cols-2">
      <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3">
        <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-emerald-700 dark:text-emerald-300">What it buys</div>
        <ul className="list-disc space-y-1 pl-4 text-sm text-zinc-700 dark:text-zinc-200">
          {m.buys.map((b, i) => (
            <li key={i}><Rich text={b} /></li>
          ))}
        </ul>
      </div>
      <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-3">
        <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-red-700 dark:text-red-300">What it costs</div>
        <ul className="list-disc space-y-1 pl-4 text-sm text-zinc-700 dark:text-zinc-200">
          {m.costs.map((c, i) => (
            <li key={i}><Rich text={c} /></li>
          ))}
        </ul>
      </div>
    </div>
  )
}

function PickWhen({ m }) {
  const p = m.pickWhen
  if (!p) return null
  return (
    <div className="mt-3 rounded-lg bg-zinc-100 p-3 dark:bg-zinc-800/60">
      <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">When you would actually pick it</div>
      <div className="flex flex-wrap gap-1.5">
        {SCENARIOS.map((s) => (
          <PickPill key={s.key} v={p[s.key]?.v} label={s.label} why={p[s.key]?.why} />
        ))}
      </div>
      {p.verdict && <p className="mt-2 text-sm text-zinc-700 dark:text-zinc-200"><span className="font-semibold">Verdict — </span><Rich text={p.verdict} /></p>}
    </div>
  )
}

export function MajorCard({ m, defaultOpen = false, vizMode = 'list' }) {
  const [openViz, setOpenViz] = useState(defaultOpen)
  // The timeline's "visuals" control re-syncs every card; local toggles still work after.
  useEffect(() => {
    setOpenViz(vizMode === 'all' ? true : vizMode === 'none' ? false : !!m.instructorList)
  }, [vizMode, m.instructorList])
  const era = eraOf(m)
  return (
    <article id={`m-${m.id}`} className="scroll-mt-24 rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900 sm:p-5" style={{ borderLeftWidth: 4, borderLeftColor: era.color }}>
      <header className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <DateBadge m={m} />
          <h3 className="mt-1 text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">{m.name}</h3>
          {m.aka && <div className="mt-0.5 text-xs text-zinc-500 dark:text-zinc-400">aka {m.aka}</div>}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <ListTag m={m} />
          <FamilyChip family={m.family} />
          <BillChips bills={m.bill} />
        </div>
      </header>

      <div className="mt-3 grid gap-3 md:grid-cols-2">
        <div className="rounded-lg border-l-4 border-zinc-300 bg-zinc-50 p-3 dark:border-zinc-600 dark:bg-zinc-800/40">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-zinc-500">The problem at that moment</div>
          <p className="mt-1 text-sm leading-relaxed text-zinc-700 dark:text-zinc-200"><Rich text={m.problem} /></p>
        </div>
        <div className="rounded-lg border-l-4 p-3" style={{ borderColor: era.color, backgroundColor: `${era.color}12` }}>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-zinc-500">The idea</div>
          <p className="mt-1 text-sm leading-relaxed text-zinc-700 dark:text-zinc-200"><Rich text={m.idea} /></p>
        </div>
      </div>

      {m.viz && m.viz.kind !== 'none' && (
        <div className="mt-4">
          <button
            type="button"
            onClick={() => setOpenViz((o) => !o)}
            className="inline-flex items-center gap-1.5 rounded-md border border-zinc-300 px-2.5 py-1 text-xs font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800"
            aria-expanded={openViz}
          >
            <span aria-hidden="true">{openViz ? '▾' : '▸'}</span>
            {openViz ? 'Hide the visual' : 'Open the visual'}
          </button>
          {openViz && (
            <div className="mt-2">
              <Viz viz={m.viz} />
            </div>
          )}
        </div>
      )}

      <Tradeoffs m={m} />
      <PickWhen m={m} />
      <Sources m={m} />
    </article>
  )
}

export function MinorCard({ m }) {
  const [open, setOpen] = useState(false)
  const era = eraOf(m)
  return (
    <article id={`m-${m.id}`} className="scroll-mt-24 rounded-xl border border-zinc-200 bg-white p-3 dark:border-zinc-800 dark:bg-zinc-900" style={{ borderLeftWidth: 3, borderLeftColor: era.color }}>
      <button type="button" onClick={() => setOpen((o) => !o)} className="flex w-full flex-wrap items-start justify-between gap-2 text-left" aria-expanded={open}>
        <div className="min-w-0 flex-1">
          <DateBadge m={m} compact />
          <div className="mt-0.5 text-sm font-semibold text-zinc-900 dark:text-zinc-50">{m.name}</div>
          {!open && <p className="mt-0.5 line-clamp-2 text-xs text-zinc-600 dark:text-zinc-300"><Rich text={m.problem} /></p>}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <ListTag m={m} />
          <FamilyChip family={m.family} small />
          <BillChips bills={m.bill} small />
          <span className="text-xs text-zinc-400" aria-hidden="true">{open ? '▾' : '▸'}</span>
        </div>
      </button>
      {open && (
        <div className="mt-2">
          <div className="grid gap-2 md:grid-cols-2">
            <div className="rounded-lg bg-zinc-50 p-2.5 text-xs dark:bg-zinc-800/40">
              <div className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500">Problem</div>
              <p className="mt-0.5 text-zinc-700 dark:text-zinc-200"><Rich text={m.problem} /></p>
            </div>
            <div className="rounded-lg p-2.5 text-xs" style={{ backgroundColor: `${era.color}12` }}>
              <div className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500">Idea</div>
              <p className="mt-0.5 text-zinc-700 dark:text-zinc-200"><Rich text={m.idea} /></p>
            </div>
          </div>
          {m.viz && m.viz.kind !== 'none' && (
            <div className="mt-3">
              <Viz viz={m.viz} />
            </div>
          )}
          <Tradeoffs m={m} />
          <PickWhen m={m} />
          <Sources m={m} />
        </div>
      )}
    </article>
  )
}
