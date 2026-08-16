import { FAMILIES, BILLS, DATE_KIND_LABEL, fmtDate } from '../data.js'

export function DateBadge({ m, compact = false }) {
  const kind = DATE_KIND_LABEL[m.dateKind] || m.dateKind
  return (
    <span className="inline-flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
      <span className={`font-mono font-semibold tabular-nums text-zinc-900 dark:text-zinc-50 ${compact ? 'text-xs' : 'text-sm'}`}>
        {fmtDate(m.date, m.datePrecision)}
      </span>
      <span className="rounded border border-zinc-300 px-1 py-px font-mono text-[10px] uppercase tracking-wide text-zinc-500 dark:border-zinc-700 dark:text-zinc-400" title="what kind of primary source the date comes from">
        {kind}
      </span>
      {m.paperDate && m.paperDate !== m.date && (
        <span className="font-mono text-[10px] text-zinc-500 dark:text-zinc-400" title="arXiv v1 date, when it differs from first public appearance">
          paper {fmtDate(m.paperDate)}
        </span>
      )}
    </span>
  )
}

export function FamilyChip({ family, small = false }) {
  const f = FAMILIES[family] || { label: family, color: '#71717a' }
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-px font-medium text-zinc-700 dark:text-zinc-200 ${small ? 'text-[10px]' : 'text-[11px]'}`}
      style={{ borderColor: f.color }}
    >
      <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ backgroundColor: f.color }} />
      {f.label}
    </span>
  )
}

export function BillChips({ bills = [], small = false }) {
  return (
    <span className="inline-flex flex-wrap gap-1">
      {bills.map((b) => {
        const d = BILLS[b] || { label: b, glyph: '·' }
        return (
          <span
            key={b}
            className={`inline-flex items-center gap-1 rounded-md bg-zinc-100 px-1.5 py-px text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300 ${small ? 'text-[10px]' : 'text-[11px]'}`}
            title={`the bill this mechanism attacks: ${d.label}`}
          >
            <span aria-hidden="true">{d.glyph}</span>
            {d.label}
          </span>
        )
      })}
    </span>
  )
}

export function ListTag({ m }) {
  if (m.instructorList) {
    return (
      <span className="rounded-md bg-brand-500/15 px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide text-brand-600 dark:text-brand-300" title="on the assignment's minimum list">
        on the list
      </span>
    )
  }
  return (
    <span className="rounded-md bg-zinc-100 px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400" title="added beyond the assignment's minimum list">
      extra
    </span>
  )
}

export const PICK_GLYPH = {
  yes: { glyph: '✓', cls: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300', label: 'yes' },
  maybe: { glyph: '△', cls: 'bg-amber-500/15 text-amber-700 dark:text-amber-300', label: 'sometimes' },
  no: { glyph: '✗', cls: 'bg-red-500/15 text-red-700 dark:text-red-300', label: 'no' },
  na: { glyph: '—', cls: 'bg-zinc-100 text-zinc-500 dark:bg-zinc-800', label: 'n/a' },
}

export function PickPill({ v, label, why }) {
  const g = PICK_GLYPH[v] || PICK_GLYPH.na
  return (
    <span className={`inline-flex items-baseline gap-1 rounded-md px-2 py-0.5 text-[11px] ${g.cls}`} title={why || g.label}>
      <span className="font-mono font-bold">{g.glyph}</span>
      <span className="font-medium">{label}</span>
      {why && <span className="opacity-80">— <Rich text={why} /></span>}
    </span>
  )
}

export function ExtLink({ href, children, className = '' }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className={`text-brand-600 hover:underline dark:text-brand-400 ${className}`}>
      {children}
    </a>
  )
}

// Minimal inline markup for the JSON prose: *emphasis* and `code`.
export function Rich({ text }) {
  if (!text) return null
  const parts = String(text).split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g)
  return (
    <>
      {parts.map((p, i) => {
        if (p.startsWith('**') && p.endsWith('**') && p.length > 4) return <b key={i}>{p.slice(2, -2)}</b>
        if (p.startsWith('*') && p.endsWith('*') && p.length > 2) return <i key={i}>{p.slice(1, -1)}</i>
        if (p.startsWith('`') && p.endsWith('`') && p.length > 2)
          return (
            <code key={i} className="rounded bg-zinc-100 px-1 font-mono text-[0.9em] dark:bg-zinc-800">
              {p.slice(1, -1)}
            </code>
          )
        return <span key={i}>{p}</span>
      })}
    </>
  )
}
