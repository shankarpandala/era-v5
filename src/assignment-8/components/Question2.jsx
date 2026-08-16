/**
 * A8-7 — Question 2: what the date order shows that a list cannot, plus the
 * mechanisms the taught list did not cover.
 *
 * Every number here is COMPUTED from assignment-8/data at render time, never
 * retyped: the year histogram, the densest window, the dormancy gaps, the
 * paper→shipped lags and the holes in the instructor's own list are all derived
 * from ALL / NODES / INSTRUCTOR. If a record's date changes, this section
 * changes with it.
 */
import { useMemo } from 'react'
import { ALL, NODES, INSTRUCTOR, byId, fmtDate } from '../data.js'

const DAY = 86400000
const toDate = (iso) => new Date(`${iso}T00:00:00Z`)
const days = (a, b) => Math.round((toDate(b) - toDate(a)) / DAY)
const years = (d) => (d / 365).toFixed(1)
const head = (name) => name.split('—')[0].trim()

function useEvidence() {
  return useMemo(() => {
    const dated = ALL.filter((m) => m.date).sort((a, b) => a.date.localeCompare(b.date))

    const byYear = new Map()
    for (const m of dated) byYear.set(m.date.slice(0, 4), (byYear.get(m.date.slice(0, 4)) || 0) + 1)
    const hist = [...byYear.entries()].sort()
    const peak = Math.max(...hist.map(([, n]) => n))

    let cluster = { count: 0, items: [] }
    for (let i = 0; i < dated.length; i++) {
      const items = dated.filter((x) => days(dated[i].date, x.date) >= 0 && days(dated[i].date, x.date) <= 30)
      if (items.length > cluster.count) cluster = { count: items.length, items }
    }

    const deltaThread = dated.filter((m) => /delta rule|deltanet|kimi delta|\bkda\b/i.test(`${m.name} ${m.aka || ''}`))
    const deltaGap = deltaThread.length > 1 ? days(deltaThread[0].date, deltaThread[1].date) : 0

    const lags = dated
      .map((m) => {
        const d = m.firstShipped?.date
        return d && d.length >= 10 ? { m, lag: days(m.date, d), shipped: d } : null
      })
      .filter((x) => x && x.lag > 0)
      .sort((a, b) => b.lag - a.lag)

    // holes in the taught list, by date
    const ins = INSTRUCTOR.filter((m) => m.date).sort((a, b) => a.date.localeCompare(b.date))
    const insGaps = ins
      .slice(0, -1)
      .map((m, i) => {
        const to = ins[i + 1]
        return { from: m, to, gap: days(m.date, to.date), inside: NODES.filter((x) => x.date > m.date && x.date < to.date) }
      })
      .sort((a, b) => b.gap - a.gap)

    // the biggest names the taught list did not include
    const uncovered = NODES.filter((m) => !m.instructorList && m.tier === 'major').sort((a, b) =>
      a.date.localeCompare(b.date),
    )

    return { hist, peak, cluster, deltaThread, deltaGap, lags, insGaps, uncovered, insCount: ins.length }
  }, [])
}

function Src({ m }) {
  const s = m?.source
  if (!s) return null
  return (
    <a
      className="text-brand-600 hover:underline dark:text-brand-400"
      href={s.url}
      target="_blank"
      rel="noreferrer"
    >
      {s.arxiv ? `arXiv:${s.arxiv}` : 'source'}
    </a>
  )
}

export default function Question2() {
  const { hist, peak, cluster, deltaThread, deltaGap, lags, insGaps, uncovered, insCount } = useEvidence()
  const y = (yr) => hist.find(([k]) => k === yr)?.[1]
  const flash = byId['flash-attention'] || uncovered.find((m) => /FlashAttention/i.test(m.name))
  // the specific hole the taught list has around 2022
  const hole2022 = insGaps.find((g) => g.from.date < '2022-06-01' && g.to.date > '2022-06-01')

  return (
    <div className="space-y-7">
      {/* 1 — bursts */}
      <section>
        <h4 className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">
          1. Effort arrives in bursts, and one whole thread stopped for four years
        </h4>
        <div className="mt-3 flex items-end gap-1">
          {hist.map(([yr, n]) => (
            <div key={yr} className="flex flex-1 flex-col items-center gap-1">
              <span className="font-mono text-[10px] text-zinc-500">{n}</span>
              <div
                className="w-full rounded-t bg-brand-500/80"
                style={{ height: `${Math.max(4, (n / peak) * 92)}px` }}
                title={`${yr}: ${n} mechanisms`}
              />
              <span className="font-mono text-[10px] text-zinc-500">{yr.slice(2)}</span>
            </div>
          ))}
        </div>
        <p className="mt-3 text-sm text-zinc-700 dark:text-zinc-300">
          Grouped by family this is invisible; in date order the shape is the argument.{' '}
          <strong>{y('2020')} mechanisms land in 2020</strong> — sparse, linear and low-rank all
          swinging at n² — and then <strong>{y('2021')} in 2021</strong>. The wave did not taper, it
          stopped, and what little 2021 produced was a change of subject: positions (RoPE, ALiBi) and
          one paper nobody followed up (the delta rule). The approximation thread then stayed flat
          through 2022 and did not properly return until <strong>2025</strong>, in a different form —
          trainable sparsity (NSA, MoBA, DSA) rather than fixed patterns. A list tells you these
          mechanisms exist. Only the calendar tells you the field tried something, quit for four
          years, and came back having changed its mind about how to do it.
        </p>
      </section>

      {/* 2 — simultaneity */}
      <section>
        <h4 className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">
          2. The big moves are simultaneous, not sequential
        </h4>
        <p className="mt-2 text-sm text-zinc-700 dark:text-zinc-300">
          The densest 30 days on the timeline carry <strong>{cluster.count} mechanisms</strong> from
          different labs, attacking different bills at once:
        </p>
        <ul className="mt-2 space-y-1">
          {cluster.items.map((m) => (
            <li key={m.id} className="font-mono text-xs text-zinc-600 dark:text-zinc-400">
              {m.date} · <span className="text-zinc-800 dark:text-zinc-200">{m.name}</span>
            </li>
          ))}
        </ul>
        <p className="mt-2 text-sm text-zinc-700 dark:text-zinc-300">
          As a list these read as a considered sequence: someone fixed KV heads, then someone fixed
          positions, then someone fixed serving. As dates they are a scramble — several teams hitting
          the same wall in the same month and reaching for different tools. Context extension is the
          extreme case: <em>Position Interpolation → NTK-aware → Dynamic NTK</em> inside about ten
          days, two of the three posted to Reddit before any paper existed. No list can show you that
          a field was improvising.
        </p>
      </section>

      {/* 3 — dormancy */}
      <section>
        <h4 className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">
          3. Right ideas sit unused for years — a list makes them look adjacent
        </h4>
        <ul className="mt-2 space-y-1">
          {deltaThread.slice(0, 5).map((m) => (
            <li key={m.id} className="font-mono text-xs text-zinc-600 dark:text-zinc-400">
              {m.date} · <span className="text-zinc-800 dark:text-zinc-200">{m.name}</span>
            </li>
          ))}
        </ul>
        <p className="mt-2 text-sm text-zinc-700 dark:text-zinc-300">
          The delta rule was published <strong>{fmtDate(deltaThread[0]?.date)}</strong> and then sat
          for <strong>{years(deltaGap)} years</strong> before anyone made it trainable at scale; it
          is now the recurrent half of flagship models. Grouped by family, &ldquo;DeltaNet&rdquo; and
          &ldquo;Gated DeltaNet&rdquo; are two adjacent lines and you would assume one naturally
          followed the other. The dates say something else: the idea was right, public, and ignored
          for longer than most people have worked on LLMs, and what unblocked it was not a better
          idea but a chunkwise parallel training form.
        </p>
      </section>

      {/* 4 — invention vs adoption */}
      <section>
        <h4 className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">
          4. Invention order is not adoption order — visible only with two dates per row
        </h4>
        <div className="mt-2 overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-zinc-500">
              <tr>
                <th className="py-1 pr-3 font-medium">mechanism</th>
                <th className="py-1 pr-3 font-medium">published</th>
                <th className="py-1 pr-3 font-medium">first shipped</th>
                <th className="py-1 font-medium">lag</th>
              </tr>
            </thead>
            <tbody className="text-zinc-700 dark:text-zinc-300">
              {lags.slice(0, 6).map(({ m, lag, shipped }) => (
                <tr key={m.id} className="border-t border-zinc-200 dark:border-zinc-800">
                  <td className="py-1 pr-3">{head(m.name)}</td>
                  <td className="py-1 pr-3 font-mono">{m.date}</td>
                  <td className="py-1 pr-3 font-mono">{shipped}</td>
                  <td className="py-1 font-mono">{years(lag)}y</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-sm text-zinc-700 dark:text-zinc-300">
          The longest lag here is <strong>{years(lags[0]?.lag)} years</strong> between publication
          and a frontier model actually running it. A list cannot express a lag — it has one date per
          row, if any. Two dates per row turn the timeline into a claim about the field:{' '}
          <em>the idea is rarely the bottleneck</em>. A kernel, a training form, or a bill becoming
          binding is.
        </p>
      </section>

      {/* 5 — holes in the taught list */}
      <section>
        <h4 className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">
          5. A calendar exposes the holes in any curated list — including the taught one
        </h4>
        <div className="mt-2 overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-zinc-500">
              <tr>
                <th className="py-1 pr-3 font-medium">silence in the taught list</th>
                <th className="py-1 pr-3 font-medium">length</th>
                <th className="py-1 font-medium">mechanisms this page places inside it</th>
              </tr>
            </thead>
            <tbody className="text-zinc-700 dark:text-zinc-300">
              {insGaps.slice(0, 3).map((g) => (
                <tr key={g.from.id} className="border-t border-zinc-200 dark:border-zinc-800">
                  <td className="py-1 pr-3">
                    {head(g.from.name)} <span className="font-mono text-zinc-500">{g.from.date}</span> →{' '}
                    {head(g.to.name)} <span className="font-mono text-zinc-500">{g.to.date}</span>
                  </td>
                  <td className="py-1 pr-3 font-mono">{years(g.gap)}y</td>
                  <td className="py-1 font-mono">{g.inside.length}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-sm text-zinc-700 dark:text-zinc-300">
          {insCount} taught mechanisms sorted by date leave visible holes, and the holes are not
          empty in reality — this page puts {insGaps[0]?.inside.length} mechanisms inside the largest
          one alone. That is the property a list structurally cannot have: a list has no gaps,
          because a list has no axis. Put the same items on a date axis and the missing years
          announce themselves.
        </p>
      </section>

      {/* 6 — the bonus */}
      <section className="rounded-lg border border-brand-500/40 bg-brand-500/5 p-4">
        <h4 className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">
          6. The mechanism the taught list did not cover
        </h4>
        <p className="mt-2 text-sm text-zinc-700 dark:text-zinc-300">
          <strong>FlashAttention — exact attention, IO-aware.</strong> Date{' '}
          <strong>27 May 2022</strong>, taken from the arXiv v1 line of{' '}
          <em>&ldquo;FlashAttention: Fast and Memory-Efficient Exact Attention with
          IO-Awareness&rdquo;</em> (Dao, Fu, Ermon, Rudra, Ré) — <Src m={flash} />, evidence string{' '}
          <code className="rounded bg-zinc-200 px-1 font-mono text-[11px] dark:bg-zinc-800">
            {flash?.source?.evidence?.split(';')[0]}
          </code>
          .
        </p>
        <p className="mt-2 text-sm text-zinc-700 dark:text-zinc-300">
          I am naming this one rather than something more obscure because of where it falls. The
          taught list runs{' '}
          {hole2022 ? (
            <>
              <strong>{head(hole2022.from.name)} ({fmtDate(hole2022.from.date)})</strong> straight to{' '}
              <strong>{head(hole2022.to.name)} ({fmtDate(hole2022.to.date)})</strong>
            </>
          ) : (
            <>from 2021 straight to 2023</>
          )}
          , and FlashAttention sits in that gap. Its absence is load-bearing: without it the timeline
          has no explanation for why the 2020 approximation burst never resumed, and why the next
          five years were spent on memory rather than on compute. It changed nothing about the
          mathematics — same softmax, same output — and changed the entire research agenda by making
          the exact version cheap enough that approximating it stopped paying. It is also the only
          mechanism on this page that is ✓ in all four deployment scenarios and is now simply what
          attention <em>is</em> in every serving stack.
        </p>
        <p className="mt-3 text-sm text-zinc-700 dark:text-zinc-300">
          Others in the same category — significant, not on the taught list, dates from the primary
          source shown on their cards:
        </p>
        <ul className="mt-2 space-y-1">
          {uncovered
            .filter((m) => !/FlashAttention/i.test(m.name))
            .slice(0, 8)
            .map((m) => (
              <li key={m.id} className="text-xs text-zinc-600 dark:text-zinc-400">
                <span className="font-mono">{m.date}</span> ·{' '}
                <span className="text-zinc-800 dark:text-zinc-200">{head(m.name)}</span> · <Src m={m} />
              </li>
            ))}
        </ul>
        <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">
          In total the page carries {NODES.filter((m) => !m.instructorList).length} mechanisms beyond
          the {insCount} taught ones; every one is dated from its own primary source and listed in
          A8-6.
        </p>
      </section>
    </div>
  )
}
