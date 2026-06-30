// Section shell shared by all four claims: anchor target, numbered badge,
// title, the claim statement in an accent callout, the interactive body, and a
// closing takeaway.
export default function ClaimCard({ id, code, accent, title, claim, children, takeaway }) {
  return (
    <section id={id} className="scroll-mt-20 py-10">
      <div className="mb-5 flex items-center gap-3">
        <span
          className="inline-flex h-8 items-center rounded-full px-3 font-mono text-xs font-semibold text-white"
          style={{ backgroundColor: accent }}
        >
          {code}
        </span>
        <h2 className="text-xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50 sm:text-2xl">
          {title}
        </h2>
      </div>

      <div
        className="mb-6 rounded-xl border-l-4 bg-white p-4 text-sm leading-relaxed text-zinc-700 shadow-sm dark:bg-zinc-900 dark:text-zinc-200"
        style={{ borderColor: accent }}
      >
        <span className="font-semibold text-zinc-900 dark:text-zinc-50">Claim. </span>
        {claim}
      </div>

      {children}

      {takeaway && (
        <div className="mt-6 rounded-xl bg-zinc-100 p-4 text-sm text-zinc-700 dark:bg-zinc-800/60 dark:text-zinc-200">
          <span className="font-semibold" style={{ color: accent }}>
            Verdict —{' '}
          </span>
          {takeaway}
        </div>
      )}
    </section>
  )
}
