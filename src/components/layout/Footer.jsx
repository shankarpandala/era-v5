export default function Footer({ note }) {
  return (
    <footer className="mt-16 border-t border-zinc-200 py-8 dark:border-zinc-800">
      <div className="mx-auto max-w-6xl px-4 text-center text-xs text-zinc-500 dark:text-zinc-400">
        <p>
          {note ??
            'ERA-V5 · Assignment 1 — interactive proofs. Every model trains live in your browser with a tiny hand-written neural net; nothing is precomputed.'}
        </p>
        <p className="mt-1">
          Built by{' '}
          <a
            href="https://shankarpandala.github.io/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-brand-600 hover:underline dark:text-brand-400"
          >
            Shankar Pandala
          </a>{' '}
          · The School of AI
        </p>
      </div>
    </footer>
  )
}
