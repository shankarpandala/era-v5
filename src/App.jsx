import Navbar from './components/layout/Navbar.jsx'
import Footer from './components/layout/Footer.jsx'
import useTheme from './hooks/useTheme.js'
import S1RingActivation from './claims/S1RingActivation.jsx'
import S1DepthLinear from './claims/S1DepthLinear.jsx'
import S1Embeddings from './claims/S1Embeddings.jsx'
import S1MemorizeGeneralize from './claims/S1MemorizeGeneralize.jsx'

const SECTIONS = [
  { id: 's1-1', code: 'S1-1', title: 'Activations exist for a reason', color: 'var(--claim-1)' },
  { id: 's1-2', code: 'S1-2', title: 'Depth without nonlinearity is a lie', color: 'var(--claim-2)' },
  { id: 's1-3', code: 'S1-3', title: 'Embeddings learn similarity from next-token', color: 'var(--claim-3)' },
  { id: 's1-4', code: 'S1-4', title: 'Memorization vs generalization', color: 'var(--claim-4)' },
]

function Hero() {
  return (
    <header id="top" className="pt-12 pb-2">
      <p className="font-mono text-xs uppercase tracking-widest text-brand-500">ERA-V5 · The School of AI</p>
      <h1 className="mt-2 text-3xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50 sm:text-4xl">
        Assignment 1
      </h1>

      <nav aria-label="Claims" className="mt-6 grid gap-2 sm:grid-cols-2">
        {SECTIONS.map((s) => (
          <a
            key={s.id}
            href={`#${s.id}`}
            className="group flex items-center gap-3 rounded-xl border border-zinc-200 bg-white p-3 transition-colors hover:border-zinc-300 hover:bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:bg-zinc-800/60"
          >
            <span
              className="inline-flex h-7 shrink-0 items-center rounded-full px-2.5 font-mono text-[11px] font-semibold text-white"
              style={{ backgroundColor: s.color }}
            >
              {s.code}
            </span>
            <span className="text-sm font-medium text-zinc-700 group-hover:text-zinc-900 dark:text-zinc-200">
              {s.title}
            </span>
          </a>
        ))}
      </nav>
    </header>
  )
}

export default function App() {
  const [theme, toggleTheme] = useTheme()

  return (
    <div className="flex min-h-screen flex-col">
      <Navbar theme={theme} onToggleTheme={toggleTheme} label="Assignment 1" />
      <main className="mx-auto w-full max-w-6xl flex-1 px-4">
        <Hero />
        <div className="divide-y divide-zinc-200 dark:divide-zinc-800">
          <S1RingActivation />
          <S1DepthLinear />
          <S1Embeddings />
          <S1MemorizeGeneralize />
        </div>
      </main>
      <Footer />
    </div>
  )
}
