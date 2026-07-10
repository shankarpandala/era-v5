// Header, adapted from the math4ai reference: same look (node-graph mark, ~/pandala.in
// tag, theme toggle, portfolio + LinkedIn + GitHub links) but rebranded to
// ERA-V5 · Assignment 1, with the react-router center nav removed (this is a
// single page — the logo is a plain anchor).

function SunIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="5" />
      <line x1="12" y1="1" x2="12" y2="3" />
      <line x1="12" y1="21" x2="12" y2="23" />
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
      <line x1="1" y1="12" x2="3" y2="12" />
      <line x1="21" y1="12" x2="23" y2="12" />
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  )
}

function GitHubIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.009-.868-.013-1.703-2.782.604-3.369-1.341-3.369-1.341-.454-1.154-1.11-1.462-1.11-1.462-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0 1 12 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.161 22 16.416 22 12c0-5.523-4.477-10-10-10z" />
    </svg>
  )
}

function LinkedInIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
    </svg>
  )
}

function UserIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  )
}

export default function Navbar({ theme, onToggleTheme, label = 'Assignment 1', crossLink = null }) {
  return (
    <header
      className="sticky top-0 z-50 h-14 w-full border-b border-zinc-200 bg-white/80 backdrop-blur-md dark:border-zinc-800 dark:bg-zinc-950/80"
      role="banner"
    >
      <div className="mx-auto flex h-full max-w-6xl items-center justify-between gap-3 px-4">
        {/* Left: brand */}
        <div className="flex items-center gap-3">
          <a
            href="https://www.pandala.in"
            className="hidden items-center font-mono text-sm text-zinc-500 transition-opacity hover:opacity-80 sm:flex"
          >
            ~/<span className="text-[#5ce0d8]">pandala.in</span>
          </a>
          <span className="hidden select-none text-zinc-300 dark:text-[#2d3a4d] sm:inline" aria-hidden="true">
            |
          </span>
          <a href="#top" className="flex items-center gap-2 select-none">
            {/* Node-graph mark — a small attention-style graph; this course is
                "building LLMs from scratch", so the app gets its own logo. */}
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <defs>
                <linearGradient id="logoGrad" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0" stopColor="#3b82f6" />
                  <stop offset="1" stopColor="#7c3aed" />
                </linearGradient>
              </defs>
              <g stroke="url(#logoGrad)" strokeWidth="1.4" opacity="0.85">
                <line x1="5" y1="6" x2="19" y2="6" />
                <line x1="5" y1="6" x2="19" y2="18" />
                <line x1="5" y1="18" x2="19" y2="6" />
                <line x1="5" y1="18" x2="19" y2="18" />
              </g>
              <g fill="url(#logoGrad)">
                <circle cx="5" cy="6" r="2.6" />
                <circle cx="5" cy="18" r="2.6" />
                <circle cx="19" cy="6" r="2.6" />
                <circle cx="19" cy="18" r="2.6" />
              </g>
            </svg>
            <span className="bg-gradient-to-r from-blue-500 to-violet-600 bg-clip-text text-lg font-bold tracking-tight text-transparent">
              ERA-V5
            </span>
            <span className="hidden text-xs font-medium text-zinc-400 sm:inline dark:text-zinc-500">· {label}</span>
          </a>
        </div>

        {/* Right: theme + links */}
        <div className="flex items-center gap-1">
          {crossLink && (
            <a
              href={crossLink.href}
              className="mr-1 hidden rounded-md px-2.5 py-1.5 text-sm font-medium text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800 sm:inline-flex"
            >
              {crossLink.text}
            </a>
          )}
          <button
            type="button"
            onClick={onToggleTheme}
            className="rounded-md p-2 text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
            aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
          </button>

          <a
            href="https://shankarpandala.github.io/"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm font-medium text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800 sm:inline-flex"
            aria-label="Shankar Pandala's portfolio"
          >
            <UserIcon />
            <span>Shankar Pandala</span>
          </a>

          <a
            href="https://www.linkedin.com/in/shankarpandala/"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-md p-2 text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800"
            aria-label="LinkedIn profile"
          >
            <LinkedInIcon />
          </a>

          <a
            href="https://github.com/shankarpandala/era-v5"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-md p-2 text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800"
            aria-label="View on GitHub"
          >
            <GitHubIcon />
          </a>
        </div>
      </div>
    </header>
  )
}
