// Small button with a primary / ghost variant.
export default function Button({ children, onClick, variant = 'primary', disabled, className = '' }) {
  const base =
    'inline-flex items-center justify-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 disabled:opacity-50 disabled:cursor-not-allowed'
  const styles =
    variant === 'primary'
      ? 'bg-brand-500 text-white hover:bg-brand-600'
      : 'border border-zinc-300 text-zinc-700 hover:bg-zinc-100 dark:border-zinc-600 dark:text-zinc-200 dark:hover:bg-zinc-800'
  return (
    <button type="button" onClick={onClick} disabled={disabled} className={`${base} ${styles} ${className}`}>
      {children}
    </button>
  )
}
