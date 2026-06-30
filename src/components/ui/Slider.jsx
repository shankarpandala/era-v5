// Labeled range slider with a live value readout.
export default function Slider({ label, value, min, max, step = 1, onChange, format, disabled }) {
  const shown = format ? format(value) : value
  return (
    <label className={`block ${disabled ? 'opacity-50' : ''}`}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-xs font-medium text-zinc-600 dark:text-zinc-300">{label}</span>
        <span className="font-mono text-xs tabular-nums text-zinc-900 dark:text-zinc-100">{shown}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 w-full accent-brand-500"
      />
    </label>
  )
}
