import { useMemo, useState } from 'react'
import Slider from '../../components/ui/Slider.jsx'
import LineChart from '../../components/ui/LineChart.jsx'
import { PRESETS, modelTotals, layoutComparison, fmtBytes, fmtFlops, fmtTokens } from '../lib/costModel.js'

// The bill, in numbers: attention FLOPs for a prefill and KV bytes for a
// context, per preset architecture, as the context grows from 128 to 1M.
// The two fixed columns answer the instructor's question directly: what is the
// bill for a 2K chatbot, and for a 1M agent?

const N_STOPS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#d946ef']

export default function CostCalculator() {
  const [nIdx, setNIdx] = useState(11) // 262144
  const [layers, setLayers] = useState(32)
  const [heads, setHeadsRaw] = useState(32)
  const [dHead, setDHead] = useState(128)
  const [groups, setGroupsRaw] = useState(8)
  const setHeads = (h) => {
    setHeadsRaw(h)
    setGroupsRaw((g) => Math.min(g, h))
  }
  const setGroups = (g) => setGroupsRaw(Math.min(g, heads))
  const [latent, setLatent] = useState(512)
  const [window, setWindow] = useState(4096)
  const [bytes, setBytes] = useState(2)
  const n = N_STOPS[nIdx]

  const kvRows = useMemo(() => layoutComparison({ layers, heads, dHead, bytes, groups, latent, ropeDim: 64, window, sinks: 4, linearDim: 128 }, n), [layers, heads, dHead, bytes, groups, latent, window, n])
  const kv2k = useMemo(() => layoutComparison({ layers, heads, dHead, bytes, groups, latent, ropeDim: 64, window, sinks: 4, linearDim: 128 }, 2048), [layers, heads, dHead, bytes, groups, latent, window])
  const kv1m = useMemo(() => layoutComparison({ layers, heads, dHead, bytes, groups, latent, ropeDim: 64, window, sinks: 4, linearDim: 128 }, 1048576), [layers, heads, dHead, bytes, groups, latent, window])

  const series = useMemo(
    () =>
      PRESETS.map((p, i) => ({
        label: p.label,
        color: COLORS[i % COLORS.length],
        data: N_STOPS.map((x) => ({ x, y: Math.log10(Math.max(1, modelTotals(p, x, bytes).kv)) })),
      })),
    [bytes],
  )
  const flopSeries = useMemo(
    () =>
      PRESETS.map((p, i) => ({
        label: p.label,
        color: COLORS[i % COLORS.length],
        data: N_STOPS.map((x) => ({ x, y: Math.log10(Math.max(1, modelTotals(p, x, bytes).flops)) })),
      })),
    [bytes],
  )

  return (
    <div className="space-y-4">
      <div className="panel p-4">
        <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
          <div className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">Real layouts, whole model: KV cache and attention FLOPs vs context</div>
          <div className="text-[11px] text-zinc-500">bf16 unless changed · idealised: attention core only</div>
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          <div>
            <div className="mb-1 font-mono text-[11px] text-zinc-500">log₁₀ KV bytes held for a context of n tokens</div>
            <LineChart width={420} height={220} xLabel="context n: 128 → 1M (log scale)" yLabel="log10 bytes" logX xMin={128} xMax={1048576} yMin={5} yMax={13} series={series} showLegend={false} />
          </div>
          <div>
            <div className="mb-1 font-mono text-[11px] text-zinc-500">log₁₀ attention FLOPs to prefill n tokens</div>
            <LineChart width={420} height={220} xLabel="context n: 128 → 1M (log scale)" yLabel="log10 FLOP" logX xMin={128} xMax={1048576} yMin={9} yMax={20} series={flopSeries} showLegend={false} />
          </div>
        </div>
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-zinc-500 dark:text-zinc-400">
          {PRESETS.map((p, i) => (
            <span key={p.key} className="inline-flex items-center gap-1">
              <span className="inline-block h-2 w-4 rounded-sm" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
              {p.label}
            </span>
          ))}
        </div>
        <div className="mt-3 overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
          <table className="w-full text-xs" style={{ minWidth: 640 }}>
            <thead>
              <tr className="border-b border-zinc-200 text-left text-[10px] uppercase tracking-wide text-zinc-500 dark:border-zinc-800">
                <th className="px-2 py-1.5">preset</th>
                <th className="px-2 py-1.5">KV @ 2K</th>
                <th className="px-2 py-1.5">KV @ 128K</th>
                <th className="px-2 py-1.5">KV @ 1M</th>
                <th className="px-2 py-1.5">attn FLOPs @ 2K</th>
                <th className="px-2 py-1.5">attn FLOPs @ 1M</th>
              </tr>
            </thead>
            <tbody>
              {PRESETS.map((p) => {
                const a = modelTotals(p, 2048, bytes)
                const b = modelTotals(p, 131072, bytes)
                const c = modelTotals(p, 1048576, bytes)
                return (
                  <tr key={p.key} className="border-b border-zinc-100 last:border-0 dark:border-zinc-800">
                    <td className="px-2 py-1.5 font-medium">{p.label}</td>
                    <td className="px-2 py-1.5 font-mono">{fmtBytes(a.kv)}</td>
                    <td className="px-2 py-1.5 font-mono">{fmtBytes(b.kv)}</td>
                    <td className="px-2 py-1.5 font-mono">{fmtBytes(c.kv)}</td>
                    <td className="px-2 py-1.5 font-mono">{fmtFlops(a.flops)}</td>
                    <td className="px-2 py-1.5 font-mono">{fmtFlops(c.flops)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel p-4">
        <div className="mb-2 text-sm font-semibold text-zinc-800 dark:text-zinc-100">Build your own: one hypothetical model, every KV layout</div>
        <div className="grid gap-4 md:grid-cols-[1fr_auto]">
          <div className="overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
            <table className="w-full text-xs" style={{ minWidth: 560 }}>
              <thead>
                <tr className="border-b border-zinc-200 text-left text-[10px] uppercase tracking-wide text-zinc-500 dark:border-zinc-800">
                  <th className="px-2 py-1.5">layout</th>
                  <th className="px-2 py-1.5">bytes / token / layer</th>
                  <th className="px-2 py-1.5">2K chatbot</th>
                  <th className="px-2 py-1.5">@ {fmtTokens(n)}</th>
                  <th className="px-2 py-1.5">1M agent</th>
                </tr>
              </thead>
              <tbody>
                {kvRows.map((r, i) => (
                  <tr key={r.key} className="border-b border-zinc-100 last:border-0 dark:border-zinc-800">
                    <td className="px-2 py-1.5 font-medium">{r.label}</td>
                    <td className="px-2 py-1.5 font-mono">{r.perTok ? r.perTok.toLocaleString() : 'state, fixed'}</td>
                    <td className="px-2 py-1.5 font-mono">{fmtBytes(kv2k[i].total)}</td>
                    <td className="px-2 py-1.5 font-mono font-semibold">{fmtBytes(r.total)}</td>
                    <td className="px-2 py-1.5 font-mono">{fmtBytes(kv1m[i].total)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="w-full space-y-2 md:w-60">
            <Slider label="context n" value={nIdx} min={0} max={N_STOPS.length - 1} onChange={setNIdx} format={(i) => fmtTokens(N_STOPS[i])} />
            <Slider label="layers" value={layers} min={8} max={128} step={8} onChange={setLayers} />
            <Slider label="query heads" value={heads} min={8} max={128} step={8} onChange={setHeads} />
            <Slider label="head dim" value={dHead} min={64} max={256} step={64} onChange={setDHead} />
            <Slider label="GQA groups" value={groups} min={1} max={heads} onChange={setGroups} />
            <Slider label="MLA latent" value={latent} min={128} max={2048} step={128} onChange={setLatent} />
            <Slider label="window" value={window} min={128} max={16384} step={128} onChange={setWindow} format={(v) => v.toLocaleString()} />
            <Slider label="bytes / value" value={bytes} min={1} max={4} step={1} onChange={setBytes} format={(v) => ({ 1: 'fp8', 2: 'bf16', 3: '3 B', 4: 'fp32' })[v]} />
          </div>
        </div>
        <p className="mt-3 text-[11px] text-zinc-500 dark:text-zinc-400">
          Read the last two columns as the instructor's question: a mechanism that is right for the 2K chatbot column and wrong for the 1M agent
          column is not a bad mechanism — it is a trade. At 2K nothing matters; at 1M the dense MHA cache is measured in terabytes and only
          bounded (window), latent (MLA) or fixed-state (linear) layouts fit on a machine.
        </p>
      </div>
    </div>
  )
}
