import { useState } from 'react'
import Slider from '../../components/ui/Slider.jsx'
import { fmtBytes } from '../lib/costModel.js'

// The KV-layout family: how many key/value vectors are stored per token, and
// therefore how big the cache is. MHA → MQA → GQA → MLA → CLA is one story:
// shrink the head axis, then the latent, then the layer axis. FlashAttention
// and PagedAttention are drawn as "same math, different plumbing".

function HeadRow({ label, count, color, note, small = false }) {
  const shown = Math.min(count, 32)
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 shrink-0 font-mono text-[11px] text-zinc-500">{label}</div>
      <div className="flex flex-wrap gap-[3px]">
        {Array.from({ length: shown }, (_, i) => (
          <span
            key={i}
            className={`${small ? 'h-3 w-2' : 'h-4 w-3'} rounded-sm`}
            style={{ backgroundColor: color, opacity: 0.85 }}
          />
        ))}
        {count > shown && <span className="font-mono text-[10px] text-zinc-400">+{count - shown}</span>}
      </div>
      {note && <span className="text-[11px] text-zinc-500">{note}</span>}
    </div>
  )
}

function Box({ children, className = '' }) {
  return <div className={`rounded-lg border border-zinc-200 bg-white p-3 dark:border-zinc-800 dark:bg-zinc-900 ${className}`}>{children}</div>
}

export default function KVCacheHeads({ mode = 'gqa' }) {
  const [heads, setHeads] = useState(32)
  const [groups, setGroups] = useState(8)
  const [dHead, setDHead] = useState(128)
  const [latent, setLatent] = useState(512)
  const [ropeDim, setRopeDim] = useState(64)
  const [layers, setLayers] = useState(32)
  const [share, setShare] = useState(2)
  const bytes = 2
  const n = 32768

  const perTokMHA = 2 * heads * dHead * bytes
  const perTokMQA = 2 * dHead * bytes
  const perTokGQA = 2 * groups * dHead * bytes
  const perTokMLA = (latent + ropeDim) * bytes
  const perTokCLA = perTokGQA / share

  const rows = {
    mha: { label: 'MHA', perTok: perTokMHA, kvHeads: heads, desc: 'every query head has its own K and V head' },
    mqa: { label: 'MQA', perTok: perTokMQA, kvHeads: 1, desc: 'one K and one V head shared by all query heads' },
    gqa: { label: 'GQA', perTok: perTokGQA, kvHeads: groups, desc: `${groups} K/V heads, each shared by ${Math.max(1, Math.round(heads / groups))} query heads` },
    mla: { label: 'MLA', perTok: perTokMLA, kvHeads: null, desc: `one latent c of ${latent} dims (+ ${ropeDim} decoupled RoPE dims) per token; K and V are up-projected from it` },
    cla: { label: 'GQA + CLA', perTok: perTokCLA, kvHeads: groups, desc: `${groups} K/V heads, and each K/V shared by ${share} adjacent layers` },
  }
  const focus = rows[mode] || rows.gqa

  if (mode === 'flash' || mode === 'paged') {
    return (
      <div className="panel p-4">
        {mode === 'flash' ? (
          <div className="grid gap-4 md:grid-cols-2">
            <Box>
              <div className="mb-2 text-xs font-semibold text-zinc-700 dark:text-zinc-200">Before: materialise the n×n matrix</div>
              <div className="grid grid-cols-[auto_1fr] items-center gap-2 text-[11px] text-zinc-500">
                <span className="font-mono">HBM</span>
                <div className="rounded border border-zinc-300 p-2 dark:border-zinc-700">
                  Q, K, V → <span className="rounded bg-red-500/20 px-1 text-red-600 dark:text-red-300">S = QKᵀ (n×n)</span> → write · read →{' '}
                  <span className="rounded bg-red-500/20 px-1 text-red-600 dark:text-red-300">P = softmax(S) (n×n)</span> → write · read → O = PV
                </div>
              </div>
              <p className="mt-2 text-[11px] text-zinc-500">Two n×n round-trips to slow memory per head per layer. Bandwidth-bound; O(n²) memory.</p>
            </Box>
            <Box>
              <div className="mb-2 text-xs font-semibold text-zinc-700 dark:text-zinc-200">FlashAttention: tile it, keep it in SRAM</div>
              <div className="grid grid-cols-6 gap-1">
                {Array.from({ length: 24 }, (_, i) => (
                  <div
                    key={i}
                    className="aspect-square rounded-sm"
                    style={{ backgroundColor: i % 7 === 0 ? '#3b82f6' : i % 5 === 0 ? '#8b5cf6' : 'rgba(59,130,246,0.25)' }}
                  />
                ))}
              </div>
              <p className="mt-2 text-[11px] text-zinc-500">
                Load one block of Q and stream blocks of K, V through on-chip SRAM; keep a running max and sum (online softmax); write only the
                output. Same numbers, O(n) memory, 2–4× wall-clock. Recompute in the backward pass instead of storing P.
              </p>
            </Box>
            <p className="text-xs text-zinc-600 md:col-span-2 dark:text-zinc-300">
              What it does <b>not</b> change: the FLOPs are still O(n²·d) and the KV cache is untouched — which is why the 2023–25 mechanisms exist at all.
            </p>
          </div>
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            <Box>
              <div className="mb-2 text-xs font-semibold text-zinc-700 dark:text-zinc-200">Before: one contiguous slab per request</div>
              <div className="space-y-1">
                {[0.9, 0.35, 0.6].map((f, i) => (
                  <div key={i} className="h-4 w-full rounded bg-zinc-200 dark:bg-zinc-800">
                    <div className="h-4 rounded bg-amber-500/70" style={{ width: `${f * 100}%` }} />
                  </div>
                ))}
              </div>
              <p className="mt-2 text-[11px] text-zinc-500">Reserve the maximum length up front: 60–80% of GPU memory sits empty (fragmentation + over-reservation).</p>
            </Box>
            <Box>
              <div className="mb-2 text-xs font-semibold text-zinc-700 dark:text-zinc-200">PagedAttention: fixed blocks + a block table</div>
              <div className="grid grid-cols-8 gap-1">
                {Array.from({ length: 32 }, (_, i) => (
                  <div
                    key={i}
                    className="h-4 rounded-sm"
                    style={{ backgroundColor: [3, 9, 14, 20, 27].includes(i) ? 'rgba(0,0,0,0.06)' : ['#3b82f6', '#10b981', '#f59e0b'][i % 3], opacity: 0.8 }}
                  />
                ))}
              </div>
              <p className="mt-2 text-[11px] text-zinc-500">
                KV lives in non-contiguous pages; a per-sequence table maps logical → physical. &lt;4% waste, copy-on-write prefix sharing, 2–4× throughput.
              </p>
            </Box>
            <p className="text-xs text-zinc-600 md:col-span-2 dark:text-zinc-300">
              A serving change, not a maths change — but it made "KV bytes" the number every 2024 mechanism is measured against.
            </p>
          </div>
        )}
      </div>
    )
  }

  const total = (perTok) => fmtBytes(perTok * layers * n)

  return (
    <div className="panel p-4">
      <div className="grid gap-4 md:grid-cols-[1fr_auto]">
        <div className="space-y-3">
          <Box>
            <div className="mb-2 text-xs font-semibold text-zinc-700 dark:text-zinc-200">
              {focus.label}: {focus.desc}
            </div>
            <div className="space-y-1.5">
              <HeadRow label="Q heads" count={heads} color="#3b82f6" />
              {focus.kvHeads != null ? (
                <>
                  <HeadRow label="K heads" count={focus.kvHeads} color="#f59e0b" note={mode === 'cla' ? `(shared across ${share} layers)` : ''} />
                  <HeadRow label="V heads" count={focus.kvHeads} color="#10b981" />
                </>
              ) : (
                <>
                  <HeadRow label="latent c" count={Math.max(1, Math.round(latent / dHead))} color="#d946ef" note={`${latent} dims (≈ ${(latent / dHead).toFixed(1)} heads' worth)`} />
                  <HeadRow label="rope k" count={1} color="#f59e0b" small note={`${ropeDim} decoupled RoPE dims`} />
                </>
              )}
            </div>
          </Box>
          <div className="overflow-x-auto rounded-lg border border-zinc-200 dark:border-zinc-800">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-zinc-200 text-left text-[10px] uppercase tracking-wide text-zinc-500 dark:border-zinc-800">
                  <th className="px-2 py-1.5">layout</th>
                  <th className="px-2 py-1.5">KV bytes / token / layer</th>
                  <th className="px-2 py-1.5">× vs MHA</th>
                  <th className="px-2 py-1.5">cache @ 32K, {layers} layers</th>
                </tr>
              </thead>
              <tbody>
                {['mha', 'mqa', 'gqa', 'mla', 'cla'].map((k) => (
                  <tr key={k} className={`border-b border-zinc-100 last:border-0 dark:border-zinc-800 ${k === mode ? 'bg-brand-50 dark:bg-brand-900/30' : ''}`}>
                    <td className="px-2 py-1.5 font-medium">{rows[k].label}</td>
                    <td className="px-2 py-1.5 font-mono">{rows[k].perTok.toLocaleString()}</td>
                    <td className="px-2 py-1.5 font-mono">{(perTokMHA / rows[k].perTok).toFixed(1)}×</td>
                    <td className="px-2 py-1.5 font-mono">{total(rows[k].perTok)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="w-full space-y-2 md:w-56">
          <Slider label="query heads h" value={heads} min={8} max={128} step={8} onChange={setHeads} />
          <Slider label="head dim" value={dHead} min={64} max={256} step={64} onChange={setDHead} />
          <Slider label="KV groups (GQA)" value={groups} min={1} max={heads} step={1} onChange={setGroups} />
          <Slider label="MLA latent d_c" value={latent} min={128} max={2048} step={128} onChange={setLatent} />
          <Slider label="MLA rope dims" value={ropeDim} min={0} max={128} step={16} onChange={setRopeDim} />
          <Slider label="CLA share factor" value={share} min={1} max={4} step={1} onChange={setShare} />
          <Slider label="layers" value={layers} min={8} max={128} step={8} onChange={setLayers} />
        </div>
      </div>
      <p className="mt-3 text-[11px] text-zinc-500 dark:text-zinc-400">
        bf16, per token per layer: MHA 2·h·d_head·2 B; GQA 2·g·d_head·2 B; MQA 2·d_head·2 B; MLA (d_c + d_r)·2 B (only the latent and the
        decoupled RoPE key are cached — the up-projections are absorbed into W_Q / W_O at decode); CLA divides by the number of layers sharing a K/V.
        DeepSeek-V2's numbers (h=128, d_head=128, d_c=512, d_r=64) give 57× vs MHA — "GQA with 2.25 groups" in the paper's words.
      </p>
    </div>
  )
}
