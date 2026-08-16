// Idealised cost model for the attention core (stated as such on the page).
//
// Two bills:
//  • attention FLOPs for a prefill of n tokens (causal), per layer, summed over
//    heads: every query pays 4·(keys it reads)·d_head per head (QKᵀ and A·V,
//    two FLOPs per multiply-add) — dense causal ⇒ Σᵢ 4·i·d = 2·n²·d with
//    d = h·d_head. Windowed / top-k / sparse variants change "keys it reads".
//    Linear / delta layers pay a per-token state update instead.
//  • KV-cache bytes held for a context of n tokens, per layer: 2 · n_kv · d_head
//    · bytes per token for MHA/GQA/MQA; (d_c + d_r) · bytes for MLA (only the
//    latent + decoupled-RoPE key is cached); bounded by the window for sliding
//    window; a fixed d_head² state per head for linear / delta layers.
// It ignores projections, MoE, softmax and the model's other layers on purpose:
// the point is the *shape* of the curve, not a benchmark.

export const DTYPE_BYTES = { fp32: 4, bf16: 2, fp8: 1 }

// One preset per architecture the timeline talks about. Values were read from
// the public config.json files below (Aug 2026) and describe the attention
// layout only:
//   Llama-2-7B / 70B   https://huggingface.co/NousResearch/Llama-2-7b-hf/blob/main/config.json (mirror of the gated meta-llama repos)
//                      https://huggingface.co/NousResearch/Llama-2-70b-hf/blob/main/config.json
//   Mistral-7B         https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json (32 L, 32 H, 8 KV, sliding_window 4096)
//   DeepSeek-V2        https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/config.json (60 L, 128 H, kv_lora_rank 512, qk_rope_head_dim 64)
//   Gemma-3-27B        https://huggingface.co/unsloth/gemma-3-27b-it/blob/main/config.json (62 L, 32 H, 16 KV, head 128, sliding_window 1024, pattern 6)
//   gpt-oss-120b       https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json (36 L, 64 H, 8 KV, head 64, sliding_window 128, 18 sliding : 18 full)
//   Qwen3-Next-80B     https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json (48 L, 16 H, 2 KV, head 256, linear 32 V / 16 K heads × 128, full_attention_interval 4)
export const PRESETS = [
  {
    key: 'llama2-7b',
    label: 'Llama-2-7B (MHA)',
    layers: 32,
    heads: 32,
    dHead: 128,
    kvHeads: 32,
    scheme: 'dense',
  },
  {
    key: 'llama2-70b',
    label: 'Llama-2-70B (GQA-8)',
    layers: 80,
    heads: 64,
    dHead: 128,
    kvHeads: 8,
    scheme: 'dense',
  },
  {
    key: 'mistral-7b',
    label: 'Mistral-7B (GQA-8 + SWA 4096)',
    layers: 32,
    heads: 32,
    dHead: 128,
    kvHeads: 8,
    scheme: 'window',
    window: 4096,
  },
  {
    key: 'deepseek-v2',
    label: 'DeepSeek-V2 (MLA, d_c 512 + d_r 64)',
    layers: 60,
    heads: 128,
    dHead: 128,
    kvHeads: 128,
    scheme: 'mla',
    latent: 512,
    ropeDim: 64,
  },
  {
    key: 'gemma3-27b',
    label: 'Gemma-3-27B (5:1 local 1024 : global)',
    layers: 62,
    heads: 32,
    dHead: 128,
    kvHeads: 16,
    scheme: 'local-global',
    window: 1024,
    ratio: 5,
  },
  {
    key: 'gpt-oss-120b',
    label: 'gpt-oss-120b (1:1 banded 128 : dense, GQA-8)',
    layers: 36,
    heads: 64,
    dHead: 64,
    kvHeads: 8,
    scheme: 'local-global',
    window: 128,
    ratio: 1,
  },
  {
    key: 'qwen3-next',
    label: 'Qwen3-Next-80B (3:1 Gated DeltaNet : attention)',
    layers: 48,
    heads: 16,
    dHead: 256,
    kvHeads: 2,
    scheme: 'linear-hybrid',
    ratio: 3,
    linearHeads: 32,
    linearDim: 128,
  },
]

// ---- attention FLOPs (prefill, causal) per layer ---------------------------

// Sum over queries i=1..n of min(i, cap) — closed form.
function sumMin(n, cap) {
  if (cap >= n) return (n * (n + 1)) / 2
  return (cap * (cap + 1)) / 2 + (n - cap) * cap
}

export function denseFlops(n, d) {
  return 4 * d * ((n * (n + 1)) / 2)
}
export function windowFlops(n, d, w) {
  return 4 * d * sumMin(n, w)
}
export function topkFlops(n, d, k, indexerDim = 128) {
  // read k keys per query + an indexer that scores every key in a small dim
  return 4 * d * sumMin(n, k) + 2 * indexerDim * ((n * (n + 1)) / 2)
}
export function nsaFlops(n, d, { stride = 16, selected = 16 * 64, window = 512 } = {}) {
  // compressed branch reads i/stride keys, selected branch a fixed block budget, window branch w
  const compressed = 4 * d * ((n * (n + 1)) / 2 / stride)
  return compressed + 4 * d * sumMin(n, selected) + 4 * d * sumMin(n, window)
}
export function linearFlops(n, heads, dHead, { delta = false } = {}) {
  // per token per head: write k·vᵀ into a dHead×dHead state and read S·q — ~4·dHead²;
  // the delta rule reads S·k first and writes a correction: ~2×.
  const perTok = (delta ? 8 : 4) * dHead * dHead * heads
  return n * perTok
}

// ---- KV bytes for a context of n tokens, per layer ------------------------

export function kvBytesDense(n, kvHeads, dHead, bytes) {
  return n * 2 * kvHeads * dHead * bytes
}
export function kvBytesWindow(n, kvHeads, dHead, bytes, w, sinks = 0) {
  return Math.min(n, w + sinks) * 2 * kvHeads * dHead * bytes
}
export function kvBytesMLA(n, latent, ropeDim, bytes) {
  return n * (latent + ropeDim) * bytes
}
export function kvBytesLinear(heads, dHead, bytes) {
  return heads * dHead * dHead * bytes
}

// ---- whole-model totals for the calculator --------------------------------

export function modelTotals(p, n, bytes = 2, opts = {}) {
  const d = p.heads * p.dHead
  const L = p.layers
  let flops = 0
  let kv = 0
  switch (p.scheme) {
    case 'dense':
      flops = L * denseFlops(n, d)
      kv = L * kvBytesDense(n, p.kvHeads, p.dHead, bytes)
      break
    case 'window':
      flops = L * windowFlops(n, d, p.window)
      kv = L * kvBytesWindow(n, p.kvHeads, p.dHead, bytes, p.window)
      break
    case 'mla':
      flops = L * denseFlops(n, d)
      kv = L * kvBytesMLA(n, p.latent, p.ropeDim, bytes)
      break
    case 'local-global': {
      const local = Math.round((L * p.ratio) / (p.ratio + 1))
      const glob = L - local
      flops = local * windowFlops(n, d, p.window) + glob * denseFlops(n, d)
      kv = local * kvBytesWindow(n, p.kvHeads, p.dHead, bytes, p.window) + glob * kvBytesDense(n, p.kvHeads, p.dHead, bytes)
      break
    }
    case 'linear-hybrid': {
      const lin = Math.round((L * p.ratio) / (p.ratio + 1))
      const att = L - lin
      flops = lin * linearFlops(n, p.linearHeads, p.linearDim, { delta: true }) + att * denseFlops(n, d)
      kv = lin * kvBytesLinear(p.linearHeads, p.linearDim, bytes) + att * kvBytesDense(n, p.kvHeads, p.dHead, bytes)
      break
    }
    case 'topk': {
      flops = L * topkFlops(n, d, opts.k ?? 2048)
      kv = L * kvBytesMLA(n, p.latent ?? 512, p.ropeDim ?? 64, bytes)
      break
    }
    default:
      flops = L * denseFlops(n, d)
      kv = L * kvBytesDense(n, p.kvHeads, p.dHead, bytes)
  }
  return { flops, kv }
}

// Compare the KV-layout family on one hypothetical model at context n.
export function layoutComparison({ layers, heads, dHead, bytes = 2, groups = 8, latent = 512, ropeDim = 64, window = 4096, sinks = 4, linearDim = 128 }, n) {
  const rows = [
    { key: 'mha', label: 'MHA', perTok: 2 * heads * dHead * bytes, total: layers * kvBytesDense(n, heads, dHead, bytes) },
    { key: 'gqa', label: `GQA (${groups} KV heads)`, perTok: 2 * groups * dHead * bytes, total: layers * kvBytesDense(n, groups, dHead, bytes) },
    { key: 'mqa', label: 'MQA (1 KV head)', perTok: 2 * dHead * bytes, total: layers * kvBytesDense(n, 1, dHead, bytes) },
    { key: 'mla', label: `MLA (latent ${latent} + rope ${ropeDim})`, perTok: (latent + ropeDim) * bytes, total: layers * kvBytesMLA(n, latent, ropeDim, bytes) },
    { key: 'swa', label: `Sliding window ${window} + ${sinks} sinks (GQA)`, perTok: 2 * groups * dHead * bytes, total: layers * kvBytesWindow(n, groups, dHead, bytes, window, sinks) },
    { key: 'linear', label: `Linear / delta state (${heads}×${linearDim}²)`, perTok: 0, total: layers * kvBytesLinear(heads, linearDim, bytes) },
  ]
  return rows
}

export function fmtBytes(b) {
  if (!Number.isFinite(b)) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
  let i = 0
  let v = b
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)} ${units[i]}`
}

export function fmtFlops(f) {
  if (!Number.isFinite(f)) return '—'
  const units = ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']
  let i = 0
  let v = f
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000
    i++
  }
  return `${v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)} ${units[i]}FLOP`
}

export function fmtTokens(n) {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n % 1_000_000 ? 1 : 0)}M`
  if (n >= 1024) return `${Math.round(n / 1024)}K`
  return String(n)
}
