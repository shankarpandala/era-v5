import { ERAS, NODES } from '../data.js'

// "Watch the field change its mind": the eras as a sequence of wants, then the
// guess the pattern licenses. The prediction is mine (the student's), written
// from the timeline, and it says so.

const CHANGES = [
  { from: 0, to: 1, text: 'quality → exactness', why: 'the Transformer bets that all-pairs attention is worth paying for parallel training' },
  { from: 1, to: 2, text: 'exactness → compute back', why: 'the n² term became the whole cost once contexts passed 1K' },
  { from: 2, to: 3, text: 'compute → positions', why: 'the approximations did not ship; the length limit came from positions instead' },
  { from: 3, to: 4, text: 'positions → exact got cheap', why: 'a kernel (FlashAttention) removed the reason to approximate' },
  { from: 4, to: 5, text: 'cheap → wants length', why: 'ChatGPT made context the product; everyone owned a 2–4K RoPE model' },
  { from: 5, to: 6, text: 'length → memory back', why: 'at 128K the KV cache, not FLOPs, set the number of users per GPU' },
  { from: 6, to: 7, text: 'memory → hybrids', why: 'linear layers for memory, sparse selection for the exact layers, positions out of the way' },
  { from: 7, to: 8, text: 'hybrids → compress, then select', why: 'DroPE removes the 2021 position lever; DeepSeek-V4 compresses the cache and lets the index choose among compressed entries; the hybrids become flagship defaults' },
]

export default function EraStory() {
  const counts = Object.fromEntries(ERAS.map((e) => [e.id, NODES.filter((m) => m.era === e.id).length]))
  return (
    <div className="space-y-6">
      <ol className="relative space-y-3 border-l-2 border-zinc-200 pl-5 dark:border-zinc-800">
        {ERAS.map((e) => (
          <li key={e.id} className="relative">
            <span className="absolute -left-[27px] top-1.5 h-3.5 w-3.5 rounded-full border-2 border-white dark:border-zinc-950" style={{ backgroundColor: e.color }} />
            <div className="flex flex-wrap items-baseline gap-2">
              <a href={`#era-${e.id}`} className="font-semibold text-zinc-900 hover:underline dark:text-zinc-50">
                {e.title}
              </a>
              <span className="font-mono text-xs text-zinc-500">
                {e.span} · {counts[e.id]} node{counts[e.id] === 1 ? '' : 's'}
              </span>
            </div>
            <div className="text-sm text-zinc-700 dark:text-zinc-200">
              wants: <b>{e.wanted}</b>
            </div>
            {(() => {
              const c = CHANGES.find((x) => x.from === e.id)
              return c ? (
                <div className="mt-1 inline-block rounded-md bg-zinc-100 px-2 py-1 text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
                  <span className="font-mono">↓ {c.text}</span> — {c.why}
                </div>
              ) : null
            })()}
          </li>
        ))}
      </ol>

      <div className="rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
        <h3 className="text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">What the pattern says comes next — my guess, from the timeline</h3>
        <p className="mt-2 text-sm leading-relaxed text-zinc-700 dark:text-zinc-200">
          Two efficiency threads ran for six years without meeting: <b>attend to few</b> (Sparse Transformer → top-k → Quest → NSA/MoBA → DSA) and{' '}
          <b>remember cheaply</b> (linear attention → delta rule → RetNet/GLA/Mamba → Gated DeltaNet → KDA). In 2025 they converged on one shape:
          most layers cheap and recurrent, a few layers exact, and the exact layers themselves reading a couple of thousand tokens chosen by a
          learned index. Every oscillation before that was hardware-triggered — FlashAttention (2022) pushed approximation out, 1M-token contexts
          pulled it back — so the honest prediction is conditional, not confident.
        </p>
        <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm text-zinc-700 dark:text-zinc-200">
          <li>
            <b>The default frontier layer stack becomes a hybrid.</b> Something like 3:1 or 7:1 cheap:exact (Qwen3-Next, Kimi Linear, MiniMax-01
            already are), and the exact layers get a DSA-style indexer so their cost is O(L·k) not O(L²). The interesting fights move to the
            ratio, the index, and how the two kinds of layer share position.
          </li>
          <li>
            <b>Positions get out of the way.</b> RoPE stays on local / recurrent layers; global layers go NoPE (Command R7B, Llama 4, SWAN,
            Kimi Linear) or lose their PE after pretraining (DroPE). "Length extension" stops being a trick applied at inference and becomes a
            recipe applied at the end of pretraining. Sinks become a learned logit (gpt-oss) or a gated softmax (Gated Attention), not a pinned token.
          </li>
          <li>
            <b>Retrofit-ability becomes a design goal.</b> GQA won because you could uptrain to it; DSA shipped because it could be
            continued-pretrained onto V3.1. Expect the next mechanisms to be judged on "can I convert the model I already have" (TransMLA,
            SWAN conversion, DroPE recalibration are early instances).
          </li>
          <li>
            <b>The next flip is a systems trick again.</b> If a kernel makes selected attention as tensor-core-friendly as dense (the NSA / DSA
            thesis), the linear layers lose some of their reason to exist at ≤256K, exactly as FlashAttention did to the 2020 approximations —
            and "attention" will mean "which 2K of the 1M do I read". If instead contexts keep outrunning hardware, the recurrent share of
            the stack grows.
          </li>
        </ol>
        <p className="mt-3 text-sm text-zinc-700 dark:text-zinc-200">
          <b>Scorecard so far (Aug 2026):</b> (1) and (4) already happened once — DeepSeek-V4 (Apr 2026) compresses every m tokens, lets the
          indexer pick the top-k compressed entries, and interleaves that with heavily-compressed layers; Qwen3.5 (Feb) and Kimi K3 (Jul) made the
          3:1 delta-rule hybrid a flagship default. (2) is half-done — NoPE global layers are common, DroPE-style removal is not yet in a frontier
          model. (3) is unproven. So the next guess, one step further: the compression ratio and the index become learned per layer, and the
          "exact" layers disappear as a category — every layer reads a compressed, selected view, at a resolution the model chooses.
        </p>
        <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">
          Caveat, honestly written: MiniMax walked back its linear hybrid in M2 (Oct 2025) citing infra maturity and eval blind spots, DroPE is
          ≤7B evidence, and DeepSeek-V4's long-context recall has only its own report behind it as of August 2026. The prediction is a bet on the
          pattern, not a report.
        </p>
      </div>
    </div>
  )
}
