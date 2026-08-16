import { ALL, DATE_KIND_LABEL, fmtDate } from '../data.js'
import { ExtLink } from './Bits.jsx'

// Dating method, the corrections the checking turned up, and the full table
// (every entry: date, kind, source, the literal evidence string).

const CORRECTIONS = [
  { what: 'NTK-aware scaled RoPE', was: '“~June 30, 2023”', is: '29 Jun 2023 08:21 UTC (Reddit post 14lz7j5, created_utc 1688026889) — a Reddit post, not a paper.' },
  { what: 'Dynamic NTK', was: '“July 2023”', is: '30 Jun 2023 05:34 UTC (post 14mrgpr), ~21 hours after bloc97’s post — still June in every US time zone.' },
  { what: 'Position Interpolation', was: 'usually dated to Meta’s paper (27 Jun 2023)', is: 'kaiokendev’s blog section is dated 20 Jun 2023 in its own changelog; the paper is a week later. Both shown.' },
  { what: 'MQA — first mainstream model', was: 'often stated as PaLM (Apr 2022)', is: 'AlphaCode, blog 2 Feb 2022 / arXiv 8 Feb 2022 (“we take advantage of multi-query attention (Shazeer 2019)”).' },
  { what: 'RoPE', was: 'arXiv 20 Apr 2021', is: 'first appeared on Su Jianlin’s blog kexue.fm/archives/8265 on 23 Mar 2021 (repo created 22 Mar); arXiv v1 is 20 Apr 2021. Both shown; the timeline sorts by first appearance.' },
  { what: 'MLA', was: 'the DeepSeek-V2 paper (7 May 2024)', is: 'the model shipped 6 May 2024, one day before the report’s arXiv v1.' },
  { what: 'Learned absolute positions', was: '“GPT/BERT (2018)”', is: 'ConvS2S v1 8 May 2017 — 35 days before the Transformer, and the reference Vaswani §3.5 cites; MemN2N’s temporal encoding (31 Mar 2015) is prehistory.' },
  { what: 'Explicit Sparse Transformer (top-k)', was: 'arXiv 25 Dec 2019', is: 'public as an ICLR 2020 OpenReview submission on 25 Sep 2019; the arXiv v1 is 25 Dec 2019 (kept as the primary, both noted).' },
  { what: 'RWKV', was: 'paper 22 May 2023', is: 'the RWKV-4 checkpoints predate the paper: HF uploads on 17 Aug 2022.' },
  { what: 'NoPE — first hybrid ship', was: 'Llama 4 iRoPE (5 Apr 2025)', is: 'Cohere Command R7B, 13 Dec 2024 (“a fourth layer uses global attention without positional embeddings”), also six weeks before the RNoPE-SWA paper.' },
  { what: 'Dual Chunk Attention — first ship', was: 'Qwen2.5-1M (Jan 2025)', is: 'Qwen2 Instruct already used YaRN + DCA for 128K (blog 7 Jun 2024; report 2407.10671 §3.2).' },
  { what: 'DeepSeek-V4 report', was: 'arXiv id 2606.19348 (a June number)', is: 'the arXiv API records v1 as published 2026-04-26; the model shipped 2026-04-24. Recorded as the sources state it; the id/month mismatch is noted in the evidence string.' },
  { what: 'DroPE', was: 'assumed a 2026 paper', is: 'arXiv v1 13 Dec 2025 (Sakana AI); public launch 12 Jan 2026. Not the same as DRoPE (Directional RoPE, Mar 2025). The arXiv title has no “DroPE:” prefix.' },
  { what: 'Sliding window', was: 'often credited to Mistral 7B (Sep 2023)', is: 'the mechanism is Longformer (10 Apr 2020), the idea older still (Image Transformer 2018, Luong local 2015); Mistral is the first mainstream decoder to ship it under that name.' },
]

const KIND_COUNTS = ALL.reduce((acc, m) => {
  acc[m.dateKind] = (acc[m.dateKind] || 0) + 1
  return acc
}, {})

export default function Sources() {
  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2">
        <div className="panel p-4">
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">How every date was determined</h3>
          <ol className="mt-2 list-decimal space-y-1.5 pl-5 text-sm text-zinc-700 dark:text-zinc-200">
            <li>
              <b>Papers:</b> the arXiv <b>v1</b> submission date, read from the literal “[v1] Submitted on …” line of the abs page (not the
              conference date, not a later revision, not the arXiv id month — 2408.00118 was submitted on 31 Jul 2024).
            </li>
            <li>
              <b>Mechanisms that first appeared elsewhere</b> (a blog, a Reddit post, a model release, a GitHub commit): the date on that primary
              page, with the paper’s arXiv v1 shown alongside as “paper …”. The timeline sorts by <i>first public appearance</i> — the
              instructor’s “the date each one actually appeared”.
            </li>
            <li>
              <b>Model releases:</b> the lab’s own announcement or model card; HF repo commit timestamps where a blog was unreachable.
            </li>
            <li>
              Every record stores the evidence string that was read. Two independent passes: a verifier fetched each source and wrote the record;
              a second, adversarial pass re-fetched 105 of the {ALL.length} records and compared to the day (0 date disagreements; the five
              attribution nits it found are folded in above). The {ALL.length - 105} records added afterwards for Oct 2025 – Aug 2026 (from Qwen3.5,
              GLM-5 and DeepSeek-V4 to Mamba-3, Gated DeltaNet-2, MiniMax M3, Gemma 4, Kimi K3 and LongCat) were checked directly against
              the arXiv API and Hugging Face repo metadata; their evidence strings say so.
            </li>
            <li>
              A third pass tried to <i>refute</i> the trade-off text: six adversarial reviewers re-read every quoted number, first-ship
              attribution and description on the 76 timeline nodes against the papers — 71 sourced findings, all applied (e.g. Falcon-40B
              shipped grouped KV heads before Llama 2 named it GQA; Jukebox used Sparse-Transformer attention before GPT-3; NSA’s 11.6× decode
              is an expected, not measured, speedup; T5’s bias extrapolates better than RoPE, not worse).
            </li>
          </ol>
          <p className="mt-2 font-mono text-[11px] text-zinc-500">
            by kind: {Object.entries(KIND_COUNTS).map(([k, v]) => `${DATE_KIND_LABEL[k] || k} ${v}`).join(' · ')}
          </p>
        </div>
        <div className="panel p-4">
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">Things the checking corrected (including my own first guesses)</h3>
          <ul className="mt-2 space-y-1.5 text-xs text-zinc-700 dark:text-zinc-200">
            {CORRECTIONS.map((c) => (
              <li key={c.what}>
                <b>{c.what}</b> — {c.was} → {c.is}
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="panel p-4">
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">Limitations, honestly</h3>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-zinc-700 dark:text-zinc-200">
          <li>“First appeared” is first <i>public</i> appearance, not invention: several ideas were in code or on OpenReview before the date shown, and internal use at labs is invisible.</li>
          <li>The cost calculator is the attention core only (idealised FLOPs and bytes); it is meant to show the shape of the curves, not to benchmark anything.</li>
          <li>The visualizers use seeded random Q/K/V on ≤64 tokens; they show the pattern each mechanism computes, not a trained model’s attention.</li>
          <li>Pros / cons / “when you would pick it” are judgements written from the papers’ own numbers and later adoption; they are opinions, and the verdict lines say so.</li>
          <li>Post-October-2025 coverage was swept twice (by hand, then by two agents once the session limit lifted) and adds 30 records — DeepSeek-V4, GLM-5, MiniMax M3, Gemma 4, Mamba-3, Gated DeltaNet-2, Kimi K3 among them — but it is still the part of the timeline most likely to be incomplete.</li>
        </ul>
      </div>

      <div className="overflow-x-auto rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
        <table className="w-full text-xs" style={{ minWidth: 820 }}>
          <thead>
            <tr className="border-b border-zinc-200 text-left text-[10px] uppercase tracking-wide text-zinc-500 dark:border-zinc-700">
              <th className="px-3 py-2">#</th>
              <th className="px-3 py-2">first appeared</th>
              <th className="px-3 py-2">kind</th>
              <th className="px-3 py-2">mechanism</th>
              <th className="px-3 py-2">source</th>
              <th className="px-3 py-2">evidence read</th>
            </tr>
          </thead>
          <tbody>
            {ALL.map((m, i) => (
              <tr key={m.id} className="border-b border-zinc-100 align-top last:border-0 dark:border-zinc-800">
                <td className="px-3 py-1.5 font-mono text-zinc-400">{i + 1}</td>
                <td className="whitespace-nowrap px-3 py-1.5 font-mono text-zinc-800 dark:text-zinc-100">
                  {fmtDate(m.date, m.datePrecision)}
                  {m.paperDate && m.paperDate !== m.date && <div className="text-[10px] text-zinc-400">paper {fmtDate(m.paperDate)}</div>}
                </td>
                <td className="whitespace-nowrap px-3 py-1.5 text-zinc-500">{DATE_KIND_LABEL[m.dateKind] || m.dateKind}</td>
                <td className="px-3 py-1.5">
                  <a href={`#m-${m.id}`} className="font-medium text-zinc-800 hover:underline dark:text-zinc-100">
                    {m.short}
                  </a>
                  {m.tier === 'footnote' && <span className="ml-1 text-[10px] text-zinc-400">(footnote)</span>}
                </td>
                <td className="px-3 py-1.5">
                  <ExtLink href={m.source.url}>{m.source.arxiv ? `arXiv ${m.source.arxiv}` : m.source.url.replace(/^https?:\/\//, '').slice(0, 48)}</ExtLink>
                </td>
                <td className="px-3 py-1.5 font-mono text-[10px] text-zinc-500 dark:text-zinc-400">{m.source.evidence}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
