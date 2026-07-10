import ClaimCard from '../../components/ClaimCard.jsx'
import { LANGS } from '../lib/loadData.js'

function Formula({ children }) {
  return (
    <div className="my-2 rounded-lg bg-zinc-100 px-3 py-2 font-mono text-[13px] text-zinc-800 dark:bg-zinc-800/60 dark:text-zinc-100">
      {children}
    </div>
  )
}

export default function Method({ stats, tok }) {
  const corpus = tok.corpus || {}
  const wpl = corpus.words_per_language || {}
  const weights = tok.mixing_weights || {}
  const equalWeights = Object.values(weights).every((w) => w === 1)

  return (
    <ClaimCard
      id="a2-1"
      code="A2-1"
      accent="var(--claim-1)"
      title="The method & the metric"
      claim={
        <>
          A single, from-scratch, byte-level BPE tokenizer with a shared vocabulary of{' '}
          <b>{tok.vocab_size.toLocaleString()} tokens</b> encodes India's Wikipedia page in four languages at a
          fertility of <b>≤ 1.2 tokens per word</b> in every language.
        </>
      }
      takeaway={
        <>
          A shared 10k-vocab BPE can compress four scripts to ≤ 1.2 tokens/word when trained on comparable amounts
          of each language. The tighter the four ratios cluster, the higher the score — here they span just{' '}
          <b>{stats.spread.toFixed(4)}</b>.
        </>
      }
    >
      <div className="panel p-5 text-sm leading-relaxed text-zinc-700 dark:text-zinc-200">
        <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
          <div className="space-y-3">
            <p>
              <b>Byte-Pair Encoding (BPE)</b> starts from raw UTF-8 bytes and repeatedly merges the most frequent
              adjacent pair into a new token. We train <b>one</b> tokenizer on all four languages at once, so the{' '}
              {tok.vocab_size.toLocaleString()}-token vocabulary is <b>shared</b> — every merge competes across
              English (Latin), Hindi &amp; Marathi (Devanagari) and Telugu (Telugu script). The whole trainer and
              encoder are hand-written (no <code>tokenizers</code>/<code>tiktoken</code>).
            </p>
            <p>The assignment scores how evenly the tokenizer treats each language:</p>
            <Formula>word = a whitespace-delimited run &nbsp;(len(text.split()))</Formula>
            <Formula>X(lang) = total&nbsp;tokens(lang) / total&nbsp;words(lang)</Formula>
            <Formula>spread = X_max − X_min &nbsp;&nbsp;·&nbsp;&nbsp; score = 1000 / spread</Formula>
            <p>
              Every X must be <b>≤ 1.2</b>. Because a word becomes a single token once the tokenizer has learned it,
              fertility approaches 1.0 when the shared vocabulary covers each language's frequent words — which it
              does at this corpus scale. <b>Nothing here is hardcoded:</b> the ratios are recomputed in your browser
              by running the downloadable tokenizer over the downloadable corpora, and a standalone Python reference
              (<code>evaluate.py</code>) reproduces the identical numbers.
            </p>
          </div>

          <div className="space-y-3">
            <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-700">
              <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">Corpus</div>
              <p className="text-[13px] leading-relaxed">
                India's Wikipedia article per language (MediaWiki plain-text extract). To compare fertility fairly
                across languages we use an <b>equal-size slice</b> — the first{' '}
                <span className="font-mono font-semibold">{(corpus.target_words || 0).toLocaleString()}</span> words
                of each article (naturally bounded by Telugu's short article). The exact frozen text is shipped and
                downloadable, so the numbers are fully reproducible.
              </p>
              <div className="mt-2 grid grid-cols-2 gap-1 font-mono text-[12px]">
                {LANGS.map((l) => (
                  <div key={l.code} className="flex justify-between rounded bg-zinc-100 px-2 py-1 dark:bg-zinc-800">
                    <span>{l.name}</span>
                    <span className="font-semibold">{(wpl[l.code] || 0).toLocaleString()}w</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="rounded-lg border border-zinc-200 p-3 text-[13px] dark:border-zinc-700">
              <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
                Training weights
              </div>
              {equalWeights ? (
                <p>
                  <b>Equal (1:1:1:1)</b> — no per-language up-sampling was needed; the four ratios already satisfy
                  ≤ 1.2 and cluster tightly on their own.
                </p>
              ) : (
                <p className="font-mono">
                  {LANGS.map((l) => `${l.name.slice(0, 2)}=${weights[l.code]}`).join('  ·  ')}
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </ClaimCard>
  )
}
