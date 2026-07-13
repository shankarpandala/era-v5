import ClaimCard from '../../components/ClaimCard.jsx'
import { LANGS } from '../lib/loadData.js'

function Formula({ children }) {
  return (
    <div className="my-2 rounded-lg bg-zinc-100 px-3 py-2 font-mono text-[13px] text-zinc-800 dark:bg-zinc-800/60 dark:text-zinc-100">
      {children}
    </div>
  )
}

export default function Method({ stats, metrics, tok, sample }) {
  const dec = tok.decode(tok.encode(sample))
  return (
    <ClaimCard
      id="a2-1"
      code="A2-1"
      accent="var(--claim-1)"
      title="The method & the metric"
      claim={
        <>
          A single shared <b>{tok.vocabSize.toLocaleString()}-token</b> tokenizer over the{' '}
          <b>wiki-faithful Markdown</b> India pages (links, URLs, tables, references preserved) that is{' '}
          <b>faithful</b>: decode(encode(text)) keeps every visible character.
        </>
      }
      takeaway={
        <>
          Grader-compatible by construction: the shipped tokenizer.json loads with{' '}
          <code>tokenizers.Tokenizer.from_file</code>, and the published course evaluator re-runs on these exact
          artifacts unchanged.
        </>
      }
    >
      <div className="panel p-5 text-sm leading-relaxed text-zinc-700 dark:text-zinc-200">
        <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
          <div className="space-y-3">
            <p>
              <b>Corpus.</b> Wikipedia REST HTML → strip only script/style/meta → <b>markdownify</b>. Links, URLs,
              tables, reference lists, image links and categories all stay — the tokenizer must represent the page
              faithfully, not a clipped version. The exact snapshots are committed and downloadable.
            </p>
            <p>
              <b>Tokenizer.</b> HuggingFace BPE, vocab 10,000, min_frequency 1, <code>[UNK]</code>; NFKC normalizer;
              Metaspace(<span className="font-mono">▁</span>, prepend="never") pre-tokenizer and decoder — so
              punctuation, brackets, URL characters, apostrophes and number separators round-trip exactly. Trained on
              the four pages with searched per-language weights{' '}
              <span className="font-mono">
                {LANGS.map((l) => `${l.code}=${metrics.weights[l.code]}`).join(' ')}
              </span>
              .
            </p>
            <p>The grader's scoring, exactly as we compute it live below:</p>
            <Formula>faithful unit = one letter/mark/number run OR one visible punctuation character</Formula>
            <Formula>fertility(lang) = tokens(lang) / faithful_units(lang)</Formula>
            <Formula>
              score = 1000 / (max − min) &nbsp;·&nbsp; hindi_penalty = exp(max(0, hi/1.2 − 1)) — ours: ×
              {stats.hindiPenalty.toFixed(3)}
            </Formula>
          </div>

          <div className="space-y-3">
            <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-700">
              <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
                Faithfulness gate (live)
              </div>
              <p className="text-[13px]">
                <span className="font-mono text-[12px]">"{sample}"</span>
                <br />
                decodes to
                <br />
                <span className="font-mono text-[12px]">"{dec}"</span>
              </p>
              <p
                className={`mt-2 font-mono text-sm font-bold ${
                  dec === sample ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-600 dark:text-red-400'
                }`}
              >
                {dec === sample ? 'EXACT ROUND-TRIP ✓' : 'ROUND-TRIP FAILED ✗'}
              </p>
            </div>
            <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-700">
              <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
                Corpus (faithful units)
              </div>
              <div className="grid grid-cols-2 gap-1 font-mono text-[12px]">
                {LANGS.map((l) => (
                  <div key={l.code} className="flex justify-between rounded bg-zinc-100 px-2 py-1 dark:bg-zinc-800">
                    <span>{l.name}</span>
                    <span className="font-semibold">{(metrics.faithful_units[l.code] || 0).toLocaleString()}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </ClaimCard>
  )
}
