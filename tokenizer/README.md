# ERA-V5 Assignment 2 (resubmission) — shared multilingual tokenizer, grader-compatible

One shared **10,000-token** tokenizer for the **wiki-faithful Markdown** India pages in
**English, Hindi, Telugu, Maithili** — the exact corpus recipe, tokenizer interface, and
scoring of the published course reference, with the per-language weights searched instead of
hand-picked.

Live widget: **https://www.pandala.in/era-v5/tokenizer/**

## Why this is a resubmission

The first submission used a custom tokenizer.json format; the grader's evaluator loads
tokenizers with `tokenizers.Tokenizer.from_file(...)` and therefore scored it 0. This version
ships a **standard HuggingFace tokenizer.json** — the instructor's published
`evaluate_tokenizer.py` runs on these artifacts **unchanged**.

## Recipe (matches the course reference interface exactly)

- Corpus: Wikipedia REST HTML → strip script/style/meta only → **markdownify** (links, URLs,
  tables, references, categories preserved). Snapshots committed under
  `../public/tokenizer/corpus/`. Our snapshots match the instructor's published snapshots
  unit-for-unit (en 186,367 · hi 88,359 · te 36,292 · mai 5,808 faithful units).
- Tokenizer: HuggingFace **BPE**, vocab 10,000, min_frequency 1, `[UNK]`, **NFKC**,
  **Metaspace(▁, prepend_scheme="never")** pre-tokenizer + decoder → punctuation, brackets,
  URL characters, apostrophes and number separators all **round-trip exactly**.
- Weights: **searched** (coordinate descent, ~40 trainings) to minimize the fertility spread
  subject to Hindi ≤ 1.2. Winner: `{en: 2, hi: 2, te: 4, mai: 4}`.

## Result (the grader's own formula, reproduced by their evaluator drop-in)

| Language | Tokens | Faithful units | Fertility |
|---|---:|---:|---:|
| English  | 112,786 | 186,367 | 0.605182 |
| Hindi    | 54,395 | 88,359 | 0.615614 |
| Telugu   | 21,896 | 36,292 | 0.603329 |
| Maithili | 3,547 | 5,808 | 0.610709 |

`spread = 0.615614 − 0.603329 = 0.012285` → **score = 1000 / 0.012285 = 81,399.95**
(reference solution: 6,502.56). Hindi penalty factor `exp(max(0, hi/1.2 − 1))` = **1.0** →
adjusted score identical. All four fertilities ≤ 1.2.

**Faithfulness:** `decode(encode(x))` preserves visible text on all four corpora, and the
grader's sample round-trips exactly:
`"India's population is 1,428,627,663."` → encode → decode → `"India's population is 1,428,627,663."`

## Reproduce

```bash
pip install -r requirements.txt          # tokenizers, regex, requests, beautifulsoup4, lxml, markdownify
python build_wiki_faithful_markdown.py   # (optional) refetch corpus — committed snapshots included
python train_tokenizer.py                # weight search + writes tokenizer.json/metrics.json/vocab.txt
python evaluate_tokenizer.py             # grader-verbatim scoring + faithfulness gate
```

Or load the artifact directly, exactly as the grader does:

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("../public/tokenizer/tokenizer.json")
ids = tok.encode("India's population is 1,428,627,663.").ids
assert tok.decode(ids) == "India's population is 1,428,627,663."
```

## Widget parity

The browser widget re-implements the HF pipeline in JS (`../src/tokenizer/lib/hfbpe.js`:
NFKC → Metaspace → BPE-by-rank → Metaspace decode) and recomputes every number live.
`node ../scripts/check_parity.mjs` proves the JS token ids, decoded text (SHA-256), and
faithful-unit counts are **identical** to the HuggingFace library's on all four corpora.
