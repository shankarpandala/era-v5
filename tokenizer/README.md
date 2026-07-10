# ERA-V5 Assignment 2 — from-scratch multilingual BPE tokenizer

A single **byte-level Byte-Pair-Encoding** tokenizer with a shared vocabulary of
**10,000 tokens**, trained from scratch (no `tokenizers` / `tiktoken` / `sentencepiece`)
on the **full** India Wikipedia article in **English, Hindi, Telugu, and Marathi**.

Live widget: **https://www.pandala.in/era-v5/tokenizer/**

## The metric

The tokenizer is graded by running it on the **full** India article per language, so we
train and evaluate on the full articles (train == eval).

```
X(lang) = total_BPE_tokens(lang) / total_words(lang)
hard constraint: X(English) <= 1.2          # X1 in the assignment
score = 1000 / (X_max - X_min)
```

Two word counts are reported (the fertility denominator is the whole game):

* **PRIMARY — whitespace words**: `len(text.split())`. Standard fertility; equals the
  word-faithful `[\p{L}\p{N}\p{M}]+` count within 1–2% on these corpora.
* **SECONDARY — `\w+` runs**: `[\p{L}\p{N}]+` (== `re.findall(r"\w+")`, the common
  Python idiom several classmates use). **Caveat:** `\w` excludes combining marks, so it
  splits Hindi/Telugu/Marathi words at matras/viramas, inflating Indic "word" counts
  2–3× (e.g. Telugu 7,370 vs 2,511 real words). It measures tokens per syllable-fragment,
  not per word — shown for comparability, never as our headline.

Both counts are computed identically in Python and JS (verified byte-for-byte, see parity).

## Results (reproducible — see below)

Weights **`{en:7, hi:3, te:3, mr:3}`** (search: English placed just under the 1.2 gate —
it is the *minimum* X, so raising it toward the gate frees merge budget for the Indic
languages and tightens the spread; per-language weighting is allowed).

**Primary (whitespace words):**

| Language | Script | words | tokens | X = tokens/word | ≤ 1.2 |
|----------|--------|------:|-------:|----------------:|:-----:|
| English  | Latin      | 10,121 | 11,996 | **1.1853** | ✓ |
| Hindi    | Devanagari |  8,078 | 11,928 | 1.4766 | ✗ |
| Telugu   | Telugu     |  2,511 |  6,323 | 2.5181 | ✗ |
| Marathi  | Devanagari |  4,605 |  8,488 | 1.8432 | ✗ |

`spread = 2.5181 − 1.1853 = 1.3329` → **self-score = 750.3**. English (the required
constraint) passes; the Indic three *cannot* all be ≤ 1.2 under real word counts — giving
the **entire** 9,744-merge budget to the Indic languages still leaves max X ≈ 1.58 on the
full articles. We report that honestly instead of switching denominators.

**Secondary (`\w+` count):** en 1.1576 ✓ · te 0.8579 ✓ · hi 0.7593 ✓ · mr 0.6956 ✓ —
all ≤ 1.2, spread 0.4620, **score 2,164.4**. Same tokenizer, same token counts; only the
denominator differs.

**English ≤ 1.2 holds under BOTH counts** — the binding gate is safe whichever method a
grader uses.

## Files

```
bpe.py            from-scratch byte-level BPE: pre-tokenizer, heap-based trainer, encoder,
                  and both word-count metrics
train.py          weight search on the full articles (en <= 1.19 hard, min spread);
                  writes all artifacts
tokenizer.py      standalone BPETokenizer.load(...).encode(...) — import this to reproduce
evaluate.py       recomputes X / spread / score under both word counts, writes stats.json
fetch_corpora.py  downloads + freezes the full India articles (run once; output committed)
requirements.txt  regex, requests
```

Artifacts (committed under `../public/tokenizer/`, single source of truth for Python + widget):

```
corpora/{en,hi,te,mr}.txt   the frozen FULL India articles (the eval set)
tokenizer.json              pattern + word_pattern + ordered merges + provenance
vocab.txt                   all 10,000 tokens, id<TAB>rendered-token (GPT-2 byte map)
stats.json                  per-language X under both word counts, spreads, scores
parity_golden.json          Python token-id stream per corpus (JS parity check)
```

## Reproduce

```bash
pip install -r requirements.txt
python fetch_corpora.py    # (optional) re-download the full articles — already committed
python train.py            # weight search + write artifacts (deterministic)
python evaluate.py         # print both metric tables, spreads, scores
```

Run our tokenizer on any text:

```python
from tokenizer import BPETokenizer
tok = BPETokenizer.load("../public/tokenizer/tokenizer.json")
print(tok.encode("भारत एक देश है।"))
```

## JS ↔ Python parity

The widget re-implements the **encoder** in JS (`../src/tokenizer/lib/bpe.js`) and recomputes
every number live in the browser. `node ../scripts/check_parity.mjs` proves the JS encoder
produces the **identical token stream** to Python AND identical word counts (both metrics) on
every corpus. The widget also lets a grader paste/upload their own cleaned India-page text and
watch both tables recompute live with this exact tokenizer.

## Known unknown

The instructor may grade on his own HTML→Markdown conversion of the pages (pipeline
undecided, "secret sauce"). No one can tune for that reliably; our byte-level encoder never
emits UNK on any input (an explicit requirement — UNK ⇒ score 0), and English remains the
best-compressed language on markdown-ish text since URLs/markup are ASCII.
