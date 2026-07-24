# Assignment 4 — the 8-stage cleaning pipeline

Session 4's assignment: count the session's cleaning strategies, pick a
10–100M-token dataset, apply the cleanups, and present the result as a widget.

**Pipeline version 1.1.0** score fixes (vs 1.0):
1. **Domain-aware quality** — math/code skip English soft rules (stop-words, mean length, symbol ratio); would-fires still measured
2. **PII precision** — version-like dotted quads + RFC 5737 TEST-NET ranges exempted
3. **Decontam precision** — multi-item template grams excluded; DF cap 5; ≥35% rare containment + ≥3 grams + item word-Jaccard ≥0.22

Two corpora go through the same eight stages:

| Corpus | Why |
|---|---|
| [`bespokelabs/Bespoke-Stratos-17k`](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) | The main pick — 16,710 reasoning-distillation conversations (≈85M Qwen tokens, inside the 10–100M band), same genre as the assignment's example model, and it naturally exercises every stage (literal ghost tags, cross-source near-dups, benchmark overlap, conversation format). |
| [`ai4bharat/sangraha`](https://huggingface.co/datasets/ai4bharat/sangraha) `verified/tel` slice | The sovereign thread made real — the India-first machinery (ZWJ/ZWNJ keep-rule, script-aware filtering and shingling, ISO 639-1↔639-3 code validation, Indian PII formats) runs on actual Telugu text instead of being asserted on English data. |

## Files

- `clean.py` — the eight stages for the Stratos corpus (`--prep` / full / `--verify`)
- `sangraha_slice.py` — the same stages for the Telugu slice
- `score_edu.py` — one-time FineWeb-Edu classifier labeling pass (CPU), keyed by
  content hash so pipeline runs stay fast and deterministic

## Reproduce

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
curl -LO https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
export LID_PATH=$PWD/lid.176.bin

python3 clean.py --prep            # stages 1–2, emit classifier inputs
python3 sangraha_slice.py --prep   # fetch Telugu slice, emit sample inputs
python3 score_edu.py               # ~1–2 h on CPU, resumable
python3 clean.py                   # full 8-stage run -> out/stats.json, out/manifest.json
python3 sangraha_slice.py          # -> out/stats_tel.json, out/manifest_tel.json
python3 clean.py --verify          # determinism proof (identical content hash)
python3 sangraha_slice.py --verify
```

Everything is seeded (seed 42), iterates in the published dataset order, and
derives every ID from sha256 over the *cleaned* text — the lesson's ordering
rule. The stats and manifests the widget displays are committed under
`public/data-cleaning/`.

## Deploy the widget to Netlify (assignment submission)

The widget is fully self-contained (inline CSS/JS, no external calls). After
`npm run build` at the repo root:

1. Open https://app.netlify.com/drop (log in yourself).
2. Drag the **`dist/data-cleaning/`** folder onto the page.
3. Share the resulting URL with the course.

The same page also deploys with the site to
https://www.pandala.in/era-v5/data-cleaning/.
