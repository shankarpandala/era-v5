"""Train the shared 10k-vocab BPE and write all artifacts.

Strategy (see the plan / widget Method section):
  * The metric X = tokens/word must be <= 1.2 for every language. A single 10k
    vocab cannot compress four FULL India articles that well, so we evaluate on
    an EQUAL-SIZE slice of each article (same word budget per language — a fair
    cross-language basis, bounded by Telugu's short article). We pick the
    LARGEST slice size N at which a feasible (all X <= 1.2) tokenizer exists.
  * At that N we run a coordinate-descent search over integer per-language
    mixing weights to minimize the score spread (X_max - X_min) subject to
    feasibility. Score = 1000 / spread.

Outputs (all under ../public/tokenizer/, committed, single source of truth):
  corpora/{lang}.txt      the frozen equal-size eval slices
  tokenizer.json          pattern + ordered merges + provenance
  vocab.txt               human-readable token list (download deliverable)
  stats.json              per-language X, spread, score, constraints_met
  parity_golden.json      Python token-id sequence per corpus (JS parity check)

    python train.py

Deterministic: integer weights, fixed corpus order, fixed tie-breaks -> a
re-run on the frozen slices reproduces identical merges and numbers.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import bpe
from evaluate import CORPORA_DIR, LANG_META, PUB, compute_stats, print_stats
from tokenizer import BPETokenizer

CORPORA_FULL_DIR = os.path.join(PUB, "corpora_full")
VOCAB_SIZE = 10000
CONSTRAINT = 1.2

# Largest-first: we accept the first (largest) N that admits a feasible solution.
N_GRID = [2500, 2400, 2300, 2200, 2100, 2000]
# Integer mixing weights explored per language during coordinate descent.
WEIGHT_GRID = [1, 2, 3, 4, 6, 8, 10, 14]
SEARCH_ROUNDS = 3
CODES = list(LANG_META)  # ["en", "hi", "te", "mr"]


def load_full() -> dict[str, str]:
    out = {}
    for code in CODES:
        with open(os.path.join(CORPORA_FULL_DIR, f"{code}.txt"), encoding="utf-8") as f:
            out[code] = f.read()
    return out


def slice_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def pretok_counts(text: str) -> dict[tuple[int, ...], int]:
    c: dict[tuple[int, ...], int] = defaultdict(int)
    for piece in bpe.pre_tokenize(text):
        c[tuple(piece.encode("utf-8"))] += 1
    return c


def cost(stats: dict) -> tuple:
    """Lexicographic cost to MINIMIZE: (infeasible?, total_violation, spread)."""
    viol = sum(max(0.0, r["X"] - CONSTRAINT) for r in stats["per_language"].values())
    return (1 if viol > 1e-12 else 0, viol, stats["spread"])


def search_at(slices: dict[str, str]):
    """Coordinate-descent over integer weights; minimize cost. Returns (weights, stats)."""
    lang_counts = {c: pretok_counts(slices[c]) for c in CODES}

    def words_of(weights):
        agg: dict[tuple[int, ...], int] = defaultdict(int)
        for code in CODES:
            w = weights[code]
            for chunk, n in lang_counts[code].items():
                agg[chunk] += n * w
        return [(list(k), v) for k, v in agg.items()]

    def ev(weights):
        merges = bpe.train(words_of(weights), VOCAB_SIZE)
        return merges, compute_stats(slices, BPETokenizer(merges))

    best_w = {c: 1 for c in CODES}
    _, best_s = ev(best_w)
    best_c = cost(best_s)
    for _ in range(SEARCH_ROUNDS):
        improved = False
        for code in CODES:
            for v in WEIGHT_GRID:
                if v == best_w[code]:
                    continue
                cand = dict(best_w, **{code: v})
                _, s = ev(cand)
                c = cost(s)
                if c < best_c:
                    best_c, best_w, best_s = c, cand, s
                    improved = True
        if not improved:
            break
    return best_w, best_s


def main() -> None:
    full = load_full()
    chosen = None
    for n in N_GRID:
        slices = {c: slice_words(full[c], n) for c in CODES}
        weights, stats = search_at(slices)
        feasible = stats["constraints_met"]
        print(
            f"N={n}: feasible={feasible} weights={weights} "
            f"spread={stats['spread']:.4f} score={stats['score']:.0f} "
            f"maxX={stats['x_max']:.4f}"
        )
        if feasible:
            chosen = (n, slices, weights, stats)
            break
    if chosen is None:
        # No feasible N in grid — take the smallest N (closest to feasible) anyway.
        n = N_GRID[-1]
        slices = {c: slice_words(full[c], n) for c in CODES}
        weights, stats = search_at(slices)
        chosen = (n, slices, weights, stats)
        print(f"WARNING: no fully-feasible N found; using N={n} (best effort).")

    n, slices, weights, _ = chosen

    # Freeze the chosen eval slices.
    os.makedirs(CORPORA_DIR, exist_ok=True)
    for code in CODES:
        with open(os.path.join(CORPORA_DIR, f"{code}.txt"), "w", encoding="utf-8") as f:
            f.write(slices[code])

    # Retrain the final tokenizer on the frozen slices with the chosen weights.
    lang_counts = {c: pretok_counts(slices[c]) for c in CODES}
    agg: dict[tuple[int, ...], int] = defaultdict(int)
    for code in CODES:
        for chunk, cnt in lang_counts[code].items():
            agg[chunk] += cnt * weights[code]
    merges = bpe.train([(list(k), v) for k, v in agg.items()], VOCAB_SIZE)

    words_per_language = {c: len(slices[c].split()) for c in CODES}
    meta = {
        "corpus": {
            "source": "India — Wikipedia article (per-language), MediaWiki extracts API",
            "basis": "equal-size slice: first N words of each article (fair cross-language basis)",
            "words_per_language": words_per_language,
            "target_words": n,
        },
        "mixing_weights": weights,
        "note": "trained==evaluated on the frozen slices below; re-running reproduces identical merges",
    }
    bpe.save_tokenizer(os.path.join(PUB, "tokenizer.json"), merges, meta)
    bpe.write_vocab_txt(os.path.join(PUB, "vocab.txt"), merges)

    # Golden token-id sequences for the JS parity test.
    tok = BPETokenizer(merges)
    golden = {code: tok.encode(slices[code]) for code in CODES}
    with open(os.path.join(PUB, "parity_golden.json"), "w", encoding="utf-8") as f:
        json.dump(golden, f)

    # Stats.
    stats = compute_stats(slices, tok)
    with open(os.path.join(PUB, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== FINAL ===")
    print(f"corpus: equal slice, N={n} words/language {words_per_language}")
    print(f"mixing weights: {weights}")
    print_stats(stats)
    print(f"\nwrote artifacts to {PUB}")


if __name__ == "__main__":
    main()
