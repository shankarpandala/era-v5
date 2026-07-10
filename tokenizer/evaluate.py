"""Recompute the fertility stats from the frozen FULL India articles + tokenizer.json.

This is the "graders run it themselves" entry point. Evaluation is on the FULL
India Wikipedia article per language (train == eval), exactly as the course
grades it. For each language:

    X(lang) = tokens(lang) / words(lang)
    spread  = max(X) - min(X)
    score   = 1000 / spread

The assignment's hard constraint is on ENGLISH: X(en) must be <= 1.2. We report
two word counts and English passes under BOTH:

  * PRIMARY   word = whitespace-delimited run, len(text.split()) — standard
    fertility. (Equals the word-faithful [\\p{L}\\p{N}\\p{M}]+ count within 1-2%.)
  * SECONDARY word = [\\p{L}\\p{N}]+ runs (== re.findall(r"\\w+") — the common
    Python idiom, used by several classmates). NOTE: \\w excludes combining
    marks, so this splits Indic words at matras/viramas and inflates Indic word
    counts 2-3x; it is shown for comparability, not as a true word count.

    python evaluate.py            # prints both tables, writes stats.json
"""

from __future__ import annotations

import json
import os

import bpe
from tokenizer import BPETokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.normpath(os.path.join(HERE, "..", "public", "tokenizer"))
CORPORA_DIR = os.path.join(PUB, "corpora")

# code -> (display name, script) — order is the display order.
LANG_META = {
    "en": ("English", "Latin"),
    "hi": ("Hindi", "Devanagari"),
    "te": ("Telugu", "Telugu"),
    "mr": ("Marathi", "Devanagari"),
}

CONSTRAINT = 1.2


def load_corpora() -> dict[str, str]:
    corpora = {}
    for code in LANG_META:
        with open(os.path.join(CORPORA_DIR, f"{code}.txt"), encoding="utf-8") as f:
            corpora[code] = f.read()
    return corpora


def _metric(corpora: dict[str, str], per_tokens: dict, word_fn) -> dict:
    per = {}
    for code, (name, script) in LANG_META.items():
        words = word_fn(corpora[code])
        tokens = per_tokens[code]
        per[code] = {
            "language": name,
            "script": script,
            "words": words,
            "tokens": tokens,
            "X": tokens / words,
            "within_constraint": (tokens / words) <= CONSTRAINT,
        }
    xs = [per[c]["X"] for c in LANG_META]
    x_max, x_min = max(xs), min(xs)
    spread = x_max - x_min
    return {
        "per_language": per,
        "x_sorted_desc": sorted(
            ({"code": c, "language": per[c]["language"], "X": per[c]["X"]} for c in LANG_META),
            key=lambda r: r["X"],
            reverse=True,
        ),
        "x_max": x_max,
        "x_min": x_min,
        "spread": spread,
        "score": 1000.0 / spread if spread > 0 else float("inf"),
        "constraints_met": all(per[c]["within_constraint"] for c in LANG_META),
        "english_within_constraint": per["en"]["within_constraint"],
    }


def compute_stats(corpora: dict[str, str], tok: BPETokenizer) -> dict:
    tokens = {c: len(tok.encode(corpora[c])) for c in LANG_META}
    primary = _metric(corpora, tokens, bpe.count_words_split)  # whitespace words
    wplus = _metric(corpora, tokens, bpe.count_words_wplus)    # [\p{L}\p{N}]+ (== \w+)
    return {
        "vocab_size": tok.vocab_size,
        "constraint": CONSTRAINT,
        "word_metric": (
            "primary = whitespace split() (standard fertility); "
            "secondary = [\\p{L}\\p{N}]+ (== \\w+, splits Indic words at combining marks)"
        ),
        "primary": primary,
        "wplus": wplus,
        "english_ok_both_counts": (
            primary["english_within_constraint"] and wplus["english_within_constraint"]
        ),
        # convenience top-level mirror of the primary metric
        "per_language": primary["per_language"],
        "spread": primary["spread"],
        "score": primary["score"],
        "constraints_met": primary["constraints_met"],
    }


def _print_metric(title: str, m: dict) -> None:
    print(f"\n--- {title} ---")
    print(f"{'lang':<9}{'script':<12}{'words':>8}{'tokens':>9}{'X=tok/word':>12}  ok")
    print("-" * 54)
    for code, row in m["per_language"].items():
        ok = "OK " if row["within_constraint"] else "XX "
        print(
            f"{row['language']:<9}{row['script']:<12}{row['words']:>8,}{row['tokens']:>9,}"
            f"{row['X']:>12.4f}  {ok}"
        )
    print("-" * 54)
    print("X sorted   : " + ", ".join(f"{r['code']}={r['X']:.4f}" for r in m["x_sorted_desc"]))
    print(f"spread     : {m['x_max']:.4f} - {m['x_min']:.4f} = {m['spread']:.4f}")
    print(f"English <= 1.2 : {m['english_within_constraint']}   (all four: {m['constraints_met']})")
    print(f"SELF SCORE : 1000 / {m['spread']:.4f} = {m['score']:.1f}")


def print_stats(stats: dict) -> None:
    print(f"vocab size : {stats['vocab_size']}")
    _print_metric("PRIMARY word count = whitespace split() (standard fertility)", stats["primary"])
    _print_metric("SECONDARY word count = [\\p{L}\\p{N}]+ (== \\w+; splits Indic at marks)", stats["wplus"])
    print(f"\nEnglish <= 1.2 under BOTH counts: {stats['english_ok_both_counts']}")


def main() -> None:
    corpora = load_corpora()
    tok = BPETokenizer.load(os.path.join(PUB, "tokenizer.json"))
    stats = compute_stats(corpora, tok)
    print_stats(stats)
    with open(os.path.join(PUB, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {os.path.join(PUB, 'stats.json')}")


if __name__ == "__main__":
    main()
