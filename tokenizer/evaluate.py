"""Recompute the fertility stats from the frozen FULL India articles + tokenizer.json.

This is the "graders run it themselves" entry point. Evaluation is on the FULL
India Wikipedia article per language (train == eval), exactly as the course
grades it. For each language:

    X(lang) = tokens(lang) / words(lang)          (must be <= 1.2)
    spread  = max(X) - min(X)
    score   = 1000 / spread

Primary word count = Unicode letter/number runs `[\\p{L}\\p{N}]+` (== re.findall
r"\\w+"); we also report the whitespace-split count for transparency. English is
<= 1.2 under BOTH, so the binding English gate holds under any grader method.

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


def _metric(per_tokens: dict, word_fn) -> dict:
    per = {}
    for code, (name, script) in LANG_META.items():
        words = word_fn(_CORP[code])
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


_CORP: dict[str, str] = {}


def compute_stats(corpora: dict[str, str], tok: BPETokenizer) -> dict:
    global _CORP
    _CORP = corpora
    tokens = {c: len(tok.encode(corpora[c])) for c in LANG_META}
    primary = _metric(tokens, bpe.count_words)          # [\p{L}\p{N}]+  (== \w+)
    split = _metric(tokens, bpe.count_words_split)       # text.split()
    return {
        "vocab_size": tok.vocab_size,
        "constraint": CONSTRAINT,
        "word_metric": "primary = [\\p{L}\\p{N}]+ (== \\w+); secondary = whitespace split()",
        "primary": primary,
        "split": split,
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
    print(f"all <= 1.2 : {m['constraints_met']}   (English <= 1.2: {m['english_within_constraint']})")
    print(f"SELF SCORE : 1000 / {m['spread']:.4f} = {m['score']:.1f}")


def print_stats(stats: dict) -> None:
    print(f"vocab size : {stats['vocab_size']}")
    _print_metric("PRIMARY word count = [\\p{L}\\p{N}]+  (== re.findall r'\\w+')", stats["primary"])
    _print_metric("SECONDARY word count = whitespace split()", stats["split"])


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
