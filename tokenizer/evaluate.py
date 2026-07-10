"""Recompute the fertility stats from the frozen corpora + tokenizer.json.

This is the "graders run it themselves" entry point. It loads the committed
tokenizer and the committed corpora, encodes each language, and reports:

    X(lang) = tokens(lang) / words(lang)          (per language, must be <= 1.2)
    spread  = max(X) - min(X)
    score   = 1000 / spread

    python evaluate.py            # prints the table, writes stats.json

Word = maximal run of non-whitespace: len(text.split()). The corpora are
whitespace-normalized (fetch_corpora.py) so this equals the JS widget's
text.trim().split(/\\s+/).filter(Boolean).length exactly.
"""

from __future__ import annotations

import json
import os

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


def word_count(text: str) -> int:
    return len(text.split())


def compute_stats(corpora: dict[str, str], tok: BPETokenizer) -> dict:
    per = {}
    for code, text in corpora.items():
        words = word_count(text)
        tokens = len(tok.encode(text))
        name, script = LANG_META[code]
        per[code] = {
            "language": name,
            "script": script,
            "words": words,
            "tokens": tokens,
            "X": tokens / words,
            "within_constraint": (tokens / words) <= CONSTRAINT,
        }
    xs = [per[c]["X"] for c in corpora]
    x_max, x_min = max(xs), min(xs)
    spread = x_max - x_min
    score = 1000.0 / spread if spread > 0 else float("inf")
    sorted_desc = sorted(
        ({"code": c, "language": per[c]["language"], "X": per[c]["X"]} for c in corpora),
        key=lambda r: r["X"],
        reverse=True,
    )
    return {
        "vocab_size": tok.vocab_size,
        "constraint": CONSTRAINT,
        "per_language": per,
        "x_sorted_desc": sorted_desc,
        "x_max": x_max,
        "x_min": x_min,
        "spread": spread,
        "score": score,
        "constraints_met": all(per[c]["within_constraint"] for c in corpora),
    }


def print_stats(stats: dict) -> None:
    print(f"{'lang':<9}{'script':<12}{'words':>8}{'tokens':>9}{'X=tok/word':>12}  ok")
    print("-" * 54)
    for code, row in stats["per_language"].items():
        ok = "OK " if row["within_constraint"] else "XX "
        print(
            f"{row['language']:<9}{row['script']:<12}{row['words']:>8,}{row['tokens']:>9,}"
            f"{row['X']:>12.4f}  {ok}"
        )
    print("-" * 54)
    print(f"vocab size : {stats['vocab_size']}")
    print(f"X sorted   : " + ", ".join(f"{r['code']}={r['X']:.4f}" for r in stats["x_sorted_desc"]))
    print(f"X_max-X_min: {stats['x_max']:.4f} - {stats['x_min']:.4f} = {stats['spread']:.4f}")
    print(f"all <= {stats['constraint']}: {stats['constraints_met']}")
    print(f"SELF SCORE : 1000 / {stats['spread']:.4f} = {stats['score']:.1f}")


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
