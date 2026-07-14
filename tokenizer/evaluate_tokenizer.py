#!/usr/bin/env python3
"""Evaluate tokenizer.json on the faithful Markdown corpus (grader-compatible).

Scoring is byte-for-byte the published course evaluator: load with
tokenizers.Tokenizer.from_file, count faithful units
([\\p{L}\\p{M}\\p{N}]+ runs or single visible punctuation chars), ratio =
tokens/units per language, score = 1000/(max-min), Hindi penalty
exp(max(0, hi/1.2 - 1)).

Additionally enforces the FAITHFULNESS gate the grader applies:
decode(encode(text)) must preserve the same visible (non-whitespace)
characters — checked on every corpus and on the grader's own sample string.

    python evaluate_tokenizer.py
"""
from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

import regex
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent / "public" / "tokenizer"
CORPUS = ROOT / "corpus"
LANGS = ["en", "hi", "te", "mai"]
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")
SAMPLE = "India's population is 1,428,627,663."


def faithful_units(text: str) -> int:
    return len(FAITHFUL_UNIT_RE.findall(text))


def visible(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def roundtrip_ok(tokenizer: Tokenizer, text: str) -> bool:
    dec = tokenizer.decode(tokenizer.encode(text).ids)
    # The tokenizer normalizes with NFKC, so faithfulness is judged against
    # the NFKC form (the grader's reference tokenizer behaves identically).
    return visible(dec) == visible(unicodedata.normalize("NFKC", text))


def main() -> int:
    tokenizer = Tokenizer.from_file(str(ROOT / "tokenizer.json"))

    # --- faithfulness gate ---------------------------------------------------
    sample_dec = tokenizer.decode(tokenizer.encode(SAMPLE).ids)
    checks = {"sample_exact": sample_dec == SAMPLE, "sample_decoded": sample_dec}
    for code in LANGS:
        text = (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        checks[f"{code}_visible_preserved"] = roundtrip_ok(tokenizer, text)

    # --- grader scoring (verbatim logic) -------------------------------------
    rows = {}
    for code in LANGS:
        text = (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        units = faithful_units(text)
        tokens = len(tokenizer.encode(text).ids)
        rows[code] = {"tokens": tokens, "faithful_units": units, "ratio": tokens / units}

    ratios = [row["ratio"] for row in rows.values()]
    spread = max(ratios) - min(ratios)
    score = 1000 / spread
    hindi_penalty = math.exp(max(0.0, rows["hi"]["ratio"] / 1.2 - 1.0))
    result = {
        "faithfulness": checks,
        "rows": rows,
        "spread": spread,
        "score": score,
        "hindi_exp1_penalty_factor": hindi_penalty,
        "hindi_exp1_adjusted_score": score / hindi_penalty,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    ok = checks["sample_exact"] and all(checks[f"{c}_visible_preserved"] for c in LANGS)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
