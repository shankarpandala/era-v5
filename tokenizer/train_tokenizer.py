#!/usr/bin/env python3
"""Train the shared 10k tokenizer on the faithful Markdown corpus (grader recipe).

Recipe (identical interface to the course reference so the published grader
evaluator runs on the output unchanged): HuggingFace BPE, vocab 10,000,
min_frequency=1, [UNK], NFKC normalizer, Metaspace(▁, prepend_scheme="never")
pre-tokenizer + decoder. Per-language weights are applied by duplicating
corpus files, exactly as the reference does.

On top of the reference recipe we SEARCH the integer weights (coordinate
descent) to minimize the faithful-unit fertility spread, subject to
Hindi ratio <= 1.2 (so the Hindi penalty factor stays 1.0).

    python train_tokenizer.py

Writes to ../public/tokenizer/:
    tokenizer.json       HF tokenizers format (Tokenizer.from_file-loadable)
    metrics.json         ratios / spread / score / hindi penalty
    vocab.txt            id<TAB>token, one per line (download deliverable)
    parity_golden.json   per-language token ids + decode SHA-256 (JS parity)
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import regex
from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).resolve().parent.parent / "public" / "tokenizer"
CORPUS = ROOT / "corpus"
LANGS = ["en", "hi", "te", "mai"]
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")

START_WEIGHTS = {"en": 3, "hi": 4, "te": 4, "mai": 2}  # reference starting point
WEIGHT_GRID = [1, 2, 3, 4, 5, 6, 8]
SEARCH_ROUNDS = 3


def faithful_units(text: str) -> int:
    return len(FAITHFUL_UNIT_RE.findall(text))


def make_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="never")
    tokenizer.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="never")
    return tokenizer


def train_with(texts: dict[str, str], weights: dict[str, int]) -> Tokenizer:
    with tempfile.TemporaryDirectory() as tmp:
        files: list[str] = []
        tmpdir = Path(tmp)
        for code, text in texts.items():
            path = tmpdir / f"{code}.txt"
            path.write_text(text, encoding="utf-8")
            files.extend([str(path)] * weights[code])
        tokenizer = make_tokenizer()
        trainer = BpeTrainer(vocab_size=10000, min_frequency=1, special_tokens=["[UNK]"])
        tokenizer.train(files, trainer)
    return tokenizer


def ratios_of(tokenizer: Tokenizer, texts: dict[str, str], units: dict[str, int]) -> dict[str, float]:
    return {c: len(tokenizer.encode(texts[c]).ids) / units[c] for c in LANGS}


def cost(r: dict[str, float]) -> tuple:
    viol = max(0.0, r["hi"] - 1.2)  # keep the Hindi penalty factor at exactly 1
    spread = max(r.values()) - min(r.values())
    return (1 if viol > 0 else 0, round(viol, 6), round(spread, 6))


def main() -> int:
    texts = {c: (CORPUS / f"{c}.faithful.txt").read_text(encoding="utf-8") for c in LANGS}
    units = {c: faithful_units(texts[c]) for c in LANGS}

    seen: dict[tuple, tuple] = {}

    def evaluate(w: dict[str, int]) -> tuple:
        key = tuple(w[c] for c in LANGS)
        if key not in seen:
            r = ratios_of(train_with(texts, w), texts, units)
            seen[key] = (cost(r), r)
            sp = max(r.values()) - min(r.values())
            print(f"  w={w} spread={sp:.6f} score={1000/sp:,.0f} hi={r['hi']:.4f}")
        return seen[key]

    best_w = dict(START_WEIGHTS)
    best_c, best_r = evaluate(best_w)
    for rnd in range(SEARCH_ROUNDS):
        improved = False
        for code in LANGS:
            for v in WEIGHT_GRID:
                if v == best_w[code]:
                    continue
                cand = dict(best_w, **{code: v})
                c, r = evaluate(cand)
                if c < best_c:
                    best_c, best_r, best_w = c, r, cand
                    improved = True
        print(f"round {rnd}: best={best_w} spread={best_c[2]:.6f}")
        if not improved:
            break

    # Final artifacts from the winning weights.
    tokenizer = train_with(texts, best_w)
    tokenizer.save(str(ROOT / "tokenizer.json"))

    r = ratios_of(tokenizer, texts, units)
    spread = max(r.values()) - min(r.values())
    score = 1000 / spread
    import math

    hindi_penalty = math.exp(max(0.0, r["hi"] / 1.2 - 1.0))
    metrics = {
        "variant": "wiki_faithful_markdown",
        "languages": {"en": "English", "hi": "Hindi", "te": "Telugu", "mai": "Maithili"},
        "weights": best_w,
        "vocab_size": tokenizer.get_vocab_size(),
        "faithful_units": units,
        "unit_policy": (
            "Counts each contiguous Unicode letter/mark/number run as one unit "
            "and each visible non-space punctuation/symbol character as one unit."
        ),
        "token_counts": {c: len(tokenizer.encode(texts[c]).ids) for c in LANGS},
        "ratios": r,
        "spread": spread,
        "score": score,
        "hindi_exp1_penalty_factor": hindi_penalty,
        "hindi_exp1_adjusted_score": score / hindi_penalty,
    }
    (ROOT / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # vocab.txt (download deliverable): id<TAB>token
    vocab = tokenizer.get_vocab()
    by_id = sorted(vocab.items(), key=lambda kv: kv[1])
    with open(ROOT / "vocab.txt", "w", encoding="utf-8") as f:
        for tok, i in by_id:
            f.write(f"{i}\t{tok}\n")

    # Parity golden for the JS reimplementation: full ids + decode hash.
    golden = {}
    for c in LANGS:
        enc = tokenizer.encode(texts[c])
        dec = tokenizer.decode(enc.ids)
        golden[c] = {
            "ids": enc.ids,
            "decode_sha256": hashlib.sha256(dec.encode("utf-8")).hexdigest(),
        }
    (ROOT / "parity_golden.json").write_text(json.dumps(golden), encoding="utf-8")

    print(json.dumps({k: metrics[k] for k in ["weights", "ratios", "spread", "score",
                                              "hindi_exp1_penalty_factor"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
