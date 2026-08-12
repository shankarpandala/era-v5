"""Deterministic arithmetic corpora: nested train sets, three eval regimes,
two task families.

Split construction guarantees, by construction rather than by checking:

  * a contiguous OPERAND HOLE [40, 59] is excluded from training entirely:
    no training pair touches any operand in the hole. eval_hole pairs all do.
    For a learned embedding table the hole tokens remain untrained random
    rows; for a deterministic scheme their embeddings are analytically
    correct — this split is the token-level generalization test. (The hole
    is an INPUT-TOKEN hole: hole-band VALUES do occur as training answers,
    equally for every arm; the manifest counts them.)
  * eval-in-range and every train set are drawn from one deterministic
    shuffle of all hole-free (op, a, b) triples with operands in [0, 100):
    eval-in-range is the first 1000 triples, train sets are prefixes of the
    remainder. Train sets of different sizes are therefore *nested*
    (train_500 ⊂ train_2000 ⊂ train_8000) and never overlap eval.
  * extrapolation triples have max(a, b) in [100, 999] (b <= a keeps each
    example in its assigned magnitude bucket) — the magnitude-extrapolation
    stress test. Values stay inside the exact-additivity domain of LIN.

Operations are add, mul, and sub — subtraction activates the SIGN dim
(in-range differences go negative). 25% of arithmetic examples use word-form
operators ("plus"/"times"/"minus").

Task families:
  * "arith" — the fixed template  <bos> a op b = <ans> <eos>
  * "nl"    — natural-language templates of varying length and answer
    position ("what is a plus b", "compute the sum of a and b", ...), the
    transfer slice showing the unified embedding is not tied to one rigid
    template. Same operand splits, same hole.
"""

from __future__ import annotations

import math

import numpy as np

from .embedding import numeric_features
from .layout import LAYOUT, TARGET_SCALE
from .util import deterministic_shuffle, rand_float, rand_int, sha256_json
from .vocab import ANS, BOS, EOS, PAD, Vocab

IN_RANGE_MAX = 100          # train/eval operands are in [0, IN_RANGE_MAX)
EXTRA_MAX = 1000            # extrapolation operands are in [0, EXTRA_MAX)
HOLE = (40, 59)             # operands never seen in training, inclusive band
EVAL_IN_SIZE = 1_000
HOLE_SIZE = 1_000
EXTRA_SIZE = 2_000
WORD_FORM_PROB = 0.25
BUCKETS = ((100, 199), (200, 499), (500, 999))

OPS = ("add", "mul", "sub")
OP_SYMBOL = {"add": "+", "mul": "*", "sub": "-"}
OP_WORD = {"add": "plus", "mul": "times", "sub": "minus"}
OP_NOUN = {"add": "sum", "mul": "product", "sub": "difference"}


def in_hole(x: int) -> bool:
    return HOLE[0] <= x <= HOLE[1]


def _apply(op: str, a: int, b: int) -> int:
    if op == "add":
        return a + b
    if op == "mul":
        return a * b
    return a - b


def _surface(seed: int, key: tuple) -> str:
    return "word" if rand_float(seed, "surface", *key) < WORD_FORM_PROB else "sym"


def _example(op: str, a: int, b: int, seed: int) -> dict:
    return {"op": op, "a": a, "b": b, "c": _apply(op, a, b),
            "surface": _surface(seed, (op, a, b))}


def build_splits(seed: int, train_size: int) -> dict:
    """All splits + manifest for one train size. Pure function of (seed, size).

    The same operand splits serve both task families — templates are applied
    at encode time, so "arith" and "nl" runs see identical (op, a, b) data.
    """
    clear = [(op, a, b)
             for op in OPS
             for a in range(IN_RANGE_MAX)
             for b in range(IN_RANGE_MAX)
             if not (in_hole(a) or in_hole(b))]
    holed = [(op, a, b)
             for op in OPS
             for a in range(IN_RANGE_MAX)
             for b in range(IN_RANGE_MAX)
             if in_hole(a) or in_hole(b)]
    pool = deterministic_shuffle(clear, seed, "in_range_pool")
    if train_size > len(pool) - EVAL_IN_SIZE:
        raise ValueError(f"train_size {train_size} exceeds pool")
    eval_in = [_example(*t, seed) for t in pool[:EVAL_IN_SIZE]]
    train = [_example(*t, seed) for t in pool[EVAL_IN_SIZE:EVAL_IN_SIZE + train_size]]
    hole_pool = deterministic_shuffle(holed, seed, "hole_pool")
    eval_hole = [_example(*t, seed) for t in hole_pool[:HOLE_SIZE]]

    extra = []
    for i in range(EXTRA_SIZE):
        lo, hi = BUCKETS[i % len(BUCKETS)]
        # op drawn independently of the bucket index — a round-robin on the
        # same modulus would confound bucket with operation
        op = OPS[rand_int(seed, 0, len(OPS), "extra_op", i)]
        a = rand_int(seed, lo, hi + 1, "extra_a", i)
        b = rand_int(seed, 0, a + 1, "extra_b", i)  # b <= a keeps the bucket
        ex = _example(op, a, b, seed)
        ex["bucket"] = f"{lo}-{hi}"
        extra.append(ex)

    splits = {"train": train, "eval_in": eval_in, "eval_hole": eval_hole,
              "eval_extra": extra}
    train_keys = {(e["op"], e["a"], e["b"]) for e in train}
    eval_keys = {(e["op"], e["a"], e["b"]) for e in eval_in}
    manifest = {
        "seed": seed,
        "train_size": train_size,
        "ops": list(OPS),
        "hole": list(HOLE),
        "counts": {k: len(v) for k, v in splits.items()},
        "hashes": {k: sha256_json(v) for k, v in splits.items()},
        "disjoint_train_eval": not (train_keys & eval_keys),
        "train_never_touches_hole": all(
            not (in_hole(e["a"]) or in_hole(e["b"])) for e in train),
        "hole_always_touched": all(
            in_hole(e["a"]) or in_hole(e["b"]) for e in eval_hole),
        # Disclosure: the hole is an INPUT-TOKEN hole only. Hole-band VALUES
        # do occur as training answers (e.g. 30 + 12 = 42) and supervise the
        # output heads of every arm equally; what no arm ever receives is a
        # hole token at an input position, which is exactly what the
        # input-embedding claim requires.
        "train_answers_in_hole_band": sum(1 for e in train
                                          if in_hole(abs(e["c"]))),
        "extra_bucket_counts": {f"{lo}-{hi}": sum(1 for e in extra
                                                  if e["bucket"] == f"{lo}-{hi}")
                                for lo, hi in BUCKETS},
        "max_product": max((e["c"] for e in extra if e["op"] == "mul"),
                           default=0),
        "word_form_fraction": round(sum(1 for e in train
                                        if e["surface"] == "word") / max(1, len(train)), 4),
    }
    return {"splits": splits, "manifest": manifest}


# ---------------------------------------------------------------------------
# Serialization: example -> token sequence (per task family)
# ---------------------------------------------------------------------------

ARITH_SEQ_LEN = 7   # <bos> a op b = <ans> <eos>
NL_SEQ_LEN = 10     # longest NL template, front-aligned, <pad> after <eos>

# NL templates: (chooser index) -> token pattern. Answer position varies by
# template, so encode() records a per-example <ans> position.
_NL_TEMPLATES = [
    lambda a, opw, opn, b: ["what", "is", a, opw, b],
    lambda a, opw, opn, b: ["compute", "the", opn, "of", a, "and", b],
    lambda a, opw, opn, b: ["tell", "me", a, opw, b],
]


def _tokens_for(example: dict, task: str, seed: int) -> list[str]:
    a, b, op = str(example["a"]), str(example["b"]), example["op"]
    if task == "arith":
        op_tok = (OP_WORD[op] if example["surface"] == "word"
                  else OP_SYMBOL[op])
        return [BOS, a, op_tok, b, "=", ANS, EOS]
    t = _NL_TEMPLATES[rand_int(seed, 0, len(_NL_TEMPLATES), "nl_template",
                               op, example["a"], example["b"])]
    body = t(a, OP_WORD[op], OP_NOUN[op], b)
    seq = [BOS] + body + [ANS, EOS]
    return seq + [PAD] * (NL_SEQ_LEN - len(seq))


def encode(examples: list[dict], vocab: Vocab, task: str = "arith",
           seed: int = 0) -> dict:
    """Token ids + supervision targets as numpy arrays.

    The regression targets ARE the numeric block of the answer's own
    embedding: scalar value/TARGET_SCALE (signed), log10(|c|+1), sign(c), and
    the 12 Fourier phase dims produced by the same ``numeric_features`` that
    builds input embeddings. The model predicts the answer's embedding; the
    analytic decoder inverts it — no softmax over answers is required.
    ``ans_pos`` is per-example: NL templates place <ans> at varying positions.
    """
    lay = LAYOUT
    n = len(examples)
    seq_len = ARITH_SEQ_LEN if task == "arith" else NL_SEQ_LEN
    n_fourier = 2 * len(lay.fourier_val_periods)
    ids = np.zeros((n, seq_len), dtype=np.int64)
    ans_pos = np.zeros(n, dtype=np.int64)
    y_lin = np.zeros(n, dtype=np.float32)
    y_log = np.zeros(n, dtype=np.float32)
    y_sign = np.zeros(n, dtype=np.float32)
    y_fourier = np.zeros((n, n_fourier), dtype=np.float32)
    y_cls = np.full(n, -100, dtype=np.int64)  # ignore_index when c not in vocab
    ops = np.zeros(n, dtype=np.int64)         # index into OPS
    values = np.zeros(n, dtype=np.int64)
    for i, e in enumerate(examples):
        seq = _tokens_for(e, task, seed)
        ids[i] = [vocab.id(t) for t in seq]
        ans_pos[i] = seq.index(ANS)
        c = e["c"]
        y_lin[i] = np.float32(c / TARGET_SCALE)
        y_log[i] = np.float32(math.log10(abs(c) + 1))
        y_sign[i] = np.float32(0.0 if c == 0 else math.copysign(1.0, c))
        y_fourier[i] = numeric_features(c)[lay.fourier_val_lo:lay.fourier_log_lo]
        if 0 <= c and str(c) in vocab.to_id:
            y_cls[i] = vocab.id(str(c))
        ops[i] = OPS.index(e["op"])
        values[i] = c
    return {"ids": ids, "ans_pos": ans_pos, "y_lin": y_lin, "y_log": y_log,
            "y_sign": y_sign, "y_fourier": y_fourier, "y_cls": y_cls,
            "ops": ops, "values": values, "task": task}
