"""Deterministic arithmetic corpus: nested train sets, three eval regimes.

Split construction guarantees, by construction rather than by checking:

  * a contiguous OPERAND HOLE [40, 59] is excluded from training entirely:
    no training pair touches any operand in the hole. eval_hole pairs all do.
    For a learned embedding table the hole tokens remain untrained random
    rows; for a deterministic scheme their embeddings are analytically
    correct — this split is the token-level generalization test.
  * eval-in-range and every train set are drawn from one deterministic
    shuffle of all hole-free (op, a, b) triples with operands in [0, 100):
    eval-in-range is the first 1000 triples, train sets are prefixes of the
    remainder. Train sets of different sizes are therefore *nested*
    (train_500 ⊂ train_2000 ⊂ train_8000) and never overlap eval.
  * extrapolation triples have max(a, b) in [100, 999] (b <= a keeps each
    example in its assigned magnitude bucket) — the magnitude-extrapolation
    stress test. Products stay below 2**20, inside the exact-additivity
    domain of the LIN dim.

25% of examples use word-form operators ("plus"/"times") so the unified
word+number embedding path is exercised by the same model.
"""

from __future__ import annotations

import math

import numpy as np

from .embedding import numeric_features
from .layout import LAYOUT, TARGET_SCALE
from .util import deterministic_shuffle, rand_float, rand_int, sha256_json
from .vocab import ANS, BOS, EOS, Vocab

IN_RANGE_MAX = 100          # train/eval operands are in [0, IN_RANGE_MAX)
EXTRA_MAX = 1000            # extrapolation operands are in [0, EXTRA_MAX)
HOLE = (40, 59)             # operands never seen in training, inclusive band
EVAL_IN_SIZE = 1_000
HOLE_SIZE = 1_000
EXTRA_SIZE = 2_000
WORD_FORM_PROB = 0.25
BUCKETS = ((100, 199), (200, 499), (500, 999))


def in_hole(x: int) -> bool:
    return HOLE[0] <= x <= HOLE[1]

OP_SYMBOL = {"add": "+", "mul": "*"}
OP_WORD = {"add": "plus", "mul": "times"}


def _apply(op: str, a: int, b: int) -> int:
    return a + b if op == "add" else a * b


def _surface(seed: int, key: tuple) -> str:
    return "word" if rand_float(seed, "surface", *key) < WORD_FORM_PROB else "sym"


def _example(op: str, a: int, b: int, seed: int) -> dict:
    return {"op": op, "a": a, "b": b, "c": _apply(op, a, b),
            "surface": _surface(seed, (op, a, b))}


def build_splits(seed: int, train_size: int) -> dict:
    """All splits + manifest for one train size. Pure function of (seed, size)."""
    clear = [(op, a, b)
             for op in ("add", "mul")
             for a in range(IN_RANGE_MAX)
             for b in range(IN_RANGE_MAX)
             if not (in_hole(a) or in_hole(b))]
    holed = [(op, a, b)
             for op in ("add", "mul")
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
        op = ("add", "mul")[i % 2]
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
        "train_answers_in_hole_band": sum(1 for e in train if in_hole(e["c"])),
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
# Encoding to tensors
# ---------------------------------------------------------------------------

SEQ_LEN = 7  # <bos> a op b = <ans> <eos>


def encode(examples: list[dict], vocab: Vocab) -> dict:
    """Token ids + supervision targets as numpy arrays.

    The regression targets ARE the numeric block of the answer's own
    embedding: scalar value/TARGET_SCALE, log10(c+1), and the 12 Fourier phase
    dims produced by the same ``numeric_features`` that builds input
    embeddings. The model predicts the answer's embedding; the analytic
    decoder inverts it — no softmax over answers is required.
    """
    lay = LAYOUT
    n = len(examples)
    n_fourier = 2 * len(lay.fourier_val_periods)
    ids = np.zeros((n, SEQ_LEN), dtype=np.int64)
    y_lin = np.zeros(n, dtype=np.float32)
    y_log = np.zeros(n, dtype=np.float32)
    y_fourier = np.zeros((n, n_fourier), dtype=np.float32)
    y_cls = np.full(n, -100, dtype=np.int64)  # ignore_index when c not in vocab
    ops = np.zeros(n, dtype=np.int64)         # 0 = add, 1 = mul
    values = np.zeros(n, dtype=np.int64)
    for i, e in enumerate(examples):
        op_tok = OP_WORD[e["op"]] if e["surface"] == "word" else OP_SYMBOL[e["op"]]
        seq = [BOS, str(e["a"]), op_tok, str(e["b"]), "=", ANS, EOS]
        ids[i] = [vocab.id(t) for t in seq]
        c = e["c"]
        y_lin[i] = np.float32(c / TARGET_SCALE)
        y_log[i] = np.float32(math.log10(c + 1))
        y_fourier[i] = numeric_features(c)[lay.fourier_val_lo:lay.fourier_log_lo]
        if str(c) in vocab.to_id:
            y_cls[i] = vocab.id(str(c))
        ops[i] = 0 if e["op"] == "add" else 1
        values[i] = c
    return {"ids": ids, "y_lin": y_lin, "y_log": y_log, "y_fourier": y_fourier,
            "y_cls": y_cls, "ops": ops, "values": values,
            "ans_pos": SEQ_LEN - 2}  # heads read the hidden state at <ans>
