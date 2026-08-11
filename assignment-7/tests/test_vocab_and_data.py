"""Vocabulary identity and data-split invariants."""

import numpy as np

from kronembed.data import (BUCKETS, EVAL_IN_SIZE, HOLE, HOLE_SIZE,
                            IN_RANGE_MAX, SEQ_LEN, build_splits, encode,
                            in_hole)
from kronembed.layout import MAX_EXACT_VALUE, TARGET_SCALE
from kronembed.vocab import Vocab

SEED = 20260811


def test_vocab_layout():
    v = Vocab()
    assert len(v) == 4 + 5 + 1000
    assert v.id("<pad>") == 0
    assert v.token(v.id("999")) == "999"
    assert v.value_of_id(v.id("42")) == 42
    assert v.value_of_id(v.id("plus")) is None


def test_splits_deterministic_and_disjoint():
    d1 = build_splits(SEED, 2000)
    d2 = build_splits(SEED, 2000)
    assert d1["manifest"]["hashes"] == d2["manifest"]["hashes"]
    assert d1["manifest"]["disjoint_train_eval"] is True
    train_keys = {(e["op"], e["a"], e["b"]) for e in d1["splits"]["train"]}
    eval_keys = {(e["op"], e["a"], e["b"]) for e in d1["splits"]["eval_in"]}
    assert not train_keys & eval_keys
    assert len(d1["splits"]["eval_in"]) == EVAL_IN_SIZE


def test_train_sets_are_nested():
    small = build_splits(SEED, 500)["splits"]["train"]
    large = build_splits(SEED, 8000)["splits"]["train"]
    assert small == large[:500]
    # eval is identical across sizes
    assert (build_splits(SEED, 500)["manifest"]["hashes"]["eval_in"]
            == build_splits(SEED, 8000)["manifest"]["hashes"]["eval_in"])


def test_train_operands_in_range_extra_out_of_range():
    d = build_splits(SEED, 2000)
    for e in d["splits"]["train"] + d["splits"]["eval_in"]:
        assert 0 <= e["a"] < IN_RANGE_MAX and 0 <= e["b"] < IN_RANGE_MAX
    for e in d["splits"]["eval_extra"]:
        lo, hi = map(int, e["bucket"].split("-"))
        assert lo <= max(e["a"], e["b"]) <= hi
        assert (lo, hi) in BUCKETS


def test_operand_hole_is_absolute():
    d = build_splits(SEED, 8000)
    for e in d["splits"]["train"] + d["splits"]["eval_in"]:
        assert not in_hole(e["a"]) and not in_hole(e["b"])
    assert len(d["splits"]["eval_hole"]) == HOLE_SIZE
    for e in d["splits"]["eval_hole"]:
        assert in_hole(e["a"]) or in_hole(e["b"])
        assert 0 <= e["a"] < IN_RANGE_MAX and 0 <= e["b"] < IN_RANGE_MAX
    m = d["manifest"]
    assert m["train_never_touches_hole"] is True
    assert m["hole_always_touched"] is True
    assert m["hole"] == list(HOLE)


def test_all_values_inside_exact_domain():
    d = build_splits(SEED, 8000)
    for split in d["splits"].values():
        for e in split:
            assert e["c"] < MAX_EXACT_VALUE
            assert e["c"] == (e["a"] + e["b"] if e["op"] == "add"
                              else e["a"] * e["b"])


def test_encode_shapes_and_targets():
    v = Vocab()
    d = build_splits(SEED, 500)
    enc = encode(d["splits"]["train"], v)
    n = len(d["splits"]["train"])
    assert enc["ids"].shape == (n, SEQ_LEN)
    assert enc["ans_pos"] == 5
    # every sequence starts <bos> and the <ans> slot is the <ans> token
    assert (enc["ids"][:, 0] == v.id("<bos>")).all()
    assert (enc["ids"][:, 5] == v.id("<ans>")).all()
    # answers never appear as input tokens
    for i, e in enumerate(d["splits"]["train"][:50]):
        assert np.float32(e["c"] / TARGET_SCALE) == enc["y_lin"][i]
        if e["c"] <= 999:
            assert enc["y_cls"][i] == v.id(str(e["c"]))
        else:
            assert enc["y_cls"][i] == -100


def test_word_form_fraction_near_quarter():
    m = build_splits(SEED, 8000)["manifest"]
    assert 0.2 < m["word_form_fraction"] < 0.3
