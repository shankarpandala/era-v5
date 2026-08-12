"""Vocabulary identity and data-split invariants, both task families."""

import numpy as np

from kronembed.data import (ARITH_SEQ_LEN, BUCKETS, EVAL_IN_SIZE, HOLE,
                            HOLE_SIZE, IN_RANGE_MAX, NL_SEQ_LEN, OPS,
                            build_splits, encode, in_hole)
from kronembed.layout import MAX_EXACT_VALUE, TARGET_SCALE
from kronembed.vocab import NL_WORDS, OPERATORS, SPECIALS, Vocab

SEED = 20260811


def test_vocab_layout():
    v = Vocab()
    assert len(v) == len(SPECIALS) + len(OPERATORS) + len(NL_WORDS) + 1000
    assert v.id("<pad>") == 0
    assert v.token(v.id("999")) == "999"
    assert v.value_of_id(v.id("42")) == 42
    assert v.value_of_id(v.id("plus")) is None
    assert v.value_of_id(v.id("minus")) is None
    assert v.value_of_id(v.id("difference")) is None


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
    assert (build_splits(SEED, 500)["manifest"]["hashes"]["eval_in"]
            == build_splits(SEED, 8000)["manifest"]["hashes"]["eval_in"])


def test_all_three_ops_present_and_correct():
    d = build_splits(SEED, 8000)
    seen_ops = {e["op"] for e in d["splits"]["train"]}
    assert seen_ops == set(OPS)
    for split in d["splits"].values():
        for e in split:
            want = {"add": e["a"] + e["b"], "mul": e["a"] * e["b"],
                    "sub": e["a"] - e["b"]}[e["op"]]
            assert e["c"] == want
            assert abs(e["c"]) < MAX_EXACT_VALUE
    # subtraction actually goes negative in-range
    assert any(e["c"] < 0 for e in d["splits"]["train"] if e["op"] == "sub")


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
    # disclosure counter: hole-band values DO occur as training answers
    assert m["train_answers_in_hole_band"] > 0


def test_encode_arith_shapes_and_targets():
    v = Vocab()
    d = build_splits(SEED, 500)
    enc = encode(d["splits"]["train"], v, "arith", SEED)
    n = len(d["splits"]["train"])
    assert enc["ids"].shape == (n, ARITH_SEQ_LEN)
    assert (enc["ans_pos"] == 5).all()
    assert (enc["ids"][:, 0] == v.id("<bos>")).all()
    assert (enc["ids"][:, 5] == v.id("<ans>")).all()
    for i, e in enumerate(d["splits"]["train"][:80]):
        c = e["c"]
        assert np.float32(c / TARGET_SCALE) == enc["y_lin"][i]
        assert enc["y_sign"][i] == (0.0 if c == 0 else np.sign(c))
        if 0 <= c <= 999:
            assert enc["y_cls"][i] == v.id(str(c))
        else:
            assert enc["y_cls"][i] == -100


def test_encode_nl_templates_and_positions():
    v = Vocab()
    d = build_splits(SEED, 500)
    enc = encode(d["splits"]["train"], v, "nl", SEED)
    n = len(d["splits"]["train"])
    assert enc["ids"].shape == (n, NL_SEQ_LEN)
    ans_id, pad_id = v.id("<ans>"), v.id("<pad>")
    for i in range(n):
        pos = enc["ans_pos"][i]
        assert enc["ids"][i, pos] == ans_id
        # tokens after <eos> are pads only
        assert all(t == pad_id for t in enc["ids"][i, pos + 2:])
    # templates vary the answer position
    assert len(set(enc["ans_pos"].tolist())) > 1
    # targets identical to the arith encoding (same underlying examples)
    enc_a = encode(d["splits"]["train"], v, "arith", SEED)
    assert np.array_equal(enc["y_lin"], enc_a["y_lin"])
    assert np.array_equal(enc["y_fourier"], enc_a["y_fourier"])


def test_word_form_fraction_near_quarter():
    m = build_splits(SEED, 8000)["manifest"]
    assert 0.2 < m["word_form_fraction"] < 0.3
