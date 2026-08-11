"""Claim A battery: the properties module itself, plus direct spot checks."""

import numpy as np

from kronembed.embedding import embed_token, numeric_features
from kronembed.layout import LAYOUT, MAX_EXACT_VALUE
from kronembed.properties import run_properties
from kronembed.util import rand_int

SEED = 20260811


def test_run_properties_all_pass():
    report = run_properties(SEED, coord="test", n_pairs=2_000, n_words=300)
    failed = [c["name"] for c in report["checks"] if not c["ok"]]
    assert report["all_ok"], failed
    assert len(report["checks"]) == 8


def test_lin_additivity_is_bit_exact_not_approximate():
    for i in range(2_000):
        a = rand_int(SEED, 0, MAX_EXACT_VALUE // 2, "a", i)
        b = rand_int(SEED, 0, MAX_EXACT_VALUE // 2, "b", i)
        lin_a = numeric_features(a)[LAYOUT.LIN]
        lin_b = numeric_features(b)[LAYOUT.LIN]
        lin_sum = numeric_features(a + b)[LAYOUT.LIN]
        # float32 '==', deliberately not np.isclose
        assert np.float32(lin_a + lin_b) == lin_sum


def test_log_dim_makes_multiplication_additive():
    worst = 0.0
    for i in range(2_000):
        a = rand_int(SEED, 1, 1000, "ma", i)
        b = rand_int(SEED, 1, 1000, "mb", i)
        la = float(numeric_features(a)[LAYOUT.LOG])
        lb = float(numeric_features(b)[LAYOUT.LOG])
        lab = float(numeric_features(a * b)[LAYOUT.LOG])
        worst = max(worst, abs(la + lb - lab))
    assert worst <= 1e-5


def test_showcase_9_plus_9_equals_18():
    s = embed_token("9") + embed_token("9")
    assert np.float32(s[LAYOUT.LIN]) == embed_token("18")[LAYOUT.LIN]


def test_properties_report_is_deterministic():
    r1 = run_properties(SEED, coord="det", n_pairs=200, n_words=50)
    r2 = run_properties(SEED, coord="det", n_pairs=200, n_words=50)
    assert r1 == r2
