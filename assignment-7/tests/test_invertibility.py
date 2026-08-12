"""Analytic-inverse guarantees: chars and values decode back exactly."""

import numpy as np

from kronembed.embedding import (codebook_max_cosine, decode_chars,
                                 decode_value, embed_token, embed_word,
                                 numeric_features)
from kronembed.layout import ALPHABET, LAYOUT, MAX_EXACT_VALUE
from kronembed.util import rand_int
from kronembed.vocab import Vocab

SEED = 20260811


def test_codebook_margin():
    # Fibonacci lattice with 44 points: worst pairwise cosine must leave a
    # real margin for nearest-neighbour decoding.
    assert codebook_max_cosine() < 0.95


def test_every_vocab_token_round_trips():
    for tok in Vocab().tokens:
        assert decode_chars(embed_token(tok)) == tok


def test_random_words_round_trip():
    for i in range(500):
        length = rand_int(SEED, 1, LAYOUT.n_slots + 1, "len", i)
        w = "".join(ALPHABET[rand_int(SEED, 0, len(ALPHABET), "ch", i, j)]
                    for j in range(length))
        assert decode_chars(embed_word(w)) == w


def test_value_round_trips_exhaustive_small():
    for v in range(0, 2001):
        assert decode_value(numeric_features(v)) == v


def test_value_round_trips_sampled_large():
    for i in range(1000):
        v = rand_int(SEED, 0, MAX_EXACT_VALUE, "v", i)
        assert decode_value(numeric_features(v)) == v


def test_negative_values_round_trip():
    for v in [-1, -57, -999, -12345, -(2 ** 19)]:
        assert decode_value(numeric_features(v)) == v


def test_nonnumeric_decodes_to_none():
    assert decode_value(embed_token("plus")) is None
    assert decode_value(embed_token("<ans>")) is None


def test_sum_of_embeddings_decodes_to_sum_of_values():
    # The homomorphism showcase, as a hard test.
    for a, b in [(9, 9), (0, 0), (123, 877), (999, 999), (2 ** 18, 2 ** 18)]:
        s = numeric_features(a) + numeric_features(b)
        assert decode_value(s) == a + b
