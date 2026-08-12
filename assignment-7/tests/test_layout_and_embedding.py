"""Layout tiling, embedding construction, and variant-zeroing invariants."""

import numpy as np
import pytest

from kronembed.embedding import (VARIANTS, build_embedding_matrix, embed_token,
                                 embed_word, numeric_features, token_value)
from kronembed.layout import ALPHABET, LAYOUT, LIN_SCALE, LOG_ZERO
from kronembed.vocab import Vocab


def test_layout_blocks_tile_without_overlap():
    lay = LAYOUT
    assert lay.char_hi == lay.n_slots * lay.char_dims
    scalar_dims = {lay.LIN, lay.SIGN, lay.LOG, lay.NUMFLAG}
    fourier_val = set(range(lay.fourier_val_lo,
                            lay.fourier_val_lo + 2 * len(lay.fourier_val_periods)))
    fourier_log = set(range(lay.fourier_log_lo,
                            lay.fourier_log_lo + 2 * len(lay.fourier_log_periods)))
    char = set(range(lay.char_lo, lay.char_hi))
    reserved = set(range(lay.reserved_lo, lay.d_model))
    groups = [char, scalar_dims, fourier_val, fourier_log, reserved]
    union = set().union(*groups)
    assert sum(len(g) for g in groups) == len(union) == lay.d_model


def test_char_block_norm_is_one_for_any_length():
    for w in ["a", "ab", "apple", "a" * 32]:
        vec = embed_word(w)
        assert np.isclose(np.linalg.norm(vec[:LAYOUT.char_hi]), 1.0, atol=1e-5)


def test_words_longer_than_32_are_rejected_not_cropped():
    with pytest.raises(ValueError):
        embed_word("a" * 33)


def test_unknown_character_rejected():
    with pytest.raises(ValueError):
        embed_word("café")


def test_numeric_token_fills_both_blocks():
    vec = embed_token("42")
    assert np.abs(vec[:LAYOUT.char_hi]).max() > 0        # char block live
    assert vec[LAYOUT.NUMFLAG] == 1.0
    assert vec[LAYOUT.LIN] == np.float32(42 / LIN_SCALE)


def test_zero_value_flagged_but_log_sentinel():
    vec = numeric_features(0)
    assert vec[LAYOUT.NUMFLAG] == 1.0
    assert vec[LAYOUT.SIGN] == 0.0
    assert vec[LAYOUT.LOG] == np.float32(LOG_ZERO)


def test_variant_zeroing_is_exactly_the_declared_dims():
    full = embed_token("137")
    char_only = embed_token("137", variant="kron_char")
    readout = embed_token("137", variant="readout_only")
    hom = embed_token("137", variant="hom_only")
    # kron_char: numeric block entirely zero, char block identical
    assert np.array_equal(char_only[:LAYOUT.char_hi], full[:LAYOUT.char_hi])
    assert np.abs(char_only[LAYOUT.char_hi:]).max() == 0.0
    # readout_only differs from full in exactly {LIN, SIGN, LOG}
    diff = np.nonzero(readout != full)[0].tolist()
    assert diff == sorted([LAYOUT.LIN, LAYOUT.SIGN, LAYOUT.LOG])
    assert all(readout[d] == 0.0 for d in diff)
    # hom_only differs from full in exactly the Fourier dims
    diff_h = np.nonzero(hom != full)[0].tolist()
    assert diff_h == list(range(LAYOUT.fourier_val_lo, LAYOUT.reserved_lo))
    assert all(hom[d] == 0.0 for d in diff_h)
    # hom_only and readout_only partition the numeric block (flag shared)
    assert np.array_equal(hom[LAYOUT.char_hi:] + readout[LAYOUT.char_hi:]
                          - np.where(np.arange(LAYOUT.d_model) == LAYOUT.NUMFLAG,
                                     1.0, 0.0)[LAYOUT.char_hi:].astype(np.float32),
                          full[LAYOUT.char_hi:])


def test_embedding_matrix_deterministic_and_variant_shapes():
    vocab = Vocab()
    m1 = build_embedding_matrix(vocab.tokens)
    m2 = build_embedding_matrix(vocab.tokens)
    assert np.array_equal(m1, m2)
    assert m1.shape == (len(vocab), LAYOUT.d_model)
    for v in VARIANTS:
        assert build_embedding_matrix(vocab.tokens, variant=v).shape == m1.shape


def test_random_matrix_is_deterministic_and_norm_matched():
    from kronembed.embedding import build_random_matrix
    vocab = Vocab()
    m1 = build_random_matrix(vocab.tokens)
    m2 = build_random_matrix(vocab.tokens)
    assert np.array_equal(m1, m2)
    ref = build_embedding_matrix(vocab.tokens)
    assert np.isclose(np.linalg.norm(m1, axis=1).mean(),
                      np.linalg.norm(ref, axis=1).mean(), rtol=1e-4)
    # and it carries no numeric structure: LIN dim does not decode values
    from kronembed.embedding import decode_value
    row = m1[vocab.id("42")]
    assert decode_value(row) != 42


def test_token_value_parsing():
    assert token_value("0") == 0
    assert token_value("999") == 999
    assert token_value("plus") is None
    assert token_value("<ans>") is None
    assert token_value("-5") is None  # negatives are out of scope, not silently parsed


def test_alphabet_covers_vocab():
    for tok in Vocab().tokens:
        assert all(c in ALPHABET for c in tok), tok
