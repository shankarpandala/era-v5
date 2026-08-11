"""The analytic decoders invert clean numeric features exactly, and degrade
gracefully under bounded phase noise."""

import numpy as np

from kronembed.embedding import numeric_features
from kronembed.layout import LAYOUT, TARGET_SCALE
from kronembed.metrics import decode_fourier, decode_lin, decode_log
from kronembed.util import rand_int

SEED = 20260811
LO, HI = LAYOUT.fourier_val_lo, LAYOUT.fourier_log_lo


def _phases(vals):
    return np.stack([numeric_features(v)[LO:HI] for v in vals])


def test_fourier_decode_inverts_numeric_features_exactly():
    vals = [0, 1, 9, 10, 42, 99, 100, 999, 1998, 9801, 123456, 998001]
    assert decode_fourier(_phases(vals)).tolist() == vals


def test_fourier_decode_inverts_random_values():
    vals = [rand_int(SEED, 0, 10 ** 6, "fd", i) for i in range(500)]
    assert decode_fourier(_phases(vals)).tolist() == vals


def test_fourier_decode_tolerates_small_phase_noise():
    vals = [rand_int(SEED, 0, 10 ** 6, "fn", i) for i in range(200)]
    phases = _phases(vals)
    rng = np.random.default_rng(0)
    noisy = phases + rng.uniform(-0.05, 0.05, phases.shape).astype(np.float32)
    assert (decode_fourier(noisy) == np.array(vals)).mean() > 0.95


def test_lin_and_log_decodes():
    y = np.array([42 / TARGET_SCALE, 1998 / TARGET_SCALE], dtype=np.float64)
    assert decode_lin(y).tolist() == [42, 1998]
    assert decode_log(np.log10(np.array([43.0, 1999.0]))).tolist() == [42, 1998]
    # negative head output clamps to zero instead of going negative
    assert decode_lin(np.array([-0.3])).tolist() == [0]
