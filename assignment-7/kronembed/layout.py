"""The embedding dimension budget — the single frozen source of truth.

Every dimension of the 128-d embedding has a declared owner and a declared
algebraic status:

  homomorphic dims   -- vector arithmetic on them mirrors math *exactly*
                        (LIN, SIGN: addition) or provably-approximately
                        (LOG: multiplication becomes addition, float32 log
                        rounding only).
  readout dims       -- exist so a tiny transformer can *read* precise digits
                        and magnitude through LayerNorm (Fourier features,
                        NUMFLAG). Not homomorphic, and never claimed to be.
  char dims          -- orthographic identity: 32 slots x 3-d character codes,
                        invertible by per-slot nearest-neighbour. This is the
                        original "Kronecker" idea (slot-position (x) char-code).

Nothing in this file is learned. Changing any constant changes the content
hash of every embedding matrix, which the audit verifies end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Characters the char block can represent. Sorted, fixed, hashed. 44 symbols:
# lowercase letters, digits, and the operators/special-token punctuation used
# by the task vocabulary.
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789+*=<>-_#"

# v / LIN_SCALE is exact in float32 for any integer |v| < 2**24 because the
# integer fits the 24-bit mantissa and dividing by a power of two only shifts
# the exponent. We cap the *claimed* exact-additivity domain at 2**20 so sums
# and products of task values always stay far inside the mantissa.
LIN_SCALE = 2 ** 14
MAX_EXACT_VALUE = 2 ** 20

# Sentinel written to the LOG dim for v == 0 (log10 undefined). Chosen far
# below log10(1) = 0 so "zero" is linearly separable from every v >= 1.
LOG_ZERO = -8.0

# Regression-head target scale (power of two, same dyadic argument as LIN):
# y_lin_target = value / TARGET_SCALE keeps in-range addition targets O(1) so
# the linear head's loss is not vanishingly small next to the other terms.
TARGET_SCALE = 2 ** 8


@dataclass(frozen=True)
class Layout:
    d_model: int = 128

    # --- character block: dims [0, 96) ------------------------------------
    n_slots: int = 32          # 32 character slots per token
    char_dims: int = 3         # slot s occupies dims [3s, 3s+3)
    char_lo: int = 0
    char_hi: int = 96

    # --- numeric block: dims [96, 128) ------------------------------------
    # NOTE on LOG conventions: the embedding's LOG dim is the algebraic object
    # log10(|v|) (sentinel LOG_ZERO at v=0) — exact math, used by Claim A. The
    # regression HEAD's log target is log10(|c|+1) instead, because answers
    # include 0 and a trainable head needs a smooth target there; the two are
    # deliberately different and this is the one place that documents why.
    LIN: int = 96              # v / LIN_SCALE, signed    (exactly additive)
    SIGN: int = 97             # sign(v) in {-1, 0, +1}   (subtraction readout)
    LOG: int = 98              # log10(|v|) for v!=0      (mult/div -> +/-)
    NUMFLAG: int = 99          # 1.0 iff token is numeric (0 disambiguates v=0)

    # sin/cos pairs of 2*pi*(v mod T)/T -- digit/period readout (NOT homomorphic)
    fourier_val_periods: tuple = (10, 100, 1_000, 10_000, 100_000, 1_000_000)
    fourier_val_lo: int = 100  # dims [100, 112): 6 periods x (sin, cos)

    # sin/cos pairs of 2*pi*log10(|v|)/L -- INPUT-side magnitude readout for
    # multiplication. Deliberately not used by any output decoder (the output
    # side decodes digits from fourier_val + sign); kept because ablating it
    # is part of the readout_only/hom_only arm design.
    fourier_log_periods: tuple = (1.0, 2.0, 4.0, 8.0)
    fourier_log_lo: int = 112  # dims [112, 120): 4 periods x (sin, cos)

    reserved_lo: int = 120     # dims [120, 128): zeros, reserved

    def describe(self) -> dict:
        """Machine-readable layout summary (hashed into run_config)."""
        return {
            "d_model": self.d_model,
            "alphabet": ALPHABET,
            "char_block": {"lo": self.char_lo, "hi": self.char_hi,
                           "n_slots": self.n_slots, "char_dims": self.char_dims},
            "numeric_block": {
                "LIN": self.LIN, "SIGN": self.SIGN, "LOG": self.LOG,
                "NUMFLAG": self.NUMFLAG,
                "fourier_val": {"lo": self.fourier_val_lo,
                                "periods": list(self.fourier_val_periods)},
                "fourier_log": {"lo": self.fourier_log_lo,
                                "periods": list(self.fourier_log_periods)},
                "reserved_lo": self.reserved_lo,
            },
            "constants": {"LIN_SCALE": LIN_SCALE, "LOG_ZERO": LOG_ZERO,
                          "MAX_EXACT_VALUE": MAX_EXACT_VALUE,
                          "TARGET_SCALE": TARGET_SCALE},
            "dim_taxonomy": {
                "homomorphic_add": ["LIN", "SIGN"],
                "homomorphic_mul_via_add": ["LOG"],
                "readout_only": ["NUMFLAG", "fourier_val", "fourier_log"],
                "orthographic": ["char_block"],
            },
        }


LAYOUT = Layout()

# Static sanity: the blocks tile [0, 128) without overlap.
assert LAYOUT.n_slots * LAYOUT.char_dims == LAYOUT.char_hi - LAYOUT.char_lo
assert LAYOUT.fourier_val_lo + 2 * len(LAYOUT.fourier_val_periods) == LAYOUT.fourier_log_lo
assert LAYOUT.fourier_log_lo + 2 * len(LAYOUT.fourier_log_periods) == LAYOUT.reserved_lo
assert LAYOUT.reserved_lo <= LAYOUT.d_model
