"""Answer decoding and metric computation — pure numpy, no model state.

The PRIMARY pre-registered decode reconstructs the integer digit-by-digit
from the predicted Fourier phases (the answer-embedding readout): the phase
pair of period 10^(k+1) fixes digit k, given the digits below it. Each digit
tolerates a phase error of a twentieth of a circle — six forgiving decodes
instead of one impossible-precision scalar regression. The scalar linear and
log heads are decoded and reported as well (the homomorphic-path readouts),
alongside the classification head with its structural <=999 ceiling.
"""

from __future__ import annotations

import numpy as np

from .layout import LAYOUT, TARGET_SCALE
from .vocab import Vocab

MAX_LOG_DECODE = 7.0  # clip before 10**x so an early wild head can't overflow


def decode_lin(y: np.ndarray) -> np.ndarray:
    """Signed: subtraction answers are negative and decode as such."""
    return np.round(y * TARGET_SCALE).astype(np.int64)


def decode_log(y: np.ndarray, sign: np.ndarray | None = None) -> np.ndarray:
    """Magnitude from the log head, re-signed from the sign head if given."""
    y = np.clip(y, 0.0, MAX_LOG_DECODE)
    mag = np.maximum(0, np.round(np.power(10.0, y) - 1.0)).astype(np.int64)
    if sign is None:
        return mag
    return np.where(sign < -0.5, -mag, mag)


def decode_fourier(phases: np.ndarray) -> np.ndarray:
    """(N, 12) predicted (sin, cos) pairs -> (N,) integers in [0, 10^6).

    Digit k comes from the period-10^(k+1) pair: theta/(2*pi)*T estimates
    v mod T; subtracting the already-reconstructed v mod 10^k and rounding
    gives the digit. Exactly inverts ``numeric_features`` on clean inputs.
    """
    n = phases.shape[0]
    v = np.zeros(n, dtype=np.int64)
    for k, T in enumerate(LAYOUT.fourier_val_periods):
        s = phases[:, 2 * k].astype(np.float64)
        c = phases[:, 2 * k + 1].astype(np.float64)
        theta = np.mod(np.arctan2(s, c), 2.0 * np.pi)
        est_mod_T = theta / (2.0 * np.pi) * T          # ~ v mod T
        base = T // 10                                  # 10^k
        digit = np.round((est_mod_T - v) / base).astype(np.int64) % 10
        v = v + digit * base
    return v


def decode_fourier_signed(phases: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Negative values: phases encode v mod 10^6 (CRT-consistent, so
    -57 -> 999943); the sign head selects the signed representative."""
    T = LAYOUT.fourier_val_periods[-1]
    v = decode_fourier(phases)
    return np.where(sign < -0.5, v - T, v)


def _rates(pred: np.ndarray, truth: np.ndarray) -> dict:
    err = np.abs(pred - truth).astype(np.float64)
    rel = err / np.maximum(1.0, np.abs(truth))
    return {
        "exact": float((pred == truth).mean()),
        "mae": float(err.mean()),
        "relerr_median": float(np.median(rel)),
        "within_1pct": float((rel <= 0.01).mean()),
        "within_01pct": float((rel <= 0.001).mean()),
        "n": int(truth.size),
    }


def evaluate_split(reg: np.ndarray, cls_ids: np.ndarray, values: np.ndarray,
                   ops: np.ndarray, vocab: Vocab,
                   buckets: list | None = None) -> dict:
    """reg: (N, 15) head outputs at <ans> — [lin, log, sign, 12 fourier];
    cls_ids: (N,) argmax token ids; values: (N,) true answers (signed);
    ops: (N,) 0=add 1=mul 2=sub; buckets: optional (N,) magnitude-bucket
    labels for the extrapolation split."""
    sign = reg[:, 2]
    c_lin = decode_lin(reg[:, 0])
    c_log = decode_log(reg[:, 1], sign)
    primary = decode_fourier_signed(reg[:, 3:15], sign)
    cls_vals = np.array(
        [v if (v := vocab.value_of_id(int(i))) is not None else -1
         for i in cls_ids], dtype=np.int64)

    out = {"overall": _rates(primary, values)}
    for op_id, op_name in ((0, "add"), (1, "mul"), (2, "sub")):
        m = ops == op_id
        if not m.any():
            continue
        out[op_name] = {
            "primary": _rates(primary[m], values[m]),
            "lin_decode": _rates(c_lin[m], values[m]),
            "log_decode": _rates(c_log[m], values[m]),
            "cls_decode": _rates(cls_vals[m], values[m]),
            "cls_expressible": float(((values[m] >= 0)
                                      & (values[m] <= 999)).mean()),
        }
    if buckets is not None:
        buckets = np.asarray(buckets)
        out["by_bucket"] = {}
        for b in sorted(set(buckets.tolist())):
            bm = buckets == b
            out["by_bucket"][b] = {"overall": _rates(primary[bm], values[bm])}
            for op_id, op_name in ((0, "add"), (1, "mul"), (2, "sub")):
                sel = bm & (ops == op_id)
                if sel.any():
                    out["by_bucket"][b][op_name] = _rates(primary[sel],
                                                          values[sel])
    return out
