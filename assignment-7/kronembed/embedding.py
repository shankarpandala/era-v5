"""Deterministic embedding constructions and their analytic decoders.

The V2 scheme ("kron_v2") composes two independent blocks:

  * char block  -- slot s holds the code of character s, scaled by 1/sqrt(len).
                   Codes are points of a Fibonacci sphere lattice: deterministic,
                   near-optimally separated, so per-slot nearest-neighbour
                   decoding is exact by construction (no seed luck involved).
  * numeric block -- filled only when the token parses as a non-negative
                   integer: exact linear value dim, sign, log10 dim, numeric
                   flag, and Fourier readout features (see layout.py for the
                   homomorphic / readout taxonomy).

Every function here is a pure function of its inputs — no RNG, no learned
state. ``build_embedding_matrix`` supports ablation variants that zero out
declared dim groups so experiment arms differ *only* in embedding content.
"""

from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np

from .layout import (ALPHABET, LAYOUT, LIN_SCALE, LOG_ZERO, MAX_EXACT_VALUE,
                     Layout)

_INT_RE = re.compile(r"^[0-9]+$")

# Embedding-matrix variants: which dim groups get zeroed relative to kron_v2.
# "readout_only" is the FoNE-style arm: it keeps every readout feature and
# drops exactly the homomorphic dims, so kron_v2 - readout_only isolates the
# contribution of {LIN, SIGN, LOG}.
VARIANTS = ("kron_v2", "kron_char", "readout_only")


def token_value(token: str) -> Optional[int]:
    """The integer value a token denotes, or None for non-numeric tokens."""
    if _INT_RE.match(token):
        return int(token)
    return None


# ---------------------------------------------------------------------------
# Character codebook — Fibonacci sphere lattice
# ---------------------------------------------------------------------------


def char_codebook() -> np.ndarray:
    """(n_chars, 3) float32 unit vectors, one per ALPHABET character.

    Character i receives point i of an n-point Fibonacci sphere lattice —
    a deterministic near-optimal packing, so the minimum pairwise angle is a
    fixed, testable constant rather than a random variable.
    """
    n = len(ALPHABET)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    codes = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        phi = golden_angle * i
        codes[i] = (r * math.cos(phi), r * math.sin(phi), z)
    return codes.astype(np.float32)


_CODEBOOK = char_codebook()
_CHAR_INDEX = {c: i for i, c in enumerate(ALPHABET)}


def codebook_max_cosine() -> float:
    """Largest pairwise cosine similarity in the codebook (decoding margin)."""
    g = _CODEBOOK.astype(np.float64) @ _CODEBOOK.astype(np.float64).T
    np.fill_diagonal(g, -1.0)
    return float(g.max())


# ---------------------------------------------------------------------------
# Char block
# ---------------------------------------------------------------------------


def embed_word(word: str, layout: Layout = LAYOUT) -> np.ndarray:
    """Char block only: (d_model,) with dims [char_lo, char_hi) populated.

    Slot s gets code(word[s]) / sqrt(len(word)) so the block norm is exactly 1
    for any word length. Words longer than n_slots are rejected loudly —
    silent cropping is precisely the V1 defect this assignment documents.
    """
    if len(word) == 0:
        raise ValueError("cannot embed empty word")
    if len(word) > layout.n_slots:
        raise ValueError(f"word longer than {layout.n_slots} slots: {word!r}")
    vec = np.zeros(layout.d_model, dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(len(word)))
    for s, ch in enumerate(word):
        if ch not in _CHAR_INDEX:
            raise ValueError(f"character {ch!r} not in ALPHABET")
        lo = layout.char_lo + s * layout.char_dims
        vec[lo:lo + layout.char_dims] = _CODEBOOK[_CHAR_INDEX[ch]] * scale
    return vec


def decode_chars(vec: np.ndarray, layout: Layout = LAYOUT) -> str:
    """Analytic inverse of the char block: per-slot nearest-neighbour."""
    chars = []
    for s in range(layout.n_slots):
        lo = layout.char_lo + s * layout.char_dims
        seg = np.asarray(vec[lo:lo + layout.char_dims], dtype=np.float64)
        norm = float(np.linalg.norm(seg))
        if norm < 1e-6:
            break  # first empty slot ends the word
        sims = _CODEBOOK.astype(np.float64) @ (seg / norm)
        chars.append(ALPHABET[int(np.argmax(sims))])
    return "".join(chars)


# ---------------------------------------------------------------------------
# Numeric block
# ---------------------------------------------------------------------------


def numeric_features(v: int, layout: Layout = LAYOUT) -> np.ndarray:
    """(d_model,) with only the numeric block [96, 128) populated for value v.

    LIN is computed as float64 v / 2**14 then cast — exact for v < 2**24
    (integer fits the float32 mantissa; a power-of-two divide only shifts the
    exponent). Fourier phases are reduced mod T *before* the trig call so
    large values lose no phase precision.
    """
    if v < 0:
        raise ValueError("this study uses non-negative integers only")
    vec = np.zeros(layout.d_model, dtype=np.float32)
    vec[layout.LIN] = np.float32(v / LIN_SCALE)
    vec[layout.SIGN] = np.float32(0.0 if v == 0 else 1.0)
    vec[layout.LOG] = np.float32(LOG_ZERO if v == 0 else math.log10(v))
    vec[layout.NUMFLAG] = np.float32(1.0)
    for k, T in enumerate(layout.fourier_val_periods):
        theta = 2.0 * math.pi * (v % T) / T
        vec[layout.fourier_val_lo + 2 * k] = np.float32(math.sin(theta))
        vec[layout.fourier_val_lo + 2 * k + 1] = np.float32(math.cos(theta))
    logv = 0.0 if v == 0 else math.log10(v)
    for k, L in enumerate(layout.fourier_log_periods):
        theta = 2.0 * math.pi * logv / L
        vec[layout.fourier_log_lo + 2 * k] = np.float32(math.sin(theta))
        vec[layout.fourier_log_lo + 2 * k + 1] = np.float32(math.cos(theta))
    return vec


def decode_value(vec: np.ndarray, layout: Layout = LAYOUT) -> Optional[int]:
    """Analytic inverse of the value dims: None unless the numeric flag is set."""
    if float(vec[layout.NUMFLAG]) <= 0.5:
        return None
    return int(round(float(vec[layout.LIN]) * LIN_SCALE))


# ---------------------------------------------------------------------------
# Full token embedding + matrix builders
# ---------------------------------------------------------------------------


def embed_token(token: str, layout: Layout = LAYOUT,
                variant: str = "kron_v2") -> np.ndarray:
    """Full deterministic embedding: char block always, numeric block iff the
    token parses as a non-negative integer. This is the unified-scheme point:
    numbers are not a special token class — "9" is simultaneously the word
    made of the character '9' and the value nine."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    vec = embed_word(token, layout)
    v = token_value(token)
    if v is not None and variant != "kron_char":
        vec = vec + numeric_features(v, layout)
        if variant == "readout_only":
            # FoNE-style ablation: keep every readout feature, drop exactly
            # the homomorphic dims so the arm difference is {LIN, SIGN, LOG}.
            vec[layout.LIN] = 0.0
            vec[layout.SIGN] = 0.0
            vec[layout.LOG] = 0.0
    return vec


def build_embedding_matrix(vocab: list[str], layout: Layout = LAYOUT,
                           variant: str = "kron_v2") -> np.ndarray:
    """(V, d_model) float32 frozen embedding table for a token vocabulary."""
    mat = np.zeros((len(vocab), layout.d_model), dtype=np.float32)
    for i, tok in enumerate(vocab):
        mat[i] = embed_token(tok, layout, variant)
    return mat
