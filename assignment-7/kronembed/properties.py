"""Claim A — training-free algebraic verification of the embedding scheme.

Every property below is checked by *computing*, not asserting: the report
records sample counts, worst-case errors, and margins. The audit re-runs the
same battery at a different PRNG coordinate, so a lucky sample cannot fake a
pass.

Properties:
  P1 lin_additivity        emb(a)[LIN] + emb(b)[LIN] == emb(a+b)[LIN], bit-exact
  P2 showcase_9_plus_9     decode_value(emb("9") + emb("9")) == 18
  P3 log_multiplicativity  |logdim(a) + logdim(b) - logdim(a*b)| <= 1e-5
  P4 char_invertibility    decode_chars(embed_word(w)) == w for vocab + random words
  P5 value_invertibility   decode_value(emb(v)) == v, exhaustive small + sampled large,
                           including negatives
  P6 nonnumeric_zero_block non-numeric tokens have an all-zero numeric block
  P7 codebook_margin       max pairwise cosine of char codes below threshold
  P8 fourier_digit_readout the 10 phase points of the T=10 pair are well separated
  P9 subtraction_and_sign  emb(a)[LIN] - emb(b)[LIN] == emb(a-b)[LIN] bit-exact,
                           negatives included; division = log-dim subtraction
  P10 multi_step_chains    long sums exact on LIN; long products via LOG; mixed
                           chains exact through analytic decode -> re-encode
"""

from __future__ import annotations

import math

import numpy as np

from .embedding import (codebook_max_cosine, decode_chars, decode_value,
                        embed_token, embed_word, numeric_features)
from .layout import ALPHABET, LAYOUT, MAX_EXACT_VALUE
from .util import rand_int

LOG_TOL = 1e-5
CODEBOOK_MAX_COSINE_ALLOWED = 0.95
FOURIER_MIN_SEPARATION = 0.5


def run_properties(seed: int, coord: str = "properties",
                   n_pairs: int = 10_000, n_words: int = 2_000) -> dict:
    layout = LAYOUT
    checks = []

    # -- P1: exact additivity of the LIN dim --------------------------------
    fails = 0
    for i in range(n_pairs):
        a = rand_int(seed, 0, MAX_EXACT_VALUE // 2, coord, "add_a", i)
        b = rand_int(seed, 0, MAX_EXACT_VALUE // 2, coord, "add_b", i)
        ea = numeric_features(a)[layout.LIN]
        eb = numeric_features(b)[layout.LIN]
        es = numeric_features(a + b)[layout.LIN]
        if not (np.float32(ea + eb) == es):  # bit-exact float32 equality
            fails += 1
    checks.append({"name": "lin_additivity", "ok": fails == 0,
                   "pairs": n_pairs, "range": [0, MAX_EXACT_VALUE // 2],
                   "bit_exact_failures": fails})

    # -- P2: the literal showcase ------------------------------------------
    s = embed_token("9") + embed_token("9")
    got = decode_value(s)
    checks.append({"name": "showcase_9_plus_9", "ok": got == 18, "decoded": got})

    # -- P3: multiplication via the log dim --------------------------------
    max_err = 0.0
    n_mul = 0
    for i in range(n_pairs):
        a = rand_int(seed, 1, 1024, coord, "mul_a", i)
        b = rand_int(seed, 1, MAX_EXACT_VALUE // 1024, coord, "mul_b", i)
        la = float(numeric_features(a)[layout.LOG])
        lb = float(numeric_features(b)[layout.LOG])
        lab = float(numeric_features(a * b)[layout.LOG])
        max_err = max(max_err, abs(la + lb - lab))
        n_mul += 1
    checks.append({"name": "log_multiplicativity", "ok": max_err <= LOG_TOL,
                   "pairs": n_mul, "max_abs_error": max_err, "tolerance": LOG_TOL})

    # -- P4: char-block invertibility --------------------------------------
    from .vocab import build_vocab
    bad = []
    for tok in build_vocab():
        if decode_chars(embed_token(tok)) != tok:
            bad.append(tok)
    for i in range(n_words):
        length = rand_int(seed, 1, layout.n_slots + 1, coord, "wlen", i)
        w = "".join(ALPHABET[rand_int(seed, 0, len(ALPHABET), coord, "wch", i, j)]
                    for j in range(length))
        if decode_chars(embed_word(w)) != w:
            bad.append(w)
    checks.append({"name": "char_invertibility", "ok": not bad,
                   "vocab_words": len(build_vocab()), "random_words": n_words,
                   "failures": bad[:10]})

    # -- P5: value invertibility (negatives included) -----------------------
    bad_v = [v for v in range(-2000, 2001)
             if decode_value(numeric_features(v)) != v]
    for i in range(5_000):
        v = rand_int(seed, -MAX_EXACT_VALUE, MAX_EXACT_VALUE, coord, "vinv", i)
        if decode_value(numeric_features(v)) != v:
            bad_v.append(v)
    checks.append({"name": "value_invertibility", "ok": not bad_v,
                   "exhaustive_range": [-2000, 2000], "sampled": 5_000,
                   "failures": bad_v[:10]})

    # -- P6: non-numeric tokens leave the numeric block empty ---------------
    dirty = []
    for tok in ["<pad>", "<bos>", "<eos>", "<ans>", "+", "*", "=", "plus", "times"]:
        vec = embed_token(tok)
        if float(np.abs(vec[layout.char_hi:]).max()) != 0.0:
            dirty.append(tok)
        if decode_value(vec) is not None:
            dirty.append(tok + ":decodes")
    checks.append({"name": "nonnumeric_zero_block", "ok": not dirty,
                   "failures": dirty})

    # -- P7: codebook decoding margin --------------------------------------
    mc = codebook_max_cosine()
    checks.append({"name": "codebook_margin", "ok": mc <= CODEBOOK_MAX_COSINE_ALLOWED,
                   "max_pairwise_cosine": mc,
                   "min_pairwise_angle_deg": math.degrees(math.acos(mc)),
                   "allowed_max_cosine": CODEBOOK_MAX_COSINE_ALLOWED})

    # -- P8: the T=10 Fourier pair separates the ten digits ----------------
    pts = []
    for d in range(10):
        vec = numeric_features(d)
        pts.append((float(vec[layout.fourier_val_lo]),
                    float(vec[layout.fourier_val_lo + 1])))
    min_sep = min(math.dist(pts[i], pts[j])
                  for i in range(10) for j in range(i + 1, 10))
    checks.append({"name": "fourier_digit_readout", "ok": min_sep >= FOURIER_MIN_SEPARATION,
                   "min_pairwise_distance": min_sep,
                   "required": FOURIER_MIN_SEPARATION})

    # -- P9: subtraction is exact and the SIGN dim reads it -----------------
    sub_fails = 0
    sign_fails = 0
    div_max_err = 0.0
    for i in range(n_pairs):
        a = rand_int(seed, 0, MAX_EXACT_VALUE // 2, coord, "sub_a", i)
        b = rand_int(seed, 0, MAX_EXACT_VALUE // 2, coord, "sub_b", i)
        diff = numeric_features(a) - numeric_features(b)
        if not (np.float32(diff[layout.LIN]) == numeric_features(a - b)[layout.LIN]):
            sub_fails += 1
        want_sign = 0.0 if a == b else math.copysign(1.0, a - b)
        if float(numeric_features(a - b)[layout.SIGN]) != want_sign:
            sign_fails += 1
        # division becomes subtraction on the log dim (real-valued quotient)
        if a >= 1 and b >= 1:
            got = (float(numeric_features(a)[layout.LOG])
                   - float(numeric_features(b)[layout.LOG]))
            div_max_err = max(div_max_err, abs(got - math.log10(a / b)))
    checks.append({"name": "subtraction_and_sign",
                   "ok": sub_fails == 0 and sign_fails == 0
                         and div_max_err <= 1e-5,
                   "pairs": n_pairs, "lin_bit_exact_failures": sub_fails,
                   "sign_failures": sign_fails,
                   "division_via_log_max_err": div_max_err})

    # -- P10: multi-step chains ---------------------------------------------
    # (a) Long sums stay exact under pure vector addition on LIN.
    chain_fails = 0
    for i in range(200):
        terms = [rand_int(seed, 0, MAX_EXACT_VALUE // 16, coord, "chain", i, j)
                 for j in range(10)]
        acc = numeric_features(terms[0])
        for t in terms[1:]:
            acc = acc + numeric_features(t)
        if decode_value(acc) != sum(terms):
            chain_fails += 1
    # (b) Long products via LOG addition (real arithmetic, tolerance-checked).
    prod_max_err = 0.0
    for i in range(200):
        terms = [rand_int(seed, 1, 16, coord, "pchain", i, j) for j in range(5)]
        acc_log = sum(float(numeric_features(t)[layout.LOG]) for t in terms)
        prod_max_err = max(prod_max_err,
                           abs(acc_log - math.log10(math.prod(terms))))
    # (c) Mixed chains through decode -> re-encode: invertibility makes the
    # algebra composable across operation types, e.g. (9 + 9) * 2 = 36.
    def reencode(vec):
        return numeric_features(decode_value(vec))

    mixed_ok = []
    for a, b, m in [(9, 9, 2), (123, 877, 3), (40, 2, 25), (0, 5, 7)]:
        summed = numeric_features(a) + numeric_features(b)          # a + b
        relogged = reencode(summed)                                  # exact re-encode
        prod_log = float(relogged[layout.LOG]) + float(numeric_features(m)[layout.LOG])
        got = round(10 ** prod_log)
        mixed_ok.append(got == (a + b) * m)
    checks.append({"name": "multi_step_chains",
                   "ok": chain_fails == 0 and prod_max_err <= 1e-5
                         and all(mixed_ok),
                   "ten_term_sum_failures": chain_fails,
                   "five_term_product_max_log_err": prod_max_err,
                   "mixed_decode_reencode_ok": mixed_ok})

    return {
        "seed": seed,
        "coord": coord,
        "all_ok": all(c["ok"] for c in checks),
        "checks": checks,
    }
