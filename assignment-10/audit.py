"""Assignment 10 — independent audit.

Re-derives every README/notebook claim from the committed artifacts
(submission_artifacts/*.json + the notebook's own JSON) WITHOUT executing the
notebook. Machine-specific timings are never asserted verbatim — only their
arithmetic and robust invariants are (the Session-9 convention).

Usage:  python audit.py            # standalone
        audit.run(check_fn)        # from run_demo.py / tests
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / "submission_artifacts"

REL = 1e-6          # tolerance for re-derived arithmetic identities
LOOSE = 1e-3


def _close(a, b, rel=REL, abs_tol=1e-12):
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


# ---------------------------------------------------------------- pure-python
# mini float encoder (independent of the notebook's) for the Task-6 re-derivation
def _encode(x: Fraction, exp_bits: int, mant_bits: int, e4m3fn: bool = False):
    bias = (1 << (exp_bits - 1)) - 1
    sign = 0 if x >= 0 else 1
    a = abs(x)
    if a == 0:
        return f"0{'0' * exp_bits}{'0' * mant_bits}", Fraction(0)
    e, t = 0, Fraction(1)
    if a >= 1:
        while t * 2 <= a:
            t *= 2
            e += 1
    else:
        while a < t:
            t /= 2
            e -= 1
    exp_field = e + bias

    def rne(scaled):
        lo = scaled.numerator // scaled.denominator
        rem = scaled - lo
        if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and lo % 2 == 1):
            lo += 1
        return lo

    if exp_field <= 0:
        mant = rne(a / Fraction(2) ** (1 - bias) * (1 << mant_bits))
        if mant == 0:
            exp_field, stored = 0, Fraction(0)
        elif mant >= (1 << mant_bits):
            exp_field, mant, stored = 1, 0, Fraction(2) ** (1 - bias)
        else:
            exp_field = 0
            stored = Fraction(mant, 1 << mant_bits) * Fraction(2) ** (1 - bias)
    else:
        mant = rne((a / t - 1) * (1 << mant_bits))
        if mant == (1 << mant_bits):
            mant, exp_field = 0, exp_field + 1
        stored = (1 + Fraction(mant, 1 << mant_bits)) * Fraction(2) ** (exp_field - bias)
    bits = f"{sign}{exp_field:0{exp_bits}b}{mant:0{mant_bits}b}"
    return bits, (stored if sign == 0 else -stored)


def run(check) -> int:
    """check(name, ok, detail='') -> None. Returns number of failures."""
    failures = 0

    def ck(name, ok, detail=""):
        nonlocal failures
        check(name, bool(ok), detail)
        if not ok:
            failures += 1

    try:
        R = json.loads((ART / "results.json").read_text())
        C = json.loads((ART / "curves.json").read_text())
        nb = json.loads((HERE / "training_loop.ipynb").read_text())
    except FileNotFoundError as exc:
        ck("artifacts present", False, str(exc))
        return failures

    cfg = R["config"]
    B, T, V, d = cfg["B"], cfg["T"], cfg["vocab"], cfg["d_model"]
    L, H = cfg["n_layer"], cfg["n_head"]

    # ------------------------------------------------ notebook executed cleanly
    counts = [c.get("execution_count") for c in nb["cells"]
              if c["cell_type"] == "code"]
    ck("notebook: every code cell executed",
       all(isinstance(c, int) for c in counts), f"counts={counts[:5]}...")
    ck("notebook: execution order monotone",
       counts == sorted(counts) and len(set(counts)) == len(counts))
    errs = [o for c in nb["cells"] if c["cell_type"] == "code"
            for o in c.get("outputs", []) if o.get("output_type") == "error"]
    ck("notebook: no error outputs", not errs)

    # ------------------------------------------------------------ SS3 shapes
    S = R["shapes"]
    ck("shapes: tokens/logits/shift/flat relations",
       S["tokens"] == [B, T] and S["logits"] == [B, T, V]
       and S["logits_shifted"] == [B, T - 1, V] and S["targets"] == [B, T - 1]
       and S["flat_logits"] == [B * (T - 1), V]
       and S["flat_targets"] == [B * (T - 1)] and S["loss"] == [])
    ck("shapes: attention score/q/qkv/mlp relations",
       S["block0.attn.scores"] == [B, H, T, T]
       and S["block0.attn.q"] == [B, H, T, d // H]
       and S["block0.attn.qkv"] == [B, T, 3 * d]
       and S["block0.mlp.hidden"] == [B, T, 4 * d])
    ck("shapes: optimizer holds 2 values per weight",
       R["shapes_step"]["optimizer_state_values"] == 2 * R["model"]["n_params"])

    # ------------------------------------------------------------ SS4 gradient
    toy = R["toy_chain"]
    ck("toy chain: hand == autograd == central == 64",
       _close(toy["hand"], 64) and _close(toy["autograd"], 64, rel=1e-9)
       and _close(toy["central"], 64, rel=1e-8))
    ck("toy chain: session's nudged loss 16.064",
       _close(toy["nudged_loss"], 16.064064, rel=1e-9),
       f"{toy['nudged_loss']}")
    ck("toy chain: forward-diff error law err = 64*eps (within 1%)",
       all(abs(z["err_over_eps"] - 64.0) < 0.64 for z in toy["fwd_error_law"]))

    fd = R["finite_diff"]
    ck("finite diff: fp64 probe certifies rel err < 1e-7",
       fd["best_rel_err"] < 1e-7, f"{fd['best_rel_err']:.2e}")
    ck("finite diff: best row is the min of the committed sweep",
       _close(fd["best_abs_err"], min(z["abs_err"] for z in fd["sweep"])))
    ck("finite diff: decimals = -log10(abs err), re-derived",
       _close(fd["best_decimals"],
              min(-math.log10(fd["best_abs_err"]), 15), rel=1e-3))
    ck("finite diff: every parameter type < 1e-5 rel err",
       all(z["rel_err"] < 1e-5 for z in fd["per_type"])
       and len(fd["per_type"]) == 5)
    ck("finite diff: directional derivative over all weights < 1e-6",
       fd["directional"]["rel_err"] < 1e-6)
    ck("finite diff: fp32 probe drowns (>=1 exact-cancellation eps) "
       "while fp32 backward matches fp64 to <1e-4",
       fd["fp32"]["n_cancelled"] >= 1
       and fd["fp32"]["grad_rel_diff_vs_fp64"] < 1e-4
       and fd["fp32"]["best_rel_err"] < 1e-2)

    # ------------------------------------------------------------ SS5 accumulation
    sm = R["static_mirror"]
    means = [sum(row) / len(row) for row in sm["per_token"]]
    cnts = [len(row) for row in sm["per_token"]]
    tw = sum(m * c for m, c in zip(means, cnts)) / sum(cnts)
    aa = sum(means) / len(means)
    ck("static mirror: per-token dump re-combines to 2.6 / 3.0",
       _close(tw, 2.6, rel=1e-5) and _close(aa, 3.0, rel=1e-5)
       and cnts == [4, 4, 2],
       f"tw={tw:.6f} aa={aa:.6f}")
    ck("static mirror: error = |3.0-2.6|/2.6 = 15.4%",
       _close(sm["err_pct"], abs(aa - tw) / tw * 100, rel=1e-4)
       and _close(sm["err_pct"], 15.3846, rel=1e-3))

    ai = R["accum_identity"]
    ck("accumulation identity: correct==big batch (fp32<1e-5, fp64<1e-12), "
       "buggy far away (>1e-2)",
       ai["rel_err_correct"] < 1e-5 and ai["rel_err_correct_f64"] < 1e-12
       and ai["rel_err_buggy"] > 1e-2 and ai["fwd_max_diff"] < 1e-5,
       f"f64={ai['rel_err_correct_f64']:.1e} buggy={ai['rel_err_buggy']:.1e}")
    ck("accumulation identity: unequal counts on the shifted targets",
       len(set(ai["counts"])) > 1 and ai["counts"] == [3, 3, 3, 33])

    at = R["accum_training"]
    an = at["analytic"]
    ck("hazard analytics re-derived: floor ln2/18, buggy p 11/12, asymptote",
       _close(an["correct_tw"], math.log(2) / 18)
       and _close(an["buggy_p"], 11 / 12)
       and _close(an["buggy_tw"],
                  (-math.log(11 / 12) - math.log(1 / 12)) / 36)
       and an["correct_dw"] > an["buggy_dw"])
    for z in at["seeds"]:
        ck(f"hazard seed {z['seed']}: gap>0.015, probes on their optima, "
           "doc-weighted flips",
           z["gap"] > 0.015 and 0.35 <= z["correct_probe"] <= 0.65
           and z["buggy_probe"] > 0.80 and z["buggy_dw"] < z["correct_dw"],
           f"gap={z['gap']:.4f} pC={z['correct_probe']:.3f} "
           f"pB={z['buggy_probe']:.3f}")
    ptf = at["per_token_final"]
    tot_c = sum(sum(row) for row in ptf["correct"])
    n_tok = sum(ptf["mask_counts"])
    tw_c = tot_c / n_tok
    tw_b = sum(sum(row) for row in ptf["buggy"]) / n_tok
    ck("hazard: final per-token dumps re-combine near the recorded evals",
       abs(tw_c - at["seeds"][0]["correct_tw"]) < 0.05
       and abs(tw_b - at["seeds"][0]["buggy_tw"]) < 0.05
       and tw_b > tw_c,
       f"re-derived tw {tw_c:.4f}/{tw_b:.4f}")

    rb = R["accum_ratio_bias"]
    num = den = Fraction(0)
    for others in product([3, 33], repeat=3):
        num += Fraction(1, 8) * Fraction(1, 3 + sum(others))
        den += Fraction(1, 8) * Fraction(1, 33 + sum(others))
    p_star = float(num / (num + den))
    ck("ratio bias: analytic p* re-derived by enumeration",
       _close(rb["analytic_p"], p_star, rel=1e-9), f"p*={p_star:.4f}")
    ck("ratio bias: measured probes sit near p*, not near 0.5",
       all(abs(p - p_star) < 0.15 for p in rb["measured_probes"])
       and all(p > 0.55 for p in rb["measured_probes"]),
       f"{rb['measured_probes']}")

    ac = R["accum_control"]
    ck("equal-count control: gradients coincide, curves stay within "
       "a quarter of the hazard gap",
       ac["grad_rel_err"] < 1e-5
       and ac["max_curve_diff"] < 0.25 * ac["hazard_gap"],
       f"curve diff {ac['max_curve_diff']:.5f} vs gap {ac['hazard_gap']:.4f}")

    mn = R["accum_mixed_negative"]
    rel_mix = mn["mix_gap"] / mn["mix_level"]
    rel_hz = ac["hazard_gap"] / at["seeds"][0]["correct_tw"]
    ck("mixed-text honest negative: relative gap under a quarter of the "
       "hazard corpus's (re-derived)",
       _close(rel_mix, mn["rel_gap"], rel=1e-6)
       and _close(rel_hz, mn["hazard_rel_gap"], rel=1e-3)
       and abs(rel_mix) < 0.25 * rel_hz,
       f"mixed {rel_mix * 100:+.1f}% vs hazard {rel_hz * 100:+.0f}% relative")

    # ------------------------------------------------------------ SS6 grad norm
    gn = R["gradnorm"]
    gc = C["gradnorm"]
    k = gn["k"]
    med = sorted(gc["A"]["norm"][40:k])[len(gc["A"]["norm"][40:k]) // 2]
    ck("grad norm: cap re-derived = 2 x median of placebo settled norms",
       _close(gn["cap"], 2 * med, rel=1e-6), f"cap={gn['cap']:.3f}")
    ck("grad norm: spike detected exactly at the injection step; placebo silent",
       gn["spike_step"] == k and gn["placebo_spike"] is None)
    ck("grad norm: clean-probe damage begins at/just after k; placebo silent",
       gn["damage_step"] is not None and k <= gn["damage_step"] <= k + 3
       and gn["placebo_damage"] is None)
    ck("grad norm: spike ratio > 4x settled median",
       gn["norm_at_k"] > 4 * gn["median_norm"],
       f"{gn['norm_at_k'] / gn['median_norm']:.1f}x")
    ck("grad norm: unclipped damage unmistakable; both caps contain >50% of it",
       (gn["probe_worst_B"] - gn["probe_pre"]) > 0.05
       and (gn["probe_worst_C"] - gn["probe_pre_C"])
       < 0.5 * (gn["probe_worst_B"] - gn["probe_pre"])
       and (gn["probe_worst_C2"] - gn["probe_pre_C2"])
       < 0.5 * (gn["probe_worst_B"] - gn["probe_pre"]),
       f"B +{gn['probe_worst_B'] - gn['probe_pre']:.3f} "
       f"C +{gn['probe_worst_C'] - gn['probe_pre_C']:.3f} "
       f"C2 +{gn['probe_worst_C2'] - gn['probe_pre_C2']:.3f}")
    ck("grad norm: loose cap re-derived (1.25 x max settled placebo norm) and "
       "binds on few SETTLED steps (the transient and the damage are anomalies)",
       _close(gn["cap_loose"], 1.25 * max(C["gradnorm"]["A"]["norm"][40:k]),
              rel=1e-6)
       and gn["cap_loose_binds_settled"] < 0.1 * gn["n_settled"],
       f"loose cap {gn['cap_loose']:.3f}, binds "
       f"{gn['cap_loose_binds_settled']}/{gn['n_settled']} settled steps")
    ck("grad norm: scale factor at k re-derived = cap / norm",
       _close(gn["scale_at_k"], min(1.0, gn["cap"] / gc["B"]["norm"][k]),
              rel=1e-6))
    ck("grad norm: session arithmetic (cap 1.0 / norm 8.4 -> x0.119)",
       abs(gn["session_check_scale"] - 0.119) < 5e-4)
    ck("grad norm: every logged value finite in healthy arms",
       all(math.isfinite(v) for arm in ("A", "B", "C")
           for series in ("norm", "probe", "train")
           for v in gc[arm][series]))
    ck("grad norm: pre-injection norms bitwise equal across A and B",
       gc["A"]["norm"][:k] == gc["B"]["norm"][:k])

    # ------------------------------------------------------------ SS7 MFU
    mf = R["mfu_flops"]
    exact = 6 * (12 * L * d * d + V * d) + 12 * L * T * d
    ck("MFU: exact FLOPs/token formula re-derived",
       mf["exact_per_token"] == exact, f"{exact:,}")
    ck("MFU: 6N-convention error table re-derived",
       all(_close(v["err_pct"],
                  (v["flops"] - exact) / exact * 100, rel=1e-6)
           for v in mf["conventions"].values()))
    n_all = R["model"]["n_params"]
    ck("MFU: cancellation identity 6*zero_flop - 12LTd == 6N - exact, "
       "with zero_flop = tok_emb + pos_emb + LayerNorms",
       mf["zero_flop_params"] == V * d + cfg["max_pos"] * d + (2 * 2 * L + 2) * d
       and 6 * n_all - exact == 6 * mf["zero_flop_params"] - 12 * L * T * d,
       f"zero_flop={mf['zero_flop_params']:,}")
    ck("MFU: 6*N_total within 1.5% at T=128 but drifts with T",
       abs(mf["conventions"]["6 x N_total"]["err_pct"]) < 1.5
       and abs(mf["t_dependence"]["32"]) > 5
       and abs(mf["t_dependence"]["512"]) > 15)
    ck("MFU: profiler forward count matches analytic (<0.1%)",
       abs(mf["profiler_fwd_per_token"] - mf["analytic_fwd_per_token"])
       / mf["analytic_fwd_per_token"] < 1e-3)
    mm = R["mfu_main"]
    ck("MFU: headline arithmetic re-derived from committed raw values",
       _close(mm["achieved"], exact * mm["tps"], rel=1e-9)
       and _close(mm["mfu"], mm["achieved"] / mm["peak"], rel=1e-9)
       and _close(mm["peak"], max(mm["proxy_before"], mm["proxy_after"]),
                  rel=1e-12))
    ck("MFU: 0 < mfu < 1 and achieved <= measured peak",
       0 < mm["mfu"] < 1 and mm["achieved"] <= mm["peak"])
    ck("MFU: tps consistent with committed per-step timings",
       _close(mm["tps"],
              cfg["B"] * cfg["T"] * len(mm["per_step_s"]) / sum(mm["per_step_s"]),
              rel=1e-6))
    we = mm["worked_example"]
    ck("MFU: session worked example 648/7912 = 8.1901%",
       _close(we["achieved"], 648e12, rel=1e-9)
       and _close(we["peak"], 7912e12, rel=1e-9)
       and _close(we["mfu"], 0.0819009, rel=1e-4))
    sw = R["mfu_sweep"]["rows"]
    ck("MFU: sweep rows internally consistent, utilization rises with width",
       all(_close(z["mfu"], z["flops_per_token"] * z["tps"] / mm["peak"],
                  rel=1e-9) for z in sw)
       and sw[-1]["mfu"] > sw[0]["mfu"]
       and all(0 < z["mfu"] < 1 for z in sw))
    ck("MFU: matmul ops own a minority of step time (the honest gap story)",
       0.05 < R["mfu_profile"]["matmul_time_share"] < 0.95)

    # ------------------------------------------------------------ SS8 floats
    fb = {z["format"]: z for z in R["float_bits"]["formats"]}
    specs = {"fp32": (8, 23, False), "bf16": (8, 7, False),
             "fp16": (5, 10, False), "fp8 E4M3": (4, 3, True)}
    tenth = Fraction(1, 10)
    for name, (eb, mb, fn) in specs.items():
        bits, stored = _encode(tenth, eb, mb, fn)
        z = fb[name]
        ok = (z["bits"].replace("|", "") == bits
              and _close(z["stored"], float(stored), rel=1e-12)
              and _close(z["abs_err"], float(abs(stored - tenth)), rel=1e-6)
              and _close(z["rel_err"], float(abs(stored - tenth) / tenth),
                         rel=1e-6))
        ck(f"floats: {name} bits/value/error re-derived by independent encoder",
           ok, z["bits"])
    ck("floats: expected canonical bit strings",
       fb["fp32"]["bits"].replace("|", "")
       == "00111101110011001100110011001101"
       and fb["bf16"]["bits"].replace("|", "") == "0011110111001101"
       and fb["fp8 E4M3"]["bits"].replace("|", "") == "00011101")
    ck("floats: only fp16 rounds down on 0.1",
       fb["fp16"]["rounded"] == "down"
       and all(fb[n]["rounded"] == "up" for n in ("fp32", "bf16", "fp8 E4M3")))
    for z in R["float_bits"]["tie_probes"]:
        n, dn = z["value"].split("/")
        frac = Fraction(int(n), int(dn))
        eb, mb, fn = specs[z["format"]]
        _, stored = _encode(frac, eb, mb, fn)
        ck(f"floats: tie probe {z['value']} in {z['format']} -> {z['stored']}",
           _close(float(stored), z["stored"], rel=1e-12))
    cliff = {f"{z['g']:.1e}": z for z in R["float_bits"]["cliff"]}
    ck("floats: fp16 cliff (1e-8 and 2.9e-8 -> 0; 3e-8 -> min subnormal; "
       "bf16 keeps 1e-8)",
       cliff["1.0e-08"]["fp16"] == 0.0 and cliff["2.9e-08"]["fp16"] == 0.0
       and _close(cliff["3.0e-08"]["fp16"], 2 ** -24, rel=1e-9)
       and cliff["1.0e-08"]["bf16"] != 0.0)
    ls = R["float_bits"]["loss_scaling"]
    ck("floats: loss scaling rescues 1e-8 into the subnormal range, "
       "but 1e-11 still dies",
       ls["scaled_1e8"] != 0.0 and 1e-4 < ls["rel_err"] < 1e-2
       and ls["dead_1e11"] == 0.0)

    # ------------------------------------------------------------ SS9 memory
    mem = R["memory"]
    n_par = R["model"]["n_params"]
    ck("memory: measured training state = exactly 16 bytes/weight",
       mem["total_bytes"] == 16 * n_par and mem["per_weight"] == 16.0)
    ck("memory: scale table re-derived (2B->29.8GiB ... 120B->1788GiB)",
       all(_close(z["gib"], 16 * z["params"] / 2 ** 30, rel=1e-9)
           for z in mem["scale_rows"])
       and _close(mem["scale_rows"][0]["gib"], 29.8, rel=1e-2)
       and _close(mem["scale_rows"][-1]["gib"], 1788.1, rel=1e-3))
    ck("memory: 80GB ceiling ~ 5.4B params",
       _close(mem["ceiling_80gb"], 80 * 2 ** 30 / 16, rel=1e-9)
       and _close(mem["ceiling_80gb"] / 1e9, 5.37, rel=1e-2))

    return failures


if __name__ == "__main__":
    def _print(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))

    n = run(_print)
    print(f"verdict: {'PASS' if n == 0 else f'{n} FAILING CHECKS'}")
    raise SystemExit(0 if n == 0 else 1)
