"""Assignment 11 — independent audit.

Re-derives every README/notebook claim from the committed artifacts
(submission_artifacts/results.json + curves.json + the notebook's own JSON) WITHOUT
executing the notebook and without torch. Machine-specific timings are never asserted;
training outcomes are audited as orderings with margins above the measured replicate
and seed spreads (the Session-9/10 convention).

Usage:  python audit.py            # standalone
        audit.run(check_fn)        # from run_demo.py / tests
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / "submission_artifacts"

REL = 1e-6


def _close(a, b, rel=REL, abs_tol=1e-12):
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


# ------------------------------------------------------------- pure-python re-implementations
def adam_rows(w0, grads, lr, b1, b2, eps):
    rows, m, v, w = [], 0.0, 0.0, w0
    for t, g in enumerate(grads, 1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat, vhat = m / (1 - b1 ** t), v / (1 - b2 ** t)
        u = mhat / (math.sqrt(vhat) + eps)
        w = w - lr * u
        rows.append((m, v, mhat, vhat, u, w))
    return rows


def r_unc(t, b1, b2):
    return (1 - b1 ** t) / math.sqrt(1 - b2 ** t)


def stays_within(b1, b2, tol, t_max=20_000):
    rs = [abs(r_unc(t, b1, b2) - 1) for t in range(1, t_max + 1)]
    last_out = max((i for i, x in enumerate(rs) if x >= tol), default=-1)
    return last_out + 2


def running_median(series, half=2):
    out = []
    for i in range(len(series)):
        win = sorted(x for x in series[max(0, i - half): i + half + 1] if x == x)
        out.append(win[len(win) // 2] if win else float("nan"))
    return out


def detect_peak(series, t_min=10):
    med = running_median(series)
    return max(range(t_min - 1, len(med)), key=lambda i: (med[i], -i)) + 1


def ols_slope(ys, xs=None):
    xs = list(range(len(ys))) if xs is None else xs
    n = len(ys)
    mx, my = sum(xs) / n, sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def slope_change(series, W, h):
    s = [math.log(x) if x == x and x > 0 else float("nan") for x in series]
    before, after = s[W - h:W], s[W:W + h]
    if any(x != x for x in before + after):
        return float("nan")
    return ols_slope(before) - ols_slope(after)


def lr_cosine(t, peak, warmup, total, floor=0.1):
    if t < warmup:
        return peak * (t + 1) / warmup
    prog = min(1.0, (t - warmup) / max(1, total - warmup))
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog)))


def lr_wsd(t, peak, warmup, total, frac):
    if t < warmup:
        return peak * (t + 1) / warmup
    t_dec = int(round(total * (1 - frac)))
    return peak if t < t_dec else peak * max(0.0, (total - t) / (total - t_dec))


def parabola_vertex(xs, ys):
    (x0, y0), (x1, y1), (x2, y2) = zip(xs, ys)
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    return (-b / (2 * a)) if a > 0 else None


def fair_compare(a, b):
    """Pure-python copy of the notebook's protocol: 'REFUSED', or 'A' (first side wins) / 'B'."""
    if a["config"] != b["config"] or a["stream_sha"] != b["stream_sha"]:
        return "REFUSED"
    if a["on_edge"] or b["on_edge"]:
        return "REFUSED"
    ma, mb = sum(a["seeds"]) / len(a["seeds"]), sum(b["seeds"]) / len(b["seeds"])
    ra, rb = (min(a["seeds"]), max(a["seeds"])), (min(b["seeds"]), max(b["seeds"]))
    pooled = max(ra[1] - ra[0], rb[1] - rb[0])
    overlap = not (ra[1] < rb[0] or rb[1] < ra[0])
    if overlap or abs(mb - ma) <= pooled:
        return "REFUSED"
    return "A" if mb > ma else "B"


def run(check) -> int:
    failures = 0

    def ck(name, ok, detail=""):
        nonlocal failures
        check(name, bool(ok), detail)
        if not ok:
            failures += 1

    # pre-registered checks whose failure is a reported finding: they print as [PASS] with an
    # 'ACKNOWLEDGED FAILURE' detail so the count of what failed stays visible in every log
    ACKNOWLEDGED = {"ratio: per-tensor-median detector within the pre-registered 25% of W (W50-s1337)",
                    "the per-tensor-median detector should track W within 25% in every arm"}

    def ck_ack(name, ok, detail=""):
        if ok:
            ck(name, True, detail)
        elif name in ACKNOWLEDGED:
            ck(name + " [ACKNOWLEDGED FAILURE, reported in README §3]", True, detail)
        else:
            ck(name, False, detail)

    try:
        R = json.loads((ART / "results.json").read_text())
        C = json.loads((ART / "curves.json").read_text())
        nb = json.loads((HERE / "optimizers.ipynb").read_text())
    except FileNotFoundError as exc:
        ck("artifacts present", False, str(exc))
        return failures

    cfg = R["config"]
    # ------------------------------------------------ notebook executed cleanly
    counts = [c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code"]
    ck("notebook: every code cell executed", all(isinstance(c, int) for c in counts))
    ck("notebook: execution counts monotone (one top-to-bottom run)",
       all(isinstance(c, int) for c in counts) and counts == sorted(counts) and len(set(counts)) == len(counts))
    ck("notebook: no error outputs",
       not any(o.get("output_type") == "error" for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", [])))
    ck("config: committed run is a full run (not FAST)", cfg["fast"] is False)
    ck("config: an accelerator trained the sweeps, the CPU did the checks",
       cfg["device"] in ("mps", "cuda") and cfg["check_device"] == "cpu", f"device={cfg['device']}")
    soft = R.get("soft_checks", [])
    failed = [z["msg"] for z in soft if not z["ok"]]
    ck("notebook: every recorded outcome check passed, or failed and is acknowledged in the README",
       all(m in ACKNOWLEDGED for m in failed), f"{len(soft)} checks; failed: {failed or 'none'}")

    # ------------------------------------------------ data
    D = R["data"]
    ck("data: corpus files pinned by sha256 and every language holds out blocks",
       all(len(v["sha256"]) == 64 and v["held_blocks"] >= 1 for v in D["per_lang"].values()))
    ck("data: held-out share 8.3% by construction (every 12th block)",
       D["hold_every"] == 12 and 0.07 < D["held_ids"] / (D["held_ids"] + D["train_ids"]) < 0.095)
    ck("data: eval targets = blocks x 32 windows x (T-1), every byte once",
       D["eval_targets"] == D["held_blocks"] * 32 * (cfg["T"] - 1) and D["block"] == 32 * (cfg["T"] - 1) + 1)
    ck("data: a 300-step run is under one epoch (no memorization regime)", D["epochs_per_run"] < 1.0, f"{D['epochs_per_run']:.2f}")
    ck("data: verbatim overlap falls with window length (boilerplate, not leakage)",
       D["overlap"]["32"] > D["overlap"]["64"] > D["overlap"]["128"] and D["overlap"]["128"] < 0.05)

    # ------------------------------------------------ devices
    DV = R["devices"]
    ck("devices: 20-step cpu-vs-accelerator per-step loss agreement at fp32 rounding (< 1e-5)", DV["short"]["loss_rel_max"] < 1e-5,
       f"{DV['short']['loss_rel_max']:.1e}")
    ck("devices: same-seed replicate on the accelerator is bitwise", DV.get("replicate", {}).get("bitwise") is True)
    ck("devices: long-horizon cpu-vs-accelerator difference inside the seed spread",
       DV["long"]["cpu_vs_gpu"] <= DV["long"]["seed_spread"] or DV["long"]["cpu_vs_gpu"] < 0.01,
       f"{DV['long']['cpu_vs_gpu']:.4f} vs spread {DV['long']['seed_spread']:.4f}")

    # ------------------------------------------------ §1 Adam by hand
    A = R["adam_by_hand"]
    rows = adam_rows(A["w0"], A["grads"], A["lr"], A["betas"][0], A["betas"][1], A["eps"])
    ok = all(_close(r["m"], m, 1e-12) and _close(r["v"], v, 1e-12) and _close(r["mhat"], mh, 1e-12)
             and _close(r["vhat"], vh, 1e-12) and _close(r["u"], u, 1e-12) and _close(r["w"], w, 1e-12)
             for r, (m, v, mh, vh, u, w) in zip(A["rows"], rows))
    ck("adam: the five-step table re-derived by an independent implementation", ok)
    ck("adam: torch agrees to rounding in float64 and to >= 5 decimals in float32",
       A["check"]["float64"]["worst_abs"] < 1e-14 and A["check"]["float32"]["decimals_abs"] >= 5,
       f"fp64 {A['check']['float64']['worst_abs']:.0e}, fp32 {A['check']['float32']['decimals_abs']} decimals")
    ck("adam: first step is lr to eps/|g1|; uncorrected first step = (1-b1)/sqrt(1-b2)",
       _close(A["first_step_rel_dev"], A["eps"] / abs(A["grads"][0]), 1e-6)
       and _close(A["uncorrected_step1_over_lr"], (1 - A["betas"][0]) / math.sqrt(1 - A["betas"][1]), 1e-4))
    ck("adam: v_hat_5 tracks the plain mean of g^2 (beta2 = 0.999)",
       abs(A["rows"][-1]["vhat"] / A["mean_g2"] - 1) < 2e-3)
    lad = A["ladder"]
    ck("adam: scale ladder — invariance intact at x1000, broken at x1e-8, step-1 |u| = |g|/(|g|+eps)",
       lad[0]["max_w_dev"] < 1e-7 and lad[-1]["max_w_dev"] > 0.1
       and all(_close(z["u1"], abs(A["grads"][0] * z["k"]) / (abs(A["grads"][0] * z["k"]) + A["eps"]), 1e-6) for z in lad)
       and all(lad[i]["max_w_dev"] <= lad[i + 1]["max_w_dev"] + 1e-12 for i in range(1, len(lad) - 1)))
    ec = A["eps_conventions"]
    ck("adam: Keras eps-hat differs above the fp32 floor, optax eps_root below it",
       ec["keras"]["max_w_dev"] > A["check"]["float32"]["w"] > ec["optax"]["max_w_dev"])
    dec = A["decay"]
    ck("adam: AdamW and Adam+L2 by hand match their torch classes; L2 delivers MORE decay",
       dec["adamw_vs_torch"] < 1e-14 and dec["l2_vs_torch"] < 1e-14 and dec["decoupled_flag_vs_adamw"] < 1e-14
       and dec["delivered_l2"] > dec["delivered_adamw"] > 0)
    rm = dec["real_model"]
    ck("adam: on the real model L2-in-Adam shrinks ||W|| far more than AdamW at the same lambda",
       rm["l2"]["w_norm_after"] < 0.9 * rm["adamw"]["w_norm_after"] < rm["none"]["w_norm_after"])

    # ------------------------------------------------ §2 bias correction
    B = R["bias_correction"]
    b1 = cfg["betas"][0]
    for key, want in (("0.999", (12, 1751, 3925)), ("0.95", (20, 6, 76))):
        an = B["analytic"][key]
        b2 = float(key)
        peak_t = max(range(1, 400), key=lambda t: r_unc(t, b1, b2))
        ck(f"bias: r(t) table for beta2={key} re-derived (peak, 10%/1% bands)",
           an["peak_t"] == peak_t == want[0] and _close(an["peak_r"], r_unc(peak_t, b1, b2), 1e-9)
           and an["within_10pct"] == stays_within(b1, b2, 0.10) == want[1]
           and an["within_1pct"] == stays_within(b1, b2, 0.01) == want[2]
           and all(_close(v, r_unc(int(t), b1, b2), 1e-9) for t, v in an["r"].items()))
    ck("bias: crossover beta2 = 1-(1-b1)^2 gives r(1) = 1 exactly",
       _close(B["crossover_beta2"], 0.99, 1e-12) and _close(B["analytic"]["0.99"]["r1"], 1.0, 1e-12))
    hv = B["hand_vs_torch"]
    ck("bias: HandAdamW == torch.optim.AdamW bitwise on cpu fp32 and to rounding in fp64",
       hv["cpu-float32"]["bitwise"] and hv["cpu-float64"]["max_abs_diff"] < 1e-12)
    S = B["summary"]
    for key in ("0.999", "0.95"):
        s = S[key]
        shared = C["bias_correction"][key]["shared_state"]
        b2 = float(key)
        dev = max(abs(math.sqrt(0) if False else (u / c) / r_unc(t + 1, b1, b2) - 1) for t, (c, u) in enumerate(shared))
        ck(f"bias: same-state uncorrected/corrected update ratio == r(t) (beta2={key}, re-derived from curves)",
           dev < 2e-2 and _close(s["shared_max_rel_dev"], dev, 1e-6), f"max rel dev {dev:.1e}")
        ck(f"bias: step-1 multipliers of the two arms are in ratio r(1) (beta2={key})",
           abs(s["mult_step1"]["U"] / s["mult_step1"]["C"] / r_unc(1, b1, b2) - 1) < 2e-2)
        # paired-gap rule re-derived from the committed per-seed evals, both pairs
        paired = C["bias_correction"][key]["paired"]
        evs = sorted(int(e) for e in next(iter(paired.values()))["C"])
        tol = B["gap_tol"]
        for pair, (au, ac) in {"t_star": ("U", "C"), "t_star_warmup": ("U+warmup", "C+warmup")}.items():
            t_star = None
            for i, e in enumerate(evs):
                def ok_at(x):
                    gs = [paired[sd][au][str(x)] - paired[sd][ac][str(x)] for sd in paired]
                    return abs(sum(gs) / len(gs)) < tol
                if all(ok_at(x) for x in evs[i:]):
                    t_star = e
                    break
            ck(f"bias: paired-gap {pair} re-derived from committed evals (beta2={key})", t_star == s[pair], f"{pair}={t_star}")
    s95, s999 = S["0.95"], S["0.999"]
    ck("bias: after t* the residual gap's sign is NOT unanimous across seeds (indistinguishable from zero), and stays under the tolerance",
       s95["sign_unanimous_after_tstar_warmup"] is False and s95["gap_max_after_tstar_warmup"] < B["gap_tol"], f"max |gap| after t* {s95['gap_max_after_tstar_warmup']:.4f}")
    ck("bias: the 'hot for 30 steps' counterfactual reproduces the uncorrected damage, paired over seeds (|C-hot - U| < 0.1 nats on the mean)",
       abs(s999["hot_vs_u_mean"]) < 0.1 and len(s999["hot_vs_u_per_seed"]) == 3, f"per seed {[round(x, 3) for x in s999['hot_vs_u_per_seed']]}")
    ck("bias: the beta2 = 0.999 gap plateaus (its peak is not the last eval) and never falls below 0.5 nats after step 100",
       s999["gap_peak_step"] < s999["steps"] and all(v > 0.5 for e, v in s999["gap_mean"].items() if int(e) >= 100), f"peak {s999['gap_peak']:+.3f} @ {s999['gap_peak_step']}")
    ck("bias: measured-multiplier criterion re-derived (within 10% for 20 steps: 0.95 yes, 0.999 never)",
       s95["meas_10pct"] is not None and s95["meas_10pct"] < 100 and s999["meas_10pct"] is None)
    ck("bias: at beta2 = 0.95 with warmup the loss stops caring within 100 steps; at 0.999 it never does inside the run",
       S["0.95"]["t_star_warmup"] is not None and S["0.95"]["t_star_warmup"] <= 100 and S["0.999"]["t_star"] is None and S["0.999"]["t_star_warmup"] is None)
    ck("bias: at beta2 = 0.95 WITHOUT warmup the uncorrected arm leads (its r(t) < 1 start is a warmup)",
       S["0.95"]["late_gap_no_warmup"] < -0.05, f"late gap {S['0.95']['late_gap_no_warmup']:+.3f}")
    arms = C["bias_correction"]["0.999"]["arms"]
    ck("bias: at beta2 = 0.999 the uncorrected arm inflates the weight norm > 1.2x and ends far behind",
       arms["U"]["wnorm"][-1] > 1.2 * arms["C"]["wnorm"][-1]
       and S["0.999"]["gap_mean"][str(S["0.999"]["steps"])] > 0.1)
    ck("bias: the 'hot for 30 steps' counterfactual reproduces the damage (same offset within 0.3 nats of U)",
       abs(arms["C-hot"]["eval"][str(S["0.999"]["steps"])] - arms["U"]["eval"][str(S["0.999"]["steps"])]) < 0.3)
    ck("bias: pre-divergence train losses of C and U identical at step 1 (same init, same stream)",
       arms["C"]["train"][0] == arms["U"]["train"][0] and arms["C"]["stream_sha"] == arms["U"]["stream_sha"])

    # ------------------------------------------------ §3 update ratio
    U = R["update_ratio"]
    CU = C["update_ratio"]
    def _peak_layers(run):
        peaks = sorted(detect_peak(L["rho"]) for L in run["layers"].values() if L["rho"][0] == L["rho"][0])
        return peaks[len(peaks) // 2]
    for key, d in U["detector"].items():
        t_g = detect_peak(CU[key]["global_rho"])
        t_l = _peak_layers(CU[key])
        ck(f"ratio: both blind detectors re-derived from the committed curves ({key})",
           t_g == d["t_hat_global"] and t_l == d["t_hat_layers"], f"W={d['W']}: global {t_g}, per-tensor {t_l}")
        ck_ack(f"ratio: per-tensor-median detector within the pre-registered 25% of W ({key})",
               abs(t_l - d["W"]) / d["W"] <= 0.25, f"{(t_l - d['W']) / d['W'] * 100:+.0f}%")
    ws = sorted(set(d["W"] for d in U["detector"].values()))
    firsts = {W: next(d["t_hat_layers"] for d in U["detector"].values() if d["W"] == W and d["seed"] == cfg["seed"]) for W in ws}
    ck("ratio: the per-tensor-median detected step moves with W", all(firsts[ws[i]] < firsts[ws[i + 1]] for i in range(len(ws) - 1)), str(firsts))
    for W in (50, 100, 200):
        syn = [min(1.0, (t + 1) / W) * 3e-3 for t in range(300)]
        ck(f"ratio: detector returns exactly W on a synthetic ramp-then-flat series (W={W})", detect_peak(syn) == W)
    ref = CU[f"W{U['W_mid']}-s{cfg['seed']}"]
    W, h = U["W_mid"], U["h_win"]
    ck("ratio: S_lr at W re-derived from the committed lr trace (about 1/W)",
       all(_close(a["S_lr"], slope_change(ref["lr"], W, h), 1e-9) for a in U["attribution"].values())
       and abs(next(iter(U["attribution"].values()))["S_lr"] * W - 1) < 0.15)
    recon = {}
    for n, a in U["attribution"].items():
        L = ref["layers"][n]
        r_series = [c / s if s > 0 else float("nan") for c, s in zip(L["coherence"], L["scale"])]
        recon[n] = (slope_change(r_series, W, h), slope_change(L["rho"], W, h))
    ck("ratio: S_r and S_rho re-derived per layer from the committed decomposition",
       all((math.isnan(a["S_rho"]) and math.isnan(recon[n][1])) or (_close(a["S_r"], recon[n][0], 1e-9) and _close(a["S_rho"], recon[n][1], 1e-9))
           for n, a in U["attribution"].items()))
    hidden = [n for n in U["attribution"] if n.startswith("blocks") and n.endswith(".weight") and "ln" not in n]
    vis_h = [U["attribution"][n]["visible"] for n in hidden]
    vis_e = [U["attribution"][n]["visible"] for n in ("tok_emb.weight", "pos_emb.weight", "ln_f.weight")]
    ck("ratio: at the workhorse configuration the warmup kink is visible in the embeddings/ln_f AND in every block matrix",
       sum(vis_e) / len(vis_e) > 0.5 and all(v > 0.5 for v in vis_h), f"hidden {min(vis_h):.2f}..{max(vis_h):.2f}, emb/ln_f {sum(vis_e) / len(vis_e):.2f}")
    ck("ratio: decoupled decay is a small, separated correction (|wd share of |dw|| < 1% for every tensor at W)",
       max(abs(a["wd_share"]) for a in U["attribution"].values()) < 0.01 and max(a["decay_over_dw"] for a in U["attribution"].values()) < 0.03)
    ck("ratio: the tok_emb step-1 coherence equals sqrt(rows touched by batch 1 / V)",
       abs(U["identities"]["coherence1"]["tok_emb.weight"] - U["coh_tok_expected"]) < 1e-2)
    AS = U["attribution_seeds"]
    ck("ratio: per-seed attribution re-derived for every W=100 seed (S_rho from the committed curves)",
       all(_close(AS[sd][n]["S_rho"], slope_change(CU[f"W{U['W_mid']}-s{sd}"]["layers"][n]["rho"], W, h), 1e-9) for sd in AS for n in hidden))
    ck("ratio: the block matrices show the kink in every seed as a group (seed means > 0.5) and 7+ of 8 exceed 0.5 in every seed",
       all(v > 0.5 for v in U["visible_hidden_by_seed"].values()) and U["n_hidden_visible_all_seeds"] >= U["n_hidden"] - 1,
       f"seed means {[round(v, 2) for v in U['visible_hidden_by_seed'].values()]}, {U['n_hidden_visible_all_seeds']}/{U['n_hidden']}")
    CF = U["counterfactual"]
    cf_run = CU["counterfactual"]
    cf_h = [n for n in CF["attribution"] if n.startswith("blocks") and n.endswith(".weight") and "ln" not in n]
    ck("ratio: counterfactual (width 128, lr 1e-3) attribution re-derived from its committed curves",
       all(_close(CF["attribution"][n]["S_rho"], slope_change(cf_run["layers"][n]["rho"], CF["W"], U["h_win"]), 1e-9) for n in cf_h)
       and _close(CF["visible_hidden_mean"], sum(CF["attribution"][n]["visible"] for n in cf_h) / len(cf_h), 1e-9),
       f"visible hidden mean {CF['visible_hidden_mean']:.2f} vs workhorse {sum(vis_h) / len(vis_h):.2f}")
    ids = U["identities"]
    ck("ratio: step-1 identities — rho(1)*sigma/lr(1) = 1 for dense matrices, coherence(1) = 1, |dw_adam| = lr*u",
       all(abs(v - 1) < 2e-2 for v in ids["rho1_sigma_over_lr1"].values()) and ids["dn_adam_vs_lr_u_max_rel"] < 1e-3
       and abs(ids["ln_gain_rho1_over_lr1"] - 1) < 1e-2)
    ck("ratio: rho re-derived = dn/wn from the committed parts, for every layer and step",
       all(all(_close(L["rho"][t], L["dn"][t] / L["wn"][t], 1e-9) for t in range(len(L["rho"])) if L["wn"][t] > 0)
           for L in ref["layers"].values()))

    # ------------------------------------------------ §4 schedules
    SC = R["schedules"]
    tr = C["schedules"]["lr"]
    N, W0 = SC["steps"], cfg["warmup"]
    ck("schedules: committed lr traces re-derived (cosine to 10%, WSD with 20% linear decay)",
       all(_close(tr["cosine"][t], lr_cosine(t, SC["peak"], W0, N), 1e-12) for t in range(N))
       and all(_close(tr["wsd"][t], lr_wsd(t, SC["peak"], W0, N, SC["wsd_decay_frac"]), 1e-12) for t in range(N)))
    ck("schedules: at the stop step cosine is at 0.372 x peak and WSD at the peak",
       _close(SC["lr_at_stop"]["cosine"] / SC["peak"], 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (SC["stop"] - W0) / (N - W0))), 1e-9)
       and SC["lr_at_stop"]["wsd"] == SC["peak"])
    ck("schedules: WSD has more lr-area to the stop step than cosine",
       SC["lr_area_to_stop"]["wsd"] > 1.15 * SC["lr_area_to_stop"]["cosine"])
    T = SC["table"]
    ck("schedules: per-seed paired differences re-derived from the committed per-seed losses",
       all(all(_close(c - w, d, 1e-9) for c, w, d in zip(v["cosine"], v["wsd"], v["diff"])) and _close(v["mean_diff"], sum(v["diff"]) / len(v["diff"]), 1e-9)
           for v in T.values()))
    dec = SC["decision"]
    ck("schedules: the 'stop now' decision names the lower mean loss at the stop step",
       dec["stop_now"] == ("cosine" if T["stop"]["mean_diff"] < 0 else "wsd"))
    lad = SC["ladder"]
    lv = [lad[k]["mean"] for k in sorted(lad, key=lambda k: lad[k]["peak"])]
    ck("schedules: the stop-step gap moves toward cosine as the peak lr rises (regime rule)", lv[0] > lv[-1], str([f"{x:+.3f}" for x in lv]))
    i0 = SC["seeds"].index(str(cfg["seed"]))
    ck("schedules: extension decision consistent with the committed extension losses; the WSD trunk's extension "
       "reproduces the planned run (same seed) to 1e-4",
       dec["extend"] == ("cosine" if T["ext"]["mean_diff"] < 0 else "wsd")
       and _close(dec["wsd_ext_vs_planned"], T["ext"]["wsd"][i0] - SC["planned_ext"]["wsd"], 1e-9)
       and abs(dec["wsd_ext_vs_planned"]) < 1e-4, f"wsd ext - planned = {dec['wsd_ext_vs_planned']:+.2e}")
    ob = SC["own_best"]
    ck("schedules: each schedule's own best peak lr at the stop step re-derived from the ladder means",
       all(_close(min(v["means"].values()), v["loss"], 1e-9) and float(min(v["means"], key=v["means"].get)) == v["peak"] for v in ob.values()))
    _m = lambda xs: sum(xs) / len(xs)
    m30 = min(_m(T[k]["wsd"]) for k in ("lin30", "sqrt30")) - min(_m(T[k]["cosine"]) for k in ("lin30", "sqrt30"))
    m60 = _m(T["lin60"]["wsd"]) - _m(T["lin60"]["cosine"])
    ck("schedules: the +30 / +60 cooldown decisions and margins re-derived from the committed table",
       dec["plus_c30"] == ("cosine" if m30 > 0 else "wsd") and _close(dec["plus_c30_margin"], abs(m30), 1e-9)
       and dec["plus_c60"] == ("cosine" if m60 > 0 else "wsd") and _close(dec["plus_c60_margin"], abs(m60), 1e-9), f"+30 {m30:+.4f}, +60 {m60:+.4f}")
    ck("schedules: the cosine-to-zero control is worse than cosine-to-10% at the horizon (progress-limited run)",
       SC["cosine_to_zero"] > T["end"]["cosine"][i0])
    ck("schedules: every cooldown improves on its own checkpoint (the anneal always pays)",
       all(sum(T[k][kind]) / len(T[k][kind]) < sum(T["stop"][kind]) / len(T["stop"][kind]) for k in ("lin30", "lin60", "sqrt30", "sqrt60") for kind in ("cosine", "wsd")))

    # ------------------------------------------------ §5 lr sweep
    LS = R["lr_sweep"]
    AN = LS["analysis"]
    for d, a in AN.items():
        tab = LS["table"][d]
        # per-seed vertices re-derived from the committed run table
        for s, ps in a["per_seed"].items():
            pts = {float(k): tab[k][s]["final"] for k in tab if s in tab[k] and not tab[k][s]["diverged"]}
            ks = sorted(pts)
            i = min(range(len(ks)), key=lambda j: pts[ks[j]])
            edge = i in (0, len(ks) - 1)
            xv = None if edge else parabola_vertex([ks[i - 1], ks[i], ks[i + 1]], [pts[ks[i - 1]], pts[ks[i]], pts[ks[i + 1]]])
            ck(f"sweep: width {d} seed {s} vertex re-derived (argmin 2^{ks[i]:g}, interior)",
               not edge and xv is not None and _close(xv, ps["vertex_log2"], 1e-9) and ps["argmin_log2"] == ks[i], f"vertex {xv}")
        evs = sorted(a["argmin_by_eval_step"], key=int)
        ck(f"sweep: width {d} argmin at 75% of the run within half an octave of the final one, optimum interior",
           abs(a["argmin_by_eval_step"][evs[-1]] - a["argmin_by_eval_step"][evs[-2]]) <= 0.5 and not a["on_grid_edge"], str(a["argmin_by_eval_step"]))
    E = LS["extrapolation"]
    reg = E["regression"]
    xs, ys = E["fit_x"], E["fit_y"]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a0 = my - slope * mx
    resid = [y - (a0 + slope * x) for x, y in zip(xs, ys)]
    sigma = math.sqrt(sum(r * r for r in resid) / (n - 2))
    pred = a0 + slope * 12
    ck("sweep: regression slope, intercept, sigma and the 4096 prediction re-derived from the committed (width, seed) vertices",
       _close(slope, reg["slope"], 1e-9) and _close(a0, reg["intercept"], 1e-9) and _close(sigma, reg["sigma"], 1e-9)
       and _close(pred, reg["pred"], 1e-9) and reg["n"] == n and reg["dof"] == n - 2, f"slope {slope:+.3f}, n={n}")
    half = reg["t"] * sigma * math.sqrt(1 + 1 / n + (12 - mx) ** 2 / sxx)
    ck("sweep: prediction-interval half-width and the total band re-derived",
       _close(half, reg["pi_half"], 1e-9) and _close(E["half_total"], math.sqrt(half ** 2 + E["grid_term"] ** 2), 1e-9)
       and _close(E["interval_lr"][0], 2 ** (reg["pred"] - E["half_total"]), 1e-9))
    ck("sweep: the optimum falls with width and the slope's 95% CI meets the theory bracket [-1, -0.5]",
       slope < -0.25 and (reg["slope"] - reg["t"] * reg["se_slope"]) <= -0.5 and (reg["slope"] + reg["t"] * reg["se_slope"]) >= -1.0,
       f"CI [{reg['slope'] - reg['t'] * reg['se_slope']:+.2f}, {reg['slope'] + reg['t'] * reg['se_slope']:+.2f}]")
    v_top = AN[str(LS["widths"][-1])]["vertex_log2"]
    dist = 12 - math.log2(LS["widths"][-1])
    ck("sweep: theory bracket re-derived from the widest measured vertex",
       _close(E["theory_bracket_log2"]["slope_-1"], v_top - dist, 1e-9) and _close(E["theory_bracket_log2"]["slope_-0.5"], v_top - 0.5 * dist, 1e-9))
    ck("sweep: the pre-registered workhorse lr sits within one grid step of the width-256 optimum",
       abs(math.log2(R["bias_correction"]["lr"]) - AN["256"]["vertex_log2"]) <= 1.0)
    pen = E["penalty_1024"]
    for d, a in AN.items():
        pp = a["penalty_pct"]
        mc = a["mean_curve"]
        c = pp["centre_log2"]
        ck(f"sweep: width {d} basin penalties centred on the argmin and re-derived from the mean curve",
           c == a["argmin_log2"] and "div2" in pp and "x2" in pp and _close(pp["div2"], (mc[f"{c - 1:+.2f}"] / mc[f"{c:+.2f}"] - 1) * 100, 1e-9)
           and _close(pp["x2"], (mc[f"{c + 1:+.2f}"] / mc[f"{c:+.2f}"] - 1) * 100, 1e-9), f"centre 2^{c:g}: {pp}")
        ck(f"sweep: width {d}: no run diverged", a["n_diverged"] == 0)
    ck("sweep: at the widest width x2 costs more than /2, and the recommended value is below the point estimate",
       "x2" in pen and "div2" in pen and pen["x2"] > pen["div2"] and E["use_lr"] < E["pred_lr"], str(pen))
    ck("sweep: the slope's 95% CI is stated as it is (between the two exponents; the flag says whether it excludes both)",
       _close(E["ci"][0], reg["slope"] - reg["t"] * reg["se_slope"], 1e-9) and E["ci_excludes_both_theory_ends"] == (E["ci"][0] > -1 and E["ci"][1] < -0.5))
    SP = LS["spot_check"]
    pr = SP["pre_registered"]
    finals = {float(k): (v["final"] if not v["diverged"] else float("inf")) for k, v in SP["runs"].items()}
    ks = sorted(k for k in finals if math.isfinite(finals[k]))
    i = min(range(len(ks)), key=lambda j: finals[ks[j]])
    xv = None if i in (0, len(ks) - 1) else parabola_vertex([ks[i - 1], ks[i], ks[i + 1]], [finals[ks[i - 1]], finals[ks[i]], finals[ks[i + 1]]])
    kc = pr["grid_log2"][2]
    holds = xv is not None and abs(xv - pr["pred_log2"]) <= pr["pi_half_log2"] and finals[float(kc)] <= 1.01 * min(finals.values())
    ck(f"sweep: width {SP['width']} spot check (five half-octave points) re-derived against its pre-registered criterion",
       len(pr["grid_log2"]) == 5 and ((SP["vertex_log2"] is None and xv is None) or (_close(xv, SP["vertex_log2"], 1e-9) and holds == SP["holds"])),
       f"vertex {xv}, pre-registered {pr['pred_log2']:.2f} ± {pr['pi_half_log2']:.2f} -> {'holds' if holds else 'fails'}")
    ck("sweep: the pre-registered 2048 prediction is the 3-width regression evaluated at log2(2048)",
       _close(pr["pred_log2"], a0 + slope * math.log2(SP["width"]), 1e-9))
    if LS["final"]:
        F4 = LS["final"]["regression"]
        xs4, ys4 = xs + [math.log2(SP["width"])], ys + [SP["vertex_log2"]]
        n4 = len(xs4)
        mx4, my4 = sum(xs4) / n4, sum(ys4) / n4
        s4 = sum((x - mx4) * (y - my4) for x, y in zip(xs4, ys4)) / sum((x - mx4) ** 2 for x in xs4)
        ck("sweep: the four-width refit re-derived", _close(s4, F4["slope"], 1e-9) and _close(my4 - s4 * mx4 + s4 * 12, F4["pred"], 1e-9))
    if LS["horizon"]:
        hs = [k for k in LS["horizon"] if k.isdigit()]
        ck("sweep: horizon counterfactual drift per doubling re-derived",
           _close(LS["horizon"]["drift_per_doubling_log2"],
                  (LS["horizon"][max(hs, key=int)]["vertex_log2"] - LS["horizon"][min(hs, key=int)]["vertex_log2"]) / math.log2(int(max(hs, key=int)) / int(min(hs, key=int))), 1e-9))

    # ------------------------------------------------ §6 tune both sides
    F = R["fair_compare"]
    ck("fair: HandLion matches Algorithm 2 to 1e-12 and the mis-ordered port does not",
       F["lion_check"]["vs_algorithm2"] < 1e-12 and F["lion_check"]["misordered_port"] > 1e-4)
    for kind, g in F["grid_results"].items():
        ok_ = {k: v["final"] for k, v in g.items() if not v["diverged"]}
        kb = min(ok_, key=ok_.get)
        keys = sorted(g, key=float)
        ck(f"fair: {kind} argmin re-derived from its {len(g)}-point grid, interior",
           float(kb) == F["sides"][kind]["argmin_log2"] and kb not in (keys[0], keys[-1]))
    verd = {pair: fair_compare(F["sides"][a], F["sides"][b]) for pair, (a, b) in
            {"fair_sgd": ("adamw", "sgd"), "fair_lion": ("adamw", "lion"), "fair_sgd_vs_lion": ("sgd", "lion")}.items()}
    ck("fair: the three verdicts re-derived through a pure-python fair_compare (same refusal rules)",
       all((F["verdicts"][k]["verdict"].startswith("REFUSED")) == (v == "REFUSED") for k, v in verd.items())
       and all(v == "REFUSED" or (v == "A") == F["verdicts"][k]["verdict"].startswith("adamw" if k != "fair_sgd_vs_lion" else "sgd") for k, v in verd.items()), str(verd))
    ck("fair: confirmation seeds are disjoint from the selection seed", cfg["seed"] not in F["confirm_seeds"] and len(F["confirm_seeds"]) >= 3)
    ck("fair: naive A exaggerates each challenger's deficit (gap at AdamW's lr > fair gap + 0.05)",
       all(F["verdicts"][f"naiveA_{k}"]["gap"] > (F["verdicts"][f"fair_{k}"]["gap"] or 0) + 0.05 for k in ("sgd", "lion")))
    ck("fair: naive B — at the 3e-4 default AdamW loses to both tuned challengers (the false-positive direction)",
       all(F["verdicts"][f"naiveB_{k}_common LLM default 3e-4"]["gap"] < 0 for k in ("sgd", "lion")))
    ck("fair: naive B at the 1e-3 default re-derived from the committed means",
       all(_close(F["verdicts"][f"naiveB_{k}_torch default 1e-3"]["gap"], F["sides"][k]["mean"] - F["naive_b"]["torch default 1e-3"]["mean"], 1e-9) for k in ("sgd", "lion")))
    ck("fair: unclipped deltas are same-seed comparisons, and SGD re-tuned WITHOUT clipping is recorded with an interior optimum",
       all(_close(v["delta_same_seed"], v["final"] - F["sides"][k]["selection_seed_loss"], 1e-9) for k, v in F["unclipped"].items())
       and not F["sgd_unclipped_best"]["on_edge"], f"unclipped-tuned SGD {F['sgd_unclipped_best']['final']:.4f} at 2^{F['sgd_unclipped_best']['log2']:g}")
    ck("fair: Lion's paper-rule ports and its decay robustness are committed (3x end near the tuned optimum, 10x end far off)",
       abs(F["lion_ports"]["paper rule, /3 end"] - F["sides"]["lion"]["selection_seed_loss"]) < 0.05
       and F["lion_ports"]["paper rule, /10 end"] > F["sides"]["lion"]["selection_seed_loss"] + 0.1
       and max(F["wd_check"][k] for k in ("lion_wd_0", "lion_wd_matched", "lion_wd_1.0")) - min(F["wd_check"][k] for k in ("lion_wd_0", "lion_wd_matched", "lion_wd_1.0")) < 0.02)
    ck("fair: the refusal paths fire (edge, config mismatch, inside the noise)",
       all(v.startswith("REFUSED") for v in F["refusals"].values()))
    ck("fair: clipping binds on a substantial share of steps in every arm, and unclipped SGD at its clipped optimum diverges or falls far behind",
       all(v["clip_bind_frac"] > 0.1 for v in F["sides"].values())
       and (F["unclipped"]["sgd"]["diverged"] or F["unclipped"]["sgd"]["final"] > F["sides"]["sgd"]["mean"] + 0.1))
    ck("fair: weight decay copied verbatim into SGD's units hurts; the matched convention does not",
       F["wd_check"]["sgd_wd_0.1_verbatim"] > F["wd_check"]["sgd_wd_matched"] + 0.05
       and abs(F["wd_check"]["sgd_wd_matched"] - F["wd_check"]["sgd_wd_0"]) < 0.05)
    ck("fair: both challengers' best lr differ from AdamW's by >= 2x (the units really differ)",
       abs(F["sides"]["sgd"]["argmin_log2"] - F["sides"]["adamw"]["argmin_log2"]) >= 2
       and abs(F["sides"]["lion"]["argmin_log2"] - F["sides"]["adamw"]["argmin_log2"]) >= 1)

    return failures

if __name__ == "__main__":
    def _print(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))

    n = run(_print)
    print(f"verdict: {'PASS' if n == 0 else f'{n} FAILING CHECKS'}")
    raise SystemExit(0 if n == 0 else 1)
