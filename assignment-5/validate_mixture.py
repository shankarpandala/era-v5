#!/usr/bin/env python3
"""
ERA-V5 Assignment 5 - the mixture self-audit.

The spec in README.md is mirrored by machine-readable numbers in mixture.json.
This script re-derives every arithmetic claim a reviewer would probe and fails
loudly on any inconsistency, so "the plan adds up" is a checked property, not
an assertion:

  1. Every stage's lane shares sum to 100%.
  2. Stage token budgets sum to the total training budget.
  3. No lane is allocated more tokens than its real supply can provide within
     the declared epoch cap plus its openly declared synthetic capacity - the
     wishful-accounting check. Implied epochs per lane are printed.
  4. The protected always-on floor holds in every stage for every protected
     lane (the selector is never allowed below it).
  5. The Indic tier split (verified / unverified / translated / synthetic)
     sums to the Indic lane total, respects per-tier epoch caps, and the
     translated tier stays under its declared ceiling.
  6. The anneal reserve is genuinely held back: main-run (pre-anneal) lane
     allocations must fit inside (unique - reserved) x epoch_cap + synthetic,
     and the anneal stage spends only what the reserve holds.
  7. Difficulty (D0-D3) and reasoning-length (R0-R4) band shares sum to 100%
     per stage; R-band token ranges are contiguous and non-overlapping.
  8. Proxy protocol is complete: five arms, primary metrics with decision
     rules, 1B/3B budgets, stage fractions matching the full curriculum.

Run:  python3 validate_mixture.py            (exit 0 = every check passes)
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 0.05  # percentage-point tolerance on share sums; billions on token sums

failures = []
warnings = []


def check(ok: bool, msg: str):
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok:
        failures.append(msg)


def warn(msg: str):
    print("  WARN  " + msg)
    warnings.append(msg)


def main():
    with open(os.path.join(HERE, "mixture.json")) as f:
        mx = json.load(f)

    lanes = list(mx["lanes"].keys())
    stages = mx["stages"]
    supply = mx["supply"]
    floors = mx["protected_floor_pct"]
    reserve = mx["anneal_reserve"]
    tiers = mx["indic_tiers"]
    total_budget = mx["meta"]["total_budget_tokens_b"]

    # ---- 1. shares sum to 100 in every stage
    print("\n[1] stage shares sum to 100%")
    for st in stages:
        s = sum(st["shares"].values())
        check(abs(s - 100.0) <= TOL, f"{st['name']}: shares sum to {s:.2f}")
        unknown = set(st["shares"]) - set(lanes)
        check(not unknown, f"{st['name']}: no unknown lanes {sorted(unknown) or ''}")

    # ---- 2. stage budgets sum to the total budget
    print("\n[2] stage budgets sum to the training budget")
    tot = sum(st["tokens_b"] for st in stages)
    check(abs(tot - total_budget) <= TOL,
          f"stages sum to {tot:,.0f}B vs budget {total_budget:,.0f}B")

    # ---- per-lane allocation, split main-run vs anneal
    alloc_main = {ln: 0.0 for ln in lanes}
    alloc_anneal = {ln: 0.0 for ln in lanes}
    for st in stages:
        tgt = alloc_anneal if st.get("anneal") else alloc_main
        for ln, pct in st["shares"].items():
            tgt[ln] += st["tokens_b"] * pct / 100.0

    # ---- 3. supply honesty: main-run allocation fits real supply
    print("\n[3] supply honesty (main run: allocation <= (unique - reserved) x "
          "epoch_cap + declared synthetic)")
    for ln in lanes:
        sp = supply[ln]
        unique = sp["unique_tokens_b"]
        reserved = sp.get("reserved_for_anneal_b", 0.0)
        cap = sp["epoch_cap"]
        synth = sp.get("synthetic_capacity_b", 0.0)
        avail = (unique - reserved) * cap + synth
        a = alloc_main[ln]
        real_base = max(0.001, unique - reserved)
        implied_epochs = max(0.0, a - synth) / real_base
        check(a <= avail + TOL,
              f"{ln}: main-run {a:,.0f}B <= ({unique:,.0f}-{reserved:,.0f})x{cap} "
              f"+ {synth:,.0f}B synth = {avail:,.0f}B "
              f"(implied ~{implied_epochs:.1f} epochs on real data)")
        if implied_epochs > cap:
            warn(f"{ln}: implied epochs {implied_epochs:.1f} exceed cap {cap} "
                 f"- only legal if synthetic fills the gap")

    # ---- 4. protected floor in every stage
    print("\n[4] protected always-on floor")
    for ln, fl in floors.items():
        for st in stages:
            share = st["shares"].get(ln, 0.0)
            check(share + 1e-9 >= fl,
                  f"{ln}: {st['name']} share {share:.1f}% >= floor {fl:.1f}%")

    # ---- 5. Indic tier split
    print("\n[5] Indic tier split")
    indic_total = alloc_main["indic"] + alloc_anneal["indic"]
    tier_sum = sum(t["total_b"] for t in tiers.values())
    check(abs(tier_sum - indic_total) <= max(1.0, 0.01 * indic_total),
          f"tiers sum to {tier_sum:,.0f}B vs Indic lane total {indic_total:,.0f}B")
    tr = tiers["translated"]
    cap_share = tr["cap_share_of_indic"]
    check(tr["total_b"] <= cap_share * indic_total + TOL,
          f"translated {tr['total_b']:,.0f}B <= {cap_share:.0%} of Indic "
          f"({cap_share * indic_total:,.0f}B)")
    for name, t in tiers.items():
        if t.get("unique_tokens_b"):
            ep = (t["total_b"] - t.get("synthetic_b", 0.0)) / t["unique_tokens_b"]
            check(ep <= t["epoch_cap"] + 0.05,
                  f"tier {name}: {ep:.1f} epochs on {t['unique_tokens_b']:,.0f}B "
                  f"unique <= cap {t['epoch_cap']}")
        else:
            check(t.get("synthetic_b", 0.0) + TOL >= t["total_b"],
                  f"tier {name}: fully synthetic ({t['total_b']:,.0f}B declared built)")

    # ---- 6. anneal reserve genuinely held back
    print("\n[6] anneal reserve")
    pools = reserve["pools_b"]
    spent = {ln: alloc_anneal.get(ln, 0.0) for ln in lanes if alloc_anneal.get(ln, 0.0) > 0}
    for ln, a in spent.items():
        pool = pools.get(ln, 0.0)
        check(a <= pool + TOL,
              f"{ln}: anneal spend {a:,.0f}B <= reserved pool {pool:,.0f}B")
    for ln, pool in pools.items():
        rsv = supply[ln].get("reserved_for_anneal_b", 0.0)
        check(pool <= rsv + supply[ln].get("synthetic_anneal_capacity_b", 0.0) + TOL,
              f"{ln}: pool {pool:,.0f}B backed by {rsv:,.0f}B reserved real data "
              f"+ {supply[ln].get('synthetic_anneal_capacity_b', 0.0):,.0f}B "
              f"anneal-grade synthetic")
    total_reserved = sum(pools.values())
    anneal_tokens = sum(st["tokens_b"] for st in stages if st.get("anneal"))
    check(anneal_tokens <= total_reserved + TOL,
          f"anneal stage {anneal_tokens:,.0f}B <= total reserved {total_reserved:,.0f}B "
          f"(remainder feeds SFT/RLVR in Sessions 17-18)")

    # ---- 7. difficulty + reasoning-length bands
    print("\n[7] difficulty and reasoning-length bands")
    stage_keys = ["A", "B", "C", "D"]
    diff = mx.get("difficulty_bands", {})
    rlen = mx.get("reasoning_length_bands", {})
    check(bool(diff.get("bands")), "difficulty_bands.bands present")
    check(bool(rlen.get("bands")), "reasoning_length_bands.bands present")
    if diff.get("bands"):
        for sk in stage_keys:
            s = sum(b["share_of_lane_pct"][sk] for b in diff["bands"].values())
            check(abs(s - 100.0) <= TOL,
                  f"difficulty shares stage {sk} sum to {s:.1f}%")
        check(set(diff["bands"]) == {"D0", "D1", "D2", "D3"},
              f"difficulty keys D0-D3 (got {sorted(diff['bands'])})")
        for name, b in diff["bands"].items():
            check(bool(b.get("example")), f"difficulty {name} has a concrete example")
    if rlen.get("bands"):
        for sk in stage_keys:
            s = sum(b["share_of_reasoning_pct"][sk] for b in rlen["bands"].values())
            check(abs(s - 100.0) <= TOL,
                  f"reasoning-length shares stage {sk} sum to {s:.1f}%")
        check(set(rlen["bands"]) == {"R0", "R1", "R2", "R3", "R4"},
              f"reasoning-length keys R0-R4 (got {sorted(rlen['bands'])})")
        # contiguous non-overlapping token ranges in band order
        ordered = [rlen["bands"][k] for k in ["R0", "R1", "R2", "R3", "R4"]]
        prev_hi = 0
        for i, b in enumerate(ordered):
            lo, hi = b["token_range"]
            check(lo <= hi, f"R{i} token_range lo<=hi ({lo},{hi})")
            if i == 0:
                check(lo == 1, f"R0 starts at 1 (got {lo})")
            else:
                check(lo == prev_hi + 1,
                      f"R{i} starts at {prev_hi + 1} (contiguous; got {lo})")
            prev_hi = hi
            check(bool(b.get("example")), f"reasoning R{i} has a concrete example")

    # ---- 8. proxy protocol completeness
    print("\n[8] proxy experiment protocol")
    proxy = mx.get("proxy_experiments", {})
    check(bool(proxy), "proxy_experiments block present")
    if proxy:
        arms = proxy.get("arms", [])
        arm_ids = {a.get("id") for a in arms}
        required_arms = {
            "H0_uniform", "H1_proposed", "H2_no_indic_floor",
            "H3_agentic_real_only", "H4_no_late_upweight",
        }
        check(required_arms <= arm_ids,
              f"five required arms present (missing {sorted(required_arms - arm_ids) or 'none'})")
        prim = proxy.get("metrics", {}).get("primary", [])
        check(len(prim) >= 4, f"≥4 primary metrics with decision rules (got {len(prim)})")
        for m in prim:
            check(bool(m.get("decision")), f"metric {m.get('name')} has a decision rule")
        scales = proxy.get("scales", {})
        for sk, want_tok in (("1B", 40), ("3B", 90)):
            sc = scales.get(sk, {})
            check(sc.get("tokens_b") == want_tok,
                  f"{sk} proxy budget is {want_tok}B tokens (got {sc.get('tokens_b')})")
        fr = proxy.get("scaling_rule", {}).get("stage_fractions_of_budget", {})
        # must match full curriculum: 12000/16000, 3000/16000, 600/16000, 400/16000
        expected_fr = {"A": 12000 / 16000, "B": 3000 / 16000,
                       "C": 600 / 16000, "D": 400 / 16000}
        for k, exp in expected_fr.items():
            got = fr.get(k, -1)
            check(abs(got - exp) <= 1e-6,
                  f"proxy stage fraction {k}={got} matches curriculum {exp}")
        # proxy stage token sums
        for sk, sc in scales.items():
            tok = sc.get("tokens_b", 0)
            rebuilt = sum(fr.get(k, 0) * tok for k in expected_fr)
            check(abs(rebuilt - tok) <= TOL,
                  f"{sk}: stage fractions × budget rebuild {rebuilt:.2f}B ≈ {tok}B")
        status = proxy.get("status", "")
        check(status in ("specified_not_run", "running", "complete"),
              f"proxy status is explicit (got {status!r})")
        if status == "specified_not_run":
            results = proxy.get("proxy_results", {})
            if any(results.get(k) for k in ("1B", "3B", "midtrain_surrogate")):
                warn("status is specified_not_run but proxy_results has non-null entries")

    # ---- ledger summary
    print("\n---- lane ledger (billions of tokens) ----")
    print(f"{'lane':<20}{'main run':>10}{'anneal':>9}{'total':>9}{'share':>8}")
    for ln in lanes:
        t = alloc_main[ln] + alloc_anneal[ln]
        print(f"{ln:<20}{alloc_main[ln]:>10,.0f}{alloc_anneal[ln]:>9,.0f}"
              f"{t:>9,.0f}{t / total_budget:>8.1%}")
    print(f"{'TOTAL':<20}{sum(alloc_main.values()):>10,.0f}"
          f"{sum(alloc_anneal.values()):>9,.0f}"
          f"{sum(alloc_main.values()) + sum(alloc_anneal.values()):>9,.0f}")

    print(f"\n{len(failures)} failures, {len(warnings)} warnings")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
