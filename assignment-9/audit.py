"""Independent audit for assignment 9.

Re-derives every claim from the files in submission_artifacts/ (and the executed
notebook itself) WITHOUT executing any notebook code — this module shares no state
with the code that produced the artifacts. Run directly:

    python audit.py        # prints [PASS]/[FAIL] per check and a final verdict

or through run_demo.py, which mirrors the same checks into run.log.

Strict thresholds apply to full-budget runs; a run recorded with fast=True keeps
the structural checks but relaxes the training-dynamics thresholds (40-step arms
haven't converged — that is what the smoke mode is for).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / "submission_artifacts"
NOTEBOOK = HERE / "loss_harness.ipynb"
README = HERE / "README.md"

REL_TOL = 1e-4


def _close(a: float, b: float, tol: float = REL_TOL) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _masked_mean(per_tok, mask):
    num = sum(p * m for row_p, row_m in zip(per_tok, mask)
              for p, m in zip(row_p, row_m))
    den = sum(m for row in mask for m in row)
    return num / max(den, 1), den


def run(check) -> int:
    """check(name, ok, detail) is called once per audit item. Returns #failures."""
    failures = [0]

    def c(name, ok, detail=""):
        if not ok:
            failures[0] += 1
        check(name, bool(ok), detail)

    try:
        results = json.loads((ART / "results.json").read_text())
        curves = json.loads((ART / "loss_curves.json").read_text())
        ptl = json.loads((ART / "per_token_losses.json").read_text())
    except Exception as exc:
        c("artifacts parse", False, repr(exc)[:200])
        return failures[0]
    c("artifacts parse", True,
      "results.json, loss_curves.json, per_token_losses.json")

    cfg = results.get("config", {})
    fast = bool(cfg.get("fast"))
    p1 = results.get("part1", {})
    V = p1.get("perplexity", {}).get("vocab_size", 259)
    d = cfg.get("d_model", 0)

    # -- requirement 2: the string audit really ran, in strings ----------------
    c("shift audit: all three alignments named at §4",
      p1.get("shift_audit_all") == {"correct": "CORRECT",
                                    "reversed": "REVERSED",
                                    "no_shift": "IDENTITY"},
      str(p1.get("shift_audit_all")))

    # -- requirement 3: padding, re-derived from raw per-token losses ----------
    pad = p1.get("padding", {})
    per_tok, mask = ptl["padding"]["per_tok"], ptl["padding"]["mask"]
    masked_mean, n_contrib = _masked_mean(per_tok, mask)
    flat = [p for row in per_tok for p in row]
    unmasked_mean = sum(flat) / len(flat)
    c("padding: masked mean re-derived", _close(masked_mean, pad["loss_masked"]),
      f"{masked_mean:.6f} vs reported {pad['loss_masked']:.6f}")
    c("padding: unmasked mean re-derived",
      _close(unmasked_mean, pad["loss_unmasked"]),
      f"{unmasked_mean:.6f} vs reported {pad['loss_unmasked']:.6f}")
    c("padding: contributing count re-derived",
      int(n_contrib) == pad["n_contributing"],
      f"{int(n_contrib)} vs reported {pad['n_contributing']}")
    c("padding: count changes under the mask",
      pad["n_contributing"] < pad["n_slots"],
      f"{pad['n_contributing']} < {pad['n_slots']}")
    c("padding: float-mask == ignore_index",
      _close(pad["loss_masked"], pad["loss_ignore_index"], 1e-4),
      f"{pad['loss_masked']:.6f} vs {pad['loss_ignore_index']:.6f}")
    wrong_expected = pad["loss_masked"] * pad["n_contributing"] / pad["n_slots"]
    c("padding: the wrong-mean bug re-derived (masked * discount)",
      _close(pad["loss_wrongmean"], wrong_expected, 1e-3)
      and _close(pad["gradient_discount"],
                 pad["n_contributing"] / pad["n_slots"], 1e-6),
      f"{pad['loss_wrongmean']:.4f} == {pad['loss_masked']:.4f} x "
      f"{pad['gradient_discount']:.3f}")

    sft = p1.get("sft", {})
    c("padding/sft: completions-only mask shrinks the denominator",
      0 < sft.get("n_completion", 0) < sft.get("n_slots", 0),
      f"{sft.get('n_completion')} of {sft.get('n_slots')} slots")

    # -- requirement 4: packing, re-derived ------------------------------------
    pk = p1.get("packing", {})
    per_k = ptl["packing"]["per_tok"]
    base = ptl["packing"]["base_mask"]
    bnd = ptl["packing"]["boundary_mask"]
    before, _ = _masked_mean(per_k, base)
    after_mask = [[b * (1 - x) for b, x in zip(rb, rx)]
                  for rb, rx in zip(base, bnd)]
    after, _ = _masked_mean(per_k, after_mask)
    c("packing: loss-before re-derived", _close(before, pk["loss_before"]),
      f"{before:.6f} vs reported {pk['loss_before']:.6f}")
    c("packing: loss-after re-derived", _close(after, pk["loss_after"]),
      f"{after:.6f} vs reported {pk['loss_after']:.6f}")
    c("packing: masking the seam lowers the mean", after < before,
      f"{after:.4f} < {before:.4f}")
    c("packing: boundary count", int(sum(map(sum, bnd))) == pk["n_boundary"],
      f"{int(sum(map(sum, bnd)))} == {pk['n_boundary']}")
    seam = [p for row_p, row_b in zip(per_k, bnd)
            for p, b in zip(row_p, row_b) if b]
    c("packing: seam per-token losses match",
      all(_close(a, b) for a, b in zip(seam, pk["boundary_token_losses"])),
      f"{[round(x, 4) for x in seam]}")
    easy, hard = sorted(seam)
    c("packing: the seam's two slots behave as claimed — one learnable "
      "(< mean), one impossible (> mean)",
      easy < before < hard,
      f"easy {easy:.4f} < mean {before:.4f} < hard {hard:.4f}")
    c("packing: block-causal + position restart makes doc2 match doc2-alone",
      pk.get("doc2_blocked_equals_alone") is True,
      "allclose atol=1e-4")
    pt = p1.get("packing_train", {})
    c("packing: the contract lives in the training loop, and learns",
      0 < pt.get("mask_dropped_frac", 0) < 0.1
      and pt.get("final_train_loss", 99) < math.log(V),
      f"drops {pt.get('mask_dropped_frac', 0):.3%} of slots; "
      f"final train loss {pt.get('final_train_loss', float('nan')):.4f}")

    # -- requirement 5: perplexity + the identity leak -------------------------
    pp = p1.get("perplexity", {})
    c("perplexity: ln(V) recomputed", _close(pp["ln_V"], math.log(V)),
      f"ln({V}) = {math.log(V):.6f}")
    c("perplexity: untrained CE sits at ln(V)",
      abs(pp["untrained_ce"] - math.log(V)) < 0.15,
      f"|{pp['untrained_ce']:.4f} - {math.log(V):.4f}| < 0.15")
    c("perplexity: ppl == exp(ce)",
      _close(pp["untrained_ppl"], math.exp(pp["untrained_ce"]), 1e-6),
      f"{pp['untrained_ppl']:.2f}")
    c("perplexity: untrained ppl within 15% of V",
      abs(pp["untrained_ppl"] - V) / V < 0.15,
      f"{pp['untrained_ppl']:.1f} vs V={V}")
    c("perplexity: loud init breaks the check (counterexample)",
      pp["bad_init_ce"] > pp["untrained_ce"] + 0.5,
      f"{pp['bad_init_ce']:.4f} >> {pp['untrained_ce']:.4f}")

    leak = p1.get("init_leak", {})
    c("init leak: tied no_shift beats uniform BEFORE any training",
      math.exp(leak["no_shift"]["tied"]) < 0.85 * V,
      f"PPL {math.exp(leak['no_shift']['tied']):.1f} << V={V}")
    c("init leak: untying removes it (all untied arms at PPL ~ V)",
      all(abs(math.exp(leak[a]["untied"]) - V) < 0.15 * V for a in leak),
      "; ".join(f"{a}: {math.exp(r['untied']):.1f}" for a, r in leak.items()))
    c("init leak: both tied SHIFTED arms stay at PPL ~ V (the leak is "
      "specific to the identity target)",
      all(abs(math.exp(leak[a]["tied"]) - V) < 0.15 * V
          for a in ("correct", "reversed")),
      f"correct {math.exp(leak['correct']['tied']):.1f}, "
      f"reversed {math.exp(leak['reversed']['tied']):.1f}")

    # -- requirement 6: tied vs untied ----------------------------------------
    tu = p1.get("tied_untied", {})
    c("tied/untied: delta equals V*d", tu["delta"] == V * d == tu["delta_formula_Vd"],
      f"{tu['delta']:,} == {V}*{d}")
    c("tied/untied: counts differ by exactly the delta",
      tu["untied"] - tu["tied"] == tu["delta"],
      f"{tu['untied']:,} - {tu['tied']:,}")

    # -- requirement 7: memory, three implementations --------------------------
    mem = p1.get("memory", {})
    for key, name in (("ratio", "row-chunked"), ("ratio_online", "online")):
        peak_key = "peak_chunked_mb" if key == "ratio" else "peak_online_mb"
        r = mem["peak_full_mb"] / max(mem[peak_key], 1e-9)
        c(f"memory: {name} ratio re-derived", _close(r, mem[key], 1e-2),
          f"{r:.2f} vs reported {mem[key]:.2f}")
        c(f"memory: {name} is materially smaller",
          mem[key] > (1.2 if fast else 2.5),
          f"{mem[key]:.1f}x on {mem['device']}"
          + (" (fast-mode threshold)" if fast else ""))
    c("memory: analytic logits size recomputed",
      _close(mem["analytic_logits_mb"], mem["N"] * mem["V"] * 4 / 2**20, 1e-6),
      f"{mem['analytic_logits_mb']:.1f} MiB for [{mem['N']:,}, {mem['V']:,}]")
    c("memory: online analytic buffer is independent of V",
      _close(mem["analytic_online_mb"],
             mem["N"] * mem["vocab_chunk"] * 4 / 2**20, 1e-6),
      f"[N, vocab_chunk] = {mem['analytic_online_mb']:.1f} MiB")
    c("fp16: naive softmax overflow demonstrated",
      p1.get("fp16_naive_is_nan") is True)

    # -- part 2 ----------------------------------------------------------------
    p2 = results.get("part2", {})
    fin = p2.get("final", {})
    logged = p2.get("final_logged", {})
    mtp = curves.get("part2", {})
    c("part2: both heads untied (the confound the review caught)",
      p2.get("heads_untied") is True)
    c("part2: train sum == L1+L2",
      _close(fin["train_sum"], fin["train_L1"] + fin["train_L2"], 1e-6))
    c("part2: held sum == L1+L2",
      _close(fin["held_sum"], fin["held_L1"] + fin["held_L2"], 1e-6))
    c("part2: held-out L2 stays above L1 (full-stream sweep)",
      fin["held_L2"] > fin["held_L1"],
      f"{fin['held_L2']:.4f} > {fin['held_L1']:.4f}")
    c("part2: logged tails match the curves",
      _close(logged["train_L1"], mtp["train_L1"][-1])
      and _close(logged["held_L2"], mtp["held_L2"][-1]),
      "results.json final_logged == loss_curves.json tail")
    c("part2: step 0 is pre-update and the gap starts at zero",
      mtp["step"][0] == 0 and abs(mtp["held_L2"][0] - mtp["held_L1"][0]) < 0.05,
      f"step {mtp['step'][0]}: gap {mtp['held_L2'][0] - mtp['held_L1'][0]:+.4f}")
    if not fast:
        c("part2: L2 above L1 at every held-out log point after step 0",
          all(b > a for a, b in zip(mtp["held_L1"][1:], mtp["held_L2"][1:])),
          f"{len(mtp['held_L1']) - 1} log points")
        train_gap = fin["train_L2"] - fin["train_L1"]
        held_gap = fin["held_L2"] - fin["held_L1"]
        c("part2: memorization asymmetry — train gap ends below held gap",
          0 < train_gap < held_gap,
          f"train {train_gap:+.4f} < held {held_gap:+.4f}")
        c("part2: same-head estimator agrees on direction (and over-states, "
          "as its bias predicts)",
          p2.get("samehead_gap", -1) > held_gap > 0,
          f"same-head {p2.get('samehead_gap', float('nan')):+.4f} > "
          f"two-heads {held_gap:+.4f}")

    # -- part 3 ----------------------------------------------------------------
    arms = results.get("part3", {}).get("arms", {})
    expect = {"correct": "CORRECT", "reversed": "REVERSED", "no_shift": "IDENTITY"}
    for name, want in expect.items():
        z = arms[name]
        c(f"part3/{name}: audit verdict is {want}", z["verdict"] == want,
          f"got {z['verdict']}")
        c(f"part3/{name}: final loss matches its curve",
          _close(z["final_loss"], curves[f"zoo_{name}"]["loss"][-1]))
    c("part3: both bugs end BELOW the correct arm (the warning, quantified)",
      arms["reversed"]["final_loss"] < arms["correct"]["final_loss"]
      and arms["no_shift"]["final_loss"] < arms["correct"]["final_loss"],
      f"correct {arms['correct']['final_loss']:.4f}, "
      f"reversed {arms['reversed']['final_loss']:.4f}, "
      f"no_shift {arms['no_shift']['final_loss']:.4f}")
    if not fast:
        c("part3: reversed's circuit named — copy-prev accuracy -> 1",
          arms["reversed"]["acc_prev"] > 0.9,
          f"acc_prev {arms['reversed']['acc_prev']:.3f}")
        c("part3: no_shift's circuit named — copy-self accuracy -> 1",
          arms["no_shift"]["acc_self"] > 0.9,
          f"acc_self {arms['no_shift']['acc_self']:.3f}")
        c("part3: the correct arm predicts, it does not copy",
          arms["correct"]["acc_next"] > arms["correct"]["acc_self"]
          and arms["correct"]["acc_next"] > arms["correct"]["acc_prev"],
          f"next {arms['correct']['acc_next']:.3f} vs "
          f"self {arms['correct']['acc_self']:.3f} / "
          f"prev {arms['correct']['acc_prev']:.3f}")

    # -- the executed notebook -------------------------------------------------
    try:
        nb = json.loads(NOTEBOOK.read_text())
        codes = [cl for cl in nb["cells"] if cl["cell_type"] == "code"]
        counts = [cl.get("execution_count") for cl in codes]
        errors = [o for cl in codes for o in cl.get("outputs", [])
                  if o.get("output_type") == "error"]
        c("notebook: executed top to bottom, in order",
          all(isinstance(x, int) for x in counts)
          and counts == sorted(counts) and len(set(counts)) == len(counts),
          f"{len(codes)} code cells")
        c("notebook: zero error outputs", not errors,
          f"{len(errors)} error outputs")
        size = NOTEBOOK.stat().st_size
        c("notebook: renders on GitHub (< 1.5 MiB)", size < 1_500_000,
          f"{size / 1024:.0f} KiB")
        stdout = "".join("".join(o.get("text", []))
                         for cl in codes for o in cl.get("outputs", [])
                         if o.get("output_type") == "stream")
        c("notebook: the strings were actually printed (␣, <bos>, audit rows)",
          "␣" in stdout and "<bos>" in stdout and "input   :" in stdout
          and "target  :" in stdout,
          "stdout carries the string audit, not just ids")
    except Exception as exc:
        c("notebook: readable", False, repr(exc)[:200])

    # -- README quotes every headline number verbatim --------------------------
    headline = results.get("headline", {})
    if README.exists():
        text = README.read_text()
        missing = [k for k, v in headline.items() if v not in text]
        c("README quotes every headline number verbatim", not missing,
          f"{len(headline)} numbers checked"
          + (f"; MISSING: {missing}" if missing else ""))
    else:
        c("README quotes every headline number verbatim", False,
          "README.md not written yet")

    return failures[0]


def main() -> int:
    def _print(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))

    n = run(_print)
    print(f"verdict: {'PASS' if n == 0 else f'FAIL ({n} checks)'}")
    return 0 if n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
