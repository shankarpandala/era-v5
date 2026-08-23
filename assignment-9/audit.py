"""Independent audit for assignment 9.

Re-derives every claim from the files in submission_artifacts/ (and the executed
notebook itself) WITHOUT executing any notebook code — this module shares no state
with the code that produced the artifacts. Run directly:

    python audit.py        # prints [PASS]/[FAIL] per check and a final verdict

or through run_demo.py, which mirrors the same checks into run.log.
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
    p1 = results.get("part1", {})
    V = p1.get("perplexity", {}).get("vocab_size", 259)
    d = cfg.get("d_model", 0)

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

    # -- requirement 4: packing, re-derived ------------------------------------
    pk = p1.get("packing", {})
    per_k = ptl["packing"]["per_tok"]
    base = ptl["packing"]["base_mask"]
    bnd = ptl["packing"]["boundary_mask"]
    before, n_base = _masked_mean(per_k, base)
    after_mask = [[b * (1 - x) for b, x in zip(rb, rx)]
                  for rb, rx in zip(base, bnd)]
    after, n_after = _masked_mean(per_k, after_mask)
    c("packing: loss-before re-derived", _close(before, pk["loss_before"]),
      f"{before:.6f} vs reported {pk['loss_before']:.6f}")
    c("packing: loss-after re-derived", _close(after, pk["loss_after"]),
      f"{after:.6f} vs reported {pk['loss_after']:.6f}")
    c("packing: masking the seam lowers the mean", after < before,
      f"{after:.4f} < {before:.4f}")
    c("packing: boundary count", int(sum(map(sum, bnd))) == pk["n_boundary"],
      f"{int(sum(map(sum, bnd)))} == {pk['n_boundary']}")
    boundary_losses = [p for row_p, row_b in zip(per_k, bnd)
                       for p, b in zip(row_p, row_b) if b]
    c("packing: seam per-token losses match",
      all(_close(a, b) for a, b in zip(boundary_losses,
                                       pk["boundary_token_losses"])),
      f"{[round(x, 4) for x in boundary_losses]}")
    avg_seam = sum(boundary_losses) / max(len(boundary_losses), 1)
    c("packing: the seam is harder than the average", avg_seam > before,
      f"seam mean {avg_seam:.4f} > packed mean {before:.4f}")

    # -- requirement 5: perplexity ---------------------------------------------
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

    # -- requirement 6: tied vs untied ----------------------------------------
    tu = p1.get("tied_untied", {})
    c("tied/untied: delta equals V*d", tu["delta"] == V * d == tu["delta_formula_Vd"],
      f"{tu['delta']:,} == {V}*{d}")
    c("tied/untied: counts differ by exactly the delta",
      tu["untied"] - tu["tied"] == tu["delta"],
      f"{tu['untied']:,} - {tu['tied']:,}")

    # -- requirement 7: memory -------------------------------------------------
    mem = p1.get("memory", {})
    ratio = mem["peak_full_mb"] / max(mem["peak_chunked_mb"], 1e-9)
    c("memory: ratio re-derived", _close(ratio, mem["ratio"], 1e-2),
      f"{ratio:.2f} vs reported {mem['ratio']:.2f}")
    c("memory: chunked is materially smaller", mem["ratio"] > 2.5,
      f"ratio {mem['ratio']:.1f}x on {mem['device']}")
    c("memory: analytic logits size recomputed",
      _close(mem["analytic_logits_mb"], mem["N"] * mem["V"] * 4 / 2**20, 1e-6),
      f"{mem['analytic_logits_mb']:.1f} MiB for [{mem['N']:,}, {mem['V']:,}]")

    # -- part 2 ----------------------------------------------------------------
    p2 = results.get("part2", {}).get("final", {})
    c("part2: train sum == L1+L2",
      _close(p2["train_sum"], p2["train_L1"] + p2["train_L2"], 1e-6))
    c("part2: held sum == L1+L2",
      _close(p2["held_sum"], p2["held_L1"] + p2["held_L2"], 1e-6))
    c("part2: held-out L2 stays above L1", p2["held_L2"] > p2["held_L1"],
      f"{p2['held_L2']:.4f} > {p2['held_L1']:.4f}")
    mtp = curves.get("part2", {})
    c("part2: finals match the logged curves",
      _close(p2["train_L1"], mtp["train_L1"][-1])
      and _close(p2["held_L2"], mtp["held_L2"][-1]),
      "results.json tail == loss_curves.json tail")
    c("part2: held-out gap starts at zero (both heads ignorant)",
      abs(mtp["held_L2"][0] - mtp["held_L1"][0]) < 0.05,
      f"step {mtp['step'][0]}: gap {mtp['held_L2'][0] - mtp['held_L1'][0]:+.4f}")
    c("part2: L2 above L1 at every held-out log point after the start",
      all(b > a for a, b in zip(mtp["held_L1"][1:], mtp["held_L2"][1:])),
      f"{len(mtp['held_L1']) - 1} log points")

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
    c("part3: only the correct arm generates corpus-like text",
      arms["correct"]["gram4"] > arms["reversed"]["gram4"]
      and arms["correct"]["gram4"] > arms["no_shift"]["gram4"],
      f"4-gram rates: correct {arms['correct']['gram4']:.3f}, "
      f"reversed {arms['reversed']['gram4']:.3f}, "
      f"no_shift {arms['no_shift']['gram4']:.3f}")

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
