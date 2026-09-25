"""Read-only audit of Assignment 13's committed evidence.

Every check re-derives a claim from the files on disk: the executed notebooks, the
per-run results.json files, the screening / max-batch / depth-sweep records, and
the README's headline numbers.  `run_demo.py --verify-only` runs it; the tests do too.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / "submission_artifacts"
NOTEBOOKS = ["01_baseline_fixed_batch.ipynb", "02_reversible_fixed_batch.ipynb",
             "03_reversible_max_batch.ipynb", "04_report.ipynb"]
RUNS = {"run1": "run1_baseline", "run2": "run2_reversible", "run3": "run3_reversible_maxbatch"}


def _load(path: Path):
    return json.loads(path.read_text())


def run(check) -> int:
    failures = 0

    def c(name, ok, detail=""):
        nonlocal failures
        ok = bool(ok)
        failures += not ok
        check(name, ok, detail)
        return ok

    # ---- notebooks -------------------------------------------------------------
    source = (HERE / "revllm.py").read_text()
    for name in NOTEBOOKS:
        path = HERE / name
        if not c(f"{name} exists", path.exists()):
            continue
        nb = _load(path)
        code = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
        executed = all(cell.get("execution_count") is not None for cell in code)
        errors = [o for cell in code for o in cell.get("outputs", []) if o.get("output_type") == "error"]
        c(f"{name}: every code cell executed", executed, f"{sum(cell.get('execution_count') is not None for cell in code)}/{len(code)} cells")
        c(f"{name}: no error outputs", not errors, "; ".join(o.get("ename", "?") for o in errors))
        exports = ["".join(cell["source"]) for cell in code if "export" in cell.get("metadata", {}).get("tags", [])]
        c(f"{name}: embedded implementation is byte-identical to revllm.py", source in exports)

    # ---- the three runs --------------------------------------------------------
    results = {}
    for key, folder in RUNS.items():
        path = ART / folder / "results.json"
        if c(f"{key}: results.json present", path.exists()):
            results[key] = _load(path)
    if len(results) < 3:
        return failures
    r1, r2, r3 = results["run1"], results["run2"], results["run3"]
    c("run1 is the plain residual baseline", r1["model_config"]["reversible"] == "none")
    c("run2 and run3 are reversible", r2["model_config"]["reversible"] != "none" and r3["model_config"]["reversible"] != "none")
    c("run2 uses run1's batch size", r2["train_config"]["batch_size"] == r1["train_config"]["batch_size"])
    c("run3 uses a larger batch than run2", r3["train_config"]["batch_size"] > r2["train_config"]["batch_size"],
      f"{r3['train_config']['batch_size']} vs {r2['train_config']['batch_size']}")
    c("all runs share the parameter count", r1["params"] == r2["params"] == r3["params"], str(r1["params"]))
    c("the model is about 20M parameters", 19_000_000 <= r1["params"]["total"] <= 22_000_000, f"{r1['params']['total']:,}")
    budgets = {r["planned_tokens"] for r in results.values()}
    c("all runs share one token budget", len(budgets) == 1, f"{budgets}")
    for key, r in results.items():
        c(f"{key}: trained the full budget", r["tokens"] >= r["planned_tokens"] and not r["diverged"],
          f"{r['tokens']:,} of {r['planned_tokens']:,}" + (" (diverged)" if r["diverged"] else ""))
        c(f"{key}: finite losses", math.isfinite(r["final_val_loss"]) and math.isfinite(r["final_train_loss_last20"]),
          f"val {r['final_val_loss']:.4f}")
        c(f"{key}: loss went down", r["evals"]["val_loss"][-1] < r["evals"]["val_loss"][0],
          f"{r['evals']['val_loss'][0]:.3f} -> {r['evals']['val_loss'][-1]:.3f}")
        c(f"{key}: throughput and memory recorded", r["tokens_per_second"] > 0 and r["peak_memory"]["peak_bytes"] > 0,
          f"{r['tokens_per_second']:,.0f} tok/s, {r['peak_memory']['peak_bytes'] / 2**30:.2f} GiB ({r['peak_memory']['kind']})")
        c(f"{key}: tokens/s consistent with tokens and time",
          abs(r["tokens_per_second"] - r["tokens"] / r["train_seconds"]) / r["tokens_per_second"] < 1e-6)
        c(f"{key}: steps x tokens/step covers the budget",
          r["steps"] * r["train_config"]["batch_size"] * r["train_config"]["seq_len"] == r["tokens"])
    c("run2 keeps far fewer activations than run1 at the same batch",
      r2["saved_for_backward"]["activation_bytes"] * 3 < r1["saved_for_backward"]["activation_bytes"],
      f"{r2['saved_for_backward']['activation_bytes'] / 2**20:,.0f} vs {r1['saved_for_backward']['activation_bytes'] / 2**20:,.0f} MiB")
    c("run2 reconstruction error is tiny", r2["reconstruction"]["max_rel_error"] < 1e-4, f"{r2['reconstruction']['max_rel_error']:.2e}")
    c("run3 peak lr follows sqrt batch scaling (capped)",
      abs(r3["train_config"]["lr"] - min(3e-3, 1e-3 * math.sqrt(r3["train_config"]["batch_size"] / r2["train_config"]["batch_size"]))) < 1e-12,
      f"{r3['train_config']['lr']:.2e}")

    # ---- screening -------------------------------------------------------------
    sp = ART / "screening.json"
    if c("screening.json present", sp.exists()):
        s = _load(sp)
        ok = {v: d for v, d in s["variants"].items() if not d["diverged"] and math.isfinite(d["final_val_loss"])}
        c("screening covers euler, midpoint (2h=1 and 2h=2) and momentum",
          set(s["variants"]) >= {"euler", "midpoint", "midpoint-2h2", "momentum"})
        c("screening winner is the best non-diverged configuration", ok and s["winner"] == min(ok, key=lambda v: ok[v]["final_val_loss"]),
          f"winner {s['winner']}: " + ", ".join(f"{v} {d['final_val_loss']:.4f}{' (diverged)' if d['diverged'] else ''}" for v, d in s["variants"].items()))
        wc = s["winner_config"]
        c("run2 and run3 use the screening winner's configuration",
          all(r["model_config"][k] == v for r in (r2, r3) for k, v in wc.items()), str(wc))
        for v, d in s["variants"].items():
            p = ART / f"screen_{v}" / "results.json"
            c(f"screen_{v}: results file matches screening.json", p.exists() and abs(_load(p)["final_val_loss"] - d["final_val_loss"]) < 1e-9)

    # ---- max batch -------------------------------------------------------------
    mp = ART / "max_batch.json"
    if c("max_batch.json present", mp.exists()):
        m = _load(mp)
        rev_key = next((k for k in m if k.startswith("reversible")), None)
        c("search covers baseline and the reversible model", "baseline" in m and rev_key is not None)
        if rev_key:
            c("reversible max batch exceeds the baseline's", m[rev_key]["max_batch"] > m["baseline"]["max_batch"],
              f"{m[rev_key]['max_batch']} vs {m['baseline']['max_batch']}")
            c("run3 trains at the reversible maximum", r3["train_config"]["batch_size"] == m[rev_key]["max_batch"])
            for k in ("baseline", rev_key):
                oks = [t for t in m[k]["trials"] if t.get("ok")]
                c(f"{k}: last verified trial fits the budget", oks and oks[-1]["batch"] == m[k]["max_batch"]
                  and oks[-1]["peak_bytes"] <= m[k]["budget_bytes"] * (1.0 if m[k]["method"].startswith("cpu") else 10.0),
                  f"batch {m[k]['max_batch']}, peak {oks[-1]['peak_bytes'] / 2**30:.2f} GiB, budget {m[k]['budget_bytes'] / 2**30:.2f} GiB" if oks else "no trials")
                peaks = [t["peak_bytes"] for t in sorted(oks, key=lambda t: t["batch"])]
                c(f"{k}: peak memory grows with batch", all(a <= b for a, b in zip(peaks, peaks[1:])))

    # ---- depth sweep & reversibility checks -------------------------------------
    dp = ART / "depth_sweep.json"
    if c("depth_sweep.json present", dp.exists()):
        rows = _load(dp)
        by = {(r["variant"], r["n_layer"]): r["activation_bytes"] for r in rows}
        depths = sorted({r["n_layer"] for r in rows})
        for v in ("euler", "midpoint", "momentum"):
            c(f"depth sweep: {v} saved bytes flat in depth", len({by[(v, L)] for L in depths}) == 1, f"{by[(v, depths[0])]:,} B")
        base = [by[("none", L)] for L in depths]
        diffs = [(b - a) / (L2 - L1) for a, b, L1, L2 in zip(base, base[1:], depths, depths[1:])]
        c("depth sweep: baseline saved bytes linear in depth", max(diffs) - min(diffs) < 1e-6 * max(diffs), f"{diffs[0]:,.0f} B/layer")
    cp = ART / "reversibility_checks.json"
    if c("reversibility_checks.json present", cp.exists()):
        for row in _load(cp):
            c(f"{row['variant']}: inverse exact and gradients match stored-activation autograd",
              row["reconstruction_max_abs_error"] < 1e-3 and row["gradient_max_rel_error"] < 1e-3,
              f"recon {row['reconstruction_max_abs_error']:.1e}, grad rel {row['gradient_max_rel_error']:.1e}")

    # ---- summary and README ---------------------------------------------------
    sp = ART / "summary.json"
    if c("summary.json present", sp.exists()):
        s = _load(sp)
        for key, r in results.items():
            c(f"summary matches {key}", key in s["runs"] and abs(s["runs"][key]["final_val_loss"] - r["final_val_loss"]) < 1e-9
              and abs(s["runs"][key]["tokens_per_second"] - r["tokens_per_second"]) < 1e-6)
    readme = (HERE / "README.md").read_text() if (HERE / "README.md").exists() else ""
    if c("README.md present", bool(readme)):
        for key, r in results.items():
            needles = [f"{r['final_val_loss']:.4f}", f"{r['tokens_per_second']:,.0f}", f"{r['peak_memory']['peak_bytes'] / 2**30:.2f} GiB"]
            missing = [n for n in needles if n not in readme]
            c(f"README quotes {key}'s val loss, tokens/s and peak memory", not missing, "missing: " + ", ".join(missing) if missing else "")
        c("README states the executed token budget", f"{r1['tokens']:,}" in readme, f"{r1['tokens']:,}")
        if (ART / "max_batch.json").exists():
            m = _load(ART / "max_batch.json")
            c("README quotes both maximum batch sizes", all(f"**{v['max_batch']}**" in readme for v in m.values()))
    lp = ART / "run.log"
    c("run.log present", lp.exists())
    return failures


if __name__ == "__main__":
    n = run(lambda name, ok, detail="": print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else "")))
    print(f"verdict: {'PASS' if n == 0 else 'FAIL'}")
    raise SystemExit(int(n != 0))
