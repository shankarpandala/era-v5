"""Claim B — the experiment matrix.

Seven arms share one architecture, optimizer, schedule, and byte-identical
batch stream; the arm name selects only the embedding provider (proved
post-hoc by the audit comparing arch hashes across every result file):

  kron_v2       char block + full numeric block, frozen        (ours)
  kron_char     char block only, frozen                        (ablation: no numeric block)
  readout_only  char + numeric minus {LIN, SIGN, LOG}, frozen  (FoNE-style ablation)
  hom_only      char + {LIN, SIGN, LOG, NUMFLAG} only, frozen  (marginal value of the algebra dims)
  frozen_rand   deterministic random rows, frozen              (capacity / frozen-ness control)
  learned       nn.Embedding, trainable                        (the conventional baseline)
  xval          value-scaled shared direction, trainable       (xVal-style baseline)

Main comparison: all seven arms at BOTH operating points (2000 = primary,
8000 = saturation), several seeds. Sample-efficiency curve extends kron_v2
vs learned down to 500 (train sets are nested, so the curve measures data
volume and nothing else). Probes run on every probe-arm seed at the primary
size. The NL transfer slice ("what is 9 plus 9") runs kron_v2 vs learned.
"""

from __future__ import annotations

import numpy as np

from .train import run_one
from .util import ensure_dir, write_json

FULL_PLAN = {
    "arms": ["kron_v2", "kron_char", "readout_only", "hom_only",
             "frozen_rand", "learned", "xval"],
    "sizes": [2000, 8000],
    "primary_size": 2000,
    "seeds": [0, 1, 2, 3, 4],
    "curve_arms": ["kron_v2", "learned"],
    "curve_sizes": [500],
    "probe_arms": ["kron_v2", "readout_only", "hom_only", "frozen_rand",
                   "learned"],
    "nl_arms": ["kron_v2", "learned"],
    "nl_sizes": [2000, 8000],
    "nl_seeds": [0, 1, 2],
}

FAST_PLAN = {
    "arms": ["kron_v2", "learned", "frozen_rand"],
    "sizes": [2000],
    "primary_size": 2000,
    "seeds": [0],
    "curve_arms": ["kron_v2", "learned"],
    "curve_sizes": [500],
    "probe_arms": ["kron_v2", "learned"],
    "nl_arms": ["kron_v2", "learned"],
    "nl_sizes": [2000],
    "nl_seeds": [0],
}


def plan_runs(plan: dict) -> list[dict]:
    runs = [{"task": "arith", "arm": a, "train_size": sz, "seed_idx": s,
             "probe": (a in plan["probe_arms"] and sz == plan["primary_size"])}
            for a in plan["arms"] for sz in plan["sizes"]
            for s in plan["seeds"]]
    runs += [{"task": "arith", "arm": a, "train_size": sz, "seed_idx": s,
              "probe": False}
             for a in plan["curve_arms"] for sz in plan["curve_sizes"]
             for s in plan["seeds"]]
    runs += [{"task": "nl", "arm": a, "train_size": sz, "seed_idx": s,
              "probe": False}
             for a in plan["nl_arms"] for sz in plan["nl_sizes"]
             for s in plan["nl_seeds"]]
    return runs


def group_key(task: str, arm: str, size: int) -> str:
    return f"{task}:{arm}@{size}"


def run_matrix(plan: dict, out_root: str, base_cfg: dict | None = None,
               progress=None) -> dict:
    """Execute every run in the plan, write per-run result.json files, and an
    aggregated results.json. ``progress`` is an optional callback(str)."""
    ensure_dir(out_root)
    runs = plan_runs(plan)
    results = []
    for i, spec in enumerate(runs):
        cfg = dict(base_cfg or {})
        cfg["probe"] = spec["probe"]
        cfg["task"] = spec["task"]
        run_dir = (f"{out_root}/runs/{spec['task']}_{spec['arm']}"
                   f"_{spec['train_size']}_s{spec['seed_idx']}")
        r = run_one(spec["arm"], spec["train_size"], spec["seed_idx"],
                    run_dir, cfg=cfg)
        results.append(r)
        if progress:
            ein = r["eval"]["eval_in"]["add"]["primary"]["exact"]
            hole = r["eval"]["eval_hole"]["add"]["primary"]["exact"]
            progress(f"run {i + 1}/{len(runs)} {spec['task']}:{spec['arm']}"
                     f"@{spec['train_size']} seed{spec['seed_idx']}:"
                     f" in-add={ein:.3f} hole-add={hole:.3f}"
                     f" ({r['wall_time_s']:.0f}s)")
    aggregated = aggregate(results, plan)
    write_json(f"{out_root}/results.json", aggregated)
    return aggregated


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

METRIC_PATHS = {
    "in_add_exact": ("eval_in", "add", "primary", "exact"),
    "in_mul_exact": ("eval_in", "mul", "primary", "exact"),
    "in_sub_exact": ("eval_in", "sub", "primary", "exact"),
    "in_mul_within_1pct": ("eval_in", "mul", "primary", "within_1pct"),
    "hole_add_exact": ("eval_hole", "add", "primary", "exact"),
    "hole_mul_exact": ("eval_hole", "mul", "primary", "exact"),
    "hole_sub_exact": ("eval_hole", "sub", "primary", "exact"),
    "hole_mul_within_1pct": ("eval_hole", "mul", "primary", "within_1pct"),
    "extra_add_exact": ("eval_extra", "add", "primary", "exact"),
    "extra_add_mae": ("eval_extra", "add", "primary", "mae"),
    "extra_mul_within_1pct": ("eval_extra", "mul", "primary", "within_1pct"),
    "in_add_cls_exact": ("eval_in", "add", "cls_decode", "exact"),
    "hole_add_cls_exact": ("eval_hole", "add", "cls_decode", "exact"),
}


def _dig(d: dict, path: tuple):
    for k in path:
        d = d[k]
    return d


def aggregate(results: list[dict], plan: dict) -> dict:
    groups: dict = {}
    for r in results:
        key = group_key(r["task"], r["arm"], r["train_size"])
        groups.setdefault(key, []).append(r)

    by_group = {}
    for key, rs in sorted(groups.items()):
        entry = {"n_seeds": len(rs),
                 "wall_time_s": [r["wall_time_s"] for r in rs]}
        for mname, path in METRIC_PATHS.items():
            vals = [float(_dig(r["eval"], path)) for r in rs]
            entry[mname] = {"mean": float(np.mean(vals)),
                            "std": float(np.std(vals)),
                            "values": vals}
        buckets = rs[0]["eval"]["eval_extra"].get("by_bucket", {})
        entry["extra_buckets"] = {
            b: {"add_exact": float(np.mean(
                    [r["eval"]["eval_extra"]["by_bucket"][b]["add"]["exact"]
                     for r in rs])),
                "add_mae": float(np.mean(
                    [r["eval"]["eval_extra"]["by_bucket"][b]["add"]["mae"]
                     for r in rs]))}
            for b, bd in buckets.items() if "add" in bd}
        by_group[key] = entry

    probes = {f"{r['arm']}@{r['train_size']}_s{r['seed_idx']}": r["probe"]
              for r in results if "probe" in r}
    arch_hashes = sorted({r["arch_hash"] for r in results})
    return {
        "plan": plan,
        "n_runs": len(results),
        "arch_hashes": arch_hashes,
        "arch_identical_across_arms": len(arch_hashes) == 1,
        "by_group": by_group,
        "probes": probes,
        "run_index": [{"task": r["task"], "arm": r["arm"],
                       "train_size": r["train_size"],
                       "seed_idx": r["seed_idx"],
                       "dir": (f"runs/{r['task']}_{r['arm']}"
                               f"_{r['train_size']}_s{r['seed_idx']}")}
                      for r in results],
    }
