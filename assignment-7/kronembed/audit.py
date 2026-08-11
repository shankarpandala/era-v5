"""Independent audit: re-derive every claim from disk, sharing no state with
the code that produced it.

The auditor trusts only ``submission_artifacts/`` plus the pure functions in
this package. It re-runs the algebra at a fresh PRNG coordinate (a lucky
sample in the producer cannot fake a pass), rebuilds data manifests from the
seed, recomputes aggregate metrics from per-run result files, and re-checks
each Claim B threshold. Output: evidence.json + evidence.md.
"""

from __future__ import annotations

import os

import numpy as np

from .data import build_splits
from .embedding import VARIANTS, build_embedding_matrix
from .properties import run_properties
from .util import read_json, sha256_array, write_json
from .vocab import Vocab

# Pre-registered Claim B thresholds (mirrored in run_demo and the README).
THRESHOLDS = {
    "hole_add_ratio_min": 2.0,     # kron_v2 hole-add exact >= 2x learned
    "in_add_margin": -0.02,        # kron_v2 in-add >= learned - 0.02 at every size
    "extra_add_exact_max": 0.10,   # magnitude extrapolation is (honestly) low
}


def _check(name: str, ok: bool, **details) -> dict:
    return {"name": name, "ok": bool(ok), **details}


def _mean(runs: list[dict], *path) -> float:
    vals = []
    for r in runs:
        d = r["eval"]
        for k in path:
            d = d[k]
        vals.append(float(d))
    return float(np.mean(vals))


def run_audit(artifacts_dir: str) -> dict:
    checks: list[dict] = []

    # -- 1. Claim A re-derived at a fresh coordinate ------------------------
    stored = read_json(os.path.join(artifacts_dir, "properties_report.json"))
    fresh = run_properties(stored["seed"], coord="audit", n_pairs=2_000,
                           n_words=300)
    checks.append(_check("claimA_reproduced_at_fresh_coordinate",
                         fresh["all_ok"] and stored["all_ok"],
                         producer_ok=stored["all_ok"], audit_ok=fresh["all_ok"]))

    # -- 2. Embedding matrices match their recorded hashes ------------------
    run_config = read_json(os.path.join(artifacts_dir, "run_config.json"))
    vocab = Vocab()
    recomputed = {v: sha256_array(build_embedding_matrix(vocab.tokens, variant=v))
                  for v in VARIANTS}
    checks.append(_check("embedding_matrix_hashes_match",
                         recomputed == run_config["embedding_hashes"],
                         recomputed=recomputed))
    checks.append(_check("vocab_hash_matches",
                         vocab.hash == run_config["vocab_hash"]))

    # -- 3. Data manifests rebuild identically ------------------------------
    results = read_json(os.path.join(artifacts_dir, "results.json"))
    sizes = sorted({int(k.split("@")[1]) for k in results["by_group"]})
    manifest_ok = []
    for size in sizes:
        rebuilt = build_splits(run_config["base_seed"], size)["manifest"]
        stored_m = read_json(os.path.join(artifacts_dir, "manifests",
                                          f"data_{size}.json"))
        manifest_ok.append(rebuilt == stored_m
                           and rebuilt["disjoint_train_eval"]
                           and rebuilt["train_never_touches_hole"]
                           and rebuilt["hole_always_touched"])
    checks.append(_check("data_manifests_rebuild_identically",
                         all(manifest_ok), sizes=sizes))

    # -- 4. Aggregates recomputed from per-run files ------------------------
    # Every metric in METRIC_PATHS (means AND stds AND per-seed values), for
    # every group — a single miswired path in experiments.py must fail here.
    from .experiments import METRIC_PATHS, _dig

    run_files = {}
    for spec in results["run_index"]:
        r = read_json(os.path.join(artifacts_dir, spec["dir"], "result.json"))
        run_files.setdefault(f"{r['arm']}@{r['train_size']}", []).append(r)
    mismatches = []
    n_compared = 0
    for key, entry in results["by_group"].items():
        runs = run_files.get(key, [])
        if not runs:
            mismatches.append(f"{key}:no_run_files")
            continue
        for mname, path in METRIC_PATHS.items():
            vals = [float(_dig(r["eval"], path)) for r in runs]
            n_compared += 1
            if (abs(float(np.mean(vals)) - entry[mname]["mean"]) > 1e-9
                    or abs(float(np.std(vals)) - entry[mname]["std"]) > 1e-9
                    or vals != entry[mname]["values"]):
                mismatches.append(f"{key}:{mname}")
    agg_ok = not mismatches and n_compared > 0
    checks.append(_check("aggregates_match_per_run_files", agg_ok,
                         metrics_compared=n_compared, mismatches=mismatches))

    # -- 5. Architecture identical across every run -------------------------
    arch_hashes = {r["arch_hash"] for runs in run_files.values() for r in runs}
    checks.append(_check("architecture_identical_across_arms",
                         len(arch_hashes) == 1, n_hashes=len(arch_hashes)))
    frozen_runs = [r for runs in run_files.values() for r in runs
                   if r["arm"] in VARIANTS]
    frozen_ok = bool(frozen_runs) and all(
        r["emb_hash"] == recomputed[r["arm"]] for r in frozen_runs)
    checks.append(_check("frozen_embeddings_match_recomputed_hashes",
                         frozen_ok, n_frozen_runs=len(frozen_runs)))

    # -- 6. Claim B thresholds re-derived from run files --------------------
    main_size = max(sizes)

    def group(arm, size):
        return run_files.get(f"{arm}@{size}", [])

    kron_hole = _mean(group("kron_v2", main_size),
                      "eval_hole", "add", "primary", "exact")
    learned_hole = _mean(group("learned", main_size),
                         "eval_hole", "add", "primary", "exact")
    ratio = kron_hole / max(learned_hole, 1e-9)
    checks.append(_check("claimB_hole_generalization",
                         ratio >= THRESHOLDS["hole_add_ratio_min"],
                         kron_v2=kron_hole, learned=learned_hole,
                         ratio=round(ratio, 2),
                         threshold=THRESHOLDS["hole_add_ratio_min"]))

    in_ok, in_detail = True, {}
    for size in sizes:
        if not group("kron_v2", size) or not group("learned", size):
            continue
        k = _mean(group("kron_v2", size), "eval_in", "add", "primary", "exact")
        l = _mean(group("learned", size), "eval_in", "add", "primary", "exact")
        in_detail[str(size)] = {"kron_v2": k, "learned": l}
        if k < l + THRESHOLDS["in_add_margin"]:
            in_ok = False
    # a check that compared nothing must not pass vacuously
    in_ok = in_ok and len(in_detail) > 0
    checks.append(_check("claimB_in_range_at_every_size", in_ok,
                         sizes_compared=len(in_detail), **in_detail))

    extra_vals = {arm: _mean(group(arm, main_size),
                             "eval_extra", "add", "primary", "exact")
                  for arm in sorted({r["arm"] for runs in run_files.values()
                                     for r in runs})
                  if group(arm, main_size)}
    checks.append(_check("claimB_extrapolation_negative_reported",
                         all(v <= THRESHOLDS["extra_add_exact_max"]
                             for v in extra_vals.values()),
                         per_arm=extra_vals))

    probes = results.get("probes", {})
    kron_probes = [p for key, p in sorted(probes.items())
                   if key.startswith("kron_v2")]
    learned_probes = [p for key, p in sorted(probes.items())
                      if key.startswith("learned")]
    if kron_probes and learned_probes:
        # aggregate if several probed seeds exist; a missing arm is a FAIL
        # with an explanation, never a crash or a vacuous pass
        kron_in = float(np.mean([p["input"]["add_lin"]["relerr_median"]
                                 for p in kron_probes]))
        kron_hid = float(np.mean([p["hidden"]["add_lin"]["relerr_median"]
                                  for p in kron_probes]))
        learned_in = float(np.mean([p["input"]["add_lin"]["relerr_median"]
                                    for p in learned_probes]))
        checks.append(_check("claimB_probe_localizes_failure_to_trunk",
                             kron_in < kron_hid and kron_in < learned_in,
                             kron_input=kron_in, kron_hidden=kron_hid,
                             learned_input=learned_in,
                             n_probes=len(kron_probes) + len(learned_probes)))
    else:
        checks.append(_check("claimB_probe_localizes_failure_to_trunk", False,
                             reason="probes missing for one or both arms",
                             kron=len(kron_probes), learned=len(learned_probes)))

    verdict = all(c["ok"] for c in checks)
    evidence = {"verdict": "PASS" if verdict else "FAIL",
                "thresholds": THRESHOLDS, "checks": checks}
    write_json(os.path.join(artifacts_dir, "evidence.json"), evidence)
    _write_md(os.path.join(artifacts_dir, "evidence.md"), evidence)
    return evidence


def _write_md(path: str, evidence: dict):
    lines = ["# Audit evidence — Kronecker Embedding V2", "",
             f"**Verdict: {evidence['verdict']}**", "",
             "Every check below was re-derived from the files in "
             "`submission_artifacts/` by `kronembed/audit.py`, which shares "
             "no state with the code that produced them.", ""]
    for c in evidence["checks"]:
        mark = "PASS" if c["ok"] else "FAIL"
        extras = {k: v for k, v in c.items() if k not in ("name", "ok")}
        lines.append(f"- [{mark}] `{c['name']}`"
                     + (f" — {extras}" if extras else ""))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
