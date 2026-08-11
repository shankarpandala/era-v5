#!/usr/bin/env python3
"""One command that runs the complete Assignment-7 demonstration.

    python run_demo.py            # full matrix (~15-20 min on a laptop CPU)
    python run_demo.py --fast     # reduced matrix (~3 min), same pipeline

Pipeline, writing ``submission_artifacts/``:

    embedding hashes -> Claim A properties (zero training) -> data manifests
      -> determinism proof -> experiment matrix -> Claim B threshold checks
      -> figures + self-contained report.html -> independent audit

Every ``[PASS]`` line is a derived result: the value it reports is computed
from artifacts, and the independent auditor re-derives the same facts from
disk afterwards. A ``[FAIL]`` anywhere makes the process exit non-zero.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from kronembed.audit import THRESHOLDS, run_audit  # noqa: E402
from kronembed.data import build_splits  # noqa: E402
from kronembed.embedding import (VARIANTS, build_embedding_matrix,  # noqa: E402
                                 decode_value, embed_token)
from kronembed.experiments import FAST_PLAN, FULL_PLAN, run_matrix  # noqa: E402
from kronembed.layout import LAYOUT  # noqa: E402
from kronembed.plots import make_all_plots, make_report  # noqa: E402
from kronembed.properties import run_properties  # noqa: E402
from kronembed.train import DEFAULT_CFG, run_one  # noqa: E402
from kronembed.util import (ensure_dir, sha256_array, write_json)  # noqa: E402
from kronembed.vocab import Vocab  # noqa: E402

ARTIFACTS = os.path.join(HERE, "submission_artifacts")


class RunLog:
    """Human-readable execution log, mirrored to stdout (assignment-6 style)."""

    def __init__(self, path: str):
        ensure_dir(os.path.dirname(path))
        self.f = open(path, "w", encoding="utf-8")
        self.t0 = time.time()
        self.failures: List[str] = []

    def _w(self, line: str):
        out = f"[{time.time() - self.t0:8.3f}s] {line}"
        self.f.write(out + "\n")
        self.f.flush()
        print(out, flush=True)

    def section(self, title: str):
        self._w("")
        self._w("=" * 72)
        self._w(f"== {title}")
        self._w("=" * 72)

    def info(self, msg: str):
        self._w(f"       {msg}")

    def check(self, name: str, ok: bool, **kw):
        if not ok:
            self.failures.append(name)
        extra = " ".join(f"{k}={_fmt(v)}" for k, v in kw.items())
        self._w(f"{'[PASS]' if ok else '[FAIL]'} {name}"
                + (f"  {extra}" if extra else ""))

    def close(self):
        self.f.close()


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True)
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="reduced matrix (~3 min) instead of the full one")
    args = ap.parse_args()
    plan = FAST_PLAN if args.fast else FULL_PLAN

    log = RunLog(os.path.join(ARTIFACTS, "run.log"))
    log.section("Kronecker Embedding V2 — math-structure embeddings")
    log.info(f"mode={'fast' if args.fast else 'full'} plan={_fmt(plan)}")

    # -- 0. identity: layout, vocab, embedding matrices ---------------------
    vocab = Vocab()
    emb_hashes = {v: sha256_array(build_embedding_matrix(vocab.tokens, variant=v))
                  for v in VARIANTS}
    run_config = {
        "base_seed": DEFAULT_CFG["base_seed"],
        "layout": LAYOUT.describe(),
        "vocab_size": len(vocab),
        "vocab_hash": vocab.hash,
        "embedding_hashes": emb_hashes,
        "train_defaults": DEFAULT_CFG,
        "plan": plan,
        "thresholds": THRESHOLDS,
    }
    write_json(os.path.join(ARTIFACTS, "run_config.json"), run_config)
    log.info(f"vocab={len(vocab)} tokens hash={vocab.hash[:12]}")
    for v, h in emb_hashes.items():
        log.info(f"embedding[{v}] sha256={h[:12]}")

    # -- 1. Claim A: algebra without training -------------------------------
    log.section("Claim A: the embedding is an algebra (zero training)")
    report = run_properties(DEFAULT_CFG["base_seed"], coord="properties",
                            n_pairs=10_000, n_words=2_000)
    write_json(os.path.join(ARTIFACTS, "properties_report.json"), report)
    for c in report["checks"]:
        detail = {k: v for k, v in c.items()
                  if k not in ("name", "ok") and not isinstance(v, list)}
        log.check(f"claimA/{c['name']}", c["ok"], **detail)
    s = embed_token("9") + embed_token("9")
    log.info(f"showcase: decode_value(emb('9')+emb('9')) = {decode_value(s)}")

    # -- 2. data manifests ---------------------------------------------------
    log.section("Data: deterministic splits with a structural operand hole")
    sizes = sorted({plan["main_size"], *plan["curve_sizes"]})
    for size in sizes:
        m = build_splits(DEFAULT_CFG["base_seed"], size)["manifest"]
        write_json(os.path.join(ARTIFACTS, "manifests", f"data_{size}.json"), m)
        log.check(f"data/size_{size}_structural_invariants",
                  m["disjoint_train_eval"] and m["train_never_touches_hole"]
                  and m["hole_always_touched"],
                  counts=m["counts"])

    # -- 3. determinism proof ------------------------------------------------
    log.section("Determinism: identical (arm, size, seed) => identical bits")
    det_cfg = {"steps": 40, "batch_size": 32, "loss_log_every": 20}
    r1 = run_one("kron_v2", 500, 0, "", cfg=det_cfg)
    r2 = run_one("kron_v2", 500, 0, "", cfg=det_cfg)
    log.check("determinism/param_hash_bit_identical",
              r1["param_hash_final"] == r2["param_hash_final"],
              hash=r1["param_hash_final"][:12])

    # -- 4. experiment matrix ------------------------------------------------
    log.section("Claim B: the experiment matrix")
    results = run_matrix(plan, ARTIFACTS, progress=log.info)
    log.check("matrix/architecture_identical_across_arms",
              results["arch_identical_across_arms"],
              n_runs=results["n_runs"])

    # -- 5. Claim B threshold checks ----------------------------------------
    g = results["by_group"]
    ms = plan["main_size"]

    def m(arm, metric):
        return g[f"{arm}@{ms}"][metric]["mean"]

    ratio = m("kron_v2", "hole_add_exact") / max(m("learned", "hole_add_exact"), 1e-9)
    log.check("claimB/hole_generalization_kron_vs_learned",
              ratio >= THRESHOLDS["hole_add_ratio_min"],
              kron_v2=m("kron_v2", "hole_add_exact"),
              learned=m("learned", "hole_add_exact"), ratio=round(ratio, 1))
    in_pairs = [sz for sz in sizes
                if f"kron_v2@{sz}" in g and f"learned@{sz}" in g]
    in_ok = bool(in_pairs) and all(
        g[f"kron_v2@{sz}"]["in_add_exact"]["mean"]
        >= g[f"learned@{sz}"]["in_add_exact"]["mean"] + THRESHOLDS["in_add_margin"]
        for sz in in_pairs)
    log.check("claimB/in_range_at_every_train_size", in_ok,
              sizes_compared=in_pairs)
    extra_low = all(g[k]["extra_add_exact"]["mean"]
                    <= THRESHOLDS["extra_add_exact_max"] for k in g
                    if k.endswith(f"@{ms}"))
    log.check("claimB/magnitude_extrapolation_reported_negative", extra_low)

    # -- 6. figures + report -------------------------------------------------
    log.section("Figures and report")
    plots = make_all_plots(os.path.join(ARTIFACTS, "plots"), results, ms)
    make_report(os.path.join(ARTIFACTS, "report.html"), plots, results,
                report, ms)
    log.info(f"{len(plots)} figures + report.html written")

    # -- 7. independent audit ------------------------------------------------
    log.section("Independent audit (re-derives everything from disk)")
    evidence = run_audit(ARTIFACTS)
    for c in evidence["checks"]:
        log.check(f"audit/{c['name']}", c["ok"])
    log.check("audit/verdict", evidence["verdict"] == "PASS")

    log.section("Done")
    ok = not log.failures
    log.info("ALL CHECKS PASSED" if ok else f"FAILURES: {log.failures}")
    log.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
