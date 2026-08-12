"""The audit re-derives a miniature artifact bundle from disk.

A tiny (deliberately undertrained) matrix is produced in tmp_path; the audit's
STRUCTURAL checks must all pass on it. Claim-threshold checks are allowed to
fail at 30 steps of training — the test asserts they exist and carry details,
not that an undertrained model wins them.
"""

import os

from kronembed.audit import run_audit
from kronembed.embedding import (VARIANTS, build_embedding_matrix,
                                 build_random_matrix)
from kronembed.experiments import run_matrix
from kronembed.data import build_splits
from kronembed.properties import run_properties
from kronembed.train import DEFAULT_CFG
from kronembed.util import read_json, sha256_array, write_json
from kronembed.vocab import Vocab

TINY_CFG = {"steps": 30, "batch_size": 16, "loss_log_every": 15,
            "eval_chunk": 512}
TINY_PLAN = {"arms": ["kron_v2", "learned"], "sizes": [300],
             "primary_size": 300, "seeds": [0],
             "curve_arms": [], "curve_sizes": [],
             "probe_arms": ["kron_v2", "learned"],
             "nl_arms": ["kron_v2", "learned"], "nl_sizes": [300],
             "nl_seeds": [0]}

STRUCTURAL = [
    "claimA_reproduced_at_fresh_coordinate",
    "embedding_matrix_hashes_match",
    "vocab_hash_matches",
    "data_manifests_rebuild_identically",
    "aggregates_match_per_run_files",
    "architecture_identical_across_arms",
    "frozen_embeddings_match_recomputed_hashes",
]


def _build_bundle(root: str):
    vocab = Vocab()
    emb_hashes = {v: sha256_array(build_embedding_matrix(vocab.tokens, variant=v))
                  for v in VARIANTS}
    emb_hashes["frozen_rand"] = sha256_array(build_random_matrix(vocab.tokens))
    write_json(os.path.join(root, "run_config.json"), {
        "base_seed": DEFAULT_CFG["base_seed"],
        "vocab_hash": vocab.hash,
        "embedding_hashes": emb_hashes,
    })
    report = run_properties(DEFAULT_CFG["base_seed"], coord="properties",
                            n_pairs=300, n_words=50)
    write_json(os.path.join(root, "properties_report.json"), report)
    m = build_splits(DEFAULT_CFG["base_seed"],
                     TINY_PLAN["primary_size"])["manifest"]
    write_json(os.path.join(root, "manifests",
                            f"data_{TINY_PLAN['primary_size']}.json"), m)
    run_matrix(TINY_PLAN, root, base_cfg=TINY_CFG)


def test_audit_structural_checks_pass_on_fresh_bundle(tmp_path):
    root = str(tmp_path)
    _build_bundle(root)
    evidence = run_audit(root)
    by_name = {c["name"]: c for c in evidence["checks"]}
    for name in STRUCTURAL:
        assert by_name[name]["ok"], (name, by_name[name])
    # claim checks exist (their outcome is training-dependent at 30 steps)
    assert "claimB_hole_generalization" in by_name
    assert "claimB_probe_localizes_failure_to_trunk" in by_name
    assert os.path.exists(os.path.join(root, "evidence.json"))
    assert os.path.exists(os.path.join(root, "evidence.md"))


def test_audit_detects_tampered_result(tmp_path):
    root = str(tmp_path)
    _build_bundle(root)
    # tamper with one per-run metric: aggregates must stop matching
    results = read_json(os.path.join(root, "results.json"))
    spec = results["run_index"][0]
    path = os.path.join(root, spec["dir"], "result.json")
    r = read_json(path)
    r["eval"]["eval_in"]["add"]["primary"]["exact"] += 0.123
    write_json(path, r)
    evidence = run_audit(root)
    by_name = {c["name"]: c for c in evidence["checks"]}
    assert not by_name["aggregates_match_per_run_files"]["ok"]
    assert evidence["verdict"] == "FAIL"
