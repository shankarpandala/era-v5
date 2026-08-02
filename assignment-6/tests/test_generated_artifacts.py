"""Assertions against the artifacts a real demonstration run produced.

These tests are skipped when ``submission_artifacts/`` is absent, so the unit
suite still runs on a clean checkout. Run ``python run_demo.py`` first to
exercise them. They check the *evidence itself* -- that the log contains the
required events, that the bundle is internally consistent, and that the numbers
in the evidence can be recomputed from the ledgers.
"""

from __future__ import annotations

import json
import os

import pytest

from datasys import perf as perf_mod
from datasys.ledger import Ledger, verify_chain
from datasys.util import read_json

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "submission_artifacts")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ART, "evidence.json")),
    reason="run `python run_demo.py` first to generate submission_artifacts/",
)


@pytest.fixture(scope="module")
def evidence():
    return read_json(os.path.join(ART, "evidence.json"))


@pytest.fixture(scope="module")
def run_log():
    with open(os.path.join(ART, "run.log"), encoding="utf-8") as f:
        return f.read()


def test_required_directory_structure_exists():
    for rel in ["run.log", "evidence.json", "evidence.md", "performance.json",
                "manifests", "ledgers", "checkpoints"]:
        assert os.path.exists(os.path.join(ART, rel)), rel


def test_run_log_contains_the_required_event_sequence(run_log):
    # Underscored [PASS]/EVENT markers used by the orchestrator…
    for marker in [
        "shards_created", "manifests_validated", "eval_shard_blocked",
        "mixture_compiled", "batches_packed", "opus_decisions_recorded",
        "checkpoint_saved", "crash_simulated", "run_resumed",
        "replay_hash_matched", "branch_forked", "audit_completed",
        "performance_measured",
    ]:
        assert marker in run_log, marker
    # …and the human-readable phrases required by the assignment brief.
    for phrase in [
        "shards created", "manifests validated", "evaluation data blocked",
        "mixture compiled", "batches packed", "OPUS decisions recorded",
        "crash simulated", "run resumed", "historical stream replayed",
        "branch forked", "audit completed", "performance measured",
    ]:
        assert phrase in run_log, phrase


def test_run_log_records_the_headline_passes(run_log):
    for marker in [
        "[PASS] tokenizer_hash_verified",
        "[PASS] eval_shard_blocked",
        "[PASS] checkpoint_saved",
        "[PASS] resume_next_batch_matched",
        "[PASS] replay_hash_matched",
        "[PASS] demonstration_complete",
    ]:
        assert marker in run_log, marker


def test_every_evidence_check_passed(evidence):
    failed = [k for k, v in evidence["checks"].items() if v["result"] != "PASS"]
    assert failed == []
    assert evidence["all_passed"] is True
    assert evidence["n_passed"] == evidence["n_checks"]


def test_evidence_points_at_files_that_exist(evidence):
    for key, check in evidence["checks"].items():
        assert check["evidence"], key
        for rel in check["evidence"]:
            assert os.path.exists(os.path.join(ART, rel)), (key, rel)


def test_evidence_md_mirrors_evidence_json(evidence):
    with open(os.path.join(ART, "evidence.md"), encoding="utf-8") as f:
        md = f.read()
    assert "| Requirement | Result | Evidence |" in md
    assert "FAIL" not in md
    n_rows = sum(1 for line in md.splitlines()
                 if line.startswith("|") and "PASS" in line)
    assert n_rows == evidence["n_checks"]


def test_all_ledger_chains_verify():
    ledgers = os.path.join(ART, "ledgers")
    files = [f for f in os.listdir(ledgers) if f.endswith(".jsonl")]
    assert files
    for f in files:
        result = verify_chain(os.path.join(ledgers, f))
        assert result["ok"], (f, result["error"])


def test_consumption_ledger_has_no_gaps_or_duplicates():
    run_id = read_json(os.path.join(ART, "evidence.json"))["run_id"]
    path = os.path.join(ART, "ledgers", f"consumption_{run_id}.jsonl")
    steps = [e["payload"]["step"] for e in Ledger(path).entries()]
    total = read_json(os.path.join(ART, "run_config.json"))["total_steps"]
    assert steps == list(range(total))


def test_resume_rewrote_the_crash_window_identically():
    """The pre-crash ledger copy and the final ledger must agree everywhere
    they overlap -- that is what 'no skipped or repeated batches' means."""
    run_id = read_json(os.path.join(ART, "evidence.json"))["run_id"]
    ledgers = os.path.join(ART, "ledgers")
    pre = os.path.join(ledgers, f"consumption_{run_id}.precrash.jsonl")
    post = os.path.join(ledgers, f"consumption_{run_id}.jsonl")
    pre_by_step = {e["payload"]["step"]: e["payload"]["batch_id"]
                   for e in Ledger(pre).entries()}
    post_by_step = {e["payload"]["step"]: e["payload"]["batch_id"]
                    for e in Ledger(post).entries()}
    assert pre_by_step
    for step, batch_id in pre_by_step.items():
        assert post_by_step[step] == batch_id, step


def test_no_blocked_content_appears_in_the_consumption_ledger():
    run_id = read_json(os.path.join(ART, "evidence.json"))["run_id"]
    fw = read_json(os.path.join(ART, "manifests", "firewall.json"))
    blocked_docs = {e["doc_id"] for e in fw["entries"]}
    blocked_hashes = {e["content_hash"] for e in fw["entries"]}
    # resolve every consumed doc_id to its text content hash via manifests
    doc_hash = {}
    man_dir = os.path.join(ART, "manifests")
    for name in os.listdir(man_dir):
        if name.endswith(".manifest.json") and "__" in name:
            m = read_json(os.path.join(man_dir, name))
            for d in m.get("documents", []):
                doc_hash[d["doc_id"]] = d["text_hash"]
    path = os.path.join(ART, "ledgers", f"consumption_{run_id}.jsonl")
    for e in Ledger(path).entries():
        for s in e["payload"]["samples"]:
            for seg in s["segments"]:
                assert seg["doc_id"] not in blocked_docs
                assert doc_hash.get(seg["doc_id"]) not in blocked_hashes
                assert "__eval" not in seg["shard_id"]
                assert "__validation" not in seg["shard_id"]


def test_performance_numbers_recompute_from_the_ledger():
    perf = read_json(os.path.join(ART, "performance.json"))
    run_id = read_json(os.path.join(ART, "evidence.json"))["run_id"]
    path = os.path.join(ART, "ledgers", f"consumption_{run_id}.jsonl")
    real = loss = total = 0
    for e in Ledger(path).entries():
        p = e["payload"]
        real += p["n_real_tokens"]
        loss += p["n_loss_tokens"]
        total += p["n_total_tokens"]
    assert perf["counters"]["real_tokens"] == real
    assert perf["counters"]["loss_tokens"] == loss
    assert perf["counters"]["total_slot_tokens"] == total
    derived = perf_mod.derive(perf["counters"])
    for key, value in derived.items():
        assert abs(perf["derived"][key] - value) < 1e-9, key


def test_learning_ledger_links_every_step_to_its_batch():
    run_id = read_json(os.path.join(ART, "evidence.json"))["run_id"]
    ledgers = os.path.join(ART, "ledgers")
    cons = {e["payload"]["step"]: e["payload"]
            for e in Ledger(os.path.join(ledgers, f"consumption_{run_id}.jsonl")).entries()}
    learn = list(Ledger(os.path.join(ledgers, f"learning_{run_id}.jsonl")).entries())
    assert len(learn) == len(cons)
    for e in learn:
        p = e["payload"]
        ce = cons[p["step"]]
        assert ce["batch_id"] == p["batch_id"]
        assert p["n_loss_tokens"] > 0
        assert sum(v["tokens"] for v in p["per_lane_loss"].values()) == p["n_loss_tokens"]
        # sample-level loss tracking: every sample hash ties back to consumption
        samples = p.get("per_sample_loss") or []
        assert samples, "missing per_sample_loss"
        cons_hashes = {s["sample_hash"] for s in ce["samples"]}
        assert sum(s["tokens"] for s in samples) == p["n_loss_tokens"]
        for s in samples:
            assert s["sample_hash"] in cons_hashes
            assert s["tokens"] >= 0
            assert "source_docs" in s


def test_audit_reverified_crash_and_replay_independently(evidence):
    """The audit must re-derive crash/replay proofs from disk, not just echo
    the orchestrator's in-memory dicts."""
    cr = evidence["checks"]["crash_recovery"]["detail"]
    assert "independent_from_disk" in cr
    ind = cr["independent_from_disk"]
    assert ind["next_batch_matched"] is True
    assert ind["rewritten_batches_identical"] is True
    assert ind["learning_state_matched"] is True
    assert ind["checkpoint_offset_matches_step"] is True
    # a resume is only trustworthy if the blob it restores from is intact
    assert ind["checkpoint_blobs_verified"] is True
    assert ind["checkpoint_blob_errors"] == []

    rp = evidence["checks"]["replay"]["detail"]
    assert rp["all_match"] is True
    assert rp.get("report_agrees_with_recompute") is True

    pk = evidence["checks"]["packing_correctness"]["detail"]
    assert "independent_repack" in pk
    assert pk["independent_repack"]["mismatches"] == []


def test_checkpoints_are_tied_to_ledger_offsets():
    ckpt_dir = os.path.join(ART, "checkpoints")
    run_id = read_json(os.path.join(ART, "evidence.json"))["run_id"]
    cons_path = os.path.join(ART, "ledgers", f"consumption_{run_id}.jsonl")
    entries = list(Ledger(cons_path).entries())
    by_seq = {e["seq"]: e["hash"] for e in entries}
    mans = [read_json(os.path.join(ckpt_dir, f)) for f in os.listdir(ckpt_dir)
            if f.startswith(run_id) and f.endswith(".manifest.json")]
    assert mans
    for m in mans:
        count = m["consumption_offset"]["count"]
        assert count == m["step"]
        # the recorded chain head must be the hash of the last committed entry
        assert m["consumption_offset"]["head"] == by_seq[count - 1]


def test_every_checkpoint_blob_still_hashes_to_its_manifest():
    """Independent of the audit: re-hash each committed blob off disk."""
    from datasys.checkpoint import verify_checkpoint

    ckpt_dir = os.path.join(ART, "checkpoints")
    tags = [f[: -len(".manifest.json")] for f in os.listdir(ckpt_dir)
            if f.endswith(".manifest.json")]
    assert tags
    for tag in tags:
        assert verify_checkpoint(ckpt_dir, tag)["ok"], tag


def test_run_log_records_no_failed_check():
    """The execution log is a deliverable: a [FAIL] anywhere in it means the
    demonstration did not hold, even if the audit summary is green."""
    with open(os.path.join(ART, "run.log"), encoding="utf-8") as f:
        failed = [l.strip() for l in f if "[FAIL]" in l]
    assert failed == [], failed


def test_fork_lineage_records_its_parent():
    ev = read_json(os.path.join(ART, "evidence.json"))
    fork_run = ev["fork_run_id"]
    lineage = read_json(os.path.join(ART, "checkpoints", f"{fork_run}.lineage.json"))
    assert lineage["parent_run_id"] == ev["run_id"]
    assert lineage["parent_checkpoint"]
    assert lineage["parent_model_tensor_hash"]
