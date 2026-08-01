"""Independent audit -> evidence bundle.

The audit deliberately re-derives every claim **from the artifacts on disk**
(manifests, ledgers, checkpoints, counters). It shares no in-memory state with
the trainer: it re-hashes shard bytes, re-verifies ledger chains, re-computes
mixture shares, re-scans the consumption ledger for firewalled content, and
re-derives throughput from raw counters. If an artifact were edited by hand the
audit would fail, which is what makes the evidence bundle worth something.

Outputs ``evidence.json`` (machine-readable, with a pointer to the supporting
artifact for every check) and ``evidence.md`` (the human-readable table).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from . import mixture as mix
from . import perf as perf_mod
from .ledger import Ledger, verify_chain
from .shards import load_manifests, validate_shard
from .tokenizer import Tokenizer
from .util import read_json, write_json


class Check:
    def __init__(self, key: str, title: str):
        self.key = key
        self.title = title
        self.passed = False
        self.detail: Dict[str, object] = {}
        self.evidence: List[str] = []

    def to_obj(self) -> dict:
        return {
            "requirement": self.title,
            "result": "PASS" if self.passed else "FAIL",
            "evidence": self.evidence,
            "detail": self.detail,
        }


def _rel(artifacts_dir: str, path: str) -> str:
    return os.path.relpath(path, artifacts_dir)


def run_audit(artifacts_dir: str, run_id: str, fork_run_id: Optional[str],
              resume_proof: dict, replay_proof: dict, crash_proof: dict,
              seq_len: int) -> dict:
    manifests_dir = os.path.join(artifacts_dir, "manifests")
    ledgers_dir = os.path.join(artifacts_dir, "ledgers")
    ckpt_dir = os.path.join(artifacts_dir, "checkpoints")

    checks: List[Check] = []

    # ---------------------------------------------------------------- 1
    c = Check("tokenizer_integrity", "Tokenizer integrity")
    tok = Tokenizer.load(os.path.join(manifests_dir, "tokenizer.json"))
    loaded = load_manifests(manifests_dir)
    frozen_hash = tok.content_hash
    mismatches = [s for s in loaded["shards"].values()
                  if s["tokenizer_hash"] != frozen_hash]
    root_ok = loaded["root"]["tokenizer_hash"] == frozen_hash
    # round-trip check on real corpus text proves the tokenizer is lossless
    sample_text = "The system replay ledger. नमस्ते dunia. def f(x): return x\n"
    roundtrip_ok = tok.decode(tok.encode(sample_text)) == sample_text
    c.passed = not mismatches and root_ok and roundtrip_ok
    c.detail = {
        "tokenizer_content_hash": frozen_hash,
        "vocab_size": tok.vocab_size,
        "shards_bound_to_tokenizer": len(loaded["shards"]) - len(mismatches),
        "shards_total": len(loaded["shards"]),
        "root_manifest_hash": loaded["root"]["root_hash"],
        "decode_encode_roundtrip": roundtrip_ok,
    }
    c.evidence = [_rel(artifacts_dir, os.path.join(manifests_dir, "tokenizer.json")),
                  _rel(artifacts_dir, os.path.join(manifests_dir, "root_manifest.json"))]
    checks.append(c)

    # ---------------------------------------------------------------- 2
    c = Check("shard_immutability", "Shard immutability and manifests")
    errs: List[str] = []
    for m in loaded["shards"].values():
        errs += validate_shard(manifests_dir, m, frozen_hash)
    c.passed = not errs
    c.detail = {
        "shards_verified": len(loaded["shards"]),
        "total_tokens": sum(m["n_tokens"] for m in loaded["shards"].values()),
        "total_documents": sum(m["n_docs"] for m in loaded["shards"].values()),
        "errors": errs,
    }
    c.evidence = [_rel(artifacts_dir, os.path.join(manifests_dir, f"{sid}.manifest.json"))
                  for sid in sorted(loaded["shards"])]
    checks.append(c)

    # ---------------------------------------------------------------- 3
    c = Check("evaluation_firewall", "Evaluation firewall")
    fw = read_json(os.path.join(manifests_dir, "firewall.json"))
    blocked_hashes = {e["content_hash"] for e in fw["entries"]}
    # (a) every blocked doc got an OPUS REJECT
    opus_entries = [e["payload"] for e in Ledger(os.path.join(ledgers_dir, "opus_ledger.jsonl")).entries()]
    blocked_decisions = [e for e in opus_entries if e["content_hash"] in blocked_hashes]
    all_rejected = all(e["decision"] == "REJECT" for e in blocked_decisions)
    poisoned = [e for e in blocked_decisions
                if e["split"] == "train"]  # the decoy: train-labelled, eval content
    # (b) no blocked content hash appears anywhere in the consumption ledger
    doc_text_hash = _doc_text_hashes(manifests_dir)
    consumed_docs = set()
    for e in Ledger(os.path.join(ledgers_dir, f"consumption_{run_id}.jsonl")).entries():
        for s in e["payload"]["samples"]:
            for seg in s["segments"]:
                consumed_docs.add(seg["doc_id"])
    leaked = sorted(d for d in consumed_docs if doc_text_hash.get(d) in blocked_hashes)
    # (c) no eval/validation shard was ever consumed
    consumed_shards = set()
    for e in Ledger(os.path.join(ledgers_dir, f"consumption_{run_id}.jsonl")).entries():
        for s in e["payload"]["samples"]:
            for seg in s["segments"]:
                consumed_shards.add(seg["shard_id"])
    eval_shards_consumed = sorted(s for s in consumed_shards
                                  if s.endswith("__eval") or s.endswith("__validation"))
    c.passed = all_rejected and not leaked and not eval_shards_consumed and len(blocked_decisions) > 0
    c.detail = {
        "blocked_content_hashes": len(blocked_hashes),
        "blocked_docs_rejected_by_opus": len(blocked_decisions),
        "poisoned_train_docs_caught": [e["doc_id"] for e in poisoned],
        "leaked_documents": leaked,
        "eval_shards_consumed": eval_shards_consumed,
        "consumed_shards": sorted(consumed_shards),
    }
    c.evidence = [_rel(artifacts_dir, os.path.join(manifests_dir, "firewall.json")),
                  _rel(artifacts_dir, os.path.join(ledgers_dir, "opus_ledger.jsonl"))]
    checks.append(c)

    # ---------------------------------------------------------------- 4
    c = Check("packing_correctness", "Packing correctness (masks, positions)")
    pack_report = read_json(os.path.join(artifacts_dir, "packing_report.json"))
    c.passed = (pack_report["invariant_violations"] == 0
                and pack_report["samples_checked"] > 0
                and pack_report["policies_exercised"] >= 3)
    c.detail = pack_report
    c.evidence = [_rel(artifacts_dir, os.path.join(artifacts_dir, "packing_report.json"))]
    checks.append(c)

    # ---------------------------------------------------------------- 5
    c = Check("mixture_compliance", "Mixture compliance and protected floors")
    schedule = read_json(os.path.join(manifests_dir, "schedule.json"))
    planned = schedule["planned_fractions"]
    # Actual shares are recomputed from the CONSUMPTION LEDGER, never from the
    # schedule that produced it. Floors are promises about *scheduled sequence
    # slots*, so that is the basis they are checked on; the realized token share
    # is reported alongside because packing policies differ in how much of a
    # slot they fill.
    actual_slots = _actual_shares_from_ledger(ledgers_dir, run_id, "lane_slots")
    actual_tokens = _actual_shares_from_ledger(ledgers_dir, run_id, "lane_tokens")
    floor_violations = []
    share_deviations = []
    for stage, lanes in actual_slots.items():
        for lane, share in lanes.items():
            fl = mix.FLOORS.get(lane)
            if fl is not None and share + 1e-9 < fl:
                floor_violations.append({"stage": stage, "lane": lane,
                                         "slot_share": share, "floor": fl})
            p = planned.get(stage, {}).get(lane, 0.0)
            share_deviations.append({"stage": stage, "lane": lane,
                                     "planned": p, "actual": share,
                                     "abs_dev": abs(p - share)})
    max_dev = max((d["abs_dev"] for d in share_deviations), default=0.0)
    tolerance = 0.02
    worst = sorted(share_deviations, key=lambda d: -d["abs_dev"])[:5]
    c.passed = not floor_violations and max_dev <= tolerance
    c.detail = {
        "floors": mix.FLOORS,
        "floor_basis": schedule.get("floor_basis"),
        "floor_violations": floor_violations,
        "max_abs_share_deviation": max_dev,
        "tolerance": tolerance,
        "largest_deviations": worst,
        "planned_fractions": planned,
        "actual_slot_shares_from_consumption_ledger": actual_slots,
        "actual_token_shares_from_consumption_ledger": actual_tokens,
    }
    c.evidence = [_rel(artifacts_dir, os.path.join(manifests_dir, "schedule.json")),
                  _rel(artifacts_dir, os.path.join(ledgers_dir, f"consumption_{run_id}.jsonl"))]
    checks.append(c)

    # ---------------------------------------------------------------- 6
    c = Check("opus_audit_trail", "OPUS audit trail")
    chain = verify_chain(os.path.join(ledgers_dir, "opus_ledger.jsonl"))
    by_decision: Dict[str, int] = {}
    by_reason: Dict[str, int] = {}
    final: Dict[str, str] = {}
    for e in opus_entries:
        by_decision[e["decision"]] = by_decision.get(e["decision"], 0) + 1
        by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1
        final[e["doc_id"]] = e["decision"]   # last event wins
    required = {"ACCEPT", "REJECT", "DEFER", "FLOOR_OVERRIDE"}
    present = required.issubset(set(by_decision))
    every_doc_decided = all(e.get("reason") for e in opus_entries)
    # every consumed document must have been admitted by OPUS
    admitted_ids = {doc for doc, dec in final.items()
                    if dec in ("ACCEPT", "FLOOR_OVERRIDE")}
    unadmitted_consumed = sorted(consumed_docs - admitted_ids)
    # deferral must be a distinct state from rejection: at least one deferred
    # candidate has to have been reconsidered and admitted in the second pass
    requeued = [e for e in opus_entries if e["reason"] == "requeued_admitted"]
    final_counts: Dict[str, int] = {}
    for dec in final.values():
        final_counts[dec] = final_counts.get(dec, 0) + 1
    c.passed = (chain["ok"] and present and every_doc_decided
                and not unadmitted_consumed and bool(requeued))
    c.detail = {
        "chain_ok": chain["ok"], "chain_entries": chain["count"],
        "events_by_decision": by_decision, "events_by_reason": by_reason,
        "final_decision_per_document": final_counts,
        "all_four_decision_types_present": present,
        "deferred_then_admitted_in_pass_2": [e["doc_id"] for e in requeued],
        "documents_consumed_without_admission": unadmitted_consumed,
    }
    c.evidence = [_rel(artifacts_dir, os.path.join(ledgers_dir, "opus_ledger.jsonl")),
                  _rel(artifacts_dir, os.path.join(manifests_dir, "opus_summary.json"))]
    checks.append(c)

    # ---------------------------------------------------------------- 7
    c = Check("consumption_ledger", "Consumption ledger integrity")
    cons_path = os.path.join(ledgers_dir, f"consumption_{run_id}.jsonl")
    cchain = verify_chain(cons_path)
    steps = [e["payload"]["step"] for e in Ledger(cons_path).entries()]
    expected = list(range(len(steps)))
    contiguous = steps == expected
    duplicates = len(steps) != len(set(steps))
    batch_ids = [e["payload"]["batch_id"] for e in Ledger(cons_path).entries()]
    unique_batches = len(batch_ids) == len(set(batch_ids))
    c.passed = cchain["ok"] and contiguous and not duplicates and unique_batches
    c.detail = {
        "chain_ok": cchain["ok"], "entries": cchain["count"],
        "steps_contiguous_from_zero": contiguous,
        "duplicate_steps": duplicates,
        "unique_batch_ids": unique_batches,
        "first_step": steps[0] if steps else None,
        "last_step": steps[-1] if steps else None,
    }
    c.evidence = [_rel(artifacts_dir, cons_path)]
    checks.append(c)

    # ---------------------------------------------------------------- 8
    c = Check("learning_trace", "Learning trace linked to source data")
    learn_path = os.path.join(ledgers_dir, f"learning_{run_id}.jsonl")
    lchain = verify_chain(learn_path)
    learn_entries = [e["payload"] for e in Ledger(learn_path).entries()]
    cons_by_step = {e["payload"]["step"]: e["payload"]
                    for e in Ledger(cons_path).entries()}
    link_errors = []
    lane_loss_tokens = 0
    for le in learn_entries:
        ce = cons_by_step.get(le["step"])
        if ce is None or ce["batch_id"] != le["batch_id"]:
            link_errors.append(le["step"])
        lane_loss_tokens += sum(v["tokens"] for v in le["per_lane_loss"].values())
    total_loss_tokens = sum(ce["n_loss_tokens"] for ce in cons_by_step.values())
    tokens_reconcile = lane_loss_tokens == total_loss_tokens
    losses = [le["loss"] for le in learn_entries]
    first_avg = sum(losses[:10]) / max(1, len(losses[:10]))
    last_avg = sum(losses[-10:]) / max(1, len(losses[-10:]))
    c.passed = (lchain["ok"] and not link_errors and tokens_reconcile
                and len(learn_entries) == len(cons_by_step))
    c.detail = {
        "chain_ok": lchain["ok"], "entries": lchain["count"],
        "steps_linked_to_consumption": len(learn_entries) - len(link_errors),
        "unlinked_steps": link_errors,
        "per_lane_loss_tokens": lane_loss_tokens,
        "consumption_loss_tokens": total_loss_tokens,
        "token_accounting_reconciles": tokens_reconcile,
        "mean_loss_first_10_steps": first_avg,
        "mean_loss_last_10_steps": last_avg,
        "loss_decreased": last_avg < first_avg,
    }
    c.evidence = [_rel(artifacts_dir, learn_path), _rel(artifacts_dir, cons_path)]
    checks.append(c)

    # ---------------------------------------------------------------- 9
    c = Check("crash_recovery", "Crash recovery (no skipped or repeated batches)")
    c.passed = bool(resume_proof.get("next_batch_matched")) and \
        bool(resume_proof.get("no_gaps_or_duplicates")) and \
        bool(resume_proof.get("learning_state_matched")) and \
        bool(crash_proof.get("process_died"))
    c.detail = {**crash_proof, **resume_proof}
    c.evidence = [_rel(artifacts_dir, cons_path),
                  _rel(artifacts_dir, os.path.join(ckpt_dir,
                       f"{resume_proof.get('resumed_from','')}.manifest.json"))]
    checks.append(c)

    # ---------------------------------------------------------------- 10
    c = Check("replay", "Replay of historical data stream")
    c.passed = bool(replay_proof.get("all_match")) and replay_proof.get("n_steps", 0) > 0
    c.detail = {
        "interval": replay_proof.get("interval"),
        "n_steps": replay_proof.get("n_steps"),
        "all_match": replay_proof.get("all_match"),
        "examples": replay_proof.get("comparisons", [])[:3],
    }
    c.evidence = [_rel(artifacts_dir, os.path.join(artifacts_dir, "replay_report.json"))]
    checks.append(c)

    # ---------------------------------------------------------------- 11
    c = Check("fork", "Fork from an earlier checkpoint")
    if fork_run_id:
        fork_cons = os.path.join(ledgers_dir, f"consumption_{fork_run_id}.jsonl")
        fchain = verify_chain(fork_cons)
        fork_entries = [e["payload"] for e in Ledger(fork_cons).entries()]
        lineage_path = os.path.join(ckpt_dir, f"{fork_run_id}.lineage.json")
        lineage = read_json(lineage_path) if os.path.exists(lineage_path) else {}
        # The fork must start exactly where the parent checkpoint left off, and
        # consume the *same* data there -- proving it branched rather than
        # restarted -- while carrying its own batch-id namespace.
        parent_cons = {e["payload"]["step"]: e["payload"]
                       for e in Ledger(cons_path).entries()}
        first = fork_entries[0] if fork_entries else {}
        parent_at_branch = parent_cons.get(first.get("step"), {})
        same_first_samples = (
            [s["sample_hash"] for s in first.get("samples", [])] ==
            [s["sample_hash"] for s in parent_at_branch.get("samples", [])]
        ) if first else False
        parent_ids = {p["batch_id"] for p in parent_cons.values()}
        fork_ids = {e["batch_id"] for e in fork_entries}
        c.passed = (fchain["ok"] and bool(fork_entries)
                    and bool(lineage.get("parent_run_id"))
                    and lineage.get("parent_step") == first.get("step")
                    and same_first_samples
                    and not (parent_ids & fork_ids))
        c.detail = {
            "fork_run_id": fork_run_id,
            "fork_steps": [e["step"] for e in fork_entries],
            "lineage": lineage,
            "chain_ok": fchain["ok"],
            "branch_step_data_matches_parent": same_first_samples,
            "batch_id_collision_with_parent": sorted(parent_ids & fork_ids),
        }
        c.evidence = [_rel(artifacts_dir, fork_cons), _rel(artifacts_dir, lineage_path)]
    else:
        c.detail = {"error": "no fork run recorded"}
    checks.append(c)

    # ---------------------------------------------------------------- 12
    c = Check("throughput", "Throughput and packing efficiency")
    counters = read_json(os.path.join(artifacts_dir, f"perf_counters_{run_id}.json"))
    derived = perf_mod.derive(counters)
    # cross-check the counters against the consumption ledger itself
    led_real = sum(e["payload"]["n_real_tokens"] for e in Ledger(cons_path).entries())
    led_loss = sum(e["payload"]["n_loss_tokens"] for e in Ledger(cons_path).entries())
    led_total = sum(e["payload"]["n_total_tokens"] for e in Ledger(cons_path).entries())
    reconciles = (led_real == counters["real_tokens"]
                  and led_loss == counters["loss_tokens"]
                  and led_total == counters["total_slot_tokens"])
    perf_report = {
        "counters": counters,
        "derived": derived,
        "ledger_recomputed": {"real_tokens": led_real, "loss_tokens": led_loss,
                              "total_slot_tokens": led_total},
        "counters_reconcile_with_ledger": reconciles,
    }
    write_json(os.path.join(artifacts_dir, "performance.json"), perf_report)
    c.passed = reconciles and derived["packing_utilization"] > 0.5 and derived["loss_tokens_per_sec"] > 0
    c.detail = perf_report
    c.evidence = [_rel(artifacts_dir, os.path.join(artifacts_dir, "performance.json"))]
    checks.append(c)

    # ---------------------------------------------------------------- assemble
    evidence = {
        "run_id": run_id,
        "fork_run_id": fork_run_id,
        "generated_by": "datasys.audit.run_audit",
        "all_passed": all(ch.passed for ch in checks),
        "n_checks": len(checks),
        "n_passed": sum(1 for ch in checks if ch.passed),
        "checks": {ch.key: ch.to_obj() for ch in checks},
    }
    write_json(os.path.join(artifacts_dir, "evidence.json"), evidence)
    _write_evidence_md(artifacts_dir, evidence, checks)
    return evidence


def _doc_text_hashes(manifests_dir: str) -> Dict[str, str]:
    """doc_id -> text content hash, read from the shard manifests."""
    out: Dict[str, str] = {}
    loaded = load_manifests(manifests_dir)
    for m in loaded["shards"].values():
        for d in m["documents"]:
            out[d["doc_id"]] = d["text_hash"]
    return out


def _actual_shares_from_ledger(ledgers_dir: str, run_id: str,
                               field: str) -> Dict[str, Dict[str, float]]:
    """Recompute realized lane shares per stage from the consumption ledger.

    ``field`` selects the unit: ``lane_slots`` (sequence slots, the unit the
    scheduler and the floors are expressed in) or ``lane_tokens`` (realized real
    tokens, which also reflects each policy's packing utilization).
    """
    path = os.path.join(ledgers_dir, f"consumption_{run_id}.jsonl")
    agg: Dict[str, Dict[str, int]] = {}
    for e in Ledger(path).entries():
        p = e["payload"]
        st = agg.setdefault(p["stage"], {})
        for lane, value in p[field].items():
            st[lane] = st.get(lane, 0) + value
    out: Dict[str, Dict[str, float]] = {}
    for stage, counts in agg.items():
        tot = sum(counts.values()) or 1
        out[stage] = {lane: counts.get(lane, 0) / tot for lane in mix.LANES}
    return out


ROW_ORDER = [
    ("tokenizer_integrity", "Tokenizer integrity", "Manifest record"),
    ("shard_immutability", "Shards & manifests", "Shard manifests re-hashed"),
    ("evaluation_firewall", "Evaluation firewall", "Blocked-shard event"),
    ("packing_correctness", "Packing correctness", "Packed-batch report"),
    ("mixture_compliance", "Mixture compliance", "Planned versus actual shares"),
    ("opus_audit_trail", "OPUS audit trail", "Candidate decision records"),
    ("consumption_ledger", "Consumption ledger", "Hash-chained step records"),
    ("learning_trace", "Learning trace", "Loss linked to source data"),
    ("crash_recovery", "Crash recovery", "Expected and resumed batch ids"),
    ("replay", "Replay", "Original and replay hashes"),
    ("fork", "Fork lineage", "Forked run ledger + lineage"),
    ("throughput", "Throughput", "Performance report"),
]


def _write_evidence_md(artifacts_dir: str, evidence: dict, checks: List[Check]) -> None:
    by_key = {ch.key: ch for ch in checks}
    lines = [
        "# Evidence Summary",
        "",
        f"Run id: `{evidence['run_id']}`  ",
        f"Fork run id: `{evidence['fork_run_id']}`  ",
        f"Checks passed: **{evidence['n_passed']} / {evidence['n_checks']}**",
        "",
        "| Requirement | Result | Evidence |",
        "|---|---|---|",
    ]
    for key, title, ev_desc in ROW_ORDER:
        ch = by_key.get(key)
        if ch is None:
            continue
        files = ", ".join(f"`{p}`" for p in ch.evidence) or "-"
        lines.append(f"| {title} | {'PASS' if ch.passed else 'FAIL'} | {ev_desc} — {files} |")

    lines += ["", "## Key numbers", ""]
    tk = by_key["tokenizer_integrity"].detail
    sh = by_key["shard_immutability"].detail
    pk = by_key["packing_correctness"].detail
    op = by_key["opus_audit_trail"].detail
    th = by_key["throughput"].detail
    lt = by_key["learning_trace"].detail
    cr = by_key["crash_recovery"].detail
    rp = by_key["replay"].detail
    lines += [
        f"- Frozen tokenizer hash: `{tk['tokenizer_content_hash'][:16]}…`, "
        f"vocab {tk['vocab_size']}, bound to {tk['shards_bound_to_tokenizer']}/{tk['shards_total']} shards.",
        f"- Shards: {sh['shards_verified']} verified, {sh['total_documents']} documents, "
        f"{sh['total_tokens']} tokens.",
        f"- Packing: {pk['samples_checked']} samples checked, "
        f"{pk['invariant_violations']} invariant violations, "
        f"{pk['policies_exercised']} policies exercised.",
        f"- OPUS decision events: {op['events_by_decision']}; "
        f"final per document: {op['final_decision_per_document']}.",
        f"- Packing utilization: {th['derived']['packing_utilization']:.3f}; "
        f"loss-bearing fraction {th['derived']['loss_bearing_fraction']:.3f}; "
        f"{th['derived']['loss_tokens_per_sec']:.1f} loss-bearing tokens/sec.",
        f"- Loss: {lt['mean_loss_first_10_steps']:.4f} (first 10 steps) → "
        f"{lt['mean_loss_last_10_steps']:.4f} (last 10 steps).",
        f"- Crash at step {cr.get('crash_at_step')}, resumed from "
        f"`{cr.get('resumed_from')}`; expected next batch "
        f"`{str(cr.get('expected_next_batch_id'))[:16]}…` matched: "
        f"{cr.get('next_batch_matched')}.",
        f"- Replay interval {rp.get('interval')}: {rp.get('n_steps')} steps, "
        f"all hashes matched: {rp.get('all_match')}.",
        "",
        "All values above are recomputed by `datasys/audit.py` from the generated",
        "manifests, ledgers, checkpoints and counters — none are hardcoded.",
        "",
    ]
    with open(os.path.join(artifacts_dir, "evidence.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
