"""Evaluation firewall, mixture floors and OPUS admission rules."""

from __future__ import annotations

import os

from datasys import mixture as mix
from datasys import opus as opus_mod
from datasys.batcher import Batcher
from datasys.firewall import Firewall
from datasys.ledger import Ledger
from datasys.util import read_json, sha256_text

from .fixtures import SEQ_LEN, SEQS_PER_STEP, TOTAL_STEPS, build_session


# --------------------------------------------------------------------- firewall

def test_firewall_blocks_by_content_hash_not_by_label():
    fw = Firewall()
    text = "EVALUATION ITEM. what is the capital of Telangana?"
    fw.register("E1", "eval", sha256_text(text))
    # the same bytes relabelled as training data are still blocked
    assert fw.check(sha256_text(text)) == "eval_firewall"
    assert fw.check(sha256_text(text + " ")) is None


def test_poisoned_training_document_is_rejected(tmp_path):
    session = build_session(str(tmp_path))
    ledger = Ledger(os.path.join(str(tmp_path), "ledgers", "opus_ledger.jsonl"))
    entries = [e["payload"] for e in ledger.entries()]
    fw = read_json(os.path.join(str(tmp_path), "manifests", "firewall.json"))
    blocked = {e["content_hash"] for e in fw["entries"]}
    poisoned = [e for e in entries if e["split"] == "train" and e["content_hash"] in blocked]
    assert poisoned, "the corpus should contain a relabelled evaluation document"
    assert all(e["decision"] == "REJECT" for e in poisoned)
    assert all(e["reason"].endswith("_firewall") for e in poisoned)


def test_no_blocked_content_reaches_the_admitted_inventory(tmp_path):
    session = build_session(str(tmp_path))
    fw = read_json(os.path.join(str(tmp_path), "manifests", "firewall.json"))
    blocked = {e["content_hash"] for e in fw["entries"]}
    by_id = {d["doc_id"]: d for d in session["documents"]}
    for lane, docs in session["inventory"].items():
        for d in docs:
            assert sha256_text(by_id[d["doc_id"]]["text"]) not in blocked


def test_eval_shards_are_never_scheduled(tmp_path):
    session = build_session(str(tmp_path))
    b = Batcher("t", session["schedule"], session["inventory"],
                os.path.join(str(tmp_path), "manifests"), SEQ_LEN)
    for step in range(TOTAL_STEPS):
        for s in b.build_step(step)["samples"]:
            for seg in s["segments"]:
                assert not seg["shard_id"].endswith("__eval")
                assert not seg["shard_id"].endswith("__validation")


# ---------------------------------------------------------------------- mixture

def test_planned_fractions_sum_to_one_and_respect_floors():
    for stage in mix.STAGE_ORDER:
        frac = mix._weights_with_floor(stage)
        assert abs(sum(frac.values()) - 1.0) < 1e-9
        for lane, floor in mix.FLOORS.items():
            assert frac[lane] >= floor - 1e-12, (stage, lane)


def test_compiled_schedule_meets_integer_floors_in_every_stage():
    sched = mix.compile_schedule(TOTAL_STEPS, SEQS_PER_STEP)
    shares = mix.scheduled_shares(sched["per_step"])
    for stage, lanes in shares.items():
        for lane, floor in mix.FLOORS.items():
            assert lanes[lane] + 1e-12 >= floor, (stage, lane, lanes[lane])


def test_every_step_gets_exactly_the_requested_number_of_slots():
    sched = mix.compile_schedule(TOTAL_STEPS, SEQS_PER_STEP)
    for rec in sched["per_step"]:
        assert len(rec["lane_slots"]) == SEQS_PER_STEP


def test_schedule_compilation_is_deterministic():
    a = mix.compile_schedule(TOTAL_STEPS, SEQS_PER_STEP)
    b = mix.compile_schedule(TOTAL_STEPS, SEQS_PER_STEP)
    assert [r["lane_slots"] for r in a["per_step"]] == \
           [r["lane_slots"] for r in b["per_step"]]


def test_realized_shares_track_the_plan_within_tolerance():
    sched = mix.compile_schedule(48, 8)
    planned = sched["planned_fractions"]
    actual = mix.scheduled_shares(sched["per_step"])
    for stage, lanes in actual.items():
        for lane, share in lanes.items():
            assert abs(share - planned[stage][lane]) <= 0.02, (stage, lane)


def test_a_lane_starved_by_weights_is_still_lifted_to_its_floor():
    """A stage that gives agentic almost nothing must still hit the 2% floor."""
    original = dict(mix.STAGE_WEIGHTS["A_foundation"])
    try:
        mix.STAGE_WEIGHTS["A_foundation"] = {**original, "agentic": 0.0001}
        frac = mix._weights_with_floor("A_foundation")
        assert frac["agentic"] >= mix.FLOORS["agentic"]
        sched = mix.compile_schedule(TOTAL_STEPS, SEQS_PER_STEP)
        shares = mix.scheduled_shares(sched["per_step"])
        assert shares["A_foundation"]["agentic"] >= mix.FLOORS["agentic"]
    finally:
        mix.STAGE_WEIGHTS["A_foundation"] = original


# ------------------------------------------------------------------------ OPUS

def test_all_four_decision_types_occur(tmp_path):
    build_session(str(tmp_path))
    entries = [e["payload"] for e in
               Ledger(os.path.join(str(tmp_path), "ledgers", "opus_ledger.jsonl")).entries()]
    decisions = {e["decision"] for e in entries}
    assert decisions == {"ACCEPT", "REJECT", "DEFER", "FLOOR_OVERRIDE"}


def test_deferral_is_distinct_from_rejection(tmp_path):
    """At least one deferred candidate is reconsidered and admitted in pass 2."""
    build_session(str(tmp_path))
    entries = [e["payload"] for e in
               Ledger(os.path.join(str(tmp_path), "ledgers", "opus_ledger.jsonl")).entries()]
    requeued = [e for e in entries if e["reason"] == "requeued_admitted"]
    assert requeued
    for e in requeued:
        assert e["pass"] == 2 and e["decision"] == "ACCEPT"
        earlier = [x for x in entries if x["doc_id"] == e["doc_id"] and x["pass"] == 1]
        assert earlier and earlier[0]["decision"] == "DEFER"


def test_floor_override_only_fires_for_floored_lanes(tmp_path):
    build_session(str(tmp_path))
    entries = [e["payload"] for e in
               Ledger(os.path.join(str(tmp_path), "ledgers", "opus_ledger.jsonl")).entries()]
    overrides = [e for e in entries if e["decision"] == "FLOOR_OVERRIDE"]
    assert overrides
    for e in overrides:
        assert e["lane"] in mix.FLOORS
        assert e["reason"] == "protected_floor_starved"
        # the override only happens when the budget would otherwise have deferred
        assert e["lane_admitted_before"] + e["n_tokens"] > e["lane_budget"]
        assert e["lane_admitted_before"] < e["floor_token_demand"]


def test_duplicate_documents_are_rejected(tmp_path):
    build_session(str(tmp_path))
    entries = [e["payload"] for e in
               Ledger(os.path.join(str(tmp_path), "ledgers", "opus_ledger.jsonl")).entries()]
    dups = [e for e in entries if e["reason"] == "duplicate"]
    assert dups
    for e in dups:
        assert e["decision"] == "REJECT"
    # an admitted document with the same hash must exist and come first
    admitted = {e["content_hash"] for e in entries
                if e["decision"] in ("ACCEPT", "FLOOR_OVERRIDE")}
    assert all(e["content_hash"] in admitted for e in dups)


def test_floor_token_demand_matches_the_schedule():
    demand = opus_mod.floor_token_demand(SEQ_LEN, SEQS_PER_STEP, TOTAL_STEPS)
    total = SEQ_LEN * SEQS_PER_STEP * TOTAL_STEPS
    for lane, floor in mix.FLOORS.items():
        assert demand[lane] == int(floor * total)


def test_admission_is_order_independent_of_input_ordering(tmp_path):
    """Admission sorts by doc_id, so shuffling the input cannot change it."""
    session = build_session(str(tmp_path))
    docs = list(reversed(session["documents"]))
    shard_index = {}
    for m in session["manifests"]["shards"]:
        for dr in m["documents"]:
            shard_index[dr["doc_id"]] = {
                "shard_id": m["shard_id"], "token_start": dr["token_start"],
                "token_end": dr["token_end"], "n_tokens": dr["n_tokens"],
                "prompt_tokens": dr["prompt_tokens"]}
    fw = Firewall.from_documents(session["documents"], sha256_text)
    demand = opus_mod.floor_token_demand(SEQ_LEN, SEQS_PER_STEP, TOTAL_STEPS)
    again = opus_mod.admit(docs, shard_index, fw, sha256_text, demand)
    assert again["inventory"].keys() == session["inventory"].keys()
    for lane in again["inventory"]:
        assert [d["doc_id"] for d in again["inventory"][lane]] == \
               [d["doc_id"] for d in session["inventory"][lane]]
