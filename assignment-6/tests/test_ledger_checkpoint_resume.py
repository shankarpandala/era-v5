"""Ledger tamper-evidence, checkpoint round-trip, crash/resume, replay and fork."""

from __future__ import annotations

import json
import os

import torch

from datasys.batcher import Batcher, batch_consumption_record, replay_interval
from datasys.checkpoint import load_checkpoint, save_checkpoint, verify_checkpoint
from datasys.ledger import GENESIS, Ledger, verify_chain
from datasys.train import build_model

from .fixtures import SEQ_LEN, TOTAL_STEPS, build_session


# ---------------------------------------------------------------------- ledger

def test_chain_verifies_and_detects_edits(tmp_path):
    path = os.path.join(str(tmp_path), "l.jsonl")
    led = Ledger(path)
    for i in range(5):
        led.append({"step": i, "value": i * 2})
    assert verify_chain(path)["ok"]
    assert verify_chain(path)["count"] == 5

    lines = open(path).read().splitlines()
    obj = json.loads(lines[2])
    obj["payload"]["value"] = 999          # tamper with a payload
    lines[2] = json.dumps(obj)
    open(path, "w").write("\n".join(lines) + "\n")
    result = verify_chain(path)
    assert not result["ok"] and "hash mismatch" in result["error"]


def test_chain_detects_a_deleted_middle_entry(tmp_path):
    path = os.path.join(str(tmp_path), "l.jsonl")
    led = Ledger(path)
    for i in range(5):
        led.append({"step": i})
    lines = open(path).read().splitlines()
    del lines[2]
    open(path, "w").write("\n".join(lines) + "\n")
    assert not verify_chain(path)["ok"]


def test_truncate_restores_the_earlier_chain_head(tmp_path):
    path = os.path.join(str(tmp_path), "l.jsonl")
    led = Ledger(path)
    for i in range(3):
        led.append({"step": i})
    head_at_3, count_at_3 = led.head, led.count
    for i in range(3, 6):
        led.append({"step": i})
    led.truncate_to(count_at_3)
    assert led.head == head_at_3
    assert led.count == count_at_3
    assert verify_chain(path)["ok"]
    # appending again continues the same chain
    led.append({"step": 3})
    assert verify_chain(path)["ok"]


def test_reopening_a_ledger_recovers_head_and_count(tmp_path):
    path = os.path.join(str(tmp_path), "l.jsonl")
    led = Ledger(path)
    for i in range(4):
        led.append({"step": i})
    reopened = Ledger(path)
    assert reopened.head == led.head and reopened.count == led.count


# ------------------------------------------------------------------ checkpoint

def _tiny(vocab=64):
    cfg = {"d_model": 32, "n_layer": 1, "n_head": 2, "lr": 1e-3}
    return build_model(vocab, 32, cfg)


def test_checkpoint_round_trip_restores_the_model_exactly(tmp_path):
    ckpt_dir = str(tmp_path)
    model, opt = _tiny()
    ids = torch.randint(0, 64, (2, 16))
    seg = torch.zeros_like(ids)
    pos = torch.arange(16).expand(2, 16)
    model(ids, seg, pos).sum().backward()
    opt.step()

    man = save_checkpoint(ckpt_dir, "t", 7, "A", model, opt,
                          {"web": {"doc": 3, "off": 12}},
                          {"count": 7, "head": "h1"}, {"count": 7, "head": "h2"},
                          {"steps": 7, "real_tokens": 1, "loss_tokens": 1,
                           "total_slot_tokens": 1, "wall_seconds": 0.5},
                          "run_x",
                          data_binding={"tokenizer_hash": "abc", "seq_len": 32})
    before = model(ids, seg, pos)

    other, other_opt = _tiny()
    restored = load_checkpoint(ckpt_dir, "t", other, other_opt)
    after = other(ids, seg, pos)
    assert torch.equal(before, after)
    assert restored["cursor"] == {"web": {"doc": 3, "off": 12}}
    assert restored["consumption_offset"] == {"count": 7, "head": "h1"}
    assert restored["perf_counters"]["steps"] == 7
    assert man["model_tensor_hash"] == restored["model_tensor_hash"]
    assert restored["data_binding"] == {"tokenizer_hash": "abc", "seq_len": 32}


def test_corrupting_a_checkpoint_blob_is_detected(tmp_path):
    ckpt_dir = str(tmp_path)
    model, opt = _tiny()
    save_checkpoint(ckpt_dir, "t", 1, "A", model, opt, {},
                    {"count": 0, "head": GENESIS}, {"count": 0, "head": GENESIS},
                    {"steps": 0, "real_tokens": 0, "loss_tokens": 0,
                     "total_slot_tokens": 0, "wall_seconds": 0.0}, "r")
    blob = os.path.join(ckpt_dir, "t.pt")
    data = bytearray(open(blob, "rb").read())
    data[-1] ^= 0xFF
    open(blob, "wb").write(bytes(data))
    assert verify_checkpoint(ckpt_dir, "t")["ok"] is False


# ------------------------------------------------------- resume / replay / fork

def _batcher(tmp_path, session, run_id="t"):
    return Batcher(run_id, session["schedule"], session["inventory"],
                   os.path.join(str(tmp_path), "manifests"), SEQ_LEN)


def test_restoring_a_cursor_reproduces_the_next_batch(tmp_path):
    session = build_session(str(tmp_path))
    a = _batcher(tmp_path, session)
    for step in range(5):
        a.build_step(step)
    saved_cursor = a.get_cursor()
    expected = a.build_step(5)

    b = _batcher(tmp_path, session)
    b.set_cursor(saved_cursor)              # nothing else carried over
    actual = b.build_step(5)
    assert actual["batch_id"] == expected["batch_id"]
    assert [s["sample_hash"] for s in actual["samples"]] == \
           [s["sample_hash"] for s in expected["samples"]]


def test_a_stale_cursor_produces_a_different_batch(tmp_path):
    """Guards against a cursor that silently does not matter."""
    session = build_session(str(tmp_path))
    a = _batcher(tmp_path, session)
    for step in range(5):
        a.build_step(step)
    expected = a.build_step(5)

    b = _batcher(tmp_path, session)         # cursor left at step 0
    assert b.build_step(5)["batch_id"] != expected["batch_id"]


def test_replay_reconstructs_an_interval_exactly(tmp_path):
    session = build_session(str(tmp_path))
    original = _batcher(tmp_path, session)
    batches = [original.build_step(s) for s in range(TOTAL_STEPS)]

    replayed = replay_interval("t", session["schedule"], session["inventory"],
                               os.path.join(str(tmp_path), "manifests"),
                               SEQ_LEN, 3, 9)
    assert len(replayed) == 6
    for got in replayed:
        want = batches[got["step"]]
        assert got["batch_id"] == want["batch_id"]
        assert [s["sample_hash"] for s in got["samples"]] == \
               [s["sample_hash"] for s in want["samples"]]
        got_spans = [(seg["shard_id"], seg["token_start"], seg["token_end"])
                     for s in got["samples"] for seg in s["segments"]]
        want_spans = [(seg["shard_id"], seg["token_start"], seg["token_end"])
                      for s in want["samples"] for seg in s["segments"]]
        assert got_spans == want_spans


def test_batch_ids_are_namespaced_by_run_id(tmp_path):
    """A fork must not collide with its parent's batch ids at the same step."""
    session = build_session(str(tmp_path))
    parent = _batcher(tmp_path, session, "run_parent")
    child = _batcher(tmp_path, session, "run_child")
    p, c = parent.build_step(0), child.build_step(0)
    assert p["batch_id"] != c["batch_id"]
    # ...while consuming exactly the same data
    assert [s["sample_hash"] for s in p["samples"]] == \
           [s["sample_hash"] for s in c["samples"]]


def test_consumption_record_accounts_for_every_token(tmp_path):
    session = build_session(str(tmp_path))
    b = _batcher(tmp_path, session)
    for step in range(4):
        rec = batch_consumption_record(b.build_step(step))
        assert rec["n_real_tokens"] + rec["n_pad_tokens"] == rec["n_total_tokens"]
        assert sum(rec["lane_tokens"].values()) == rec["n_real_tokens"]
        assert sum(rec["lane_slots"].values()) == rec["n_samples"]
        assert rec["n_loss_tokens"] <= rec["n_real_tokens"]
