"""Replay: reconstruct a historical interval of the data stream from artifacts.

Replay uses only immutable inputs -- the schedule, the OPUS-admitted inventory and
the shard bytes -- plus the run id. It never reads the consumption ledger to build
batches; the ledger is only used afterwards to *compare*. That is what makes the
match meaningful: two independent derivations of the same stream agree on batch
ids, per-sample hashes and token spans.
"""

from __future__ import annotations

import os
from typing import Dict, List

from .batcher import replay_interval
from .ledger import Ledger
from .util import read_json


def load_consumption(ledgers_dir: str, run_id: str) -> Dict[int, dict]:
    path = os.path.join(ledgers_dir, f"consumption_{run_id}.jsonl")
    out: Dict[int, dict] = {}
    for e in Ledger(path).entries():
        out[e["payload"]["step"]] = e["payload"]
    return out


def replay_and_compare(artifacts_dir: str, run_id: str, start: int, end: int,
                       seq_len: int) -> dict:
    manifests_dir = os.path.join(artifacts_dir, "manifests")
    ledgers_dir = os.path.join(artifacts_dir, "ledgers")
    schedule = read_json(os.path.join(manifests_dir, "schedule.json"))
    inventory = read_json(os.path.join(manifests_dir, "inventory.json"))

    replayed = replay_interval(run_id, schedule, inventory, manifests_dir,
                               seq_len, start, end)
    original = load_consumption(ledgers_dir, run_id)

    comparisons: List[dict] = []
    all_match = True
    for b in replayed:
        step = b["step"]
        orig = original.get(step)
        if orig is None:
            all_match = False
            comparisons.append({"step": step, "match": False, "error": "missing in ledger"})
            continue
        batch_match = orig["batch_id"] == b["batch_id"]
        orig_hashes = [s["sample_hash"] for s in orig["samples"]]
        new_hashes = [s["sample_hash"] for s in b["samples"]]
        hash_match = orig_hashes == new_hashes
        orig_spans = [
            (seg["shard_id"], seg["token_start"], seg["token_end"])
            for s in orig["samples"] for seg in s["segments"]
        ]
        new_spans = [
            (seg["shard_id"], seg["token_start"], seg["token_end"])
            for s in b["samples"] for seg in s["segments"]
        ]
        span_match = orig_spans == new_spans
        ok = batch_match and hash_match and span_match
        all_match = all_match and ok
        comparisons.append({
            "step": step,
            "match": ok,
            "original_batch_id": orig["batch_id"],
            "replay_batch_id": b["batch_id"],
            "sample_hashes_match": hash_match,
            "token_spans_match": span_match,
            "n_segments": len(new_spans),
        })

    return {
        "run_id": run_id,
        "interval": [start, end],
        "all_match": all_match,
        "n_steps": len(replayed),
        "comparisons": comparisons,
    }
