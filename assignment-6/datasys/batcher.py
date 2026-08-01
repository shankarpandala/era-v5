"""Deterministic batch stream.

The batcher is the pure function at the centre of the system:

    batch(step) = f(run_id, schedule, admitted_inventory, shard_bytes, cursor)

For a step it reads the schedule's per-step lane slots and, for each slot, packs
one sequence from that lane's stream, advancing that lane's cursor. The only
mutable state is ``cursor`` -- ``{lane: {"doc", "off"}}`` -- which is exactly what
the checkpoint stores. Because the cursor fully determines each lane's next
sequence, restoring it reproduces the next batch bit for bit (crash/resume), and
fast-forwarding a fresh cursor from step 0 reproduces any historical interval
without a model (replay).

    batch_id = sha256(run_id, step, [sample_hash ...])

binds a step to the exact content it consumed, in order.
"""

from __future__ import annotations

from typing import Dict, List

from .mixture import LANES
from .packing import LaneStream, new_cursor, pack_sample
from .shards import load_shard_tokens
from .util import canonical_json, sha256_text


class Batcher:
    def __init__(self, run_id: str, schedule: dict, inventory: Dict[str, List[dict]],
                 shard_dir: str, seq_len: int):
        self.run_id = run_id
        self.schedule = schedule
        self.seq_len = seq_len
        self.shard_dir = shard_dir
        self.streams = {lane: LaneStream(inventory.get(lane, [])) for lane in LANES}
        self.cursor: Dict[str, dict] = {lane: new_cursor() for lane in LANES}
        self._tok_cache: Dict[str, List[int]] = {}

    # -- cursor persistence -------------------------------------------------
    def get_cursor(self) -> Dict[str, dict]:
        return {lane: dict(c) for lane, c in self.cursor.items()}

    def set_cursor(self, cursor: Dict[str, dict]) -> None:
        self.cursor = {
            lane: {"doc": int(cursor.get(lane, {}).get("doc", 0)),
                   "off": int(cursor.get(lane, {}).get("off", 0))}
            for lane in LANES
        }

    def _tokens(self, shard_id: str) -> List[int]:
        toks = self._tok_cache.get(shard_id)
        if toks is None:
            toks = load_shard_tokens(self.shard_dir, shard_id)
            self._tok_cache[shard_id] = toks
        return toks

    # -- batch construction -------------------------------------------------
    def build_step(self, step: int, advance: bool = True) -> dict:
        """Build the batch for ``step`` from the current cursor.

        ``advance=False`` leaves the cursor untouched (peeking / verification).
        """
        rec = self.schedule["per_step"][step]
        cursor = self.get_cursor()
        samples = []
        for slot_i, lane in enumerate(rec["lane_slots"]):
            stream = self.streams[lane]
            if stream.n == 0:
                raise RuntimeError(
                    f"lane '{lane}' is scheduled at step {step} but has no admitted "
                    f"inventory -- the mixture cannot be honoured")
            sample = pack_sample(lane, stream, cursor[lane], self.seq_len, self._tokens)
            cursor[lane] = sample["cursor_after"]
            sample["slot"] = slot_i
            samples.append(sample)
        batch = self._assemble(step, rec["stage"], samples)
        if advance:
            self.cursor = cursor
        return batch

    def _assemble(self, step: int, stage: str, samples: List[dict]) -> dict:
        sample_hashes = [s["sample_hash"] for s in samples]
        batch_id = sha256_text(canonical_json({
            "run_id": self.run_id, "step": step, "sample_hashes": sample_hashes,
        }))
        return {
            "step": step,
            "stage": stage,
            "batch_id": batch_id,
            "samples": samples,
            "n_samples": len(samples),
            "n_real_tokens": sum(s["n_real_tokens"] for s in samples),
            "n_loss_tokens": sum(s["n_loss_tokens"] for s in samples),
            "n_pad_tokens": sum(s["n_pad_tokens"] for s in samples),
            "n_total_tokens": len(samples) * self.seq_len,
        }


def replay_interval(run_id: str, schedule: dict, inventory: Dict[str, List[dict]],
                    shard_dir: str, seq_len: int, start: int, end: int) -> List[dict]:
    """Reconstruct batches for steps [start, end) from artifacts alone.

    Fast-forwards a fresh cursor from step 0 to ``start`` (discarding those
    batches), then returns the requested window. Uses only the schedule, the
    admitted inventory and the shard bytes -- never the consumption ledger.
    """
    b = Batcher(run_id, schedule, inventory, shard_dir, seq_len)
    for s in range(start):
        b.build_step(s, advance=True)
    return [b.build_step(s, advance=True) for s in range(start, end)]


def batch_consumption_record(batch: dict) -> dict:
    """A compact, source-linked record for the consumption ledger."""
    samples = []
    lane_tokens: Dict[str, int] = {}
    lane_slots: Dict[str, int] = {}
    for s in batch["samples"]:
        samples.append({
            "slot": s["slot"],
            "lane": s["lane"],
            "policy": s["policy"],
            "sample_hash": s["sample_hash"],
            "n_real_tokens": s["n_real_tokens"],
            "n_loss_tokens": s["n_loss_tokens"],
            "n_pad_tokens": s["n_pad_tokens"],
            "cursor_before": s["cursor_before"],
            "cursor_after": s["cursor_after"],
            "segments": [
                {"doc_id": seg["doc_id"], "shard_id": seg["shard_id"],
                 "token_start": seg["token_start"], "token_end": seg["token_end"],
                 "epoch": seg["epoch"], "continued": seg["continued"]}
                for seg in s["segments"]
            ],
        })
        lane_tokens[s["lane"]] = lane_tokens.get(s["lane"], 0) + s["n_real_tokens"]
        lane_slots[s["lane"]] = lane_slots.get(s["lane"], 0) + 1
    return {
        "step": batch["step"],
        "stage": batch["stage"],
        "batch_id": batch["batch_id"],
        "n_samples": batch["n_samples"],
        "n_real_tokens": batch["n_real_tokens"],
        "n_loss_tokens": batch["n_loss_tokens"],
        "n_pad_tokens": batch["n_pad_tokens"],
        "n_total_tokens": batch["n_total_tokens"],
        "lane_tokens": lane_tokens,
        "lane_slots": lane_slots,
        "samples": samples,
    }
