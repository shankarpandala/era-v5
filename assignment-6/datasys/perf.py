"""Throughput and packing-efficiency accounting.

Only *raw counters* are persisted (tokens by class, wall time, steps). Every
derived number -- packing utilization, useful loss-bearing tokens per second --
is recomputed by the audit from those counters and cross-checked against the
reported values, so no efficiency claim is taken on faith.

Definitions:
  packing_utilization      = real_tokens / total_slot_tokens   (1 - pad fraction)
  loss_bearing_fraction    = loss_tokens / real_tokens
  tokens_per_sec           = real_tokens / wall_seconds
  loss_tokens_per_sec      = loss_tokens / wall_seconds         (useful work rate)
"""

from __future__ import annotations

from typing import Dict


class PerfCounter:
    def __init__(self):
        self.steps = 0
        self.real_tokens = 0
        self.loss_tokens = 0
        self.total_slot_tokens = 0
        self.wall_seconds = 0.0

    def add_step(self, real_tokens: int, loss_tokens: int, total_slot_tokens: int,
                 dt: float):
        self.steps += 1
        self.real_tokens += real_tokens
        self.loss_tokens += loss_tokens
        self.total_slot_tokens += total_slot_tokens
        self.wall_seconds += dt

    def to_counters(self) -> dict:
        return {
            "steps": self.steps,
            "real_tokens": self.real_tokens,
            "loss_tokens": self.loss_tokens,
            "total_slot_tokens": self.total_slot_tokens,
            "wall_seconds": self.wall_seconds,
        }

    def load(self, counters: dict) -> None:
        """Restore counters from a checkpoint, so work done after that
        checkpoint but lost to a crash is not counted twice on resume."""
        self.steps = counters["steps"]
        self.real_tokens = counters["real_tokens"]
        self.loss_tokens = counters["loss_tokens"]
        self.total_slot_tokens = counters["total_slot_tokens"]
        self.wall_seconds = counters["wall_seconds"]


def derive(counters: dict) -> Dict[str, float]:
    real = counters["real_tokens"]
    loss = counters["loss_tokens"]
    total = counters["total_slot_tokens"]
    wall = counters["wall_seconds"] or 1e-9
    return {
        "packing_utilization": real / total if total else 0.0,
        "loss_bearing_fraction": loss / real if real else 0.0,
        "tokens_per_sec": real / wall,
        "loss_tokens_per_sec": loss / wall,
        "pad_fraction": 1.0 - (real / total if total else 0.0),
    }


def merge_counters(a: dict, b: dict) -> dict:
    return {
        "steps": a["steps"] + b["steps"],
        "real_tokens": a["real_tokens"] + b["real_tokens"],
        "loss_tokens": a["loss_tokens"] + b["loss_tokens"],
        "total_slot_tokens": a["total_slot_tokens"] + b["total_slot_tokens"],
        "wall_seconds": a["wall_seconds"] + b["wall_seconds"],
    }
