"""Mixture schedule: curriculum stages, lane weights and protected floors.

This is the miniature execution of the Assignment-5 spec. Four curriculum stages
(A foundation -> B rebalance -> C long-context -> D anneal) each declare a lane
weight vector. Three lanes carry a protected floor that the compiled schedule may
never cross in any stage:

    indic 8%, reasoning 3%, agentic 2%   (mirrors assignment-5 s5)

``compile_schedule`` turns (total_steps, sequences_per_step) into a concrete,
deterministic per-step assignment of sequences to lanes, using largest-remainder
rounding so planned shares are hit as exactly as integer batches allow, and so
that the *floor is applied before rounding* -- a floored lane always gets at least
its floor share of sequences. The audit later compares planned vs actual.
"""

from __future__ import annotations

import math
from typing import Dict, List

from .util import deterministic_shuffle, largest_remainder

LANES = ["web", "code", "math_science", "indic", "reasoning", "agentic"]

# Fixed seed for slot spreading inside a stage. Part of the schedule's identity:
# changing it changes the compiled schedule and therefore every batch id.
SCHEDULE_SEED = 611_2026

# Protected floors (fraction of each step's sequences), enforced every stage.
FLOORS: Dict[str, float] = {"indic": 0.08, "reasoning": 0.03, "agentic": 0.02}

# Per-stage lane weights (unnormalized; compiled to fractions). Compressed from
# the assignment-5 A/B/C/D curriculum.
STAGE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "A_foundation":   {"web": 40, "code": 18, "math_science": 10, "indic": 16, "reasoning": 5,  "agentic": 2},
    "B_rebalance":    {"web": 18, "code": 30, "math_science": 20, "indic": 8,  "reasoning": 12, "agentic": 8},
    "C_long_context": {"web": 10, "code": 15, "math_science": 5,  "indic": 8,  "reasoning": 7,  "agentic": 5},
    "D_anneal":       {"web": 10, "code": 25, "math_science": 15, "indic": 15, "reasoning": 20, "agentic": 12},
}

STAGE_ORDER = ["A_foundation", "B_rebalance", "C_long_context", "D_anneal"]


def stage_boundaries(total_steps: int) -> List[tuple]:
    """Split total steps into the four stages by fixed fractions (40/30/15/15)."""
    fracs = [0.40, 0.30, 0.15, 0.15]
    bounds = []
    start = 0
    acc = 0.0
    for i, name in enumerate(STAGE_ORDER):
        acc += fracs[i]
        end = total_steps if i == len(STAGE_ORDER) - 1 else round(acc * total_steps)
        bounds.append((name, start, end))
        start = end
    return bounds


def stage_for_step(step: int, total_steps: int) -> str:
    for name, s, e in stage_boundaries(total_steps):
        if s <= step < e:
            return name
    return STAGE_ORDER[-1]


def _weights_with_floor(stage: str) -> Dict[str, float]:
    """Stage weights normalized to fractions, with protected floors enforced.

    Any lane below its floor is pinned *at* its floor; the remaining probability
    mass is then distributed across the unpinned lanes in proportion to their
    declared weights. Pinning first and renormalizing only the remainder is what
    makes the floor a hard guarantee -- a later renormalization can never push a
    floored lane back under its floor.
    """
    w = STAGE_WEIGHTS[stage]
    total_w = sum(w.values())
    frac = {k: w[k] / total_w for k in LANES}

    pinned: Dict[str, float] = {}
    free = [l for l in LANES]
    # iteratively pin lanes that fall below their floor after renormalization
    while True:
        free_mass = 1.0 - sum(pinned.values())
        free_weight = sum(w[l] for l in free) or 1.0
        candidate = {l: free_mass * w[l] / free_weight for l in free}
        violating = [l for l in free if l in FLOORS and candidate[l] < FLOORS[l] - 1e-12]
        if not violating:
            frac = {**pinned, **candidate}
            break
        for l in violating:
            pinned[l] = FLOORS[l]
            free.remove(l)
        if not free:
            frac = dict(pinned)
            break
    return {l: frac[l] for l in LANES}


def _enforce_integer_floors(counts: Dict[str, int], total_slots: int) -> Dict[str, int]:
    """Guarantee each floored lane's *integer* slot count meets its floor.

    Proportional rounding can leave a floored lane one slot short (2% of 152
    slots is 3.04, which rounds down to 3 and realizes 1.97%). A floor that
    rounding can breach is not a floor, so any shortfall is taken from the
    largest lane that carries no floor.
    """
    out = dict(counts)
    for lane, fl in sorted(FLOORS.items()):
        need = math.ceil(fl * total_slots)
        while out[lane] < need:
            donors = [l for l in LANES if l not in FLOORS and out[l] > 1]
            if not donors:
                donors = [l for l in LANES if l != lane and out[l] > 1]
            if not donors:
                break
            donor = max(donors, key=lambda l: (out[l], l))
            out[donor] -= 1
            out[lane] += 1
    return out


def compile_schedule(total_steps: int, seqs_per_step: int) -> dict:
    """Produce the full deterministic schedule.

    Returns a dict with, per step: the stage, and a list of lane names of length
    ``seqs_per_step`` (which lane each sequence slot is drawn from). Also returns
    planned fractional shares per stage for the audit.
    """
    bounds = stage_boundaries(total_steps)
    per_step: List[dict] = []
    for stage, s_start, s_end in bounds:
        frac = _weights_with_floor(stage)
        n_stage_steps = s_end - s_start
        # Apportion the *whole stage's* slots first, then deal them out across
        # its steps. Rounding to 8 slots per step alone would let a small lane
        # (3% of 8 slots = 0.24) be rounded up in every step and badly overshoot;
        # apportioning at stage level and dealing the surplus keeps the realized
        # share close to plan while never dropping a floored lane to zero.
        stage_slots = n_stage_steps * seqs_per_step
        stage_counts = largest_remainder(frac, stage_slots)
        stage_counts = _enforce_integer_floors(stage_counts, stage_slots)
        pool: List[str] = []
        for lane in LANES:
            pool.extend([lane] * stage_counts[lane])
        # Spread the stage's slots across its steps with a seeded shuffle, so a
        # small lane is not clustered into the first few steps. The shuffle is a
        # pure function of (seed, stage), so the schedule stays reproducible.
        pool = deterministic_shuffle(pool, SCHEDULE_SEED, stage, n_stage_steps)
        for i, step in enumerate(range(s_start, s_end)):
            slots = pool[i * seqs_per_step:(i + 1) * seqs_per_step]
            counts = {l: slots.count(l) for l in LANES}
            per_step.append({"step": step, "stage": stage, "lane_slots": slots,
                             "lane_counts": counts})

    planned = {name: _weights_with_floor(name) for name in STAGE_ORDER}

    return {
        "total_steps": total_steps,
        "seqs_per_step": seqs_per_step,
        "floors": FLOORS,
        "floor_basis": "fraction of scheduled sequence slots per stage",
        "stage_boundaries": [
            {"stage": n, "start": s, "end": e} for n, s, e in bounds
        ],
        "planned_fractions": planned,
        "per_step": per_step,
    }


def scheduled_shares(per_step: List[dict]) -> Dict[str, Dict[str, float]]:
    """Realized lane slot-shares per stage, from the compiled per-step slots."""
    agg: Dict[str, Dict[str, int]] = {}
    for rec in per_step:
        st = agg.setdefault(rec["stage"], {l: 0 for l in LANES})
        for lane in rec["lane_slots"]:
            st[lane] += 1
    out: Dict[str, Dict[str, float]] = {}
    for stage, counts in agg.items():
        tot = sum(counts.values()) or 1
        out[stage] = {l: counts[l] / tot for l in LANES}
    return out
