"""OPUS -- the online data-admission controller.

Every candidate document that wants to enter the training pool is judged, and
every judgement is written to an auditable ledger with a reason code:

  ``ACCEPT``          passes the firewall, dedup and quality gates, and fits
                      inside the lane's admission budget.
  ``REJECT``          firewalled evaluation/validation content (by content hash,
                      so relabelling does not help), an exact duplicate, or a
                      failed quality gate. Terminal.
  ``DEFER``           the lane's budget for this pass is already committed. The
                      candidate is *not* rejected: it is re-queued for the next
                      admission pass.
  ``FLOOR_OVERRIDE``  a protected-floor lane cannot supply the tokens its floor
                      will demand over the run, so a candidate that the budget
                      would otherwise defer is force-admitted, with the reason
                      recorded so the exception is visible.

Admission runs in two passes, which is what makes ``DEFER`` a real state rather
than a synonym for rejection: pass 1 applies the base budgets; pass 2 reopens the
budgets by a fixed margin (the same way a later curriculum stage reopens supply)
and reconsiders only the deferred candidates. Some are admitted with reason
``requeued_admitted``; the rest remain deferred. Both events stay in the ledger,
so the history of a candidate is visible, and the *final* decision per document
is its last event.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from .firewall import Firewall
from .mixture import FLOORS

# Per-lane admission budget in tokens for pass 1. Set below the lane's available
# supply for the abundant lanes so the deferral path is genuinely exercised.
LANE_BUDGET_TOKENS: Dict[str, int] = {
    "web": 2000,
    "code": 1700,
    "math_science": 1500,
    "indic": 1000,
    "reasoning": 1200,
    "agentic": 400,
}

# Pass 2 reopens each budget by this factor and reconsiders deferred candidates.
REQUEUE_BUDGET_FACTOR = 1.15

MIN_TOKENS_QUALITY = 12  # documents shorter than this fail the quality gate


class Decision:
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    DEFER = "DEFER"
    FLOOR_OVERRIDE = "FLOOR_OVERRIDE"


TERMINAL = {Decision.ACCEPT, Decision.REJECT, Decision.FLOOR_OVERRIDE}


def floor_token_demand(seq_len: int, seqs_per_step: int, total_steps: int) -> Dict[str, int]:
    """Tokens each protected-floor lane will be asked for across the whole run.

    A floor of f means the scheduler will hand the lane f of every step's
    sequence slots; over the run that is ``f * seq_len * seqs_per_step *
    total_steps`` token positions. If a lane's admission budget cannot supply
    that, the floor is structurally unmeetable and OPUS must override.
    """
    total_slots = seq_len * seqs_per_step * total_steps
    return {lane: int(f * total_slots) for lane, f in FLOORS.items()}


def admit(documents: List[dict], shard_index: Dict[str, dict], firewall: Firewall,
          text_hash_fn: Callable[[str], str],
          floor_demand: Dict[str, int]) -> dict:
    """Run two-pass admission deterministically (document order = doc_id order)."""
    state = _State(firewall, text_hash_fn, floor_demand)

    # ---- pass 1: base budgets -------------------------------------------
    deferred: List[dict] = []
    for d in sorted(documents, key=lambda x: x["doc_id"]):
        sidx = shard_index.get(d["doc_id"])
        entry = state.judge(d, sidx, pass_no=1, budgets=LANE_BUDGET_TOKENS)
        if entry["decision"] == Decision.DEFER:
            deferred.append(d)

    # ---- pass 2: reopened budgets, deferred candidates only --------------
    reopened = {k: int(v * REQUEUE_BUDGET_FACTOR) for k, v in LANE_BUDGET_TOKENS.items()}
    for d in deferred:
        sidx = shard_index.get(d["doc_id"])
        state.judge(d, sidx, pass_no=2, budgets=reopened, requeue=True)

    return state.result()


class _State:
    def __init__(self, firewall, text_hash_fn, floor_demand):
        self.fw = firewall
        self.hash_fn = text_hash_fn
        self.floor_demand = floor_demand
        self.admitted_tokens: Dict[str, int] = {}
        self.seen_hashes: set = set()
        self.ledger: List[dict] = []
        self.inventory: Dict[str, List[dict]] = {}
        self.final: Dict[str, str] = {}  # doc_id -> final decision

    def judge(self, d: dict, sidx, pass_no: int, budgets: Dict[str, int],
              requeue: bool = False) -> dict:
        lane = d["lane"]
        chash = self.hash_fn(d["text"])
        n_tok = sidx["n_tokens"] if sidx else 0
        decision, reason = self._rule(d, lane, chash, n_tok, budgets)
        if requeue and decision == Decision.ACCEPT:
            reason = "requeued_admitted"

        entry = {
            "doc_id": d["doc_id"],
            "lane": lane,
            "split": d["split"],
            "content_hash": chash,
            "n_tokens": n_tok,
            "pass": pass_no,
            "decision": decision,
            "reason": reason,
            "lane_budget": budgets.get(lane, 0),
            "lane_admitted_before": self.admitted_tokens.get(lane, 0),
            "floor_token_demand": self.floor_demand.get(lane),
        }
        if decision in (Decision.ACCEPT, Decision.FLOOR_OVERRIDE):
            self.seen_hashes.add(chash)
            self.admitted_tokens[lane] = self.admitted_tokens.get(lane, 0) + n_tok
            self.inventory.setdefault(lane, []).append({
                "doc_id": d["doc_id"],
                "shard_id": sidx["shard_id"],
                "token_start": sidx["token_start"],
                "token_end": sidx["token_end"],
                "n_tokens": n_tok,
                "prompt_tokens": sidx.get("prompt_tokens", 0),
                "type": d["type"],
                "decision": decision,
                "admitted_in_pass": pass_no,
            })
        entry["lane_admitted_after"] = self.admitted_tokens.get(lane, 0)
        self.ledger.append(entry)
        self.final[d["doc_id"]] = decision
        return entry

    def _rule(self, d, lane, chash, n_tok, budgets) -> Tuple[str, str]:
        # 1. firewall by content hash -- catches relabelled evaluation content
        fw = self.fw.check(chash)
        if fw is not None:
            return Decision.REJECT, fw
        # 2. defense in depth: a document declaring a blocked split never trains
        if d["split"] in ("eval", "validation"):
            return Decision.REJECT, f"{d['split']}_split"
        # 3. exact duplicate of something already admitted
        if chash in self.seen_hashes:
            return Decision.REJECT, "duplicate"
        # 4. quality gate
        if d.get("quality") == "low":
            return Decision.REJECT, "quality_flagged"
        if n_tok < MIN_TOKENS_QUALITY:
            return Decision.REJECT, "quality_too_short"
        # 5. budget, with the protected floor able to override it
        cur = self.admitted_tokens.get(lane, 0)
        cap = budgets.get(lane, 0)
        if cur + n_tok > cap:
            if lane in FLOORS and cur < self.floor_demand.get(lane, 0):
                return Decision.FLOOR_OVERRIDE, "protected_floor_starved"
            return Decision.DEFER, "budget_exhausted"
        return Decision.ACCEPT, "ok"

    def result(self) -> dict:
        for lane in self.inventory:
            self.inventory[lane].sort(key=lambda x: x["doc_id"])
        by_decision: Dict[str, int] = {}
        by_reason: Dict[str, int] = {}
        for e in self.ledger:
            by_decision[e["decision"]] = by_decision.get(e["decision"], 0) + 1
            by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1
        final_counts: Dict[str, int] = {}
        for dec in self.final.values():
            final_counts[dec] = final_counts.get(dec, 0) + 1
        return {
            "ledger": self.ledger,
            "inventory": self.inventory,
            "admitted_tokens": self.admitted_tokens,
            "summary": {
                "total_events": len(self.ledger),
                "total_documents": len(self.final),
                "by_decision": by_decision,
                "by_reason": by_reason,
                "final_by_decision": final_counts,
                "admitted_tokens_per_lane": self.admitted_tokens,
                "budgets": LANE_BUDGET_TOKENS,
                "requeue_budget_factor": REQUEUE_BUDGET_FACTOR,
                "floor_token_demand": self.floor_demand,
            },
        }
