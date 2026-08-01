"""Packing policies, loss masks, attention masks and position ids.

Three policies, chosen by data type, with a real trade-off between them. Every
packed sequence obeys the same correctness contract, asserted by
``verify_sample_invariants`` and by the tests:

  ``input_ids``     length L: real tokens followed by PAD.
  ``segment_ids``   which segment each position belongs to. Pad positions get a
                    unique negative id so they only self-attend -- block-diagonal
                    attention means no cross-document leakage and no NaN rows.
  ``position_ids``  reset to 0 at the start of every segment, since a segment is
                    self-contained under the block-diagonal mask.
  ``loss_mask``     1 exactly where a next-token loss is legitimate: a real,
                    non-segment-initial, non-prompt token. Never on padding.

Policies:

  ``concat``            (web, math_science, indic) -- fills the sequence
      completely from a continuous token stream, splitting a document across the
      sequence boundary and continuing it in the next sequence. Utilization is
      ~1.0; the cost is that a long document's context is cut at the boundary.

  ``whole_unit``        (code) -- packs only *complete* files and never splits
      one; a file that does not fit waits for the next sequence and the tail is
      padded. Utilization is lower by design, and the packing report quantifies
      exactly what that locality costs.

  ``prompt_completion`` (reasoning, agentic) -- one trajectory per sequence with
      the prompt prefix attention-visible but loss-masked, so the model is only
      trained to produce the completion.

The cursor is ``{"doc": index, "off": token offset}`` per lane, so a lane's
stream position is a small, checkpointable value from which the next sequence is
fully determined.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from .tokenizer import PAD_ID
from .util import canonical_json, sha256_text

CONCAT = "concat"
WHOLE_UNIT = "whole_unit"
PROMPT_COMPLETION = "prompt_completion"

POLICY_BY_LANE = {
    "web": CONCAT,
    "math_science": CONCAT,
    "indic": CONCAT,
    "code": WHOLE_UNIT,
    "reasoning": PROMPT_COMPLETION,
    "agentic": PROMPT_COMPLETION,
}

POLICY_RATIONALE = {
    CONCAT: "prose-like lanes: maximize utilization, documents may continue "
            "across the sequence boundary",
    WHOLE_UNIT: "code: never split a file across sequences; accept padding to "
                "preserve file locality",
    PROMPT_COMPLETION: "reasoning/agentic: one trajectory per sequence, prompt "
                       "prefix visible but loss-masked",
}


def new_cursor() -> Dict[str, int]:
    return {"doc": 0, "off": 0}


class LaneStream:
    """Deterministic, epoch-wrapping stream over a lane's admitted documents.

    Position is ``{"doc", "off"}``. Document ``d`` of the stream is
    ``inventory[d % N]`` at epoch ``d // N``, so the stream is a pure function of
    (inventory, cursor): restore the cursor and the next sequence is provably
    identical. That property is what makes crash-resume and replay verifiable.
    """

    def __init__(self, inventory: List[dict]):
        self.inv = inventory
        self.n = len(inventory)

    def doc_at(self, d: int) -> dict:
        return self.inv[d % self.n]

    def epoch(self, d: int) -> int:
        return d // self.n if self.n else 0


def pack_sample(lane: str, stream: LaneStream, cursor: Dict[str, int], seq_len: int,
                token_provider: Callable[[str], List[int]]) -> dict:
    policy = POLICY_BY_LANE[lane]
    if policy == CONCAT:
        return _pack_concat(lane, stream, cursor, seq_len, token_provider)
    if policy == WHOLE_UNIT:
        return _pack_whole_unit(lane, stream, cursor, seq_len, token_provider)
    return _pack_prompt_completion(lane, stream, cursor, seq_len, token_provider)


def _doc_tokens(doc: dict, token_provider) -> List[int]:
    toks = token_provider(doc["shard_id"])
    return toks[doc["token_start"]: doc["token_end"]]


class _Buf:
    """Accumulates a packed sequence, one segment at a time."""

    def __init__(self):
        self.input_ids: List[int] = []
        self.segment_ids: List[int] = []
        self.position_ids: List[int] = []
        self.loss_mask: List[int] = []
        self.segments: List[dict] = []

    def add_segment(self, tokens: List[int], doc: dict, epoch: int,
                    doc_token_offset: int, prompt_tokens: int = 0,
                    continued: bool = False):
        seg_idx = len(self.segments)
        seg_start = len(self.input_ids)
        for k, tid in enumerate(tokens):
            self.input_ids.append(tid)
            self.segment_ids.append(seg_idx)
            self.position_ids.append(k)
            # a segment's first token has no in-segment predecessor; prompt
            # prefixes are visible context, not training targets
            self.loss_mask.append(0 if (k == 0 or k < prompt_tokens) else 1)
        self.segments.append({
            "doc_id": doc["doc_id"],
            "shard_id": doc["shard_id"],
            "token_start": doc["token_start"] + doc_token_offset,
            "token_end": doc["token_start"] + doc_token_offset + len(tokens),
            "doc_token_start": doc["token_start"],
            "doc_token_end": doc["token_end"],
            "seg_start": seg_start,
            "seg_len": len(tokens),
            "prompt_tokens": prompt_tokens,
            "epoch": epoch,
            "continued": continued,
        })

    def pad_to(self, seq_len: int):
        p = 0
        while len(self.input_ids) < seq_len:
            self.input_ids.append(PAD_ID)
            self.segment_ids.append(-(p + 1))  # unique negative -> singleton
            self.position_ids.append(0)
            self.loss_mask.append(0)
            p += 1


def _pack_concat(lane, stream, cursor, seq_len, token_provider) -> dict:
    buf = _Buf()
    d, off = cursor["doc"], cursor["off"]
    while len(buf.input_ids) < seq_len:
        doc = stream.doc_at(d)
        toks = _doc_tokens(doc, token_provider)
        remaining_in_doc = toks[off:]
        room = seq_len - len(buf.input_ids)
        take = remaining_in_doc[:room]
        if not take:  # degenerate empty doc; skip forward
            d += 1
            off = 0
            continue
        buf.add_segment(take, doc, stream.epoch(d), off, continued=(off > 0))
        off += len(take)
        if off >= len(toks):
            d += 1
            off = 0
    buf.pad_to(seq_len)
    return _finalize(lane, CONCAT, buf, cursor, {"doc": d, "off": off}, seq_len)


def _pack_whole_unit(lane, stream, cursor, seq_len, token_provider) -> dict:
    buf = _Buf()
    d = cursor["doc"]
    placed = 0
    while True:
        doc = stream.doc_at(d)
        toks = _doc_tokens(doc, token_provider)
        if len(toks) > seq_len:
            # a single unit larger than the sequence can never be packed whole;
            # it is truncated once, alone, and the truncation is recorded.
            if placed == 0:
                buf.add_segment(toks[:seq_len], doc, stream.epoch(d), 0)
                d += 1
                placed += 1
            break
        if len(buf.input_ids) + len(toks) > seq_len:
            break
        buf.add_segment(toks, doc, stream.epoch(d), 0)
        d += 1
        placed += 1
        if len(buf.input_ids) == seq_len:
            break
    buf.pad_to(seq_len)
    return _finalize(lane, WHOLE_UNIT, buf, cursor, {"doc": d, "off": 0}, seq_len)


def _pack_prompt_completion(lane, stream, cursor, seq_len, token_provider) -> dict:
    buf = _Buf()
    d = cursor["doc"]
    doc = stream.doc_at(d)
    toks = _doc_tokens(doc, token_provider)[:seq_len]
    prompt_tokens = min(doc.get("prompt_tokens", 0), len(toks))
    buf.add_segment(toks, doc, stream.epoch(d), 0, prompt_tokens=prompt_tokens)
    buf.pad_to(seq_len)
    return _finalize(lane, PROMPT_COMPLETION, buf, cursor, {"doc": d + 1, "off": 0},
                     seq_len)


def _finalize(lane, policy, buf: _Buf, cursor_before, cursor_after, seq_len) -> dict:
    n_real = sum(1 for s in buf.segment_ids if s >= 0)
    n_loss = sum(buf.loss_mask)
    n_prompt_masked = sum(min(s["prompt_tokens"], s["seg_len"]) for s in buf.segments)
    sample = {
        "lane": lane,
        "policy": policy,
        "input_ids": buf.input_ids,
        "segment_ids": buf.segment_ids,
        "position_ids": buf.position_ids,
        "loss_mask": buf.loss_mask,
        "segments": buf.segments,
        "n_real_tokens": n_real,
        "n_loss_tokens": n_loss,
        "n_pad_tokens": seq_len - n_real,
        "n_prompt_masked_tokens": n_prompt_masked,
        "cursor_before": dict(cursor_before),
        "cursor_after": dict(cursor_after),
    }
    # The sample hash binds token content *and* provenance, so replay can only
    # match if it re-read the same bytes from the same spans.
    sample["sample_hash"] = sha256_text(canonical_json({
        "lane": lane,
        "policy": policy,
        "input_ids": buf.input_ids,
        "loss_mask": buf.loss_mask,
        "position_ids": buf.position_ids,
        "segments": [
            {"doc_id": s["doc_id"], "shard_id": s["shard_id"],
             "token_start": s["token_start"], "token_end": s["token_end"],
             "seg_start": s["seg_start"], "seg_len": s["seg_len"]}
            for s in buf.segments
        ],
    }))
    return sample


def verify_sample_invariants(sample: dict) -> List[str]:
    """Independent checker for every packing invariant. Empty list == correct."""
    errs: List[str] = []
    L = len(sample["input_ids"])
    if not (len(sample["segment_ids"]) == len(sample["position_ids"]) ==
            len(sample["loss_mask"]) == L):
        errs.append("array length mismatch")
        return errs

    # 1. position ids reset at each segment start and increment inside a segment
    seen: Dict[int, int] = {}
    for i in range(L):
        seg = sample["segment_ids"][i]
        if seg not in seen:
            seen[seg] = i
            if sample["position_ids"][i] != 0:
                errs.append(f"position not reset at segment start {i}")
        else:
            if sample["position_ids"][i] != sample["position_ids"][i - 1] + 1:
                errs.append(f"position not contiguous at {i}")
            if sample["segment_ids"][i - 1] != seg:
                errs.append(f"segment {seg} is not contiguous at {i}")

    # 2. no loss on padding or on a segment's first token
    for i in range(L):
        if sample["loss_mask"][i] == 1:
            if sample["segment_ids"][i] < 0:
                errs.append(f"loss on padding at {i}")
            if sample["position_ids"][i] == 0:
                errs.append(f"loss on segment-initial token at {i}")

    # 3. padding is contiguous at the tail, uses PAD_ID, and self-attends only
    pad_positions = [i for i in range(L) if sample["segment_ids"][i] < 0]
    if pad_positions:
        if pad_positions != list(range(pad_positions[0], L)):
            errs.append("padding is not a contiguous tail")
        if len(set(sample["segment_ids"][i] for i in pad_positions)) != len(pad_positions):
            errs.append("pad positions share segment ids (would attend to each other)")
        for i in pad_positions:
            if sample["input_ids"][i] != PAD_ID:
                errs.append(f"non-PAD token at pad position {i}")

    # 4. prompt prefixes are fully loss-masked
    for s in sample["segments"]:
        for k in range(min(s["prompt_tokens"], s["seg_len"])):
            idx = s["seg_start"] + k
            if sample["loss_mask"][idx] == 1:
                errs.append(f"loss on prompt token {idx}")

    # 5. declared counts agree with the arrays
    if sample["n_real_tokens"] != sum(1 for s in sample["segment_ids"] if s >= 0):
        errs.append("n_real_tokens disagrees with segment ids")
    if sample["n_loss_tokens"] != sum(sample["loss_mask"]):
        errs.append("n_loss_tokens disagrees with loss mask")

    # 6. policy-specific guarantees
    if sample["policy"] == CONCAT and sample["n_real_tokens"] != L:
        errs.append("concat policy left padding in the sequence")
    if sample["policy"] == WHOLE_UNIT:
        for s in sample["segments"]:
            whole = s["doc_token_end"] - s["doc_token_start"]
            if s["seg_len"] != whole and whole <= L:
                errs.append(f"whole_unit split a unit ({s['doc_id']})")
    if sample["policy"] == PROMPT_COMPLETION and len(sample["segments"]) != 1:
        errs.append("prompt_completion packed more than one trajectory")
    return errs
