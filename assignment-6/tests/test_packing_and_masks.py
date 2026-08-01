"""Packing policies, loss masks, attention masks and position ids."""

from __future__ import annotations

import os

import torch

from datasys.batcher import Batcher
from datasys.model import _block_causal_mask, compute_loss
from datasys.packing import (CONCAT, PROMPT_COMPLETION, WHOLE_UNIT,
                             verify_sample_invariants)
from datasys.tokenizer import PAD_ID

from .fixtures import SEQ_LEN, build_session


def _all_samples(tmp_path, steps=8):
    session = build_session(str(tmp_path))
    b = Batcher("t", session["schedule"], session["inventory"],
                os.path.join(str(tmp_path), "manifests"), SEQ_LEN)
    out = []
    for s in range(steps):
        out.extend(b.build_step(s)["samples"])
    return out


def test_every_packed_sample_satisfies_all_invariants(tmp_path):
    samples = _all_samples(tmp_path)
    assert samples
    for s in samples:
        assert verify_sample_invariants(s) == [], (s["lane"], s["policy"])


def test_all_three_policies_are_exercised(tmp_path):
    policies = {s["policy"] for s in _all_samples(tmp_path)}
    assert policies == {CONCAT, WHOLE_UNIT, PROMPT_COMPLETION}


def test_concat_policy_leaves_no_padding(tmp_path):
    for s in _all_samples(tmp_path):
        if s["policy"] == CONCAT:
            assert s["n_pad_tokens"] == 0
            assert PAD_ID not in s["input_ids"] or s["input_ids"].count(PAD_ID) == 0


def test_whole_unit_policy_never_splits_a_unit(tmp_path):
    for s in _all_samples(tmp_path):
        if s["policy"] != WHOLE_UNIT:
            continue
        for seg in s["segments"]:
            whole = seg["doc_token_end"] - seg["doc_token_start"]
            if whole <= SEQ_LEN:
                assert seg["seg_len"] == whole, "a code unit was split"


def test_prompt_tokens_are_never_loss_bearing(tmp_path):
    seen_prompt = False
    for s in _all_samples(tmp_path):
        if s["policy"] != PROMPT_COMPLETION:
            continue
        for seg in s["segments"]:
            n = min(seg["prompt_tokens"], seg["seg_len"])
            if n:
                seen_prompt = True
            for k in range(n):
                assert s["loss_mask"][seg["seg_start"] + k] == 0
    assert seen_prompt, "no prompt-masked trajectory was packed"


def test_position_ids_reset_per_segment(tmp_path):
    for s in _all_samples(tmp_path):
        seen = set()
        for i, seg in enumerate(s["segment_ids"]):
            if seg not in seen:
                seen.add(seg)
                assert s["position_ids"][i] == 0


def test_attention_mask_blocks_cross_segment_and_future(tmp_path):
    samples = _all_samples(tmp_path, steps=2)
    seg = torch.tensor([s["segment_ids"] for s in samples[:4]], dtype=torch.long)
    mask = _block_causal_mask(seg)[:, 0]  # (B, L, L)
    B, L, _ = mask.shape
    for b in range(B):
        for q in range(0, L, 7):          # sample the grid; full L^2 is slow
            for k in range(0, L, 5):
                allowed = mask[b, q, k].item() == 0.0
                same = seg[b, q].item() == seg[b, k].item()
                causal = k <= q
                expected = (same and causal) or (q == k)
                assert allowed == expected, (b, q, k)


def test_no_attention_row_is_entirely_masked(tmp_path):
    """Every query must attend to at least itself, or softmax produces NaN."""
    samples = _all_samples(tmp_path, steps=2)
    seg = torch.tensor([s["segment_ids"] for s in samples[:4]], dtype=torch.long)
    mask = _block_causal_mask(seg)[:, 0]
    assert torch.isfinite(mask).any(dim=-1).all()


def test_padding_never_contributes_to_loss(tmp_path):
    """Changing only pad positions must not change the loss."""
    samples = _all_samples(tmp_path, steps=3)
    pad_sample = next(s for s in samples if s["n_pad_tokens"] > 0)
    ids = torch.tensor([pad_sample["input_ids"]], dtype=torch.long)
    mask = torch.tensor([pad_sample["loss_mask"]], dtype=torch.long)
    vocab = 1024
    torch.manual_seed(0)
    logits = torch.randn(1, ids.shape[1], vocab)
    base, _, _ = compute_loss(logits, ids, mask)

    tampered = ids.clone()
    for i, seg in enumerate(pad_sample["segment_ids"]):
        if seg < 0:
            tampered[0, i] = vocab - 1  # arbitrary non-PAD token id
    after, _, _ = compute_loss(logits, tampered, mask)
    assert torch.allclose(base, after)


def test_loss_token_counts_match_the_mask(tmp_path):
    for s in _all_samples(tmp_path):
        assert s["n_loss_tokens"] == sum(s["loss_mask"])
        assert s["n_real_tokens"] == sum(1 for x in s["segment_ids"] if x >= 0)
        assert s["n_pad_tokens"] == len(s["input_ids"]) - s["n_real_tokens"]


def test_packing_is_a_pure_function_of_the_cursor(tmp_path):
    """Same schedule + same cursor => byte-identical samples."""
    session = build_session(str(tmp_path))
    shard_dir = os.path.join(str(tmp_path), "manifests")
    a = Batcher("t", session["schedule"], session["inventory"], shard_dir, SEQ_LEN)
    b = Batcher("t", session["schedule"], session["inventory"], shard_dir, SEQ_LEN)
    for step in range(6):
        ba, bb = a.build_step(step), b.build_step(step)
        assert ba["batch_id"] == bb["batch_id"]
        assert [s["sample_hash"] for s in ba["samples"]] == \
               [s["sample_hash"] for s in bb["samples"]]


def test_sample_level_loss_reconciles_with_lane_loss_and_mask(tmp_path):
    """per_sample_loss tokens must sum to the batch's loss-token count."""
    from datasys.model import compute_loss, per_lane_loss, per_sample_loss

    session = build_session(str(tmp_path))
    b = Batcher("t", session["schedule"], session["inventory"],
                os.path.join(str(tmp_path), "manifests"), SEQ_LEN)
    batch = b.build_step(0)
    input_ids = torch.tensor([s["input_ids"] for s in batch["samples"]], dtype=torch.long)
    loss_mask = torch.tensor([s["loss_mask"] for s in batch["samples"]], dtype=torch.long)
    torch.manual_seed(0)
    logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 1024)
    _, per_tok, flat_mask = compute_loss(logits, input_ids, loss_mask)
    lanes = []
    for s in batch["samples"]:
        lanes.extend([s["lane"]] * (SEQ_LEN - 1))
    lane = per_lane_loss(per_tok, flat_mask, lanes)
    sample = per_sample_loss(per_tok, flat_mask, batch["samples"], SEQ_LEN)
    assert sum(s["tokens"] for s in sample) == batch["n_loss_tokens"]
    assert sum(v["tokens"] for v in lane.values()) == batch["n_loss_tokens"]
    assert {s["sample_hash"] for s in sample} == {s["sample_hash"] for s in batch["samples"]}
