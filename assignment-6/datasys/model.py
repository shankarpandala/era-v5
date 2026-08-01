"""A tiny GPT that honours packed attention masks, position ids and loss masks.

Small on purpose (two layers, d_model 64 in the demonstration). What matters is
that it consumes exactly the packed structure the data system produces:

  * ``position_ids`` (reset per segment) drive the positional embedding, so a
    packed document behaves as if it started at position 0.
  * ``segment_ids`` build a block-diagonal *causal* attention mask: a position
    attends only to earlier positions in the *same* segment. No cross-document
    leakage, and pad rows (unique negative segment ids) self-attend so softmax is
    well-defined.
  * ``loss_mask`` selects which next-token predictions contribute to the loss, so
    prompt tokens / pad / segment-initial tokens never train the model.

Per-lane loss is computed by grouping loss positions by the lane of their sample,
which is what links the learning ledger back to source data.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_layer: int = 2,
                 n_head: int = 4, max_pos: int = 512, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_pos, d_model)
        self.blocks = nn.ModuleList([_Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """GPT-style init. Without it the tied output head produces logits of
        order sqrt(d_model), and step-0 loss lands far above ln(vocab) -- which
        would make the learning ledger's numbers meaningless."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, segment_ids, position_ids):
        # input_ids: (B, L)
        B, L = input_ids.shape
        x = self.tok_emb(input_ids) + self.pos_emb(position_ids.clamp(min=0))
        attn_mask = _block_causal_mask(segment_ids)  # (B, 1, L, L) additive
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.ln_f(x)
        return self.head(x)  # logits (B, L, V)


class _Block(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = _SelfAttention(d_model, n_head, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class _SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.hd = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_head, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, L, hd)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)  # (B,H,L,L)
        att = att + attn_mask  # additive -inf where disallowed
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v  # (B,H,L,hd)
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(y)


def _block_causal_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """Additive attention mask: 0 where allowed, -inf where disallowed.

    Allowed iff same segment id and key position <= query position. Pad tokens
    carry unique negative segment ids so each only attends to itself.
    """
    B, L = segment_ids.shape
    seg = segment_ids.unsqueeze(-1)  # (B,L,1)
    same = seg == seg.transpose(1, 2)  # (B,L,L): [b,q,k] same segment?
    idx = torch.arange(L, device=segment_ids.device)
    # [q,k] is allowed only when key k is at or before query q
    causal = idx.unsqueeze(0) <= idx.unsqueeze(1)  # (L,L)
    allowed = same & causal.unsqueeze(0)
    # guarantee at least the diagonal is allowed (self-attention) to avoid NaN
    diag = torch.eye(L, dtype=torch.bool, device=segment_ids.device).unsqueeze(0)
    allowed = allowed | diag
    mask = torch.zeros(B, L, L, device=segment_ids.device)
    mask = mask.masked_fill(~allowed, float("-inf"))
    return mask.unsqueeze(1)  # (B,1,L,L)


def compute_loss(logits, input_ids, loss_mask):
    """Causal next-token loss over positions where loss_mask==1.

    Prediction of position t uses context < t (enforced by the attention mask);
    the target for position t is input_ids[t]. We therefore compare logits at t-1
    against input_ids[t]. loss_mask[t]==1 marks the target positions that count.
    """
    B, L, V = logits.shape
    # shift: logits at positions 0..L-2 predict tokens at 1..L-1
    pred = logits[:, :-1, :].reshape(-1, V)
    target = input_ids[:, 1:].reshape(-1)
    mask = loss_mask[:, 1:].reshape(-1).float()
    per_tok = F.cross_entropy(pred, target, reduction="none")
    masked = per_tok * mask
    denom = mask.sum().clamp(min=1.0)
    return masked.sum() / denom, per_tok, mask


def per_lane_loss(per_tok, mask, lane_of_position) -> Dict[str, dict]:
    """Aggregate masked loss by lane. ``lane_of_position`` is a flat list aligned
    with the shifted (B, L-1) positions giving each position's lane."""
    out: Dict[str, dict] = {}
    per_tok = per_tok.detach()
    mask = mask.detach()
    for i, lane in enumerate(lane_of_position):
        if mask[i].item() <= 0:
            continue
        rec = out.setdefault(lane, {"loss_sum": 0.0, "tokens": 0})
        rec["loss_sum"] += float(per_tok[i].item())
        rec["tokens"] += 1
    for lane, rec in out.items():
        rec["mean_loss"] = rec["loss_sum"] / max(1, rec["tokens"])
    return out


def per_sample_loss(per_tok, mask, samples: List[dict], seq_len: int) -> List[dict]:
    """Sample-level loss linked to source data (sample_hash + document spans).

    ``per_tok`` / ``mask`` are flat over B x (L-1). Each sample occupies a contiguous
    block of (seq_len - 1) shifted positions. This is the assignment's
    sample-level loss trace: every learning entry can point at the exact packed
    sample (and therefore the shard spans) that produced that loss.
    """
    per_tok = per_tok.detach()
    mask = mask.detach()
    out: List[dict] = []
    width = seq_len - 1
    for i, s in enumerate(samples):
        start = i * width
        end = start + width
        loss_sum = 0.0
        tokens = 0
        for j in range(start, end):
            if mask[j].item() <= 0:
                continue
            loss_sum += float(per_tok[j].item())
            tokens += 1
        out.append({
            "slot": s["slot"],
            "lane": s["lane"],
            "policy": s["policy"],
            "sample_hash": s["sample_hash"],
            "loss_sum": loss_sum,
            "tokens": tokens,
            "mean_loss": loss_sum / max(1, tokens),
            "source_docs": sorted({seg["doc_id"] for seg in s["segments"]}),
        })
    return out
