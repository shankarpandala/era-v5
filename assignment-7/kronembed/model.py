"""KronGPT — assignment-6's TinyGPT adapted for pluggable embeddings.

Differences from the assignment-6 model, each forced by the experiment design
and applied identically to every arm:

  * NO weight tying. Three of five arms have no learnable embedding table, so
    the output heads are independent modules. "Identical architecture apart
    from the embedding" is enforced by ``arch_hash`` — a content hash of every
    non-embedding parameter's (name, shape).
  * Dual answer heads read the hidden state at the <ans> position:
      - regression head (primary) reads the PRE-LayerNorm residual stream and
        predicts (value/TARGET_SCALE, log10(value+1)). LayerNorm divides by a
        per-example std, which would corrupt the exactly-linear signal the
        homomorphic LIN dim carries through the residual connections.
      - classification head reads the post-LN stream, softmax over the vocab.
        It cannot express answers above the largest integer token — that
        structural ceiling is part of what the experiment measures.
  * A learnable scalar ``input_gain`` lets each arm adapt the overall scale of
    its (possibly frozen) embeddings without touching their content.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import sha256_json


# ---------------------------------------------------------------------------
# Embedding providers — the ONLY thing that differs between experiment arms
# ---------------------------------------------------------------------------


class FrozenEmbedding(nn.Module):
    """A deterministic matrix registered as a buffer: no gradients, ever."""

    def __init__(self, matrix: np.ndarray):
        super().__init__()
        self.register_buffer("matrix", torch.from_numpy(matrix.copy()))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.matrix[ids]


class LearnedEmbedding(nn.Module):
    """Standard trainable lookup table (the conventional baseline)."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids)


class XValEmbedding(nn.Module):
    """xVal-style (Golkar et al. 2023): numeric tokens share one learned
    direction scaled by the token's normalized value; other tokens use a
    learned table. Gives the learned family a genuine value-extrapolation
    mechanism (linear in v), unlike untrained rows."""

    def __init__(self, vocab_size: int, d_model: int, values: np.ndarray,
                 v_max: float):
        super().__init__()
        self.base = nn.Embedding(vocab_size, d_model)
        self.num_direction = nn.Parameter(torch.randn(d_model) * 0.02)
        is_num = ~np.isnan(values)
        scaled = np.where(is_num, values / v_max, 0.0).astype(np.float32)
        self.register_buffer("scaled_value", torch.from_numpy(scaled))
        self.register_buffer("is_num", torch.from_numpy(is_num))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        base = self.base(ids)
        val = self.scaled_value[ids].unsqueeze(-1) * self.num_direction
        return torch.where(self.is_num[ids].unsqueeze(-1), val, base)


# ---------------------------------------------------------------------------
# The transformer
# ---------------------------------------------------------------------------


class KronGPT(nn.Module):
    def __init__(self, vocab_size: int, embedding: nn.Module, d_model: int = 128,
                 n_layer: int = 2, n_head: int = 4, max_pos: int = 16):
        super().__init__()
        self.d_model = d_model
        self.embedding = embedding
        self.input_gain = nn.Parameter(torch.ones(()))
        self.pos_emb = nn.Embedding(max_pos, d_model)
        self.blocks = nn.ModuleList(
            [_Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, vocab_size, bias=False)
        # 15 outputs = the numeric block of the answer's embedding:
        # [value/TARGET_SCALE (signed), log10(|c|+1), sign(c),
        #  6 x (sin, cos) Fourier phases].
        # Two parallel paths: a pure linear map (can extrapolate a linear
        # signal beyond the training range — a GELU MLP saturates there) plus
        # an MLP correction for in-range precision.
        self.reg_lin = nn.Linear(d_model, 15)
        self.reg_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 15))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def trunk_layers(self, ids: torch.Tensor) -> list:
        """Residual stream at every depth: [post-embedding, after block 1,
        ..., after block N]. Used by the per-layer probe analysis that asks
        WHERE linearly decodable structure is lost."""
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0)
        x = self.embedding(ids) * self.input_gain + self.pos_emb(pos)
        states = [x]
        mask = _causal_mask(L, ids.device)
        for blk in self.blocks:
            x = blk(x, mask)
            states.append(x)
        return states

    def trunk(self, ids: torch.Tensor) -> torch.Tensor:
        """Pre-LN residual stream after all blocks."""
        return self.trunk_layers(ids)[-1]

    def forward(self, ids: torch.Tensor) -> dict:
        x = self.trunk(ids)
        return {
            "reg": self.reg_lin(x) + self.reg_mlp(x),  # pre-LN residual stream
            "cls_logits": self.cls_head(self.ln_f(x)),
        }

    def arch_hash(self) -> str:
        """Hash of every non-embedding parameter's (name, shape): equal across
        arms iff the architectures are identical apart from the embedding."""
        shapes = sorted(
            (name, list(p.shape))
            for name, p in self.named_parameters()
            if not name.startswith("embedding."))
        return sha256_json(shapes)


class _Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = _SelfAttention(d_model, n_head)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model))

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class _SelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.hd = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_head, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        att = att + mask
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(y)


def _causal_mask(L: int, device) -> torch.Tensor:
    idx = torch.arange(L, device=device)
    allowed = idx.unsqueeze(0) <= idx.unsqueeze(1)  # key <= query
    mask = torch.zeros(L, L, device=device)
    return mask.masked_fill(~allowed, float("-inf"))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def compute_loss(out: dict, batch: dict, weights: dict) -> dict:
    """Joint answer loss at each example's own <ans> position.

    The regression head is supervised on the answer's OWN numeric-block
    features (signed value, log-magnitude, sign, Fourier phases), so training
    teaches the model to emit the answer's embedding. ``batch["ans_pos"]`` is
    per-example — NL templates place <ans> at varying positions. The CE term
    skips batches whose every answer is outside the token vocabulary
    (large products, negative differences)."""
    idx = batch["ans_pos"]
    rows = torch.arange(idx.shape[0], device=idx.device)
    reg = out["reg"][rows, idx, :]
    loss_lin = F.huber_loss(reg[:, 0], batch["y_lin"])
    loss_log = F.huber_loss(reg[:, 1], batch["y_log"])
    loss_sign = F.mse_loss(reg[:, 2], batch["y_sign"])
    loss_fourier = F.mse_loss(reg[:, 3:], batch["y_fourier"])
    y_cls = batch["y_cls"]
    if (y_cls != -100).any():
        loss_cls = F.cross_entropy(out["cls_logits"][rows, idx, :], y_cls,
                                   ignore_index=-100)
    else:
        loss_cls = torch.zeros((), device=reg.device)
    total = (weights["w_lin"] * loss_lin + weights["w_log"] * loss_log
             + weights["w_sign"] * loss_sign
             + weights["w_fourier"] * loss_fourier + weights["w_cls"] * loss_cls)
    return {"total": total, "lin": loss_lin.detach(), "log": loss_log.detach(),
            "sign": loss_sign.detach(), "fourier": loss_fourier.detach(),
            "cls": loss_cls.detach()}
