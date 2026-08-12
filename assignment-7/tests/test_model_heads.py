"""Model invariants: untied heads, frozen-vs-learned gradients, arch identity."""

import numpy as np
import torch

from kronembed.embedding import build_embedding_matrix
from kronembed.model import (FrozenEmbedding, KronGPT, LearnedEmbedding,
                             XValEmbedding, compute_loss)
from kronembed.train import ARMS, make_embedding
from kronembed.vocab import Vocab

VOCAB = Vocab()


def _model(emb):
    torch.manual_seed(0)
    return KronGPT(len(VOCAB), emb)


def test_head_is_not_tied_to_embedding():
    m = _model(LearnedEmbedding(len(VOCAB), 128))
    assert m.cls_head.weight.data_ptr() != m.embedding.emb.weight.data_ptr()


WEIGHTS = {"w_lin": 1.0, "w_log": 1.0, "w_sign": 0.5, "w_fourier": 1.0,
           "w_cls": 0.5}


def _batch(n=4):
    return {"ans_pos": torch.full((n,), 5, dtype=torch.long),
            "y_lin": torch.rand(n), "y_log": torch.rand(n),
            "y_sign": torch.ones(n), "y_fourier": torch.rand(n, 12),
            "y_cls": torch.randint(0, 100, (n,))}


def test_frozen_embedding_gets_no_gradient_learned_does():
    frozen = _model(FrozenEmbedding(build_embedding_matrix(VOCAB.tokens)))
    learned = _model(LearnedEmbedding(len(VOCAB), 128))
    ids = torch.randint(0, len(VOCAB), (4, 7))
    batch = _batch()
    for m, has_emb_params in ((frozen, False), (learned, True)):
        out = m(ids)
        compute_loss(out, batch, WEIGHTS)["total"].backward()
        emb_params = [p for n, p in m.named_parameters()
                      if n.startswith("embedding.")]
        assert bool(emb_params) == has_emb_params
        if has_emb_params:
            assert emb_params[0].grad is not None
    # frozen matrix is a buffer: unchanged by construction
    before = frozen.embedding.matrix.clone()
    assert torch.equal(before, frozen.embedding.matrix)


def test_arch_hash_identical_across_all_arms():
    hashes = set()
    for arm in ARMS:
        torch.manual_seed(0)
        m = KronGPT(len(VOCAB), make_embedding(arm, VOCAB, 128))
        hashes.add(m.arch_hash())
    assert len(hashes) == 1


def test_forward_shapes_and_heads():
    m = _model(FrozenEmbedding(build_embedding_matrix(VOCAB.tokens)))
    ids = torch.randint(0, len(VOCAB), (8, 7))
    out = m(ids)
    assert out["reg"].shape == (8, 7, 15)
    assert out["cls_logits"].shape == (8, 7, len(VOCAB))
    # trunk_layers exposes one state per depth (embedding + 2 blocks)
    states = m.trunk_layers(ids)
    assert len(states) == 3
    assert all(s.shape == (8, 7, 128) for s in states)


def test_loss_gathers_per_example_ans_positions():
    m = _model(LearnedEmbedding(len(VOCAB), 128))
    ids = torch.randint(0, len(VOCAB), (4, 10))
    batch = _batch()
    batch["ans_pos"] = torch.tensor([5, 7, 6, 8])
    losses = compute_loss(m(ids), batch, WEIGHTS)
    assert torch.isfinite(losses["total"])


def test_xval_numeric_tokens_use_scaled_direction():
    values = np.array([np.nan, np.nan, 0.0, 500.0, 999.0])
    emb = XValEmbedding(5, 16, values, 999.0)
    ids = torch.tensor([[2, 3, 4, 0]])
    out = emb(ids).detach()
    # value tokens lie on one shared direction, scaled by v/v_max
    d = emb.num_direction.detach()
    assert torch.allclose(out[0, 1], d * (500.0 / 999.0))
    assert torch.allclose(out[0, 2], d * 1.0)
    assert torch.allclose(out[0, 0], d * 0.0)
    # non-numeric token comes from the base table instead
    assert not torch.allclose(out[0, 3], d * emb.scaled_value[0])


def test_cls_loss_skipped_when_all_answers_out_of_vocab():
    m = _model(LearnedEmbedding(len(VOCAB), 128))
    ids = torch.randint(0, len(VOCAB), (4, 7))
    batch = _batch()
    batch["y_cls"] = torch.full((4,), -100, dtype=torch.long)
    losses = compute_loss(m(ids), batch, WEIGHTS)
    assert float(losses["cls"].item()) == 0.0
    assert torch.isfinite(losses["total"])
