"""Reversibility is exact, its gradients are the autograd gradients, and it saves memory in depth."""
from __future__ import annotations

import copy
import dataclasses
import math

import pytest
import torch

SMALL = dict(vocab_size=64, seq_len=16, d_model=32, n_layer=4, n_head=4)
REVERSIBLE = ("euler", "midpoint", "momentum")


def _batch(R, cfg, n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, cfg.vocab_size, (n, cfg.seq_len), generator=g)
    y = torch.randint(0, cfg.vocab_size, (n, cfg.seq_len), generator=g)
    return x, y


@pytest.mark.parametrize("variant", REVERSIBLE)
def test_reversible_backward_equals_plain_autograd(R, variant):
    cfg = R.ModelConfig(reversible=variant, **SMALL)
    model = R.build_model(cfg, torch.device("cpu"), seed=1).double()
    twin = copy.deepcopy(model)
    twin._materialize = True  # same recurrence, activations stored, plain autograd
    x, y = _batch(R, cfg)
    _, l1 = model(x, y); l1.backward()
    _, l2 = twin(x, y); l2.backward()
    assert l1.item() == pytest.approx(l2.item(), abs=1e-12)
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), twin.named_parameters()):
        assert n1 == n2
        assert p1.grad is not None and p2.grad is not None, n1
        assert torch.allclose(p1.grad, p2.grad, rtol=1e-9, atol=1e-12), n1


@pytest.mark.parametrize("variant", REVERSIBLE)
def test_inverse_reconstructs_the_input_state(R, variant):
    cfg = R.ModelConfig(reversible=variant, **SMALL)
    model = R.build_model(cfg, torch.device("cpu"), seed=2)
    x, _ = _batch(R, cfg)
    h = model.wte(x) + model.wpe(torch.arange(cfg.seq_len))
    rec = model.reconstruction_error(h)
    assert rec["max_rel_error"] < 1e-5
    expected = {"euler": 2, "midpoint": 1, "momentum": 2}[variant] * cfg.n_layer
    assert rec["half_steps"] == expected


def test_baseline_is_not_invertible(R):
    model = R.build_model(R.ModelConfig(**SMALL), torch.device("cpu"))
    with pytest.raises(ValueError):
        model.reconstruction_error(torch.zeros(1, SMALL["seq_len"], SMALL["d_model"]))


def test_saved_activation_bytes_flat_in_depth_for_reversible_linear_for_baseline(R):
    base = R.ModelConfig(**SMALL)
    rows = R.depth_sweep_saved_bytes(base, depths=(2, 4, 8), batch=2, device=torch.device("cpu"))
    by = {(r["variant"], r["n_layer"]): r["activation_bytes"] for r in rows}
    for variant in REVERSIBLE:
        assert by[(variant, 2)] == by[(variant, 4)] == by[(variant, 8)], variant
    b2, b4, b8 = by[("none", 2)], by[("none", 4)], by[("none", 8)]
    assert b4 > b2 and b8 > b4
    per_layer = (b8 - b4) / 4
    assert per_layer == pytest.approx((b4 - b2) / 2, rel=1e-6)  # exactly linear
    assert by[("euler", 8)] < b8 / 4


def test_reversible_variants_keep_the_baseline_parameter_count(R):
    base = R.GPT(R.ModelConfig(**SMALL)).num_params()
    for variant in REVERSIBLE:
        assert R.GPT(R.ModelConfig(reversible=variant, **SMALL)).num_params() == base


def test_full_size_model_is_about_twenty_million_parameters(R):
    counts = R.GPT(R.ModelConfig()).num_params()
    assert 19_000_000 <= counts["total"] <= 22_000_000
    assert counts["non_embedding"] > 0.8 * counts["total"]


def test_unknown_variant_rejected(R):
    with pytest.raises(ValueError):
        R.ModelConfig(reversible="rk4")


def test_midpoint_with_unit_residual_scale_matches_block_level_recurrence(R):
    """2h = 1: x[n+1] = x[n-1] + Delta_n(x[n]); checked against a hand-written loop."""
    cfg = R.ModelConfig(reversible="midpoint", midpoint_h=0.5, **SMALL)
    model = R.build_model(cfg, torch.device("cpu"), seed=3)
    x, _ = _batch(R, cfg)
    with torch.no_grad():
        h = model.wte(x) + model.wpe(torch.arange(cfg.seq_len))
        prev, cur = h, h
        for block in model.blocks:
            prev, cur = cur, prev + block.delta(cur)
        assert torch.allclose(model.stack(h), cur, atol=1e-6)


def test_momentum_recurrence_matches_hand_written_loop(R):
    cfg = R.ModelConfig(reversible="momentum", momentum_gamma=0.7, **SMALL)
    model = R.build_model(cfg, torch.device("cpu"), seed=4)
    x, _ = _batch(R, cfg)
    with torch.no_grad():
        h = model.wte(x) + model.wpe(torch.arange(cfg.seq_len))
        v, pos = torch.zeros_like(h), h
        for block in model.blocks:
            v = 0.7 * v + 0.3 * block.delta(pos)
            pos = pos + v
        assert torch.allclose(model.stack(h), pos, atol=1e-6)


def test_euler_coupling_matches_hand_written_loop(R):
    cfg = R.ModelConfig(reversible="euler", **SMALL)
    model = R.build_model(cfg, torch.device("cpu"), seed=5)
    x, _ = _batch(R, cfg)
    with torch.no_grad():
        h = model.wte(x) + model.wpe(torch.arange(cfg.seq_len))
        y1, y2 = h, h
        for block in model.blocks:
            y1 = y1 + block.f_attn(y2)
            y2 = y2 + block.f_mlp(y1)
        assert torch.allclose(model.stack(h), y1 + y2, atol=1e-6)


def test_generation_extends_and_stops_at_eot(R):
    cfg = R.ModelConfig(reversible="euler", **SMALL)
    model = R.build_model(cfg, torch.device("cpu"), seed=6)
    out = model.generate(torch.zeros(1, 3, dtype=torch.long), 5, temperature=0.0)
    assert out.shape[1] <= 8 and out.shape[1] >= 4
