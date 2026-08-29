"""Task-4/5 invariants: clipping semantics, the detectors, FLOP accounting."""
import math

import torch


def test_clip_preserves_direction_and_caps_norm(nb):
    torch.manual_seed(9)
    m = nb.TinyLM(d=32, n_layer=1, n_head=2)
    tb = torch.tensor([nb.hazard_doc(9, 12)])
    nb.token_weighted_loss(m(tb), tb).backward()
    before = torch.cat([p.grad.flatten().double() for p in m.parameters()])
    pre = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1e-3))
    after = torch.cat([p.grad.flatten().double() for p in m.parameters()])
    cos = float((before @ after) / (before.norm() * after.norm()))
    assert abs(cos - 1.0) < 1e-6, "clipping must preserve direction"
    assert abs(nb.global_grad_norm(m) - 1e-3) < 1e-6, "post-clip norm == cap"
    assert pre > 1e-3, "clip_grad_norm_ returns the PRE-clip norm"


def test_session_clip_arithmetic():
    assert abs(min(1.0, 1.0 / 8.4) - 0.119) < 5e-4


def test_norm_spike_detector(nb):
    flat = [1.0] * 100
    assert nb.detect_norm_spike(flat, burn_in=40) is None
    spiked = flat.copy()
    spiked[70] = 8.0
    assert nb.detect_norm_spike(spiked, burn_in=40) == 70


def test_probe_damage_detector(nb):
    flat = [2.0] * 100
    assert nb.detect_probe_damage(flat, burn_in=40) is None
    hurt = flat.copy()
    for j in range(70, 80):
        hurt[j] = 2.5
    assert nb.detect_probe_damage(hurt, burn_in=40) == 70


def test_clip_grad_norm_inf_never_scales(nb):
    torch.manual_seed(10)
    m = nb.TinyLM(d=32, n_layer=1, n_head=2)
    tb = torch.tensor([nb.hazard_doc(5, 10)])
    nb.token_weighted_loss(m(tb), tb).backward()
    g0 = [p.grad.clone() for p in m.parameters()]
    norm = float(torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf")))
    assert all(torch.equal(a, p.grad) for a, p in zip(g0, m.parameters()))
    assert abs(norm - nb.global_grad_norm(m)) < 1e-4


def test_exact_flops_formula(nb):
    d, T, L, V = 8, 4, 1, 11
    # by hand: linears = 12*1*64 + 88 = 856; attention = 12*1*4*8 = 384
    assert nb.flops_per_token_exact(d, T, L, V) == 6 * 856 + 384
    assert nb.flops_forward_matmul(d, T, L, V) == 2 * 856 + 128


def test_profiler_flops_match_analytic_tiny(nb):
    from torch.profiler import ProfilerActivity, profile
    torch.manual_seed(11)
    m = nb.TinyLM(d=32, n_layer=1, n_head=2)
    tb = torch.tensor([nb.hazard_doc(9, 12)] * 2)
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        with torch.no_grad():
            m(tb)
    measured = sum(e.flops for e in prof.key_averages() if e.flops)
    analytic = nb.flops_forward_matmul(32, 12, 1, nb.VOCAB) * 2 * 12
    # at tiny shapes the profiler's op bookkeeping differs from the analytic count
    # by a fraction of a percent; the notebook checks < 1e-3 at the real shapes
    assert abs(measured - analytic) / analytic < 1e-2


def test_worked_example():
    assert abs(6 * 9e9 * 12_000 / (8 * 989e12) - 0.0819009) < 1e-4


def test_sixteen_bytes_per_weight(nb):
    torch.manual_seed(12)
    m = nb.TinyLM(d=32, n_layer=1, n_head=2)
    n = nb.count_params(m)
    w = sum(p.detach().to(torch.bfloat16).numel() * 2 for p in m.parameters())
    total = w * 2 + n * 4 + n * 8      # bf16 w + bf16 g, fp32 master, 2 moments
    assert total == 16 * n
