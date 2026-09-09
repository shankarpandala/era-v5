"""The hand optimizers against torch, and the identities the notebook leans on."""
import copy
import math

import torch


def _pair(nb, d=64, seed=7):
    mA = nb.make_model(d, seed=seed, device="cpu")
    return mA, copy.deepcopy(mA), nb.decay_mask(mA)


def test_hand_adamw_bitwise_vs_torch(nb):
    mA, mB, dm = _pair(nb)
    optA = nb.HandAdamW(mA.parameters(), 1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, decay=dm)
    optB = torch.optim.AdamW([{"params": [p for p, d in zip(mB.parameters(), dm) if d], "weight_decay": 0.1},
                              {"params": [p for p, d in zip(mB.parameters(), dm) if not d], "weight_decay": 0.0}],
                             lr=1e-3, betas=(0.9, 0.95), eps=1e-8, foreach=False)
    sA, sB = nb.CropStream(5, B=4), nb.CropStream(5, B=4)
    for _ in range(8):
        for m, opt, st in ((mA, optA, sA), (mB, optB, sB)):
            b = st.next("cpu")
            loss = nb.token_weighted_loss(m(b), b)
            m.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    assert all(torch.equal(a, b) for a, b in zip(mA.parameters(), mB.parameters()))


def test_hand_sgd_bitwise_vs_torch(nb):
    mA, mB, _ = _pair(nb)
    optA = nb.HandSGD(mA.parameters(), 0.1, momentum=0.9, nesterov=True)
    optB = torch.optim.SGD(mB.parameters(), lr=0.1, momentum=0.9, nesterov=True, foreach=False)
    sA, sB = nb.CropStream(5, B=4), nb.CropStream(5, B=4)
    for _ in range(6):
        for m, opt, st in ((mA, optA, sA), (mB, optB, sB)):
            b = st.next("cpu")
            nb.token_weighted_loss(m(b), b).backward()
            opt.step()
            m.zero_grad(set_to_none=True)
    assert all(torch.equal(a, b) for a, b in zip(mA.parameters(), mB.parameters()))


def test_lion_matches_paper_algorithm(nb):
    g = torch.Generator().manual_seed(3)
    w = torch.randn(16, 8, generator=g); m0 = torch.randn(16, 8, generator=g); grad = torch.randn(16, 8, generator=g)
    p = torch.nn.Parameter(w.clone()); p.grad = grad.clone()
    lion = nb.HandLion([p], 1e-3, weight_decay=0.5)
    lion.m[0].copy_(m0)
    lion.step()
    c = 0.9 * m0 + 0.1 * grad
    assert torch.allclose(p.detach(), w - 1e-3 * (torch.sign(c) + 0.5 * w), rtol=1e-6, atol=0)
    assert torch.allclose(lion.m[0], 0.99 * m0 + 0.01 * grad, rtol=1e-6, atol=0)


def test_uncorrected_equals_corrected_with_r_schedule(nb):
    """The identity behind §2: corrected AdamW driven at lr*r(t) reproduces the uncorrected arm.
    Exact only up to eps (eps enters as eps*sqrt(1-b2^t) on the corrected side), so eps is
    made negligible here (not zero: untouched embedding rows have m = v = 0)."""
    mA, mB, dm = _pair(nb)
    b1, b2 = 0.9, 0.999
    optA = nb.HandAdamW(mA.parameters(), 1e-3, betas=(b1, b2), eps=1e-30, weight_decay=0.0, bias_correction=False, decay=dm)
    optB = nb.HandAdamW(mB.parameters(), 1e-3, betas=(b1, b2), eps=1e-30, weight_decay=0.0, bias_correction=True, decay=dm)
    sA, sB = nb.CropStream(5, B=4), nb.CropStream(5, B=4)
    for t in range(1, 7):
        r = (1 - b1 ** t) / math.sqrt(1 - b2 ** t)
        for m, opt, st, lr in ((mA, optA, sA, 1e-3), (mB, optB, sB, 1e-3 * r)):
            b = st.next("cpu")
            nb.token_weighted_loss(m(b), b).backward()
            opt.step(lr)
            m.zero_grad(set_to_none=True)
    diff = max(float((a - b).abs().max()) for a, b in zip(mA.parameters(), mB.parameters()))
    assert diff < 1e-6, diff


def test_track_both_ratio_is_r_of_t(nb):
    m = nb.make_model(64, seed=3, device="cpu")
    opt = nb.HandAdamW(m.parameters(), 1e-3, betas=(0.9, 0.999), weight_decay=0.0, decay=nb.decay_mask(m))
    opt.track_both = True
    st = nb.CropStream(9, B=4)
    for t in range(1, 6):
        b = st.next("cpu")
        nb.token_weighted_loss(m(b), b).backward()
        opt.step()
        m.zero_grad(set_to_none=True)
        cor, unc = opt.last_both
        r = (1 - 0.9 ** t) / math.sqrt(1 - 0.999 ** t)
        assert abs(math.sqrt(float(unc)) / math.sqrt(float(cor)) / r - 1) < 2e-2


def test_schedule_pins(nb):
    peak, W, N = 1.0, 30, 300
    assert nb.lr_cosine(0, peak, W, N) == peak / W
    assert nb.lr_cosine(W - 1, peak, W, N) == peak
    assert abs(nb.lr_cosine(200, peak, W, N) - 0.3718) < 5e-4
    assert abs(nb.lr_cosine(N, peak, W, N) - 0.1) < 1e-12
    assert nb.lr_wsd(200, peak, W, N) == peak and nb.lr_wsd(239, peak, W, N) == peak
    assert nb.lr_wsd(N, peak, W, N) == 0.0 and 0 < nb.lr_wsd(N - 1, peak, W, N) < 0.05
    assert nb.lr_constant(W + 50, peak, W) == peak
    assert nb.lr_linear_to_zero(200, 0.5, 200, 30) == 0.5 and nb.lr_linear_to_zero(230, 0.5, 200, 30) == 0.0


def test_opt_step_dispatch_torch_and_hand(nb):
    m = nb.make_model(64, seed=3, device="cpu")
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    nb.opt_step(opt, 0.25)
    assert opt.param_groups[0]["lr"] == 0.25
