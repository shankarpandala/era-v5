"""chunked_ce and online_ce must be the same estimator as plain_ce: same loss,
same gradients, for divisor and non-divisor chunk sizes, with and without
ignored rows."""
import pytest
import torch


def _problem(nb, N=257, V=613, d=32, with_ignore=True):
    torch.manual_seed(7)
    hidden = torch.randn(N, d, requires_grad=True)
    W = (torch.randn(V, d) * 0.02).requires_grad_()
    targets = torch.randint(0, V, (N,))
    if with_ignore:
        targets[::5] = -100
    return hidden, W, targets


@pytest.mark.parametrize("chunk_rows", [64, 128, 1000])
def test_loss_and_grads_match(nb, chunk_rows):
    h, W, t = _problem(nb)
    la = nb.plain_ce(h, W, t)
    ga_h, ga_W = torch.autograd.grad(la, [h, W])
    lb = nb.chunked_ce(h, W, t, chunk_rows=chunk_rows)
    gb_h, gb_W = torch.autograd.grad(lb, [h, W])
    assert torch.allclose(la, lb, atol=1e-6)
    assert torch.allclose(ga_h, gb_h, atol=1e-6)
    assert torch.allclose(ga_W, gb_W, atol=1e-6)


@pytest.mark.parametrize("vocab_chunk", [128, 300, 4096])
def test_online_loss_and_grads_match(nb, vocab_chunk):
    h, W, t = _problem(nb)
    la = nb.plain_ce(h, W, t)
    ga_h, ga_W = torch.autograd.grad(la, [h, W])
    lc = nb.online_ce(h, W, t, vocab_chunk=vocab_chunk)
    gc_h, gc_W = torch.autograd.grad(lc, [h, W])
    assert torch.allclose(la, lc, atol=1e-6)
    assert torch.allclose(ga_h, gc_h, atol=1e-6)
    assert torch.allclose(ga_W, gc_W, atol=1e-6)


def test_no_ignore_rows(nb):
    h, W, t = _problem(nb, with_ignore=False)
    assert torch.allclose(nb.plain_ce(h, W, t),
                          nb.chunked_ce(h, W, t, chunk_rows=100), atol=1e-6)
    assert torch.allclose(nb.plain_ce(h, W, t),
                          nb.online_ce(h, W, t, vocab_chunk=250), atol=1e-6)


def test_all_rows_ignored_is_finite(nb):
    h, W, t = _problem(nb, N=64)
    t[:] = -100
    loss = nb.chunked_ce(h, W, t, chunk_rows=16)
    assert torch.isfinite(loss) and loss.detach().item() == 0.0
    loss_o = nb.online_ce(h, W, t, vocab_chunk=100)
    assert torch.isfinite(loss_o) and loss_o.detach().item() == 0.0


def test_matches_shifted_batch_loss(nb):
    """The harness path (masked_ce over [B, P, V]) and the head path
    (plain_ce over flattened rows) are the same estimator."""
    nb.seed_all()
    model = nb.TinyLM().to(nb.DEVICE)
    import random
    tokens = nb.sample_batch(nb.TRAIN_STREAM, 4, 32, random.Random(nb.SEED))
    with torch.no_grad():
        hidden = model.trunk(tokens)
        pred, tgt, _ = nb.arm_correct(model.head(hidden), tokens)
        loss_a, _ = nb.masked_ce(pred, tgt)
        loss_b = nb.plain_ce(hidden[:, :-1].reshape(-1, hidden.shape[-1]),
                             model.head.weight, tgt.reshape(-1))
    assert torch.allclose(loss_a, loss_b, atol=1e-6)


def test_ce_impl_src_is_the_source(nb):
    """The measured subprocesses receive CE_IMPL_SRC verbatim; the notebook's
    plain_ce/chunked_ce must come from that same string."""
    ns = {}
    exec(nb.CE_IMPL_SRC, ns)
    h, W, t = _problem(nb, N=50, V=97, d=8)
    assert torch.allclose(ns["chunked_ce"](h, W, t, chunk_rows=7),
                          nb.chunked_ce(h, W, t, chunk_rows=7), atol=1e-7)
