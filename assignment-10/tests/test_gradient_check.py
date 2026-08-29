"""Task-2 invariants: the toy chain, central differences vs backward(), and the
finite-difference protocol, on the notebook's own code."""
import copy
import math

import torch


def test_toy_chain_is_64_exactly():
    x, w1v, w2v, t = 2.0, 3.0, 4.0, 20.0
    w1 = torch.tensor(w1v, dtype=torch.float64, requires_grad=True)
    ((torch.tensor(w2v, dtype=torch.float64) * w1 * x - t) ** 2).backward()
    assert abs(float(w1.grad) - 64.0) < 1e-10


def test_toy_forward_diff_error_law():
    """Forward difference error is exactly 64*eps for this quadratic loss."""
    def L(w):
        return (4.0 * w * 2.0 - 20.0) ** 2
    for k in range(2, 7):
        eps = 10.0 ** -k
        err = (L(3.0 + eps) - L(3.0)) / eps - 64.0
        assert abs(err / eps - 64.0) < 0.64, f"eps={eps}: {err / eps}"


def test_central_diff_matches_backward_float64(nb):
    torch.manual_seed(4)
    m = nb.TinyLM(d=32, n_layer=1, n_head=2).double().eval()
    tb = torch.tensor([nb.hazard_doc(9, 12), nb.hazard_doc(2, 12)])
    for p in m.parameters():
        p.grad = None
    nb.token_weighted_loss(m(tb), tb).backward()
    W = m.blocks[0].mlp.fc1.weight
    fi = int(W.grad.abs().argmax())
    ga = float(W.grad.view(-1)[fi])
    assert abs(ga) > 1e-8, "selection rule must dodge zero gradients"
    flat = W.data.view(-1)
    orig = flat[fi].clone()
    eps = 1e-5
    with torch.no_grad():
        flat[fi] = orig + eps
        Lp = float(nb.token_weighted_loss(m(tb), tb))
        flat[fi] = orig - eps
        Lm = float(nb.token_weighted_loss(m(tb), tb))
        flat[fi] = orig
    fd = (Lp - Lm) / (2 * eps)
    assert abs(fd - ga) / abs(ga) < 1e-7
    assert torch.equal(flat[fi], orig), "restoration must be bit-exact"


def test_unused_embedding_rows_have_exactly_zero_grad(nb):
    torch.manual_seed(5)
    m = nb.TinyLM(d=32, n_layer=1, n_head=2)
    tb = torch.tensor([nb.hazard_doc(5, 10)])
    nb.token_weighted_loss(m(tb), tb).backward()
    used = set(tb.flatten().tolist())
    unused = next(i for i in range(nb.VOCAB) if i not in used)
    assert float(m.tok_emb.weight.grad[unused].abs().sum()) == 0.0


def test_directional_derivative_whole_gradient(nb):
    torch.manual_seed(6)
    m = nb.TinyLM(d=32, n_layer=1, n_head=2).double().eval()
    tb = torch.tensor([nb.hazard_doc(9, 12)])
    for p in m.parameters():
        p.grad = None
    nb.token_weighted_loss(m(tb), tb).backward()
    gen = torch.Generator().manual_seed(7)
    vecs = [torch.randn(p.shape, generator=gen, dtype=torch.float64)
            for p in m.parameters()]
    vn = math.sqrt(sum(float(v.pow(2).sum()) for v in vecs))
    vecs = [v / vn for v in vecs]
    gv = sum(float((p.grad * v).sum()) for p, v in zip(m.parameters(), vecs))
    eps = 1e-6
    with torch.no_grad():
        for p, v in zip(m.parameters(), vecs):
            p.add_(eps * v)
        Lp = float(nb.token_weighted_loss(m(tb), tb))
        for p, v in zip(m.parameters(), vecs):
            p.add_(-2 * eps * v)
        Lm = float(nb.token_weighted_loss(m(tb), tb))
        for p, v in zip(m.parameters(), vecs):
            p.add_(eps * v)
    assert abs((Lp - Lm) / (2 * eps) - gv) / abs(gv) < 1e-6


def test_model_runs_identically_in_double(nb):
    """The bool causal mask keeps forward dtype-agnostic (SS4's requirement)."""
    torch.manual_seed(8)
    m = nb.TinyLM(d=32, n_layer=1, n_head=2)
    m64 = copy.deepcopy(m).double()
    tb = torch.tensor([nb.hazard_doc(5, 10)])
    with torch.no_grad():
        a = m(tb)
        b = m64(tb)
    assert float((a.double() - b).abs().max()) < 1e-5
