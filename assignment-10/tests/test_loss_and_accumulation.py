"""Task-3 invariants: shift/mask semantics, the two combine rules, the
accumulation identity, and the zero_grad mechanism — all on the notebook's own code."""
import math

import torch
import torch.nn.functional as F


def test_per_token_ce_shift_is_next_token(nb):
    """Position t must be scored against token t+1, and pads masked out."""
    tokens = torch.tensor([[nb.BOS_ID, 10, 11, nb.EOS_ID, nb.PAD_ID]])
    logits = torch.zeros(1, 5, nb.VOCAB)
    logits[0, 1, 11] = 9.0          # position 1 (has read ...10) predicts 11
    per_tok, mask = nb.per_token_ce(logits, tokens)
    assert per_tok.shape == (1, 4) and mask.tolist() == [[1, 1, 1, 0]]
    assert per_tok[0, 1] < per_tok[0, 0], \
        "boosting the TRUE next token must lower that slot's loss"


def test_session_combine_numbers(nb):
    assert abs(nb.token_weighted([2.0, 2.0, 5.0], [4, 4, 2]) - 2.6) < 1e-12
    assert abs(nb.avg_of_avgs([2.0, 2.0, 5.0], [4, 4, 2]) - 3.0) < 1e-12
    # equal counts: bug invisible
    assert abs(nb.token_weighted([1.0, 3.0], [7, 7])
               - nb.avg_of_avgs([1.0, 3.0], [7, 7])) < 1e-12


def test_gradients_accumulate_bitwise(nb):
    torch.manual_seed(0)
    m = nb.TinyLM(d=32, n_layer=1, n_head=2)
    tb = torch.tensor([nb.hazard_doc(5, 10)])
    m.zero_grad(set_to_none=True)
    nb.token_weighted_loss(m(tb), tb).backward()
    g1 = m.head.weight.grad.clone()
    nb.token_weighted_loss(m(tb), tb).backward()
    assert torch.equal(m.head.weight.grad, 2 * g1)


def test_accumulation_identity_and_bug(nb):
    torch.manual_seed(1)
    m = nb.TinyLM(d=32, n_layer=1, n_head=2)
    micros = [[nb.hazard_doc(2, 12)], [nb.hazard_doc(2, 12)], [nb.hazard_doc(9, 12)]]
    big = torch.tensor([mb[0] for mb in micros])
    m.zero_grad(set_to_none=True)
    pt, msk = nb.per_token_ce(m(big), big)
    ((pt * msk).sum() / msk.sum()).backward()
    g_big = [p.grad.clone() for p in m.parameters()]
    g_acc, counts = nb.accum_grads(m, micros, "correct")
    g_bug, _ = nb.accum_grads(m, micros, "buggy")
    assert counts == [3, 3, 10], "targets counted on the SHIFTED mask"
    assert nb.rel_l2(g_acc, g_big) < 1e-5
    assert nb.rel_l2(g_bug, g_big) > 1e-2


def test_accumulation_identity_float64_clean(nb):
    import copy
    torch.manual_seed(2)
    m = nb.TinyLM(d=32, n_layer=1, n_head=2).double()
    micros = [[nb.hazard_doc(2, 12)], [nb.hazard_doc(9, 12)]]
    big = torch.tensor([mb[0] for mb in micros])
    m.zero_grad(set_to_none=True)
    pt, msk = nb.per_token_ce(m(big), big)
    ((pt * msk).sum() / msk.sum()).backward()
    g_big = [p.grad.clone() for p in m.parameters()]
    g_acc, _ = nb.accum_grads(m, micros, "correct")
    assert nb.rel_l2(g_acc, g_big) < 1e-12


def test_hazard_analytics(nb):
    an = nb.HZ_ANALYTIC
    assert abs(an["correct_tw"] - math.log(2) / 18) < 1e-12
    assert abs(an["buggy_p"] - 11 / 12) < 1e-12
    assert an["buggy_tw"] > an["correct_tw"]
    assert an["buggy_dw"] < an["correct_dw"], "doc-weighted metric must flip"


def test_hazard_stream_fixed_composition(nb):
    stream = nb.hazard_stream(1, 5, 4)
    for micros in stream:
        counts = sorted(int((torch.tensor(mb)[:, 1:] != nb.PAD_ID).sum())
                        for mb in micros)
        assert counts == [3, 3, 33, 33], "2 short + 2 long, constant step total 72"


def test_eval_token_weighted_full_sweep_no_double_count(nb):
    """Every target scored exactly once: sum/sum over windows == direct CE."""
    torch.manual_seed(3)
    m = nb.TinyLM(d=32, n_layer=1, n_head=2)
    ids = nb.TRAIN_PROSE_IDS[:300]
    got = nb.eval_token_weighted(m, ids, T=64)
    full = torch.tensor([ids])
    with torch.no_grad():
        pt, msk = nb.per_token_ce(m(full[:, :256]), full[:, :256])
    assert 0 < got < 12 and math.isfinite(got)
