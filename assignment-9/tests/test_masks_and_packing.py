"""Masking: exact contributing counts, pad-tamper invariance, the two masking
styles agreeing, and the boundary mask removing exactly the seam."""
import torch
import torch.nn.functional as F


def _padded_batch(nb, texts, T=48):
    rows = [nb.encode(s)[:T] for s in texts]
    tokens = torch.full((len(rows), T), nb.PAD_ID, dtype=torch.long)
    for i, r in enumerate(rows):
        tokens[i, :len(r)] = torch.tensor(r)
    return tokens, [len(r) for r in rows]


def test_contributing_count_is_exact(nb):
    tokens, lens = _padded_batch(nb, ["short one.", "a much longer sentence here."])
    tgt = tokens[:, 1:]
    mask = (tgt != nb.PAD_ID).long()
    # row i contributes exactly len_i - 1 targets (every token except the first)
    assert int(mask.sum()) == sum(n - 1 for n in lens)
    assert int(mask.numel()) == 2 * (tokens.shape[1] - 1)


def test_pad_tamper_invariance(nb):
    nb.seed_all()
    model = nb.TinyLM()
    tokens, _ = _padded_batch(nb, ["print the strings.", "count the tokens."])
    tgt_mask = (tokens[:, 1:] != nb.PAD_ID).long()

    def masked_loss(toks):
        with torch.no_grad():
            pred, tgt, _ = nb.arm_correct(model(toks), toks)
            loss, _ = nb.masked_ce(pred, tgt.masked_fill(tgt_mask == 0, 0),
                                   tgt_mask)
        return float(loss)

    baseline = masked_loss(tokens)
    tampered = tokens.clone()
    tampered[tokens == nb.PAD_ID] = nb.BYTE_OFFSET + ord("!")
    assert abs(masked_loss(tampered) - baseline) < 1e-6, \
        "rewriting pads moved the masked loss — pads are participating"


def test_float_mask_equals_ignore_index(nb):
    nb.seed_all()
    model = nb.TinyLM()
    tokens, _ = _padded_batch(nb, ["a loss is a promise.", "padding exists."])
    with torch.no_grad():
        pred, tgt, _ = nb.arm_correct(model(tokens), tokens)
    mask = (tgt != nb.PAD_ID).long()
    loss_float, _ = nb.masked_ce(pred, tgt, mask)
    loss_ignore = F.cross_entropy(pred.reshape(-1, nb.VOCAB_SIZE),
                                  tgt.masked_fill(mask == 0, -100).reshape(-1),
                                  ignore_index=-100)
    assert abs(float(loss_float) - float(loss_ignore)) < 1e-6


def test_boundary_mask_removes_exactly_the_seam(nb):
    ids1, ids2 = nb.encode("doc one."), nb.encode("doc two.")
    T = 40
    tokens = torch.full((1, T), nb.PAD_ID, dtype=torch.long)
    tokens[0, :len(ids1)] = torch.tensor(ids1)
    tokens[0, len(ids1):len(ids1) + len(ids2)] = torch.tensor(ids2)
    tgt = tokens[:, 1:]
    base = (tgt != nb.PAD_ID).long()
    boundary = torch.zeros_like(base)
    boundary[0, len(ids1) - 1] = 1
    boundary[0, len(ids1)] = 1
    after = base * (1 - boundary)
    assert int(base.sum()) - int(after.sum()) == 2
    # the removed targets are doc2's <bos> and doc2's first real token
    assert int(tgt[0, len(ids1) - 1]) == nb.BOS_ID
    assert int(tgt[0, len(ids1)]) == ids2[1]


def test_crop_contract_semantics(nb):
    """segments increment at <bos>, positions restart at <bos>, and the loss
    mask drops exactly the read==<eos> slots."""
    B_, E_ = nb.BOS_ID, nb.EOS_ID
    a = nb.BYTE_OFFSET + ord("a")
    tokens = torch.tensor([[a, a, E_, B_, a, a, E_, B_, a]])
    segs, pos, mask = nb.crop_contract(tokens)
    assert segs[0].tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2]
    assert pos[0].tolist() == [0, 1, 2, 0, 1, 2, 3, 0, 1]
    # slots read tokens[:-1]; the two <eos> reads (idx 2 and 6) are dropped
    assert mask[0].tolist() == [1, 1, 0, 1, 1, 1, 0, 1]


def test_block_causal_mask_stops_cross_document_attention(nb):
    segs = torch.tensor([[1, 1, 1, 2, 2, -1]])
    m = nb.block_causal_mask(segs)[0, 0]
    inf = float("-inf")
    assert m[3, 2] == inf, "doc2 must not read doc1"
    assert m[4, 3] == 0.0, "doc2 reads doc2"
    assert m[2, 0] == 0.0, "doc1 reads doc1"
    assert m[1, 2] == inf, "no reading the future"
    assert m[5, 5] == 0.0 and m[5, 4] == inf, "pads self-attend only"
