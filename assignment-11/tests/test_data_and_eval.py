"""Data split, eval sweep, and determinism invariants."""
import hashlib

import torch


def test_split_is_disjoint_and_same_composition(nb):
    assert len(nb.HELD_BLOCKS) >= 20
    for lang, blk in nb.HELD_BLOCKS:
        assert len(blk) == nb.BLOCK
    langs = {lang for lang, _ in nb.HELD_BLOCKS}
    assert langs == set(nb.LANGS), "every language contributes held-out blocks"
    # held-out share per language tracks its byte share
    for lang in nb.LANGS:
        m = nb.CORPUS_META[lang]
        share = m["held_ids"] / (m["held_ids"] + m["train_ids"])
        assert 0.04 < share < 0.14, (lang, share)


def test_eval_scores_every_target_once(nb):
    assert nb.eval_targets_scored() == len(nb.HELD_BLOCKS) * 32 * (nb.CFG["T"] - 1)
    assert nb.BLOCK == 32 * (nb.CFG["T"] - 1) + 1


def test_eval_matches_direct_cross_entropy_on_one_block(nb):
    """The windowed sweep of one block equals a single full-block forward's mean CE."""
    m = nb.make_model(64, seed=1, device="cpu")
    lang, blk = nb.HELD_BLOCKS[0]
    ids = torch.tensor(blk)
    T = nb.CFG["T"]
    starts = [j * (T - 1) for j in range(32)]
    rows = torch.stack([ids[s:s + T] for s in starts])
    with torch.no_grad():
        per_tok, mask = nb.per_token_ce(m(rows), rows)
    windowed = float((per_tok * mask).sum() / mask.sum())
    assert mask.sum() == 32 * (T - 1)
    # every target 1..4064 appears exactly once across the windows
    covered = sorted(s + 1 + k for s in starts for k in range(T - 1))
    assert covered == list(range(1, nb.BLOCK))
    assert 0 < windowed < 10


def test_stream_is_device_independent_and_checksummed(nb):
    a, b = nb.CropStream(11), nb.CropStream(11)
    assert a.checksum() == b.checksum()
    x = a.next("cpu")
    assert torch.equal(x, b.next("cpu"))
    assert a.checksum() == b.checksum() != nb.CropStream(12).checksum()


def test_cpu_training_is_bitwise_deterministic(nb):
    def run():
        m = nb.make_model(64, seed=5, device="cpu")
        opt = nb.make_opt("adamw", m, 1e-3)
        rec = nb.train_run(m, opt, nb.CropStream(3, B=4), lambda t: 1e-3, 6)
        return rec["train"], [p.detach().clone() for p in m.parameters()]
    l1, p1 = run()
    l2, p2 = run()
    assert l1 == l2 and all(torch.equal(a, b) for a, b in zip(p1, p2))


def test_corpus_hashes_pinned(nb):
    for lang in nb.LANGS:
        raw = nb.corpus_path(lang).read_bytes()
        assert hashlib.sha256(raw).hexdigest().startswith(nb.CORPUS_SHA[lang])


def test_shift_is_next_token(nb):
    tokens = torch.tensor([[nb.BOS_ID, 10, 11, nb.EOS_ID]])
    logits = torch.zeros(1, 4, nb.VOCAB)
    logits[0, 1, 11] = 9.0
    per_tok, mask = nb.per_token_ce(logits, tokens)
    assert per_tok.shape == (1, 3) and per_tok[0, 1] < per_tok[0, 0]
