"""The causal shift: exact tensor identities, the string-audit's verdicts, and
the ln(V) step-0 invariant."""
import math
import random

import torch


def _tokens(nb, B=4, T=32):
    nb.seed_all()
    return nb.sample_batch(nb.TRAIN_STREAM, B, T, random.Random(nb.SEED))


def test_correct_targets_are_next_tokens(nb):
    tokens = _tokens(nb)
    logits = torch.zeros(*tokens.shape, nb.VOCAB_SIZE, device=tokens.device)
    pred, tgt, pos = nb.arm_correct(logits, tokens)
    assert torch.equal(tgt, tokens[:, 1:])
    assert pred.shape[1] == tokens.shape[1] - 1
    # target i answers for the token AFTER the prediction position
    assert torch.equal(tokens[:, pos + 1], tgt)


def test_t_plus_2_targets_skip_exactly_one(nb):
    tokens = _tokens(nb)
    # the part-2 alignment: logits2[:, :-2] answers for tokens[:, 2:]
    tgt2 = tokens[:, 2:]
    for i in range(tokens.shape[1] - 2):
        assert torch.equal(tgt2[:, i], tokens[:, i + 2])


def test_audit_names_all_three_arms(nb):
    tokens = _tokens(nb, B=8, T=64)
    verdicts = {name: nb.audit_shift(tokens, fn, verbose=False)["verdict"]
                for name, fn in nb.ARMS.items()}
    assert verdicts == {"correct": "CORRECT", "reversed": "REVERSED",
                        "no_shift": "IDENTITY"}


def test_audit_correct_rate_is_exactly_one(nb):
    tokens = _tokens(nb)
    audit = nb.audit_shift(tokens, nb.arm_correct, verbose=False)
    assert audit["rates"]["correct(+1)"] == 1.0
    assert audit["rates"]["identity(0)"] < 0.5
    assert audit["rates"]["reversed(-1)"] < 0.5


def test_audit_table_positions_are_not_smashed(nb, capsys):
    """The v1 table concatenated positions >= 10 into '910111213' — its own
    wall of integers, in the one instrument that had to stay readable."""
    tokens = _tokens(nb, B=2, T=40)
    nb.audit_shift(tokens, nb.arm_correct, cols=16, verbose=True)
    out = capsys.readouterr().out
    pos_line = next(l for l in out.splitlines() if l.strip().startswith("position:"))
    cells = pos_line.split(":", 1)[1].split()
    assert cells == [str(i) for i in range(16)], f"position row unreadable: {cells}"
    inp_line = next(l for l in out.splitlines() if "input   :" in l)
    assert len(pos_line.split(":", 1)[1].rstrip()) <= len(inp_line), \
        "rows are not column-aligned"


def test_copy_stats_expose_the_tied_identity_leak_at_init(nb):
    """The §7 leak, in argmax space: an UNTRAINED tied model already agrees
    with the copy-self hypothesis far above chance (1/V ~ 0.004) — the wire is
    architectural. Untying removes it: every hypothesis drops to noise."""
    tokens = _tokens(nb, B=8, T=64)
    nb.seed_all()
    tied = nb.copy_stats(nb.TinyLM(tie_weights=True).to(nb.DEVICE), tokens)
    nb.seed_all()
    untied = nb.copy_stats(nb.TinyLM(tie_weights=False).to(nb.DEVICE), tokens)
    assert tied["self"] > 0.2, f"tied identity leak missing: {tied}"
    assert tied["next"] < 0.2 and tied["prev"] < 0.2, f"shifted hypotheses: {tied}"
    for k, v in untied.items():
        assert v < 0.1, f"untied untrained argmax must be near chance ({k}={v})"


def test_sample_batch_reaches_every_window(nb):
    """v1 used randrange(0, len-T-2), silently dropping the last two windows;
    a stream of exactly length T must be croppable (start 0 only)."""
    stream = list(range(10, 10 + 32))
    batch = nb.sample_batch(stream, 3, 32, random.Random(0))
    assert batch.shape == (3, 32)
    assert batch[0].tolist() == stream


def test_untrained_loss_sits_at_ln_v(nb):
    nb.seed_all()
    model = nb.TinyLM().to(nb.DEVICE)
    tokens = _tokens(nb, B=8, T=64)
    with torch.no_grad():
        pred, tgt, _ = nb.arm_correct(model(tokens), tokens)
        loss, _ = nb.masked_ce(pred, tgt)
    assert abs(float(loss) - math.log(nb.VOCAB_SIZE)) < 0.15


def test_loud_init_breaks_the_invariant(nb):
    nb.seed_all()
    model = nb.TinyLM(init_std=1.0).to(nb.DEVICE)
    tokens = _tokens(nb, B=8, T=64)
    with torch.no_grad():
        pred, tgt, _ = nb.arm_correct(model(tokens), tokens)
        loss, _ = nb.masked_ce(pred, tgt)
    assert float(loss) > math.log(nb.VOCAB_SIZE) + 0.5
