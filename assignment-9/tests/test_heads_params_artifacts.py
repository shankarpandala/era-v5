"""Tied vs untied arithmetic, training determinism, t+2 head wiring, and the
committed artifacts' integrity."""
import json

import pytest
import torch

from conftest import A9

ART = A9 / "submission_artifacts"


def test_tied_untied_delta_is_exactly_vd(nb):
    nb.seed_all()
    tied = nb.TinyLM(tie_weights=True)
    untied = nb.TinyLM(tie_weights=False)
    n_tied = sum(p.numel() for p in tied.parameters())
    n_untied = sum(p.numel() for p in untied.parameters())
    assert n_untied - n_tied == nb.VOCAB_SIZE * nb.CFG["d_model"]


def test_tied_head_shares_storage(nb):
    m = nb.TinyLM(tie_weights=True)
    assert m.head.weight is m.tok_emb.weight
    m2 = nb.TinyLM(tie_weights=False)
    assert m2.head.weight is not m2.tok_emb.weight


def test_second_head_is_separate_and_untied(nb):
    m = nb.TinyLM(two_heads=True)
    assert m.head2 is not None
    assert m.head2.weight is not m.head.weight
    assert m.head2.weight is not m.tok_emb.weight
    assert m.head2.weight.shape == (nb.VOCAB_SIZE, nb.CFG["d_model"])


def test_training_is_deterministic(nb):
    _, c1 = nb.train_arm("correct", 5, log_every=5, quiet=True)
    _, c2 = nb.train_arm("correct", 5, log_every=5, quiet=True)
    assert c1["loss"] == c2["loss"], "same seed must give bit-identical training"


def test_two_head_losses_use_different_shifts(nb):
    nb.seed_all()
    m = nb.TinyLM(two_heads=True)
    import random
    tokens = nb.sample_batch(nb.TRAIN_STREAM, 2, 24, random.Random(nb.SEED))
    with torch.no_grad():
        hidden = m.trunk(tokens)
        l1 = m.head(hidden)[:, :-1]
        l2 = m.head2(hidden)[:, :-2]
    assert l1.shape[1] == tokens.shape[1] - 1
    assert l2.shape[1] == tokens.shape[1] - 2


# ---- committed artifacts (skip on a pre-run tree) ---------------------------

def _results():
    if not (ART / "results.json").exists():
        pytest.skip("artifacts not generated yet — run python run_demo.py")
    return json.loads((ART / "results.json").read_text())


def test_results_have_all_required_numbers():
    r = _results()
    p1 = r["part1"]
    for key in ("shapes", "shift_audit", "padding", "packing", "perplexity",
                "tied_untied", "memory"):
        assert key in p1, f"part1.{key} missing"
    for key in ("train_L1", "train_L2", "train_sum",
                "held_L1", "held_L2", "held_sum"):
        assert key in r["part2"]["final"]
    assert set(r["part3"]["arms"]) == {"correct", "reversed", "no_shift"}
    assert r["headline"], "headline numbers missing"


def test_run_log_has_no_failures():
    log = ART / "run.log"
    if not log.exists():
        pytest.skip("run.log not generated yet")
    text = log.read_text()
    assert "[FAIL]" not in text, "run.log records failing checks"
    assert "verdict: PASS" in text


def test_plots_exist():
    if not (ART / "results.json").exists():
        pytest.skip("artifacts not generated yet")
    for name in ("bug_zoo_curves.png", "mtp_gap.png", "memory_bars.png"):
        assert (ART / "plots" / name).exists(), f"plots/{name} missing"
