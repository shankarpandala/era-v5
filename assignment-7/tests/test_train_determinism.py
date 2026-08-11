"""Two identical short runs must be bit-identical; different arms must not be."""

import pytest

from kronembed.train import run_one

FAST = {"steps": 40, "batch_size": 16, "loss_log_every": 10, "eval_chunk": 256}


@pytest.mark.parametrize("arm", ["kron_v2", "learned"])
def test_repeat_run_is_bit_identical(arm, tmp_path):
    r1 = run_one(arm, 200, 0, str(tmp_path / "a"), cfg=FAST)
    r2 = run_one(arm, 200, 0, str(tmp_path / "b"), cfg=FAST)
    assert r1["param_hash_final"] == r2["param_hash_final"]
    assert r1["loss_curve"] == r2["loss_curve"]
    assert r1["eval"] == r2["eval"]


def test_different_seed_changes_params(tmp_path):
    r1 = run_one("kron_v2", 200, 0, str(tmp_path / "a"), cfg=FAST)
    r2 = run_one("kron_v2", 200, 1, str(tmp_path / "b"), cfg=FAST)
    assert r1["param_hash_final"] != r2["param_hash_final"]


def test_result_records_identity_fields(tmp_path):
    r = run_one("kron_char", 200, 0, str(tmp_path / "a"), cfg=FAST)
    assert r["arch_hash"] and r["emb_hash"].startswith("")
    assert r["data_manifest"]["disjoint_train_eval"] is True
    assert (tmp_path / "a" / "result.json").exists()
