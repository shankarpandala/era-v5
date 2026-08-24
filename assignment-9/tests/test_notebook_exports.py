"""The notebook's export surface: symbols exist, cells are defs-only (fast to
exec), and the committed notebook really ran top to bottom."""
import json
import time

from conftest import A9, load_export_module

EXPECTED = [
    "CFG", "SEED", "VOCAB_SIZE", "PAD_ID", "BOS_ID", "EOS_ID",
    "encode", "decode", "decode_token", "sample_batch",
    "TinyLM", "causal_mask", "block_causal_mask",
    "arm_correct", "arm_reversed", "arm_no_shift", "ARMS",
    "masked_ce", "audit_shift", "explain",
    "train_arm", "eval_sweep", "copy_stats", "generate", "four_gram_hit_rate",
    "plain_ce", "chunked_ce", "online_ce", "CE_IMPL_SRC",
    "fmt", "seed_all", "TRAIN_STREAM", "HELD_STREAM", "CORPUS_TEXT",
]


def test_export_symbols_exist(nb):
    missing = [name for name in EXPECTED if not hasattr(nb, name)]
    assert not missing, f"export cells no longer define: {missing}"


def test_export_cells_are_cheap():
    # Definitions only: a re-exec must not train, measure, or write artifacts.
    t0 = time.time()
    load_export_module()
    assert time.time() - t0 < 30, "an export cell is doing heavy top-level work"


def test_export_cells_write_nothing():
    assert not (A9 / "tests" / "submission_artifacts").exists()


def test_committed_notebook_ran_top_to_bottom():
    raw = json.loads((A9 / "loss_harness.ipynb").read_text())
    codes = [c for c in raw["cells"] if c["cell_type"] == "code"]
    counts = [c.get("execution_count") for c in codes]
    if all(x is None for x in counts):
        import pytest
        pytest.skip("notebook not executed yet (pre-run tree)")
    assert all(isinstance(x, int) for x in counts), "some cells never ran"
    assert counts == sorted(counts) and len(set(counts)) == len(counts), \
        "cells were run out of order — not a top-to-bottom run"
    errors = [o for c in codes for o in c.get("outputs", [])
              if o.get("output_type") == "error"]
    assert not errors, f"{len(errors)} cells errored"
