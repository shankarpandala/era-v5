"""The executed notebooks and saved evidence are part of the submission, not placeholders."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parents[1]
ART = HERE / "submission_artifacts"


def test_saved_evidence_passes_independent_read_only_audit(auditor):
    failed = []
    count = auditor.run(lambda name, ok, detail="": failed.append(f"{name}: {detail}") if not ok else None)
    assert count == 0, "\n".join(failed)


def test_every_notebook_embeds_the_exact_module_and_it_runs_standalone():
    source = (HERE / "revllm.py").read_text()
    for name in ("01_baseline_fixed_batch.ipynb", "02_reversible_fixed_batch.ipynb",
                 "03_reversible_max_batch.ipynb", "04_report.ipynb"):
        nb = json.loads((HERE / name).read_text())
        exports = ["".join(c["source"]) for c in nb["cells"]
                   if c["cell_type"] == "code" and "export" in c.get("metadata", {}).get("tags", [])]
        assert source in exports, f"{name} must embed revllm.py byte-for-byte for standalone Colab execution"
    module = types.ModuleType("assignment13_notebook_export")
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, "<notebook export>", "exec"), module.__dict__)
        import torch
        cfg = module.ModelConfig(vocab_size=16, seq_len=8, d_model=16, n_layer=2, n_head=2, reversible="euler")
        model = module.GPT(cfg)
        _, loss = model(torch.zeros(1, 8, dtype=torch.long), torch.zeros(1, 8, dtype=torch.long))
        loss.backward()
        assert all(p.grad is not None for p in model.parameters())
    finally:
        sys.modules.pop(module.__name__, None)


def test_committed_runs_are_the_full_protocol_not_a_smoke_test():
    runs = {k: json.loads((ART / f / "results.json").read_text())
            for k, f in (("run1", "run1_baseline"), ("run2", "run2_reversible"), ("run3", "run3_reversible_maxbatch"))}
    assert all(r["tokens"] >= 1_000_000 for r in runs.values()), "commit at least the CPU pilot budget"
    assert runs["run1"]["model_config"]["reversible"] == "none"
    assert runs["run2"]["model_config"]["reversible"] == runs["run3"]["model_config"]["reversible"] != "none"
    assert runs["run3"]["train_config"]["batch_size"] > runs["run2"]["train_config"]["batch_size"] == runs["run1"]["train_config"]["batch_size"]


def test_committed_tokenizer_matches_manifest_hash():
    import hashlib
    manifest = json.loads((HERE / "data" / "manifest.json").read_text())
    digest = hashlib.sha256((HERE / "data" / "tokenizer.json").read_bytes()).hexdigest()
    assert digest == manifest["tokenizer_sha256"]
    assert manifest["vocab_size"] == 8192 and manifest["eot_id"] == 0
