"""The notebook and saved evidence are part of the submission, not placeholders."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parents[1]


def test_saved_evidence_passes_independent_read_only_audit(auditor):
    failed = []
    count = auditor.run(lambda name, ok, detail="": failed.append(f"{name}: {detail}") if not ok else None)
    assert count == 0, "\n".join(failed)


def test_notebook_export_is_exact_standalone_simulator_source():
    notebook = json.loads((HERE / "zero_simulation.ipynb").read_text())
    exports = ["".join(c["source"]) for c in notebook["cells"]
               if c["cell_type"] == "code" and "export" in c.get("metadata", {}).get("tags", [])]
    source = (HERE / "zero_simulator.py").read_text()
    assert source in exports, "The complete simulator must be embedded for standalone Colab execution."
    # Execute in a fresh module with no local simulator import available.
    module = types.ModuleType("assignment12_notebook_export")
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, "<zero_simulation.ipynb export>", "exec"), module.__dict__)
        cfg = module.Config(world_size=1, dims=(2, 3, 1), local_batch=2, steps=1, workers=1)
        result = module.Simulator(cfg, stage=3).train()
        assert len(result["loss_curve"]) == 1
        assert result["communication"]["bytes_sent_per_rank_per_step"] == 0
    finally:
        sys.modules.pop(module.__name__, None)


def test_committed_configuration_uses_32_virtual_ranks():
    config = json.loads((HERE / "submission_artifacts" / "run_config.json").read_text())
    assert config["world_size"] == 32
    assert config["steps"] >= 20, "Commit a full training demonstration, not a smoke test."
