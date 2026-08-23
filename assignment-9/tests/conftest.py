"""The notebook is the single source of truth: tests exec its export-tagged
cells into a fresh module, so nothing is retyped here."""
import json
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

A9 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(A9))


def load_export_module() -> types.ModuleType:
    raw = json.loads((A9 / "loss_harness.ipynb").read_text())
    mod = types.ModuleType("lossharness_nb")
    mod.__file__ = str(A9 / "loss_harness.ipynb")
    n = 0
    for i, cell in enumerate(raw["cells"]):
        if cell["cell_type"] != "code":
            continue
        if "export" not in cell.get("metadata", {}).get("tags", []):
            continue
        src = "".join(cell["source"])
        exec(compile(src, f"<loss_harness.ipynb export cell {i}>", "exec"),
             mod.__dict__)
        n += 1
    assert n >= 5, f"expected the export cells, found {n}"
    return mod


@pytest.fixture(scope="session")
def nb() -> types.ModuleType:
    return load_export_module()
