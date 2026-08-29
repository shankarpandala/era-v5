"""The notebook is the single source of truth: tests exec its export-tagged
cells into a fresh module, so nothing is retyped here. Export cells run under
A10_FAST budgets and write any artifacts to a throwaway dir."""
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

A10 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(A10))

os.environ["A10_FAST"] = "1"
os.environ["A10_ART_DIR"] = tempfile.mkdtemp(prefix="a10_test_art_")


def load_export_module() -> types.ModuleType:
    raw = json.loads((A10 / "training_loop.ipynb").read_text())
    mod = types.ModuleType("traininloop_nb")
    mod.__file__ = str(A10 / "training_loop.ipynb")
    n = 0
    for i, cell in enumerate(raw["cells"]):
        if cell["cell_type"] != "code":
            continue
        if "export" not in cell.get("metadata", {}).get("tags", []):
            continue
        src = "".join(cell["source"])
        exec(compile(src, f"<training_loop.ipynb export cell {i}>", "exec"),
             mod.__dict__)
        n += 1
    assert n >= 8, f"expected the export cells, found {n}"
    return mod


@pytest.fixture(scope="session")
def nb() -> types.ModuleType:
    return load_export_module()
