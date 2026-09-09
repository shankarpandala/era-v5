"""The notebook is the single source of truth: tests exec its export-tagged cells
into a fresh module, so nothing is retyped here. Export cells run under A11_FAST
budgets, on the CPU, and write any artifacts to a throwaway dir."""
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

A11 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(A11))

os.environ["A11_FAST"] = "1"
os.environ["A11_ART_DIR"] = tempfile.mkdtemp(prefix="a11_test_art_")
os.environ.setdefault("A11_DEVICE", "cpu")


def load_export_module() -> types.ModuleType:
    raw = json.loads((A11 / "optimizers.ipynb").read_text())
    mod = types.ModuleType("optimizers_nb")
    mod.__file__ = str(A11 / "optimizers.ipynb")
    n = 0
    cwd = os.getcwd()
    os.chdir(A11)          # the data cell resolves ../public/tokenizer/corpus relative to the notebook
    try:
        for i, cell in enumerate(raw["cells"]):
            if cell["cell_type"] != "code":
                continue
            if "export" not in cell.get("metadata", {}).get("tags", []):
                continue
            src = "".join(cell["source"])
            exec(compile(src, f"<optimizers.ipynb export cell {i}>", "exec"), mod.__dict__)
            n += 1
    finally:
        os.chdir(cwd)
    assert n >= 5, f"expected the export cells, found {n}"
    return mod


@pytest.fixture(scope="session")
def nb() -> types.ModuleType:
    return load_export_module()
