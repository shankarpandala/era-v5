"""The committed artifacts must satisfy the independent audit, and the notebook
must be committed fully executed. (These run against the repo's committed full-run
artifacts, not the FAST fixtures the other tests build.)"""
import json
import sys
from pathlib import Path

A10 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(A10))


def test_committed_artifacts_pass_the_full_audit():
    import audit

    failures = []

    def check(name, ok, detail=""):
        if not ok:
            failures.append(f"{name} | {detail}")

    n = audit.run(check)
    assert n == 0, "audit failures:\n" + "\n".join(failures)


def test_committed_run_is_full_not_fast():
    R = json.loads((A10 / "submission_artifacts" / "results.json").read_text())
    assert R["config"]["fast"] is False, \
        "the committed artifacts must come from a full run"


def test_notebook_committed_executed():
    raw = json.loads((A10 / "training_loop.ipynb").read_text())
    code_cells = [c for c in raw["cells"] if c["cell_type"] == "code"]
    assert all(isinstance(c.get("execution_count"), int) for c in code_cells)
    assert not any(o.get("output_type") == "error"
                   for c in code_cells for o in c.get("outputs", []))


def test_determinism_of_seeded_training(nb):
    """Two hazard arms with the same seed produce identical curves."""
    _, c1, i1, s1 = nb.train_hazard("correct", 4242, steps=12)
    _, c2, i2, s2 = nb.train_hazard("correct", 4242, steps=12)
    assert c1["tw"] == c2["tw"] and i1 == i2 and s1 == s2
