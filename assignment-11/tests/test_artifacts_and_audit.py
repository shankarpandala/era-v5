"""The committed artifacts must satisfy the independent audit, and the notebook must be
committed fully executed (these run against the committed full-run artifacts)."""
import json
import sys
from pathlib import Path

A11 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(A11))


def test_committed_artifacts_pass_the_full_audit():
    import audit

    failures = []

    def check(name, ok, detail=""):
        if not ok:
            failures.append(f"{name} | {detail}")

    n = audit.run(check)
    assert n == 0, "audit failures:\n" + "\n".join(failures)


def test_committed_run_is_full_not_fast():
    R = json.loads((A11 / "submission_artifacts" / "results.json").read_text())
    assert R["config"]["fast"] is False


def test_committed_run_log_agrees_with_the_audit():
    log = (A11 / "submission_artifacts" / "run.log").read_text()
    assert "verdict: PASS" in log and "[FAIL]" not in log


def test_notebook_committed_executed():
    raw = json.loads((A11 / "optimizers.ipynb").read_text())
    code_cells = [c for c in raw["cells"] if c["cell_type"] == "code"]
    assert all(isinstance(c.get("execution_count"), int) for c in code_cells)
    assert not any(o.get("output_type") == "error" for c in code_cells for o in c.get("outputs", []))
