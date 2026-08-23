"""Assignment 9 — one command re-derives everything.

    python run_demo.py               # full run: execute the notebook (~5-10 min CPU),
                                     # write it back with outputs, then audit
    python run_demo.py --fast        # ~1-2 min smoke run (A9_FAST=1 budgets)
    python run_demo.py --verify-only # no execution: audit the committed artifacts (~5 s)

Pipeline:

    loss_harness.ipynb --nbclient--> executed notebook + submission_artifacts/*
                                          |
                            audit.py (independent, reads disk only)
                                          |
                        [PASS]/[FAIL] lines -> submission_artifacts/run.log

The notebook is the single source of truth for all harness code; this script only
executes and verifies it. NOTE: --fast overwrites submission_artifacts/ with the
reduced budgets, so the LAST run before committing must be the full one.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "loss_harness.ipynb"
ART = HERE / "submission_artifacts"


class RunLog:
    """Mirrors every line to stdout and submission_artifacts/run.log."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8")
        self._t0 = time.time()
        self.failures: list[str] = []

    def say(self, msg: str) -> None:
        line = f"[{time.time() - self._t0:8.1f}s] {msg}"
        print(line)
        self._fh.write(line + "\n")
        self._fh.flush()

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        tag = "[PASS]" if ok else "[FAIL]"
        self.say(f"{tag} {name}" + (f" | {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)

    def close(self) -> None:
        self._fh.close()


def execute_notebook(log: RunLog, fast: bool) -> None:
    import nbformat
    from nbclient import NotebookClient

    if fast:
        os.environ["A9_FAST"] = "1"
    else:
        os.environ.pop("A9_FAST", None)

    nb = nbformat.read(str(NOTEBOOK), as_version=4)
    client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(HERE)}})
    log.say(f"executing {NOTEBOOK.name} top to bottom "
            f"({'fast' if fast else 'full'} budgets) ...")
    t0 = time.time()
    client.execute()
    log.say(f"notebook executed in {time.time() - t0:.1f}s")
    nbformat.write(nb, str(NOTEBOOK))
    log.say(f"executed notebook written back to {NOTEBOOK.name} "
            f"({NOTEBOOK.stat().st_size / 1024:.0f} KiB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true",
                    help="reduced budgets via A9_FAST=1 (overwrites artifacts)")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip execution; audit the committed artifacts")
    args = ap.parse_args()

    log = RunLog(ART / "run.log")
    log.say(f"assignment-9 run_demo | mode="
            f"{'verify-only' if args.verify_only else 'fast' if args.fast else 'full'}")

    if not args.verify_only:
        try:
            execute_notebook(log, fast=args.fast)
        except Exception as exc:  # a notebook that cannot run is a hard failure
            log.check("notebook executes top to bottom", False, repr(exc)[:300])
            log.say("verdict: FAIL")
            log.close()
            return 1
        log.check("notebook executes top to bottom", True)

    sys.path.insert(0, str(HERE))
    import audit

    n_fail = audit.run(log.check)
    log.say(f"audit: {('PASS' if n_fail == 0 else f'{n_fail} FAILING CHECKS')}")
    log.say(f"verdict: {'PASS' if not log.failures else 'FAIL'}")
    log.close()
    return 0 if not log.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
