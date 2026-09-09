"""Assignment 11 — one command re-derives everything.

    python run_demo.py               # full run: execute the notebook (~45-60 min on an
                                     # Apple M5 Pro / MPS; longer on CPU), write it back
                                     # with outputs, then audit
    python run_demo.py --fast        # ~4 min smoke run (A11_FAST=1 budgets)
    python run_demo.py --resume      # full run, reusing sweep runs already cached under
                                     # submission_artifacts/runs/ (after a crash)
    python run_demo.py --verify-only # no execution: audit the committed artifacts (~2 s)

Pipeline:

    optimizers.ipynb --nbclient--> executed notebook + submission_artifacts/*
                                        |
                          audit.py (independent, reads disk only)
                                        |
                      [PASS]/[FAIL] lines -> submission_artifacts/run.log

The notebook is the single source of truth for all harness code; this script only
executes and verifies it. NOTE: --fast overwrites submission_artifacts/ with the reduced
budgets, so the LAST run before committing must be the full one.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "optimizers.ipynb"
ART = HERE / "submission_artifacts"


class RunLog:
    """Mirrors every line to stdout and submission_artifacts/run.log."""

    def __init__(self, path: Path, to_file: bool = True):
        self._fh = None
        if to_file:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "w", encoding="utf-8")
        self._t0 = time.time()
        self.failures: list[str] = []

    def say(self, msg: str) -> None:
        line = f"[{time.time() - self._t0:8.1f}s] {msg}"
        print(line, flush=True)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        tag = "[PASS]" if ok else "[FAIL]"
        self.say(f"{tag} {name}" + (f" | {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


def execute_notebook(log: RunLog, fast: bool, resume: bool) -> None:
    import nbformat
    from nbclient import NotebookClient

    os.environ["A11_FAST"] = "1" if fast else ""
    os.environ["A11_RESUME"] = "1" if resume else ""
    if not fast:
        os.environ.pop("A11_FAST", None)
    nb = nbformat.read(str(NOTEBOOK), as_version=4)
    # no per-cell timeout: the sweep cells legitimately run for tens of minutes
    client = NotebookClient(nb, timeout=None, kernel_name="python3",
                            resources={"metadata": {"path": str(HERE)}})
    log.say(f"executing {NOTEBOOK.name} top to bottom "
            f"({'fast' if fast else 'full'} budgets{', resuming cached runs' if resume else ''}) ...")
    t0 = time.time()
    try:
        client.execute()
    except Exception as exc:
        # keep the partial outputs for diagnosis without clobbering the committed notebook
        failed = NOTEBOOK.with_name("optimizers.failed.ipynb")
        nbformat.write(nb, str(failed))
        log.say(f"execution failed after {time.time() - t0:.1f}s; partial outputs in {failed.name}")
        log.say("error tail: " + str(exc)[-1500:].replace("\n", " | "))
        raise
    log.say(f"notebook executed in {time.time() - t0:.1f}s")
    nbformat.write(nb, str(NOTEBOOK))
    log.say(f"executed notebook written back to {NOTEBOOK.name} "
            f"({NOTEBOOK.stat().st_size / 1024:.0f} KiB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="reduced budgets via A11_FAST=1 (overwrites artifacts)")
    ap.add_argument("--resume", action="store_true", help="reuse sweep runs cached under submission_artifacts/runs/")
    ap.add_argument("--verify-only", action="store_true", help="skip execution; audit the committed artifacts")
    args = ap.parse_args()

    log = RunLog(ART / "run.log", to_file=not args.verify_only)
    log.say(f"assignment-11 run_demo | mode="
            f"{'verify-only' if args.verify_only else 'fast' if args.fast else 'full'}")
    if args.verify_only:
        log.say("execution evidence: the committed optimizers.ipynb itself — the audit checks its "
                "execution counts are monotone and its outputs error-free")
    else:
        try:
            execute_notebook(log, fast=args.fast, resume=args.resume)
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
