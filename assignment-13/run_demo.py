"""Execute the four notebooks in order (saving outputs), then audit the saved evidence.

    python run_demo.py                 # full pilot as committed (budgets below), then audit
    python run_demo.py --fast          # tiny smoke budgets; overwrites outputs
    python run_demo.py --verify-only   # read-only audit of the committed artifacts
    python run_demo.py --tokens 50000000 --screen-tokens 5000000   # the assignment's full scale (GPU advised)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
NOTEBOOKS = ["01_baseline_fixed_batch.ipynb", "02_reversible_fixed_batch.ipynb",
             "03_reversible_max_batch.ipynb", "04_report.ipynb"]
PILOT_TOKENS = 5_000_000        # what a 4-core CPU affords in an afternoon (see README)
PILOT_SCREEN_TOKENS = 1_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="read-only artifact audit")
    parser.add_argument("--fast", action="store_true", help="smoke budgets (a few steps); overwrites outputs")
    parser.add_argument("--tokens", type=int, default=PILOT_TOKENS, help="training tokens per arm")
    parser.add_argument("--screen-tokens", type=int, default=PILOT_SCREEN_TOKENS, help="tokens per screened variant")
    parser.add_argument("--batch", type=int, default=32, help="the fixed batch size")
    parser.add_argument("--only", nargs="*", default=None, help="subset of notebooks to execute")
    args = parser.parse_args()
    log_path = HERE / "submission_artifacts" / "run.log"
    if not args.verify_only:
        import nbformat
        from nbclient import NotebookClient

        env = {"A13_TOKENS": str(40_000 if args.fast else args.tokens),
               "A13_SCREEN_TOKENS": str(16_000 if args.fast else args.screen_tokens),
               "A13_BATCH": str(args.batch), "A13_MAXBATCH_CEILING": "64" if args.fast else "8192",
               "A13_ART_DIR": str(HERE / "submission_artifacts"), "A13_DATA_DIR": str(HERE / "data")}
        os.environ.update(env)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"run_demo.py started {time.strftime('%Y-%m-%d %H:%M:%S')} mode={'fast' if args.fast else 'pilot'} env={env}"]
        for name in NOTEBOOKS:
            if args.only and name not in args.only:
                continue
            path = HERE / name
            nb = nbformat.read(path, as_version=4)
            start = time.perf_counter()
            print(f"Executing {name} ...", flush=True)
            NotebookClient(nb, timeout=None, kernel_name="python3",
                           resources={"metadata": {"path": str(HERE)}}).execute()
            nbformat.write(nb, path)
            elapsed = time.perf_counter() - start
            lines.append(f"{name}: executed top to bottom in {elapsed / 60:.1f} min")
            print(lines[-1], flush=True)
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    import audit
    messages = []

    def check(name, ok, detail=""):
        line = f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else "")
        print(line)
        messages.append(line)

    failures = audit.run(check)
    verdict = f"verdict: {'PASS' if failures == 0 else 'FAIL'}"
    print(verdict)
    if not args.verify_only:
        with log_path.open("a", encoding="utf-8") as out:
            out.write("\n".join(messages + [verdict]) + "\n")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
