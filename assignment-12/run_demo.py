"""Execute the standalone notebook, preserve outputs, then audit saved evidence."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='read-only artifact audit')
    parser.add_argument('--fast', action='store_true', help='8 training steps; overwrites outputs')
    args = parser.parse_args()
    if not args.verify_only:
        import nbformat
        from nbclient import NotebookClient

        # These limits also reach the fresh notebook kernel, before NumPy is imported.
        for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                         'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[variable] = '1'
        os.environ['A12_FAST'] = '1' if args.fast else '0'
        os.environ['A12_ART_DIR'] = str(HERE / 'submission_artifacts')
        notebook_path = HERE / 'zero_simulation.ipynb'
        notebook = nbformat.read(notebook_path, as_version=4)
        start = time.perf_counter()
        print('Executing all notebook cells on CPU...', flush=True)
        NotebookClient(notebook, timeout=600, kernel_name='python3',
                       resources={'metadata': {'path': str(HERE)}}).execute()
        nbformat.write(notebook, notebook_path)
        elapsed = time.perf_counter() - start
        print(f'Executed and saved notebook in {elapsed:.2f}s.', flush=True)
        (HERE / 'submission_artifacts' / 'run.log').write_text(
            f'Executed zero_simulation.ipynb top to bottom in {elapsed:.2f}s.\n'
            f'Mode: {"fast" if args.fast else "full"}; CPU simulation.\n', encoding='utf-8')

    import audit
    messages = []

    def check(name, ok, detail=''):
        line = f'[{"PASS" if ok else "FAIL"}] {name}' + (f' | {detail}' if detail else '')
        print(line)
        messages.append(line)

    failures = audit.run(check)
    verdict = f'verdict: {"PASS" if failures == 0 else "FAIL"}'
    print(verdict)
    if not args.verify_only:
        with (HERE / 'submission_artifacts' / 'run.log').open('a', encoding='utf-8') as output:
            output.write('\n'.join(messages + [verdict]) + '\n')
    return int(failures != 0)


if __name__ == '__main__':
    raise SystemExit(main())
