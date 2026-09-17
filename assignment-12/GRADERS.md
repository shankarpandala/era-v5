# Assignment 12 — grader card

Open [`zero_simulation.ipynb`](zero_simulation.ipynb), already executed. The complete
engine is embedded, so **Run all** works on Colab CPU without cloning this repo.

| Requirement | Where to look |
|---|---|
| 32 virtual GPUs | Notebook §§2–3: 32 rank objects, 32-thread pool, separate state buffers, eight local examples per rank. |
| Real demo model | Notebook §3: 7,312-parameter MLP, Adam, 40 steps; loss falls and weights match the full-batch oracle. |
| ZeRO-1/2/3 | Notebook §§1–2: real shards and NumPy collective operations; DDP control. |
| Memory changes | Notebook §4: owned `nbytes`, scoped tracked payload peak, precision assumptions, separate 1B-parameter projection. |
| Computation changes | Notebook §5: unchanged model matmul work, sharded Adam updates, collective trace and ring sent-byte model. |
| Scaling explanation | Notebook §6: analytical 1–64-rank sweep and N=1 boundary. |
| Detailed explanation | [`README.md`](README.md): assumptions, algorithms, results, limitations, sources, reproduction. |
| Agent collaboration | Agents split conventions research, simulator implementation, mathematics review, and independent tests/audit. |

```bash
cd assignment-12
python -m pip install -r requirements.txt
python run_demo.py --verify-only
python -m pytest tests -q
python run_demo.py
```

**Interpretation:** real numerical CPU training; measured explicit array payloads;
analytical network bytes and billion-parameter projection. No CUDA allocation,
network throughput, or 32-GPU speedup is claimed. Full-batch reference and stage
parity are enforced at an absolute parameter tolerance of `1e-10`.

Repository: [shankarpandala/era-v5](https://github.com/shankarpandala/era-v5).
