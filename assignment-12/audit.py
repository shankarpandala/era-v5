"""Read-only audit of Assignment 12's executed notebook and saved evidence.

Recompute storage, ring traffic, dense-matmul work, padding, and mixed-precision
formulas without importing or executing the simulator. Timings are checked only
for finiteness and internal consistency, never against a performance threshold.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / "submission_artifacts"


def _close(a, b, *, rel=1e-10, absolute=1e-10):
    return math.isclose(a, b, rel_tol=rel, abs_tol=absolute)


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def run(check=None) -> int:
    """Call check(name, ok, detail='') for each assertion; return failure count.

    Omitting ``check`` is supported for scripts. No files are created or changed.
    Missing or malformed artifacts produce a failing audit, not a traceback.
    """
    failures = 0

    def ck(name, ok, detail=""):
        nonlocal failures
        ok = bool(ok)
        failures += int(not ok)
        if check is not None:
            check(name, ok, detail)

    try:
        results = json.loads((ART / "results.json").read_text())
        saved_config = json.loads((ART / "run_config.json").read_text())
        notebook = json.loads((HERE / "zero_simulation.ipynb").read_text())
        source = (HERE / "zero_simulator.py").read_text()
    except (OSError, ValueError) as exc:
        ck("required artifacts can be read", False, str(exc))
        return failures

    try:
        ck("all saved numeric metrics are finite", _finite_tree(results))
        config = results["config"]
        ck("run_config is the exact recorded experiment configuration", config == saved_config)
        n, steps, dims = config["world_size"], config["steps"], config["dims"]
        batch = config["local_batch"]
        ck("experiment ran on 32 virtual ranks", n == 32)
        ck("experiment contains at least eight optimizer steps (full or smoke run)", steps >= 8)
        ck("configuration dimensions and minibatches are positive",
           len(dims) >= 2 and all(isinstance(d, int) and d > 0 for d in dims)
           and batch > 0 and config["workers"] > 0)
        counts = [(a + 1) * b for a, b in zip(dims[:-1], dims[1:])]
        padded = [math.ceil(p / n) * n for p in counts]
        p, q = sum(counts), sum(padded)
        model = results["model"]
        ck("live model parameters re-derived from every weight and bias", model["parameters"] == p)
        ck("layer-wise padded capacity re-derived", model["padded_parameters"] == q)
        ck("padding overhead re-derived", model["padding_parameters"] == q - p)

        code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        execution = [c.get("execution_count") for c in code]
        ck("notebook contains executed code cells", bool(code) and all(type(v) is int and v > 0 for v in execution))
        if all(type(v) is int for v in execution):
            ck("notebook execution order is unique and increasing", execution == sorted(set(execution)))
        errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
        ck("notebook contains no execution errors", not errors)
        ck("standalone notebook embeds exact simulator source",
           source in ["".join(c["source"]) for c in code if "export" in c.get("metadata", {}).get("tags", [])])
        ck("notebook retains visible execution output", any(c.get("outputs") for c in code))
        for name in ("memory_comparison", "training_parity", "compute_communication", "scaling"):
            path = ART / "plots" / f"{name}.png"
            ck(f"plot {name} is a nonempty PNG", path.is_file() and path.stat().st_size > 1000
               and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n")
        ck("run log exists and records output", (ART / "run.log").is_file() and (ART / "run.log").stat().st_size > 50)

        reference = results["reference"]
        ck("single-batch reference has a full loss curve", len(reference["loss_curve"]) == steps)
        ck("single-batch reference genuinely learns", reference["final_loss"] < 0.9 * reference["loss_curve"][0])
        stages = results["stages"]
        ck("all four stages were executed exactly once", sorted(s["stage"] for s in stages) == list(range(4)))
        fields = ("parameters", "gradients", "first_moment", "second_moment")
        forward = sum(2 * batch * a * b for a, b in zip(dims[:-1], dims[1:]))
        backward = forward + sum(2 * batch * a * b for a, b in zip(dims[1:-1], dims[2:]))
        for result in stages:
            stage = result["stage"]
            label = f"stage {stage}"
            state = result["state_bytes_per_rank"]
            shard_fields = ({}, {"first_moment", "second_moment"},
                            {"gradients", "first_moment", "second_moment"}, set(fields))[stage]
            expected = {field: q * 8 // (n if field in shard_fields else 1) for field in fields}
            ck(f"{label}: actual FP64 persistent memory follows ownership", all(state[field] == expected[field] for field in fields))
            ck(f"{label}: persistent total sums components", state["total"] == sum(state[field] for field in fields))
            ck(f"{label}: tracked peak includes transient live arrays",
               result["transient_peak_bytes_per_rank"] > 0
               and result["peak_tracked_bytes_per_rank"] == state["total"] + result["transient_peak_bytes_per_rank"])
            ck(f"{label}: full finite training curve", len(result["loss_curve"]) == steps
               and all(math.isfinite(x) and x >= 0 for x in result["loss_curve"]))
            ck(f"{label}: every training loss matches independent batch grouping",
               len(result["loss_curve"]) == len(reference["loss_curve"])
               and all(_close(a, b) for a, b in zip(result["loss_curve"], reference["loss_curve"])))
            ck(f"{label}: final evaluation matches reference", _close(result["final_loss"], reference["final_loss"]))
            ck(f"{label}: parameters match DDP and single-batch reference",
               0 <= result["max_parameter_difference_vs_ddp"] < 1e-10
               and 0 <= result["max_parameter_difference_vs_reference"] < 1e-10)
            ck(f"{label}: actual thread workers recorded", 1 <= result["thread_worker_count"] <= config["workers"])
            times = result["step_seconds"]
            ck(f"{label}: positive machine-specific step measurements", len(times) == steps and all(t > 0 for t in times))
            ck(f"{label}: timing average is internally consistent",
               _close(result["seconds_per_step"], sum(times) / steps))
            ck(f"{label}: total includes measured steps and executor overhead", result["seconds_total"] >= sum(times))
            communication = result["communication"]
            events = communication["events_first_step"]
            expected_bytes = (2, 2, 2, 3)[stage] * (n - 1) * (q // n) * 8
            ck(f"{label}: ring send bytes re-derived for its collective schedule",
               communication["bytes_sent_per_rank_per_step"] == expected_bytes)
            ck(f"{label}: collective events sum to reported traffic",
               sum(event["analytic_bytes_sent_per_rank"] for event in events) == expected_bytes)
            ck(f"{label}: collective count matches layer schedule", len(events) == len(counts) * (1, 2, 2, 3)[stage])
            ck(f"{label}: collective counters agree with event log",
               dict(Counter(e["kind"] for e in events)) == communication["collective_counts_per_step"])
            for index, event in enumerate(events):
                kind = event["kind"].replace("-", "_").lower()
                factor = 2 if kind in ("allreduce", "all_reduce") else 1
                ck(f"{label}: collective {index} counts payload and ring sends",
                   event["payload_bytes"] == event["payload_elements"] * 8
                   and event["analytic_bytes_sent_per_rank"] == factor * (n - 1) * (event["payload_elements"] // n) * 8)
            compute = result["computation"]
            ck(f"{label}: dense forward and backward FLOPs independently re-derived",
               compute["forward_matmul_flops_per_rank_per_step"] == forward
               and compute["backward_matmul_flops_per_rank_per_step"] == backward)
            ck(f"{label}: optimizer work follows actual padded ownership",
               compute["optimizer_element_updates_per_rank_per_step"] == q // (n if stage else 1))

        theory = results["theory"]
        tp, tn = theory["parameter_count"], theory["world_size"]
        ck("mixed precision example uses 32 ranks", tp > 0 and tn == 32)
        ck("mixed precision accounting has FP16 weights/gradients and FP32 master+Adam states",
           theory["bytes_per_parameter"] == {"parameters": 2, "gradients": 2, "optimizer": 12})
        ck("mixed precision has all four stage rows", sorted(s["stage"] for s in theory["stages"]) == list(range(4)))
        for row in theory["stages"]:
            stage = row["stage"]
            parameters = 2 * tp / (tn if stage == 3 else 1)
            gradients = 2 * tp / (tn if stage >= 2 else 1)
            optimizer = 12 * tp / (tn if stage >= 1 else 1)
            ck(f"mixed precision stage {stage}: unpadded 2/2/12-byte formula",
               _close(row["parameters"], parameters) and _close(row["gradients"], gradients)
               and _close(row["optimizer"], optimizer) and _close(row["total"], parameters + gradients + optimizer))
        scaling = results["scaling"]
        expected_pairs = {(ranks, stage) for ranks in (1, 2, 4, 8, 16, 32, 64) for stage in range(4)}
        ck("scaling table includes all rank counts and ZeRO stages",
           len(scaling) == len(expected_pairs) and {(r["world_size"], r["stage"]) for r in scaling} == expected_pairs)
        for row in scaling:
            ranks, stage = row["world_size"], row["stage"]
            value = 2 * tp / (ranks if stage == 3 else 1) + 2 * tp / (ranks if stage >= 2 else 1) + 12 * tp / (ranks if stage >= 1 else 1)
            ck(f"scaling N={ranks}, stage={stage}: theoretical per-rank bytes", _close(row["bytes_per_rank"], value))
    except (KeyError, TypeError, IndexError, ValueError, ZeroDivisionError, OSError) as exc:
        ck("artifact schema and values are complete", False, f"{type(exc).__name__}: {exc}")
    return failures


if __name__ == "__main__":
    def report(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))

    failed = run(report)
    print("verdict: PASS" if failed == 0 else f"verdict: {failed} FAILING CHECKS")
    raise SystemExit(int(failed != 0))
