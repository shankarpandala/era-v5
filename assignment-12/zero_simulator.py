"""A numerical, CPU-thread simulation of ZeRO stages 0, 1, 2 and 3.

Virtual ranks own real NumPy parameter, gradient and Adam-state buffers. The
coordinator performs mathematically correct collectives in process; network
traffic is an analytic ideal-ring estimate, NOT measured network traffic.

Memory means ndarray payload bytes for explicitly tracked rank-owned buffers.
It excludes datasets, the coordinator's reduction/gather scratch, Python object
overhead, NumPy/BLAS internal workspaces and allocator/RSS overhead. Consequently
it must not be presented as a process RSS or physical GPU peak measurement.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import os
import threading
import time
from typing import Callable

import numpy as np


DTYPE = np.float64
ITEMSIZE = np.dtype(DTYPE).itemsize
BETA1, BETA2, EPSILON = 0.9, 0.999, 1e-8
STAGE_NAMES = {0: "DDP baseline", 1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3"}


@dataclass(frozen=True)
class Config:
    """Small defaults keep the complete 32-rank demonstration CPU friendly."""

    world_size: int = 32
    dims: tuple[int, ...] = (32, 64, 64, 16)
    local_batch: int = 8
    steps: int = 40
    learning_rate: float = 0.01
    seed: int = 12
    workers: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(self, "dims", tuple(self.dims))
        for name in ("world_size", "local_batch", "steps", "workers"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if len(self.dims) < 2 or any(
            not isinstance(d, int) or isinstance(d, bool) or d < 1 for d in self.dims
        ):
            raise ValueError("dims must contain at least two positive integers")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")


@dataclass(frozen=True)
class LayerLayout:
    input_size: int
    output_size: int
    parameter_count: int
    padded_count: int
    shard_count: int


def layer_layouts(config: Config) -> list[LayerLayout]:
    result = []
    for fan_in, fan_out in zip(config.dims[:-1], config.dims[1:]):
        count = (fan_in + 1) * fan_out
        padded = ((count + config.world_size - 1) // config.world_size) * config.world_size
        result.append(LayerLayout(fan_in, fan_out, count, padded, padded // config.world_size))
    return result


def _unpack(flat: np.ndarray, layout: LayerLayout) -> tuple[np.ndarray, np.ndarray]:
    weights_count = layout.input_size * layout.output_size
    return (
        flat[:weights_count].reshape(layout.input_size, layout.output_size),
        flat[weights_count:layout.parameter_count],
    )


def make_problem(config: Config) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Return padded initial layers and independent, fixed per-rank X/Y arrays.

    A seeded teacher MLP supplies deterministic regression targets. All stages
    receive the same initialization and global batch, reused at every step so
    loss curves and parameter comparisons isolate the sharding algorithm.
    """
    rng = np.random.default_rng(config.seed)
    initial, teacher = [], []
    layouts = layer_layouts(config)
    for layout in layouts:
        flat = np.zeros(layout.padded_count, dtype=DTYPE)
        weights, _ = _unpack(flat, layout)
        weights[:] = rng.normal(0, np.sqrt(1.0 / layout.input_size), weights.shape)
        initial.append(flat)
        teacher_flat = np.zeros_like(flat)
        teacher_w, teacher_b = _unpack(teacher_flat, layout)
        teacher_w[:] = rng.normal(0, np.sqrt(1.0 / layout.input_size), teacher_w.shape)
        teacher_b[:] = rng.normal(0, 0.1, teacher_b.shape)
        teacher.append(teacher_flat)
    inputs, targets = [], []
    for _ in range(config.world_size):
        x = rng.normal(size=(config.local_batch, config.dims[0])).astype(DTYPE)
        y = x
        for index, (flat, layout) in enumerate(zip(teacher, layouts)):
            weights, bias = _unpack(flat, layout)
            y = y @ weights + bias
            if index < len(layouts) - 1:
                y = np.tanh(y)
        inputs.append(x)
        targets.append(y.copy())
    return initial, inputs, targets


def loss_and_grad(
    parameters: list[np.ndarray], x: np.ndarray, y: np.ndarray, dims: tuple[int, ...]
) -> tuple[float, list[np.ndarray]]:
    """Ordinary unsharded MLP reference; padding has zero gradient."""
    layouts = [
        LayerLayout(a, b, (a + 1) * b, len(p), len(p))
        for a, b, p in zip(dims[:-1], dims[1:], parameters)
    ]
    activations = [x]
    for index, (flat, layout) in enumerate(zip(parameters, layouts)):
        weights, bias = _unpack(flat, layout)
        out = activations[-1] @ weights + bias
        activations.append(np.tanh(out) if index < len(layouts) - 1 else out)
    delta = activations[-1] - y
    loss = float(np.mean(delta * delta))
    delta *= 2.0 / delta.size
    gradients = [np.zeros_like(p) for p in parameters]
    for index in range(len(layouts) - 1, -1, -1):
        weights, _ = _unpack(parameters[index], layouts[index])
        grad_w, grad_b = _unpack(gradients[index], layouts[index])
        grad_w[:] = activations[index].T @ delta
        grad_b[:] = delta.sum(axis=0)
        if index:
            delta = (delta @ weights.T) * (1.0 - activations[index] ** 2)
    return loss, gradients


def full_batch_reference(config: Config) -> dict:
    """Single global-batch Adam reference, outside virtual-rank timing/memory."""
    parameters, inputs, targets = make_problem(config)
    x, y = np.concatenate(inputs), np.concatenate(targets)
    first = [np.zeros_like(p) for p in parameters]
    second = [np.zeros_like(p) for p in parameters]
    losses = []
    for step in range(1, config.steps + 1):
        loss, gradients = loss_and_grad(parameters, x, y, config.dims)
        losses.append(loss)
        for p, g, m, v in zip(parameters, gradients, first, second):
            m *= BETA1
            m += (1.0 - BETA1) * g
            v *= BETA2
            v += (1.0 - BETA2) * g * g
            p -= config.learning_rate * (m / (1.0 - BETA1**step)) / (
                np.sqrt(v / (1.0 - BETA2**step)) + EPSILON
            )
    final_loss, _ = loss_and_grad(parameters, x, y, config.dims)
    return {"loss_curve": losses, "final_loss": final_loss, "parameters": parameters}


@dataclass
class RankState:
    """The storage owned by one virtual GPU; arrays never alias another rank."""

    rank: int
    inputs: np.ndarray
    targets: np.ndarray
    parameters: list[np.ndarray]
    gradients: list[np.ndarray]
    first_moment: list[np.ndarray]
    second_moment: list[np.ndarray]
    activations: list[np.ndarray] = field(default_factory=list)
    materialized_layer: np.ndarray | None = None
    local_gradient: np.ndarray | None = None
    delta: np.ndarray | None = None
    next_delta: np.ndarray | None = None
    derivative_workspace: np.ndarray | None = None
    optimizer_workspace: list[np.ndarray] = field(default_factory=list)
    peak_tracked_bytes: int = 0
    peak_transient_bytes: int = 0
    transient_component_peaks: dict = field(default_factory=dict)
    worker_ids: set = field(default_factory=set)

    def state_bytes(self) -> dict[str, int]:
        result = {
            name: sum(array.nbytes for array in getattr(self, name))
            for name in ("parameters", "gradients", "first_moment", "second_moment")
        }
        result["total"] = sum(result.values())
        return result

    def observe(self) -> None:
        # The input batch is data, not an activation allocation. All other
        # entries here own independent buffers; views are not counted twice.
        transient = {
            "activation_cache": sum(a.nbytes for a in self.activations[1:]),
            "materialized_layer": 0 if self.materialized_layer is None else self.materialized_layer.nbytes,
            "local_gradient": 0 if self.local_gradient is None else self.local_gradient.nbytes,
            "backprop_signals": sum(a.nbytes for a in (self.delta, self.next_delta) if a is not None),
            "derivative_workspace": 0 if self.derivative_workspace is None else self.derivative_workspace.nbytes,
            "optimizer_workspace": sum(a.nbytes for a in self.optimizer_workspace),
        }
        transient_total = sum(transient.values())
        self.peak_transient_bytes = max(self.peak_transient_bytes, transient_total)
        self.peak_tracked_bytes = max(self.peak_tracked_bytes, self.state_bytes()["total"] + transient_total)
        for name, value in transient.items():
            self.transient_component_peaks[name] = max(self.transient_component_peaks.get(name, 0), value)


class Simulator:
    """One ZeRO stage, numerical collectives, and a configurable thread pool.

    The executor schedules logical ranks onto CPU threads; it does not promise
    32 simultaneous OS threads or assign a permanent thread to each rank.
    A stage-3 rank retains only parameter shards between module operations.
    """

    def __init__(self, config: Config, stage: int):
        if stage not in STAGE_NAMES:
            raise ValueError("stage must be 0, 1, 2 or 3")
        self.config, self.stage = config, stage
        self.layouts = layer_layouts(config)
        initial, inputs, targets = make_problem(config)
        self.ranks = []
        for rank in range(config.world_size):
            parameters, gradients, first, second = [], [], [], []
            for flat, layout in zip(initial, self.layouts):
                owned = slice(rank * layout.shard_count, (rank + 1) * layout.shard_count)
                parameters.append(flat[owned].copy() if stage == 3 else flat.copy())
                gradients.append(np.zeros(layout.shard_count if stage >= 2 else layout.padded_count, dtype=DTYPE))
                first.append(np.zeros(layout.shard_count if stage >= 1 else layout.padded_count, dtype=DTYPE))
                second.append(np.zeros_like(first[-1]))
            state = RankState(rank, inputs[rank], targets[rank], parameters, gradients, first, second)
            state.observe()
            self.ranks.append(state)
        # No initial full-model copy is retained on the coordinator.
        self.events_first_step: list[dict] = []
        self.phase_seconds = {name: 0.0 for name in ("forward", "backward", "communication", "optimizer")}
        self._step = 0
        self._trained = False

    def state_bytes_per_rank(self) -> dict[str, int]:
        counts = [rank.state_bytes() for rank in self.ranks]
        if any(count != counts[0] for count in counts):
            raise AssertionError("Layer padding should give all ranks equal storage")
        return counts[0]

    def _dispatch(self, pool: ThreadPoolExecutor, phase: str, function: Callable) -> list:
        started = time.perf_counter()
        def invoke(rank: RankState):
            rank.worker_ids.add(threading.get_ident())
            return function(rank)
        result = list(pool.map(invoke, self.ranks))
        self.phase_seconds[phase] += time.perf_counter() - started
        return result

    def _event(self, kind: str, phase: str, layer: int) -> None:
        if self._step != 1:
            return
        count = self.layouts[layer].padded_count
        multiplier = 2 if kind == "all_reduce" else 1
        self.events_first_step.append({
            "kind": kind,
            "phase": phase,
            "layer": layer,
            "payload_elements": count,
            "payload_bytes": count * ITEMSIZE,
            # Integer arithmetic prevents roundoff/truncation for sizes such
            # as N=7. Layer padding makes count / world_size exact.
            "analytic_bytes_sent_per_rank": multiplier * (self.config.world_size - 1) * (count // self.config.world_size) * ITEMSIZE,
        })

    def _gather(self, layer: int, phase: str) -> None:
        started = time.perf_counter()
        if self.stage == 3:
            full = np.concatenate([rank.parameters[layer] for rank in self.ranks])
            for rank in self.ranks:
                rank.materialized_layer = full.copy()
                rank.observe()
        else:
            # Stages 1/2 updated only each rank's owned parameter segment.
            layout = self.layouts[layer]
            full = np.concatenate([
                rank.parameters[layer][rank.rank * layout.shard_count:(rank.rank + 1) * layout.shard_count]
                for rank in self.ranks
            ])
            for rank in self.ranks:
                np.copyto(rank.parameters[layer], full)
        self._event("all_gather", phase, layer)
        self.phase_seconds["communication"] += time.perf_counter() - started

    def _release_layer(self) -> None:
        for rank in self.ranks:
            rank.materialized_layer = None
            rank.observe()

    def _forward_layer(self, rank: RankState, layer: int) -> None:
        layout = self.layouts[layer]
        flat = rank.materialized_layer if self.stage == 3 else rank.parameters[layer]
        weights, bias = _unpack(flat, layout)
        output = np.empty((self.config.local_batch, layout.output_size), dtype=DTYPE)
        previous = rank.activations[-1]
        rank.activations.append(output)
        rank.observe()
        np.matmul(previous, weights, out=output)
        output += bias
        if layer < len(self.layouts) - 1:
            np.tanh(output, out=output)

    def _output_delta(self, rank: RankState) -> float:
        rank.delta = np.empty_like(rank.targets)
        rank.observe()
        np.subtract(rank.activations[-1], rank.targets, out=rank.delta)
        loss = float(np.einsum("ij,ij->", rank.delta, rank.delta) / rank.delta.size)
        rank.delta *= 2.0 / rank.delta.size
        return loss

    def _backward_layer(self, rank: RankState, layer: int) -> None:
        layout = self.layouts[layer]
        flat = rank.materialized_layer if self.stage == 3 else rank.parameters[layer]
        weights, _ = _unpack(flat, layout)
        if self.stage >= 2:
            rank.local_gradient = np.zeros(layout.padded_count, dtype=DTYPE)
            gradient = rank.local_gradient
        else:
            gradient = rank.gradients[layer]
            gradient.fill(0)
        rank.observe()
        gradient_w, gradient_b = _unpack(gradient, layout)
        np.matmul(rank.activations[layer].T, rank.delta, out=gradient_w)
        np.sum(rank.delta, axis=0, out=gradient_b)
        if layer:
            rank.next_delta = np.empty_like(rank.activations[layer])
            rank.derivative_workspace = np.empty_like(rank.next_delta)
            rank.observe()
            np.matmul(rank.delta, weights.T, out=rank.next_delta)
            np.multiply(rank.activations[layer], rank.activations[layer], out=rank.derivative_workspace)
            np.subtract(1.0, rank.derivative_workspace, out=rank.derivative_workspace)
            rank.next_delta *= rank.derivative_workspace
            rank.delta = rank.next_delta
            rank.next_delta = None
            rank.derivative_workspace = None
        else:
            rank.delta = None
        rank.observe()

    def _reduce_gradients(self, layer: int) -> None:
        started = time.perf_counter()
        layout = self.layouts[layer]
        # This centralized collective scratch is explicitly outside rank-local
        # memory accounting. Its contents implement the mean global gradient.
        reduced = np.zeros(layout.padded_count, dtype=DTYPE)
        for rank in self.ranks:
            source = rank.local_gradient if self.stage >= 2 else rank.gradients[layer]
            reduced += source
        reduced /= self.config.world_size
        for rank in self.ranks:
            owned = slice(rank.rank * layout.shard_count, (rank.rank + 1) * layout.shard_count)
            if self.stage == 0:
                np.copyto(rank.gradients[layer], reduced)
            elif self.stage == 1:
                np.copyto(rank.gradients[layer][owned], reduced[owned])
            else:
                np.copyto(rank.gradients[layer], reduced[owned])
                rank.local_gradient = None
            rank.observe()
        self._event("all_reduce" if self.stage == 0 else "reduce_scatter", "backward", layer)
        self.phase_seconds["communication"] += time.perf_counter() - started

    def _update(self, rank: RankState) -> None:
        for layer, layout in enumerate(self.layouts):
            owned = slice(rank.rank * layout.shard_count, (rank.rank + 1) * layout.shard_count)
            p = rank.parameters[layer]
            g = rank.gradients[layer]
            if self.stage in (1, 2):
                p = p[owned]
            if self.stage == 1:
                g = g[owned]
            m, v = rank.first_moment[layer], rank.second_moment[layer]
            rank.optimizer_workspace = [np.empty_like(m), np.empty_like(m)]
            rank.observe()
            temporary, update = rank.optimizer_workspace
            np.multiply(g, 1.0 - BETA1, out=temporary)
            m *= BETA1
            m += temporary
            np.multiply(g, g, out=temporary)
            temporary *= 1.0 - BETA2
            v *= BETA2
            v += temporary
            np.divide(v, 1.0 - BETA2**self._step, out=temporary)
            np.sqrt(temporary, out=temporary)
            temporary += EPSILON
            np.divide(m, 1.0 - BETA1**self._step, out=update)
            update /= temporary
            update *= self.config.learning_rate
            p -= update
            # Remove local references as well as rank-owned workspace slots.
            del temporary, update
            rank.optimizer_workspace = []
            rank.observe()

    def snapshot_parameters(self) -> list[np.ndarray]:
        """Reconstruct a full model for verification OUTSIDE the timed loop."""
        if self.stage == 3:
            return [np.concatenate([rank.parameters[layer] for rank in self.ranks]) for layer in range(len(self.layouts))]
        return [flat.copy() for flat in self.ranks[0].parameters]

    def train(self) -> dict:
        if self._trained:
            raise RuntimeError("A Simulator instance can be trained only once")
        self._trained = True
        losses, step_seconds = [], []
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.config.workers, thread_name_prefix="virtual-gpu") as pool:
            for step in range(1, self.config.steps + 1):
                self._step = step
                step_started = time.perf_counter()
                for rank in self.ranks:
                    rank.activations = [rank.inputs]
                for layer in range(len(self.layouts)):
                    if self.stage == 3:
                        self._gather(layer, "forward")
                    self._dispatch(pool, "forward", lambda rank, layer=layer: self._forward_layer(rank, layer))
                    if self.stage == 3:
                        self._release_layer()
                rank_losses = self._dispatch(pool, "backward", self._output_delta)
                losses.append(float(np.mean(rank_losses)))
                for layer in range(len(self.layouts) - 1, -1, -1):
                    if self.stage == 3:
                        self._gather(layer, "backward")
                    self._dispatch(pool, "backward", lambda rank, layer=layer: self._backward_layer(rank, layer))
                    self._reduce_gradients(layer)
                    if self.stage == 3:
                        self._release_layer()
                for rank in self.ranks:
                    rank.activations = []
                    rank.observe()
                self._dispatch(pool, "optimizer", self._update)
                if self.stage in (1, 2):
                    for layer in range(len(self.layouts)):
                        self._gather(layer, "optimizer")
                step_seconds.append(time.perf_counter() - step_started)
        seconds_total = time.perf_counter() - started
        # Verification materializes parameters and the global data batch only
        # after timing and tracked peak memory are complete.
        snapshot = self.snapshot_parameters()
        x = np.concatenate([rank.inputs for rank in self.ranks])
        y = np.concatenate([rank.targets for rank in self.ranks])
        final_loss, _ = loss_and_grad(snapshot, x, y, self.config.dims)
        batch = self.config.local_batch
        forward_flops = 2 * batch * sum(layout.input_size * layout.output_size for layout in self.layouts)
        backward_flops = forward_flops + 2 * batch * sum(
            layout.input_size * layout.output_size for layout in self.layouts[1:]
        )
        padded_parameters = sum(layout.padded_count for layout in self.layouts)
        worker_ids = set().union(*(rank.worker_ids for rank in self.ranks))
        component_names = self.ranks[0].transient_component_peaks
        return {
            "stage": self.stage,
            "name": STAGE_NAMES[self.stage],
            "loss_curve": losses,
            "final_loss": final_loss,
            "seconds_total": seconds_total,
            "seconds_per_step": float(np.mean(step_seconds)),
            "step_seconds": step_seconds,
            "phase_seconds": dict(self.phase_seconds),
            "state_bytes_per_rank": self.state_bytes_per_rank(),
            "peak_tracked_bytes_per_rank": max(rank.peak_tracked_bytes for rank in self.ranks),
            "transient_peak_bytes_per_rank": max(rank.peak_transient_bytes for rank in self.ranks),
            "transient_component_peaks_per_rank": {
                name: max(rank.transient_component_peaks[name] for rank in self.ranks)
                for name in component_names
            },
            "communication": {
                "events_first_step": list(self.events_first_step),
                "bytes_sent_per_rank_per_step": sum(event["analytic_bytes_sent_per_rank"] for event in self.events_first_step),
                "collective_counts_per_step": dict(Counter(event["kind"] for event in self.events_first_step)),
                "model": "Ideal ring, bytes SENT per rank; received bytes equal sent bytes. Not measured network traffic.",
            },
            "computation": {
                "forward_matmul_flops_per_rank_per_step": forward_flops,
                "backward_matmul_flops_per_rank_per_step": backward_flops,
                "optimizer_element_updates_per_rank_per_step": padded_parameters if self.stage == 0 else padded_parameters // self.config.world_size,
                "matmul_note": "Analytic dense GEMM count: multiply+add=2 FLOPs; no input-gradient GEMM for first layer. Excludes bias, tanh, MSE, Adam arithmetic and collective reductions.",
                "optimizer_note": "Element updates, not FLOPs; includes zero padding. DDP repeats full Adam update on every rank; ZeRO updates disjoint shards.",
            },
            "thread_worker_count": len(worker_ids),
            "executor_max_workers": self.config.workers,
        }


def run_experiment(config: Config | None = None) -> dict:
    """Train all stages, check against DDP/global-batch Adam, return JSON data."""
    config = Config() if config is None else config
    layouts = layer_layouts(config)
    reference = full_batch_reference(config)
    stages, baseline = [], None
    for stage in range(4):
        simulator = Simulator(config, stage)
        result = simulator.train()
        snapshot = simulator.snapshot_parameters()
        if baseline is None:
            baseline = snapshot
        result["max_parameter_difference_vs_ddp"] = float(max(
            np.max(np.abs(a - b)) for a, b in zip(snapshot, baseline)
        ))
        result["max_parameter_difference_vs_reference"] = float(max(
            np.max(np.abs(a - b)) for a, b in zip(snapshot, reference["parameters"])
        ))
        result["max_loss_curve_difference_vs_reference"] = float(np.max(np.abs(
            np.asarray(result["loss_curve"]) - np.asarray(reference["loss_curve"])
        )))
        stages.append(result)
        del simulator
    return {
        "config": {**asdict(config), "dims": list(config.dims)},
        "model": {
            "parameters": sum(layout.parameter_count for layout in layouts),
            "padded_parameters": sum(layout.padded_count for layout in layouts),
            "padding_parameters": sum(layout.padded_count - layout.parameter_count for layout in layouts),
            "layers": [asdict(layout) for layout in layouts],
            "dtype": str(np.dtype(DTYPE)),
            "itemsize": ITEMSIZE,
            "global_batch": config.world_size * config.local_batch,
        },
        "accounting": {
            "memory": "Exact ndarray.nbytes of rank-owned persistent arrays and explicitly tracked transient semantic buffers; reported rank peak is maximum across ranks.",
            "excluded_from_peak": "Input/target datasets, coordinator collective scratch, verification snapshots, Python objects, NumPy/BLAS internal workspaces, allocator overhead and process RSS.",
            "transient_components": "Activation outputs (all layers cached until backward finishes), gathered layer, local layer gradient, old/new backprop signals, tanh derivative workspace, two Adam work buffers. Component peaks need not occur simultaneously.",
            "dtype": "All actual training arrays are float64: parameter 8, gradient 8, Adam first+second moments 16 bytes per element. There is no separate master parameter copy.",
            "padding": "Each packed layer [weights,bias] is padded to a multiple of world_size. Padding is included in storage, communication and Adam element updates; dense GEMM FLOPs exclude padding.",
            "communication": "Central in-process numerical collectives; ideal-ring sent bytes are an analytic counter. No NCCL, MPI, real distributed workers or physical GPU allocations.",
            "stage3_schedule": "Gather one layer for forward, release; gather it again for backward, reduce-scatter its gradient, release. No prefetch, communication overlap or activation checkpointing.",
            "timing": "Measured CPU wall time includes Python/thread/collective coordination. Does not predict hardware GPU speedup; phases exclude small bookkeeping and executor startup/shutdown overhead.",
            "loss": "Each loss_curve entry is pre-update MSE over the fixed global batch; final_loss is post-final-update MSE, evaluated outside timing.",
            "threads": "Virtual ranks are state objects scheduled on ThreadPoolExecutor, not a guarantee of one OS thread per rank. BLAS may use its own threads; set BLAS thread limits before importing NumPy for repeatable demos.",
        },
        "environment": {"numpy_version": np.__version__, "logical_cpu_count": os.cpu_count()},
        "reference": {"loss_curve": reference["loss_curve"], "final_loss": reference["final_loss"]},
        "stages": stages,
    }
