"""Numerical correctness, real rank ownership, and communication accounting.

No assertions depend on a particular computer's execution speed. The reference
below uses its own forward/backward and Adam implementation on the global batch.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest


def oracle_loss_and_grad(parameters, x, target, dims):
    """Independent, unpadded-layer MLP oracle; padding has exactly zero gradient."""
    activations = [x]
    weights = []
    for index, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        values = parameters[index]
        weight = values[:din * dout].reshape(din, dout)
        bias = values[din * dout:din * dout + dout]
        weights.append(weight)
        value = activations[-1] @ weight + bias
        activations.append(np.tanh(value) if index < len(dims) - 2 else value)
    residual = activations[-1] - target
    loss = float(np.mean(residual ** 2))
    delta = 2 * residual / residual.size
    grads = [np.zeros_like(p) for p in parameters]
    for index in range(len(weights) - 1, -1, -1):
        weight_grad = activations[index].T @ delta
        bias_grad = delta.sum(axis=0)
        packed = np.concatenate((weight_grad.ravel(), bias_grad))
        grads[index][:packed.size] = packed
        if index:
            delta = (delta @ weights[index].T) * (1 - activations[index] ** 2)
    return loss, grads


def oracle_training(config, parameters, x, target):
    parameters = [p.copy() for p in parameters]
    first = [np.zeros_like(p) for p in parameters]
    second = [np.zeros_like(p) for p in parameters]
    losses = []
    for step in range(1, config.steps + 1):
        loss, gradients = oracle_loss_and_grad(parameters, x, target, config.dims)
        losses.append(loss)
        for p, m, v, g in zip(parameters, first, second, gradients):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            p -= config.learning_rate * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
    final, _ = oracle_loss_and_grad(parameters, x, target, config.dims)
    return parameters, losses, final


def test_mlp_gradient_matches_finite_differences_including_padding(sim):
    config = sim.Config(world_size=4, dims=(2, 3, 2), local_batch=3, steps=1, workers=2)
    parameters, inputs, targets = sim.make_problem(config)
    x, target = np.concatenate(inputs), np.concatenate(targets)
    loss, gradient = sim.loss_and_grad(parameters, x, target, config.dims)
    oracle_loss, oracle_gradient = oracle_loss_and_grad(parameters, x, target, config.dims)
    np.testing.assert_allclose(loss, oracle_loss, rtol=0, atol=1e-13)
    for layer, values in enumerate(parameters):
        np.testing.assert_allclose(gradient[layer], oracle_gradient[layer], rtol=1e-12, atol=1e-13)
        for index in range(len(values)):
            hi, lo = [v.copy() for v in parameters], [v.copy() for v in parameters]
            hi[layer][index] += 1e-6
            lo[layer][index] -= 1e-6
            # Oracle forward makes this independent of the tested backward code.
            high_loss = oracle_loss_and_grad(hi, x, target, config.dims)[0]
            low_loss = oracle_loss_and_grad(lo, x, target, config.dims)[0]
            finite_difference = (high_loss - low_loss) / 2e-6
            np.testing.assert_allclose(gradient[layer][index], finite_difference, rtol=2e-6, atol=2e-9)


@pytest.mark.parametrize("stage", range(4))
def test_32_rank_training_matches_independent_global_batch_adam(sim, stage):
    config = sim.Config(world_size=32, dims=(3, 4, 3), local_batch=2, steps=6, workers=4, seed=19)
    initial, inputs, targets = sim.make_problem(config)
    expected, curve, final_loss = oracle_training(config, initial, np.concatenate(inputs), np.concatenate(targets))
    distributed = sim.Simulator(config, stage=stage)
    result = distributed.train()
    np.testing.assert_allclose(result["loss_curve"], curve, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result["final_loss"], final_loss, rtol=1e-12, atol=1e-12)
    for actual, reference in zip(distributed.snapshot_parameters(), expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-11, atol=1e-12)
    assert final_loss < curve[0], "The demonstration must actually learn."
    assert result["thread_worker_count"] >= 1


@pytest.mark.parametrize("stage", range(4))
def test_real_rank_buffers_have_expected_ownership_without_aliasing(sim, stage):
    config = sim.Config(world_size=32, dims=(3, 4, 3), local_batch=1, steps=2, workers=4)
    distributed = sim.Simulator(config, stage=stage)
    layouts = sim.layer_layouts(config)
    assert len(distributed.ranks) == 32
    fields = ("parameters", "gradients", "first_moment", "second_moment")
    expected_sizes = {}
    for field in fields:
        sharded = (field in ("first_moment", "second_moment") and stage >= 1
                   or field == "gradients" and stage >= 2
                   or field == "parameters" and stage == 3)
        expected_sizes[field] = [layer.shard_count if sharded else layer.padded_count for layer in layouts]
        buffers = []
        for rank in distributed.ranks:
            arrays = getattr(rank, field)
            assert [a.size for a in arrays] == expected_sizes[field]
            assert all(a.dtype == np.float64 for a in arrays)
            buffers.extend(arrays)
        # Array ownership must be real; counting the same buffer 32 times is invalid.
        assert all(not np.shares_memory(left, right) for left, right in combinations(buffers, 2))
    distributed.train()
    memory = distributed.state_bytes_per_rank()
    for field in fields:
        expected = sum(expected_sizes[field]) * np.dtype(np.float64).itemsize
        assert memory[field] == expected
        assert all(sum(a.nbytes for a in getattr(rank, field)) == expected for rank in distributed.ranks)
    assert memory["total"] == sum(memory[field] for field in fields)
    # Full gathered layers/local gradients must be released after their use;
    # otherwise the apparent sharded persistent footprint would be misleading.
    for rank in distributed.ranks:
        assert rank.materialized_layer is None
        assert rank.local_gradient is None
        assert rank.delta is None and rank.next_delta is None
        assert rank.derivative_workspace is None
        assert not rank.optimizer_workspace and not rank.activations
    parameters = distributed.snapshot_parameters()
    for layer, values in zip(layouts, parameters):
        assert layer.padded_count % config.world_size == 0
        assert layer.padded_count - layer.parameter_count < config.world_size
        np.testing.assert_array_equal(values[layer.parameter_count:], 0)
    # Every shard rank includes its own zero-padded tail and never duplicates
    # any live parameter's owner. Concatenation reconstructs exactly one model.
    if stage == 3:
        for index in range(len(layouts)):
            shards = np.concatenate([rank.parameters[index] for rank in distributed.ranks])
            np.testing.assert_array_equal(shards, parameters[index])
    else:
        for rank in distributed.ranks:
            for replica, expected in zip(rank.parameters, parameters):
                np.testing.assert_array_equal(replica, expected)


@pytest.mark.parametrize("world_size", [1, 4, 7, 32])
@pytest.mark.parametrize("stage", range(4))
def test_ring_communication_and_optimizer_work_are_rederived(sim, world_size, stage):
    config = sim.Config(world_size=world_size, dims=(3, 4, 3), local_batch=2, steps=2, workers=min(4, world_size))
    result = sim.Simulator(config, stage=stage).train()
    layouts = sim.layer_layouts(config)
    capacity = sum(layer.padded_count for layer in layouts)
    communication = result["communication"]
    events = communication["events_first_step"]
    factor = (2, 2, 2, 3)[stage]
    expected_bytes = factor * (world_size - 1) * (capacity // world_size) * 8
    assert communication["bytes_sent_per_rank_per_step"] == expected_bytes
    assert sum(event["analytic_bytes_sent_per_rank"] for event in events) == expected_bytes
    assert all(event["payload_bytes"] == event["payload_elements"] * 8 for event in events)
    assert len(events) == len(layouts) * (1, 2, 2, 3)[stage]
    for event in events:
        kind = event["kind"].replace("-", "_").lower()
        multiplier = 2 if kind in ("all_reduce", "allreduce") else 1
        assert event["analytic_bytes_sent_per_rank"] == multiplier * (world_size - 1) * (event["payload_elements"] // world_size) * 8
    computation = result["computation"]
    assert computation["optimizer_element_updates_per_rank_per_step"] == capacity // (world_size if stage else 1)
    forward = sum(2 * config.local_batch * a * b for a, b in zip(config.dims[:-1], config.dims[1:]))
    assert computation["forward_matmul_flops_per_rank_per_step"] == forward
    # Input gradients of the first layer are not needed; every layer needs dW.
    backward = forward + sum(2 * config.local_batch * a * b for a, b in zip(config.dims[1:-1], config.dims[2:]))
    assert computation["backward_matmul_flops_per_rank_per_step"] == backward
    memory = result["state_bytes_per_rank"]
    assert result["peak_tracked_bytes_per_rank"] == memory["total"] + result["transient_peak_bytes_per_rank"]
    assert result["transient_peak_bytes_per_rank"] > 0


def test_deterministic_threads_repeat_exactly(sim):
    config = sim.Config(world_size=8, dims=(3, 7, 2), local_batch=2, steps=3, workers=4, seed=44)
    left, right = sim.Simulator(config, 3), sim.Simulator(config, 3)
    a, b = left.train(), right.train()
    assert a["loss_curve"] == b["loss_curve"]
    assert a["final_loss"] == b["final_loss"]
    assert a["state_bytes_per_rank"] == b["state_bytes_per_rank"]
    assert a["communication"] == b["communication"]
    assert a["computation"] == b["computation"]
    for first, second in zip(left.snapshot_parameters(), right.snapshot_parameters()):
        np.testing.assert_array_equal(first, second)
