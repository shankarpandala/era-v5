"""Schedule, optimiser grouping, data windows, memory meters and the batch search logic."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch


def test_lr_schedule_is_defined_in_tokens_not_steps(R):
    a = R.TrainConfig(total_tokens=1_000_000, batch_size=8, seq_len=64, lr=1e-3)
    b = R.TrainConfig(total_tokens=1_000_000, batch_size=64, seq_len=64, lr=1e-3)
    for t in (0, 10_000, 25_000, 300_000, 999_999):
        assert R.lr_at(t, a) == R.lr_at(t, b)
    warm = int(a.warmup_frac * a.total_tokens)
    assert R.lr_at(warm - 1, a) == pytest.approx(1e-3, rel=1e-6)
    assert R.lr_at(a.total_tokens, a) == pytest.approx(1e-4, rel=1e-6)
    assert R.lr_at(0, a) < R.lr_at(warm // 2, a) < R.lr_at(warm, a)
    assert a.steps == math.ceil(1_000_000 / 512) and b.steps == math.ceil(1_000_000 / 4096)


def test_sqrt_batch_scaling_is_capped(R):
    assert R.scaled_lr(1e-3, 32, 128) == pytest.approx(2e-3)
    assert R.scaled_lr(1e-3, 32, 32) == pytest.approx(1e-3)
    assert R.scaled_lr(1e-3, 32, 4096) == 3e-3


def test_weight_decay_only_on_matrices(R):
    model = R.GPT(R.ModelConfig(vocab_size=32, seq_len=8, d_model=16, n_layer=1, n_head=2))
    opt = R.make_optimizer(model, R.TrainConfig())
    decayed = {id(p) for p in opt.param_groups[0]["params"]}
    for n, p in model.named_parameters():
        assert (id(p) in decayed) == (p.dim() >= 2), n
    assert opt.param_groups[0]["weight_decay"] == 0.1 and opt.param_groups[1]["weight_decay"] == 0.0


def test_token_windows_are_shifted_by_one_and_val_windows_are_fixed(R, tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / "train.bin").write_bytes(rng.integers(0, 100, 5000, dtype=np.uint16).tobytes())
    val = np.arange(1000, dtype=np.uint16)
    (tmp_path / "val.bin").write_bytes(val.tobytes())
    data = R.TokenData(tmp_path, seq_len=16, seed=1)
    x, y = data.batch(4, torch.device("cpu"))
    assert x.shape == (4, 16) and torch.equal(x[:, 1:], y[:, :-1])
    vx, vy = data.val_windows(3, torch.device("cpu"))
    assert torch.equal(vx[1], torch.arange(16, 32)) and torch.equal(vy[1], torch.arange(17, 33))
    vx2, _ = R.TokenData(tmp_path, seq_len=16, seed=99).val_windows(3, torch.device("cpu"))
    assert torch.equal(vx, vx2)
    assert data.val_windows(10_000, torch.device("cpu"))[0].shape[0] == (1000 - 1) // 16


def test_tokenizer_round_trip_and_eot_id(R, tmp_path):
    text = ("Once upon a time, there was a little girl named Lily. She loved to play." + R.EOT) * 200
    tok = R.train_tokenizer(text, 300, tmp_path / "tok.json")
    assert tok.token_to_id(R.EOT) == 0
    s = "Once upon a time, Lily played."
    assert tok.decode(tok.encode(s).ids) == s
    ids = R.encode_stories(tok, R._split_stories(text)[:3], 0)
    assert ids.dtype == np.uint16 and (ids == 0).sum() == 3 and ids[-1] == 0


def test_saved_tensor_meter_separates_parameters_from_activations(R):
    model = R.GPT(R.ModelConfig(vocab_size=32, seq_len=8, d_model=16, n_layer=2, n_head=2))
    x = torch.randint(0, 32, (2, 8)); y = torch.randint(0, 32, (2, 8))
    saved = R.measure_saved_bytes(model, x, y, torch.device("cpu"))
    assert saved["activation_bytes"] > 0 and saved["parameter_bytes"] > 0 and saved["saved_tensors"] > 0
    assert saved["tokens"] == 16
    assert all(p.grad is None for p in model.parameters())  # meter leaves no gradients behind


def test_peak_memory_meter_on_cpu_reports_rss(R):
    with R.PeakMemory(torch.device("cpu"), interval_s=0.001) as mem:
        junk = torch.ones(8 << 20)  # 32 MiB
        junk.sum()
    assert mem.result["kind"] == "cpu_peak_rss"
    assert mem.result["peak_bytes"] >= mem.result["baseline_bytes"] > 0


def test_one_training_step_reduces_loss_on_a_memorisable_batch(R):
    torch.manual_seed(0)
    cfg = R.ModelConfig(vocab_size=32, seq_len=8, d_model=32, n_layer=2, n_head=2, reversible="euler")
    model = R.build_model(cfg, torch.device("cpu"))
    tc = R.TrainConfig(lr=3e-3, grad_clip=1.0)
    opt = R.make_optimizer(model, tc)
    x = torch.randint(0, 32, (4, 8)); y = torch.randint(0, 32, (4, 8))
    first = None
    for _ in range(30):
        loss, gnorm = R.one_step(model, opt, x, y, torch.device("cpu"), 3e-3, tc)
        first = first or loss.item()
        assert math.isfinite(gnorm)
    assert loss.item() < first * 0.5


def test_cpu_max_batch_search_never_exceeds_budget(R, monkeypatch):
    """Drive the CPU search with a fake trial whose peak grows linearly in the batch."""
    calls = []

    def fake_trial(model_cfg, train_cfg, batch, device, data, n_steps=2):
        calls.append(batch)
        return {"batch": batch, "peak_bytes": 1_000 + 100 * batch, "baseline_bytes": 1_000, "step_s": 1.0,
                "tokens_per_second": 1.0}

    monkeypatch.setattr(R, "_trial_step", fake_trial)
    res = R.find_max_batch(R.ModelConfig(), R.TrainConfig(), torch.device("cpu"), data=None,
                           budget_bytes=1_000 + 100 * 1000, start=8, log=lambda *a: None)
    assert res["max_batch"] % 8 == 0 and 900 <= res["max_batch"] <= 1000
    assert all(1_000 + 100 * b <= res["budget_bytes"] for b in calls)
    assert res["trials"][-1]["ok"]


def test_cuda_search_bisects_when_oom_is_raised(R, monkeypatch):
    if not hasattr(torch.cuda, "OutOfMemoryError"):
        pytest.skip("old torch")
    limit = 300

    def fake_trial(model_cfg, train_cfg, batch, device, data, n_steps=2):
        if batch > limit:
            raise torch.cuda.OutOfMemoryError("fake")
        return {"batch": batch, "peak_bytes": batch, "baseline_bytes": 0, "step_s": 1.0, "tokens_per_second": 1.0}

    monkeypatch.setattr(R, "_trial_step", fake_trial)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    res = R.find_max_batch(R.ModelConfig(), R.TrainConfig(), torch.device("cuda"), data=None,
                           budget_bytes=10**9, start=8, log=lambda *a: None)
    assert 256 <= res["max_batch"] <= limit
    assert any(not t["ok"] for t in res["trials"])
