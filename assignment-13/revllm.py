"""Assignment 13 — Reversible LLM training lab.

A ~21M-parameter GPT trained on TinyStories three ways:

  1. baseline residual transformer at a fixed batch size,
  2. a *reversible* transformer at the same batch size (activations are not
     stored; they are reconstructed layer by layer during backward), and
  3. the reversible transformer pushed to the largest batch that fits.

Everything the notebooks need lives in this one file so that it can be embedded
verbatim in each notebook and run on Colab without cloning the repository.

Reversible variants (all keep exactly the baseline's parameters):

  euler     RevNet / Reformer additive coupling on two streams (y1, y2):
              y1' = y1 + Attn(LN(y2));   y2' = y2 + MLP(LN(y1'))
            This is the symplectic-Euler discretisation of a two-stream system.
  midpoint  explicit midpoint (leapfrog) on one stream carrying two states:
              x[n+1] = x[n-1] + 2h * Delta_n(x[n]),  Delta = full block residual
  momentum  Momentum ResNet (Sander et al. 2021):
              v' = g v + (1-g) Delta(x);   x' = x + v'

Each variant is expressed as a chain of *half-steps*  p' = alpha*p + beta*f(q)
followed by the state swap (p, q) <- (q, p').  A half-step is inverted exactly:
p = (p' - beta*f(q)) / alpha.  The backward pass reconstructs the input of every
half-step from its output, recomputes f with autograd on, and never stores the
per-layer activations of the forward pass.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ("none", "euler", "midpoint", "momentum")
EOT = "<|endoftext|>"
TINYSTORIES_BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"
VALID_FILE = "TinyStoriesV2-GPT4-valid.txt"


# --------------------------------------------------------------------------- #
# Device, timing and memory helpers
# --------------------------------------------------------------------------- #
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def configure_runtime(device: torch.device) -> dict:
    """Numerics/perf settings that must be identical for every arm."""
    info = {"device": str(device), "torch": torch.__version__, "platform": platform.platform(),
            "python": platform.python_version()}
    if device.type == "cpu":
        # Subnormal floats in the backward matmuls made a CPU step 2.3x slower in profiling.
        torch.set_flush_denormal(True)
        info["cpu_threads"] = torch.get_num_threads()
        info["cpu_model"] = platform.processor() or "unknown"
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    elif device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return info


def autocast_dtype(device: torch.device) -> Optional[torch.dtype]:
    """bf16 on Ampere+ GPUs, fp16 (with a loss scaler) on older GPUs, fp32 elsewhere."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None


def process_rss_bytes() -> int:
    try:
        import psutil  # type: ignore
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:  # pragma: no cover - psutil missing
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class PeakMemory:
    """Peak memory of a phase.

    CUDA: allocator statistics (exact for tensors).  MPS: driver allocation.
    CPU: a sampling thread records the peak resident set size (RSS) of the process,
    which is an upper bound that includes the allocator's cache and Python itself.
    """

    def __init__(self, device: torch.device, interval_s: float = 0.02):
        self.device = device
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_rss = 0
        self.baseline = 0

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, process_rss_bytes())
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "PeakMemory":
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.baseline = torch.cuda.memory_allocated()
        elif self.device.type == "mps":
            self.baseline = torch.mps.current_allocated_memory()
        else:
            self.baseline = process_rss_bytes()
            self.peak_rss = self.baseline
            self._stop.clear()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            self.result = {"kind": "cuda_max_memory_allocated",
                           "peak_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                           "baseline_bytes": int(self.baseline)}
        elif self.device.type == "mps":
            self.result = {"kind": "mps_driver_allocated",
                           "peak_bytes": int(torch.mps.driver_allocated_memory()),
                           "baseline_bytes": int(self.baseline)}
        else:
            self._stop.set()
            if self._thread is not None:
                self._thread.join()
            self.peak_rss = max(self.peak_rss, process_rss_bytes())
            self.result = {"kind": "cpu_peak_rss", "peak_bytes": int(self.peak_rss),
                           "baseline_bytes": int(self.baseline)}


class SavedTensorMeter:
    """Counts the bytes autograd saves for backward during a forward pass.

    Device independent and exact: every tensor that a backward node keeps alive
    passes through the pack hook.  Tensors that alias a parameter's storage are
    counted separately from activations.
    """

    def __init__(self, model: nn.Module):
        self.param_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
        self.activation_bytes = 0
        self.parameter_bytes = 0
        self.count = 0

    def _pack(self, t: torch.Tensor):
        nbytes = t.numel() * t.element_size()
        if t.untyped_storage().data_ptr() in self.param_storages:
            self.parameter_bytes += nbytes
        else:
            self.activation_bytes += nbytes
        self.count += 1
        return t

    @staticmethod
    def _unpack(t):
        return t

    def __enter__(self):
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    vocab_size: int = 8192
    seq_len: int = 256
    d_model: int = 384
    n_layer: int = 10
    n_head: int = 6
    reversible: str = "none"      # none | euler | midpoint | momentum
    midpoint_h: float = 0.5       # x[n+1] = x[n-1] + 2h Delta(x[n]); 2h = 1 matches a residual step
    momentum_gamma: float = 0.9   # v' = g v + (1 - g) Delta(x)

    def __post_init__(self):
        if self.reversible not in VARIANTS:
            raise ValueError(f"reversible must be one of {VARIANTS}, got {self.reversible!r}")
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    """Pre-LN transformer block exposing its two residual increments."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def f_attn(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.ln1(x))

    def f_mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln2(x))

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        """The whole block's residual increment: block(x) - x."""
        a = self.f_attn(x)
        return a + self.f_mlp(x + a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.f_attn(x)
        return x + self.f_mlp(x)


@dataclass
class HalfStep:
    """p' = alpha * p + beta * f(q), then (p, q) <- (q, p').  f=None means identity."""
    fn: Optional[Callable[[torch.Tensor], torch.Tensor]]
    alpha: float
    beta: float
    param_index: tuple  # indices into the flat parameter list handed to the Function
    name: str = ""


class _ReversibleStack(torch.autograd.Function):
    """Runs a chain of half-steps without storing intermediate activations.

    forward  keeps only the final (p, q).
    backward walks the chain in reverse: reconstruct the half-step input from its
    output, recompute f on the reconstructed input with autograd enabled, and
    propagate the two state gradients.  Memory is O(1) in depth.
    """

    @staticmethod
    def forward(ctx, p, q, steps, *params):
        with torch.no_grad():
            for step in steps:
                fq = q if step.fn is None else step.fn(q)
                p, q = q, step.alpha * p + step.beta * fq
        ctx.save_for_backward(p, q)
        ctx.steps = steps
        ctx.params = params
        return p, q

    @staticmethod
    def backward(ctx, dp, dq):
        p, q = ctx.saved_tensors
        steps, params = ctx.steps, ctx.params
        grads: list = [None] * len(params)
        if dp is None:
            dp = torch.zeros_like(p)
        if dq is None:
            dq = torch.zeros_like(q)
        for step in reversed(steps):
            # Output of this half-step: (p, q) = (q_prev, alpha * p_prev + beta * f(q_prev)).
            x = p.detach().requires_grad_(True)
            if step.fn is None:
                fx = x
                gx = step.beta * dq
            else:
                with torch.enable_grad():
                    fx = step.fn(x)
                fn_params = [params[i] for i in step.param_index]
                got = torch.autograd.grad(fx, [x] + fn_params, grad_outputs=step.beta * dq, allow_unused=True)
                gx = got[0] if got[0] is not None else torch.zeros_like(x)
                for i, g in zip(step.param_index, got[1:]):
                    if g is None:
                        continue
                    grads[i] = g if grads[i] is None else grads[i] + g
            with torch.no_grad():
                p_prev = (q - step.beta * fx.detach()) / step.alpha
                q_prev = p.detach()
                dp_prev = step.alpha * dq
                dq_prev = dp + gx
            p, q, dp, dq = p_prev, q_prev, dp_prev, dq_prev
        return dp, dq, None, *grads


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight  # tied
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
        self._materialize = False  # debug/test switch: run reversible recurrences with plain autograd

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---- parameter accounting -------------------------------------------------
    def num_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        emb = self.wte.weight.numel() + self.wpe.weight.numel()
        return {"total": int(total), "embedding": int(emb), "non_embedding": int(total - emb)}

    # ---- the reversible plan --------------------------------------------------
    def _flat_params(self) -> list:
        return list(self.blocks.parameters())

    def _plan(self) -> list:
        offsets, flat = {}, []
        for li, block in enumerate(self.blocks):
            for name, p in block.named_parameters():
                offsets[(li, name)] = len(flat)
                flat.append(p)
        steps = []
        v = self.cfg.reversible
        for li, block in enumerate(self.blocks):
            attn_idx = tuple(offsets[(li, n)] for n, _ in block.named_parameters() if n.startswith(("ln1", "attn")))
            mlp_idx = tuple(offsets[(li, n)] for n, _ in block.named_parameters() if n.startswith(("ln2", "mlp")))
            all_idx = tuple(offsets[(li, n)] for n, _ in block.named_parameters())
            if v == "euler":
                steps.append(HalfStep(block.f_attn, 1.0, 1.0, attn_idx, f"L{li}.attn"))
                steps.append(HalfStep(block.f_mlp, 1.0, 1.0, mlp_idx, f"L{li}.mlp"))
            elif v == "midpoint":
                steps.append(HalfStep(block.delta, 1.0, 2.0 * self.cfg.midpoint_h, all_idx, f"L{li}.delta"))
            elif v == "momentum":
                g = self.cfg.momentum_gamma
                steps.append(HalfStep(block.delta, g, 1.0 - g, all_idx, f"L{li}.velocity"))
                steps.append(HalfStep(None, 1.0, 1.0, (), f"L{li}.position"))
            else:
                raise ValueError(v)
        return steps

    def _initial_state(self, x: torch.Tensor):
        v = self.cfg.reversible
        if v == "euler":
            return x, x                       # (y1, y2) both start at the embedding
        if v == "midpoint":
            return x, x                       # (x[-1], x[0]) — x[-1] := x[0]
        if v == "momentum":
            return torch.zeros_like(x), x     # (v, x) with zero initial velocity
        raise ValueError(v)

    def _final_state(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        v = self.cfg.reversible
        if v == "euler":
            return p + q                      # sum of the two streams; LN follows
        return q                              # midpoint: x[L]; momentum: position

    def stack(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.reversible == "none":
            for block in self.blocks:
                x = block(x)
            return x
        p, q = self._initial_state(x)
        steps = self._plan()
        if self._materialize:
            for step in steps:
                fq = q if step.fn is None else step.fn(q)
                p, q = q, step.alpha * p + step.beta * fq
        else:
            p, q = _ReversibleStack.apply(p, q, steps, *self._flat_params())
        return self._final_state(p, q)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        x = self.ln_f(self.stack(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, eot_id: Optional[int] = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < vals[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eot_id is not None and bool((nxt == eot_id).all()):
                break
        return idx

    # ---- reconstruction check ---------------------------------------------------
    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> dict:
        """Run the reversible chain forward, invert it, and compare with the input state."""
        if self.cfg.reversible == "none":
            raise ValueError("baseline model is not invertible")
        p0, q0 = self._initial_state(x)
        p, q = p0, q0
        steps = self._plan()
        for step in steps:
            fq = q if step.fn is None else step.fn(q)
            p, q = q, step.alpha * p + step.beta * fq
        for step in reversed(steps):
            fx = p if step.fn is None else step.fn(p)
            p, q = (q - step.beta * fx) / step.alpha, p
        err = max((p - p0).abs().max().item(), (q - q0).abs().max().item())
        scale = max(p0.abs().max().item(), q0.abs().max().item(), 1e-12)
        return {"max_abs_error": err, "max_rel_error": err / scale, "half_steps": len(steps)}


def build_model(cfg: ModelConfig, device: torch.device, seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(cfg).to(device)


# --------------------------------------------------------------------------- #
# Data: TinyStories -> byte-level BPE (8192) -> uint16 token streams
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path, max_bytes: Optional[int] = None) -> None:
    """Download `url` (optionally only its first `max_bytes`) to `dest`, resuming partial files."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if max_bytes is not None and have >= max_bytes:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "era-v5-assignment-13"})
    if have or max_bytes is not None:
        end = "" if max_bytes is None else str(max_bytes - 1)
        req.add_header("Range", f"bytes={have}-{end}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        if have and resp.status != 206:
            have = 0  # server ignored the range: start over
        mode = "ab" if have else "wb"
        with dest.open(mode) as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                if max_bytes is not None and out.tell() >= max_bytes:
                    break
    if max_bytes is None or dest.stat().st_size >= max_bytes:
        return
    if dest.stat().st_size < max_bytes and dest.stat().st_size < 1024:
        raise RuntimeError(f"download of {url} produced {dest.stat().st_size} bytes")


def _split_stories(text: str) -> list:
    return [s.strip() for s in text.split(EOT) if s.strip()]


def train_tokenizer(text: str, vocab_size: int, path: Path):
    """Byte-level BPE (GPT-2 style pre-tokeniser) with EOT as id 0. Deterministic for fixed text."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # type: ignore

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOT], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(_split_stories(text), trainer=trainer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return tok


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer  # type: ignore
    return Tokenizer.from_file(str(path))


def encode_stories(tok, stories: list, eot_id: int, batch: int = 2048) -> np.ndarray:
    out = []
    for i in range(0, len(stories), batch):
        for enc in tok.encode_batch(stories[i:i + batch]):
            out.extend(enc.ids)
            out.append(eot_id)
    arr = np.asarray(out, dtype=np.uint16)
    return arr


def prepare_data(data_dir: Path, train_tokens: int, vocab_size: int = 8192, val_tokens: int = 2_000_000,
                 tokenizer_train_bytes: int = 20 << 20, chars_per_token: float = 4.0, log=print) -> dict:
    """Fetch just enough TinyStories text, train the 8192 BPE, and write train.bin / val.bin.

    Returns a manifest with byte/token counts and sha256 of the tokenizer file.
    `train_tokens` is the *minimum* number of training tokens required (a 10% margin is added).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("train_tokens", 0) >= train_tokens and manifest.get("vocab_size") == vocab_size \
                and (data_dir / "train.bin").exists() and (data_dir / "val.bin").exists():
            log(f"[data] reusing {data_dir} ({manifest['train_tokens']:,} train tokens)")
            return manifest
    need_bytes = int(train_tokens * 1.1 * chars_per_token) + tokenizer_train_bytes
    train_raw = data_dir / "train_raw.txt"
    val_raw = data_dir / "valid_raw.txt"
    log(f"[data] fetching the first {need_bytes / 1e6:.0f} MB of {TRAIN_FILE} and all of {VALID_FILE}")
    _download(TINYSTORIES_BASE + TRAIN_FILE, train_raw, max_bytes=need_bytes)
    _download(TINYSTORIES_BASE + VALID_FILE, val_raw)
    train_text = train_raw.read_text(encoding="utf-8", errors="ignore")
    val_text = val_raw.read_text(encoding="utf-8", errors="ignore")
    # The last story of a byte-range fetch is truncated: drop it.
    train_stories = _split_stories(train_text)[:-1]
    val_stories = _split_stories(val_text)

    tok_path = data_dir / "tokenizer.json"
    log(f"[data] training a {vocab_size}-token byte-level BPE on the first {tokenizer_train_bytes >> 20} MB")
    tok = train_tokenizer(train_text[:tokenizer_train_bytes], vocab_size, tok_path)
    eot_id = tok.token_to_id(EOT)
    assert eot_id == 0, eot_id

    log(f"[data] encoding {len(train_stories):,} train and {len(val_stories):,} validation stories")
    train_ids = encode_stories(tok, train_stories, eot_id)
    val_ids = encode_stories(tok, val_stories, eot_id)[:val_tokens]
    if len(train_ids) < train_tokens:
        raise RuntimeError(f"only {len(train_ids):,} train tokens after encoding; need {train_tokens:,}. "
                           f"Increase chars_per_token.")
    train_ids.tofile(data_dir / "train.bin")
    val_ids.tofile(data_dir / "val.bin")
    train_chars = sum(len(s) + len(EOT) for s in train_stories)
    manifest = {
        "source": TINYSTORIES_BASE + TRAIN_FILE, "validation_source": TINYSTORIES_BASE + VALID_FILE,
        "fetched_train_bytes": train_raw.stat().st_size, "train_stories": len(train_stories),
        "val_stories": len(val_stories), "vocab_size": vocab_size, "eot_id": eot_id,
        "train_tokens": int(len(train_ids)), "val_tokens": int(len(val_ids)),
        "train_chars_per_token": train_chars / len(train_ids),
        "tokenizer_sha256": hashlib.sha256(tok_path.read_bytes()).hexdigest(),
        "tokenizer_train_bytes": tokenizer_train_bytes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"[data] {manifest['train_tokens']:,} train tokens, {manifest['val_tokens']:,} val tokens, "
        f"{manifest['train_chars_per_token']:.2f} chars/token")
    return manifest


class TokenData:
    """Random contiguous windows from a uint16 token stream (nanoGPT style), seeded."""

    def __init__(self, data_dir: Path, seq_len: int, seed: int = 0):
        self.train = np.memmap(Path(data_dir) / "train.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(Path(data_dir) / "val.bin", dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size: int, device: torch.device, split: str = "train"):
        data = self.train if split == "train" else self.val
        T = self.seq_len
        ix = self.rng.integers(0, len(data) - T - 1, size=batch_size)
        x = torch.from_numpy(np.stack([data[i:i + T].astype(np.int64) for i in ix]))
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + T].astype(np.int64) for i in ix]))
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def val_windows(self, n_seqs: int, device: torch.device):
        """Fixed, non-overlapping validation windows (the same for every arm)."""
        T = self.seq_len
        n_seqs = min(n_seqs, (len(self.val) - 1) // T)
        x = torch.from_numpy(np.stack([self.val[i * T:(i + 1) * T].astype(np.int64) for i in range(n_seqs)]))
        y = torch.from_numpy(np.stack([self.val[i * T + 1:(i + 1) * T + 1].astype(np.int64) for i in range(n_seqs)]))
        return x.to(device), y.to(device)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    total_tokens: int = 50_000_000
    batch_size: int = 32
    seq_len: int = 256
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_frac: float = 0.025
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    eval_every_steps: int = 100
    eval_seqs: int = 128
    final_eval_seqs: int = 512
    log_every_steps: int = 10
    seed: int = 1234
    ckpt_every_steps: int = 500
    sample_tokens: int = 120

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def steps(self) -> int:
        return max(1, math.ceil(self.total_tokens / self.tokens_per_step))


def lr_at(tokens_seen: int, cfg: TrainConfig) -> float:
    """Linear warmup over warmup_frac of the budget, then cosine to min_lr_ratio * lr. Batch-size agnostic."""
    warmup = max(1, int(cfg.warmup_frac * cfg.total_tokens))
    if tokens_seen < warmup:
        return cfg.lr * (tokens_seen + 1) / warmup
    progress = min(1.0, (tokens_seen - warmup) / max(1, cfg.total_tokens - warmup))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def scaled_lr(base_lr: float, base_batch: int, batch: int, cap: float = 3e-3) -> float:
    """Square-root batch scaling for Adam-type optimisers, capped for safety."""
    return min(cap, base_lr * math.sqrt(batch / base_batch))


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    fused = torch.cuda.is_available() and next(model.parameters()).is_cuda
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, fused=fused if fused else None)


@torch.no_grad()
def evaluate(model: GPT, x: torch.Tensor, y: torch.Tensor, device: torch.device, batch: int = 32) -> float:
    model.eval()
    dtype = autocast_dtype(device)
    losses, n = [], 0
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            _, loss = model(xb, yb)
        losses.append(loss.item() * xb.size(0))
        n += xb.size(0)
    model.train()
    return sum(losses) / n


def one_step(model: GPT, opt: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor, device: torch.device,
             lr: float, cfg: TrainConfig, scaler=None) -> tuple:
    for g in opt.param_groups:
        g["lr"] = lr
    dtype = autocast_dtype(device)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
        _, loss = model(x, y)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
    else:
        loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    if scaler is not None:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()
    opt.zero_grad(set_to_none=True)
    return loss, gnorm


def measure_saved_bytes(model: GPT, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> dict:
    """Bytes autograd keeps alive for the backward pass of one forward at this batch."""
    dtype = autocast_dtype(device)
    with SavedTensorMeter(model) as meter:
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            _, loss = model(x, y)
    loss.backward()
    model.zero_grad(set_to_none=True)
    return {"activation_bytes": meter.activation_bytes, "parameter_bytes": meter.parameter_bytes,
            "saved_tensors": meter.count, "tokens": int(x.numel())}


def train_run(model_cfg: ModelConfig, train_cfg: TrainConfig, data: TokenData, device: torch.device,
              out_dir: Path, run_name: str, log=print, tokenizer=None, resume: bool = True) -> dict:
    """Train one arm to its token budget; save results.json, curves and samples under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    runtime = configure_runtime(device)
    model = build_model(model_cfg, device, seed=train_cfg.seed)
    opt = make_optimizer(model, train_cfg)
    scaler = torch.cuda.amp.GradScaler() if autocast_dtype(device) == torch.float16 else None
    torch.manual_seed(train_cfg.seed)
    counts = model.num_params()
    steps = train_cfg.steps
    log(f"[{run_name}] {counts['total'] / 1e6:.2f}M params ({counts['non_embedding'] / 1e6:.2f}M non-embedding), "
        f"variant={model_cfg.reversible}, batch={train_cfg.batch_size}x{train_cfg.seq_len}="
        f"{train_cfg.tokens_per_step:,} tokens/step, {steps:,} steps for {train_cfg.total_tokens:,} tokens, "
        f"peak lr={train_cfg.lr:.2e}, device={device}")

    val_x, val_y = data.val_windows(train_cfg.final_eval_seqs, device)
    quick_x, quick_y = val_x[:train_cfg.eval_seqs], val_y[:train_cfg.eval_seqs]

    history = {"step": [], "tokens": [], "train_loss": [], "lr": [], "grad_norm": [], "step_ms": []}
    evals = {"step": [], "tokens": [], "val_loss": []}
    start_step, train_seconds = 0, 0.0
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        history, evals = ck["history"], ck["evals"]
        start_step, train_seconds = ck["step"], ck["train_seconds"]
        data.rng = np.random.default_rng(train_cfg.seed + start_step)
        log(f"[{run_name}] resumed from step {start_step}")

    # Exact activation footprint at this batch (same measurement on every device).
    xb, yb = data.batch(train_cfg.batch_size, device)
    saved = measure_saved_bytes(model, xb, yb, device)
    recon = model.reconstruction_error(model.wte(xb) + model.wpe(torch.arange(xb.size(1), device=device))) \
        if model_cfg.reversible != "none" else None
    log(f"[{run_name}] autograd keeps {saved['activation_bytes'] / 2**20:.1f} MiB of activations for one "
        f"{train_cfg.tokens_per_step:,}-token step" + (f"; reconstruction max|err|={recon['max_abs_error']:.2e}" if recon else ""))

    model.train()
    diverged = False
    with PeakMemory(device) as mem:
        val0 = evaluate(model, quick_x, quick_y, device) if start_step == 0 else None
        if val0 is not None:
            evals["step"].append(0); evals["tokens"].append(0); evals["val_loss"].append(val0)
            log(f"[{run_name}] step 0: val {val0:.4f}")
        step = start_step - 1
        for step in range(start_step, steps):
            tokens_seen = step * train_cfg.tokens_per_step
            lr = lr_at(tokens_seen, train_cfg)
            x, y = data.batch(train_cfg.batch_size, device)
            sync(device)
            t0 = time.perf_counter()
            loss, gnorm = one_step(model, opt, x, y, device, lr, train_cfg, scaler)
            sync(device)
            dt = time.perf_counter() - t0
            train_seconds += dt
            loss_v = loss.item()
            if not math.isfinite(loss_v):
                diverged = True
                log(f"[{run_name}] step {step}: loss is {loss_v}; stopping")
                break
            if step % train_cfg.log_every_steps == 0 or step == steps - 1:
                history["step"].append(step + 1); history["tokens"].append(tokens_seen + train_cfg.tokens_per_step)
                history["train_loss"].append(loss_v); history["lr"].append(lr)
                history["grad_norm"].append(float(gnorm)); history["step_ms"].append(dt * 1000)
            if (step + 1) % train_cfg.eval_every_steps == 0 and step + 1 != steps:
                v = evaluate(model, quick_x, quick_y, device)
                evals["step"].append(step + 1); evals["tokens"].append(tokens_seen + train_cfg.tokens_per_step)
                evals["val_loss"].append(v)
                tps = (step + 1 - start_step) * train_cfg.tokens_per_step / max(train_seconds, 1e-9) if start_step == 0 else None
                log(f"[{run_name}] step {step + 1}/{steps}: train {loss_v:.4f} val {v:.4f} lr {lr:.2e} "
                    f"{dt * 1000:.0f} ms/step" + (f" {tps:,.0f} tok/s" if tps else ""))
            if (step + 1) % train_cfg.ckpt_every_steps == 0 and step + 1 != steps:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "history": history,
                            "evals": evals, "step": step + 1, "train_seconds": train_seconds}, ckpt_path)
    steps_done = (step + 1) if not diverged else step
    tokens_done = steps_done * train_cfg.tokens_per_step
    final_val = evaluate(model, val_x, val_y, device)
    evals["step"].append(steps_done); evals["tokens"].append(tokens_done); evals["val_loss"].append(final_val)
    tail = history["train_loss"][-max(1, min(20, len(history["train_loss"]))):]
    samples = []
    if tokenizer is not None:
        eot = tokenizer.token_to_id(EOT)
        prompt = torch.tensor([[eot] + tokenizer.encode("Once upon a time").ids], device=device)
        for temp in (0.0, 0.8):
            out = model.generate(prompt, train_cfg.sample_tokens, temperature=temp, top_k=50, eot_id=eot)
            samples.append({"temperature": temp, "text": tokenizer.decode(out[0, 1:].tolist())})
    step_ms = history["step_ms"][1:] if len(history["step_ms"]) > 1 else history["step_ms"]
    results = {
        "run_name": run_name, "model_config": asdict(model_cfg), "train_config": asdict(train_cfg),
        "runtime": runtime, "params": counts, "steps": steps_done, "tokens": tokens_done,
        "planned_tokens": train_cfg.total_tokens, "diverged": diverged,
        "final_train_loss_last20": float(np.mean(tail)), "final_val_loss": final_val,
        "final_val_ppl": math.exp(final_val), "train_seconds": train_seconds,
        "tokens_per_second": tokens_done / max(train_seconds, 1e-9),
        "step_ms_mean": float(np.mean(step_ms)), "step_ms_std": float(np.std(step_ms)),
        "peak_memory": mem.result, "saved_for_backward": saved, "reconstruction": recon,
        "history": history, "evals": evals, "samples": samples,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    if ckpt_path.exists():
        ckpt_path.unlink()
    log(f"[{run_name}] done: val {final_val:.4f} (ppl {math.exp(final_val):.1f}), "
        f"{results['tokens_per_second']:,.0f} tok/s, peak {mem.result['peak_bytes'] / 2**30:.2f} GiB "
        f"({mem.result['kind']}), {train_seconds / 60:.1f} min of training")
    return results


# --------------------------------------------------------------------------- #
# Maximum batch size search
# --------------------------------------------------------------------------- #
def release_memory(device: torch.device) -> None:
    """Free cached memory so the next measurement starts from a clean footprint."""
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)  # hand freed heap back to the OS so RSS drops
        except OSError:
            pass


def _trial_step(model_cfg: ModelConfig, train_cfg: TrainConfig, batch: int, device: torch.device,
                data: TokenData, n_steps: int = 2) -> dict:
    """Build a fresh model+optimizer and run n_steps full training steps at `batch`."""
    release_memory(device)
    model = build_model(model_cfg, device, seed=0)
    opt = make_optimizer(model, train_cfg)
    scaler = torch.cuda.amp.GradScaler() if autocast_dtype(device) == torch.float16 else None
    with PeakMemory(device) as mem:
        for _ in range(n_steps):
            x, y = data.batch(batch, device)
            sync(device); t0 = time.perf_counter()
            one_step(model, opt, x, y, device, train_cfg.lr, train_cfg, scaler)
            sync(device); dt = time.perf_counter() - t0
    del model, opt
    release_memory(device)
    return {"batch": batch, "peak_bytes": mem.result["peak_bytes"], "baseline_bytes": mem.result["baseline_bytes"],
            "step_s": dt, "tokens_per_second": batch * train_cfg.seq_len / dt}


def find_max_batch(model_cfg: ModelConfig, train_cfg: TrainConfig, device: torch.device, data: TokenData,
                   budget_bytes: Optional[int] = None, start: int = 8, ceiling: int = 8192, log=print) -> dict:
    """Largest batch whose full training step fits.

    CUDA: double until the step raises OOM, then bisect (each trial is caught in-process).
    CPU/MPS: the OS kills an over-committed process instead of raising, so trials are gated by
    a linear peak-memory model fitted to the measured trials (budget: 75% of the free RAM); the final
    answer is verified by running real steps at that batch and checking the measured peak against it.
    """
    trials = []
    if device.type == "cuda":
        budget_bytes = budget_bytes or torch.cuda.get_device_properties(0).total_memory
        b, last_ok, first_fail = start, None, None
        while b <= ceiling:
            try:
                t = _trial_step(model_cfg, train_cfg, b, device, data); t["ok"] = True; trials.append(t)
                log(f"[max-batch] batch {b}: ok, peak {t['peak_bytes'] / 2**30:.2f} GiB, {t['tokens_per_second']:,.0f} tok/s")
                last_ok, b = b, b * 2
            except torch.cuda.OutOfMemoryError:
                release_memory(device)
                trials.append({"batch": b, "ok": False}); first_fail = b
                log(f"[max-batch] batch {b}: out of memory")
                break
        if first_fail is not None and last_ok is not None:
            lo, hi = last_ok, first_fail
            while hi - lo > max(1, last_ok // 16):
                mid = (lo + hi) // 2
                try:
                    t = _trial_step(model_cfg, train_cfg, mid, device, data); t["ok"] = True; trials.append(t)
                    log(f"[max-batch] batch {mid}: ok, peak {t['peak_bytes'] / 2**30:.2f} GiB")
                    lo = mid
                except torch.cuda.OutOfMemoryError:
                    release_memory(device)
                    trials.append({"batch": mid, "ok": False}); hi = mid
                    log(f"[max-batch] batch {mid}: out of memory")
            last_ok = lo
        method = "cuda doubling + bisection until OutOfMemoryError"
        max_batch = last_ok
    else:
        import psutil  # type: ignore
        avail = psutil.virtual_memory().available
        budget_bytes = budget_bytes or int(0.75 * avail)
        log(f"[max-batch] budget {budget_bytes / 2**30:.2f} GiB of {avail / 2**30:.2f} GiB available RAM")
        b = start
        while True:
            t = _trial_step(model_cfg, train_cfg, b, device, data); t["ok"] = True; trials.append(t)
            log(f"[max-batch] batch {b}: peak RSS {t['peak_bytes'] / 2**30:.2f} GiB, {t['tokens_per_second']:,.0f} tok/s")
            if len(trials) >= 2:
                (b1, m1), (b2, m2) = [(u["batch"], u["peak_bytes"]) for u in trials[-2:]]
                local = (m2 - m1) / (b2 - b1)
                average = (m2 - trials[0]["baseline_bytes"]) / b2
                slope = max(local, average, 1.0)  # the steeper estimate is the safer one
                intercept = m2 - slope * b2
                predicted_next = intercept + slope * (2 * b)
                if predicted_next > budget_bytes or 2 * b > ceiling:
                    break
            b *= 2
        # Largest batch predicted to fit, then verify with a real trial.
        max_batch = int(max(b, (budget_bytes - intercept) / slope)) if len(trials) >= 2 else b
        max_batch = min(max_batch, ceiling)
        max_batch -= max_batch % 8
        for _ in range(4):
            t = _trial_step(model_cfg, train_cfg, max_batch, device, data); t["ok"] = t["peak_bytes"] <= budget_bytes
            trials.append(t)
            log(f"[max-batch] verify batch {max_batch}: peak RSS {t['peak_bytes'] / 2**30:.2f} GiB "
                f"({'fits' if t['ok'] else 'over budget'})")
            if t["ok"]:
                break
            max_batch = int(max_batch * 0.9); max_batch -= max_batch % 8
        method = "cpu doubling gated by a linear RSS model, verified by a real step"
    return {"max_batch": int(max_batch), "budget_bytes": int(budget_bytes), "trials": trials, "method": method,
            "variant": model_cfg.reversible, "seq_len": train_cfg.seq_len}


def depth_sweep_saved_bytes(base_cfg: ModelConfig, depths, batch: int, device: torch.device) -> list:
    """Autograd-saved activation bytes vs depth for baseline and every reversible variant (exact)."""
    rows = []
    torch.manual_seed(0)
    x = torch.randint(0, base_cfg.vocab_size, (batch, base_cfg.seq_len), device=device)
    y = torch.randint(0, base_cfg.vocab_size, (batch, base_cfg.seq_len), device=device)
    for variant in VARIANTS:
        for L in depths:
            cfg = dataclasses.replace(base_cfg, n_layer=L, reversible=variant)
            model = build_model(cfg, device)
            saved = measure_saved_bytes(model, x, y, device)
            rows.append({"variant": variant, "n_layer": L, **saved})
            del model
    return rows


def bytes_gib(n: float) -> str:
    return f"{n / 2**30:.2f} GiB"


def save_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
