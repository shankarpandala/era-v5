"""The training loop: consumption ledger, learning ledger, checkpoints, crash.

Invoked as a *subprocess* by ``run_demo.py`` so the deliberate crash can be a real
process death (``os._exit``) rather than a caught exception -- nothing survives in
memory, which is what makes the resume proof honest.

Per step it:
  1. builds the batch from the deterministic batcher (cursor is the only state),
  2. appends a consumption entry (batch id, per-sample source spans, lane tokens)
     BEFORE the optimizer step -- so the ledger records intent to consume,
  3. runs forward/backward with the packed masks, records a learning entry
     (total loss, per-lane token loss, lr, param hash),
  4. checkpoints on cadence, pinning the ledger offsets.

Crash/resume contract: a checkpoint stores the committed ledger entry counts. On
resume the ledgers are truncated back to those counts, so the steps that ran
after the last checkpoint are *rewritten identically* by the deterministic
batcher rather than skipped or double-counted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List

import numpy as np
import torch

from .batcher import Batcher, batch_consumption_record
from .checkpoint import load_checkpoint, save_checkpoint
from .ledger import Ledger
from .model import TinyGPT, compute_loss, per_lane_loss
from .perf import PerfCounter
from .tokenizer import Tokenizer
from .util import ensure_dir, read_json, sha256_json, write_json


def set_global_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)


def build_model(vocab_size: int, seq_len: int, cfg: dict):
    model = TinyGPT(vocab_size=vocab_size, d_model=cfg["d_model"],
                    n_layer=cfg["n_layer"], n_head=cfg["n_head"],
                    max_pos=max(seq_len, 512))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95),
                            weight_decay=0.01)
    return model, opt


def to_tensors(batch: dict):
    input_ids = torch.tensor([s["input_ids"] for s in batch["samples"]], dtype=torch.long)
    segment_ids = torch.tensor([s["segment_ids"] for s in batch["samples"]], dtype=torch.long)
    position_ids = torch.tensor([s["position_ids"] for s in batch["samples"]], dtype=torch.long)
    loss_mask = torch.tensor([s["loss_mask"] for s in batch["samples"]], dtype=torch.long)
    return input_ids, segment_ids, position_ids, loss_mask


def lane_positions(batch: dict, seq_len: int) -> List[str]:
    """Lane label for each flattened shifted position (B x (L-1))."""
    out: List[str] = []
    for s in batch["samples"]:
        out.extend([s["lane"]] * (seq_len - 1))
    return out


def lr_at(step: int, total_steps: int, base_lr: float) -> float:
    """Warmup + cosine decay to zero (WSD-ish anneal at the end)."""
    warm = max(1, int(0.1 * total_steps))
    if step < warm:
        return base_lr * (step + 1) / warm
    t = (step - warm) / max(1, total_steps - warm)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def param_hash(model) -> str:
    return sha256_json({
        k: round(float(v.detach().float().sum().item()), 6)
        for k, v in sorted(model.state_dict().items())
    })


def run_training(args) -> int:
    art = args.artifacts
    manifests_dir = os.path.join(art, "manifests")
    ledgers_dir = ensure_dir(os.path.join(art, "ledgers"))
    ckpt_dir = ensure_dir(os.path.join(art, "checkpoints"))

    cfg = read_json(os.path.join(art, "run_config.json"))
    schedule = read_json(os.path.join(manifests_dir, "schedule.json"))
    inventory = read_json(os.path.join(manifests_dir, "inventory.json"))
    tok = Tokenizer.load(os.path.join(manifests_dir, "tokenizer.json"))

    seq_len = cfg["seq_len"]
    total_steps = cfg["total_steps"]
    run_id = args.run_id

    set_global_seed(cfg["seed"])
    model, opt = build_model(tok.vocab_size, seq_len, cfg)
    batcher = Batcher(run_id, schedule, inventory, manifests_dir, seq_len)

    cons = Ledger(os.path.join(ledgers_dir, f"consumption_{run_id}.jsonl"))
    learn = Ledger(os.path.join(ledgers_dir, f"learning_{run_id}.jsonl"))

    start_step = 0
    lineage = {}
    perf = PerfCounter()

    if args.resume_from:
        man = load_checkpoint(ckpt_dir, args.resume_from, model, opt)
        start_step = man["step"]
        batcher.set_cursor(man["cursor"])
        # roll ledgers back to the committed prefix pinned by the checkpoint, so
        # steps executed after it are rewritten identically rather than doubled
        cons.truncate_to(man["consumption_offset"]["count"])
        learn.truncate_to(man["learning_offset"]["count"])
        assert cons.head == man["consumption_offset"]["head"], "consumption chain head mismatch"
        assert learn.head == man["learning_offset"]["head"], "learning chain head mismatch"
        lineage = man.get("lineage", {})
        # performance counters roll back with the ledgers -- otherwise the work
        # done between the checkpoint and the crash would be counted twice
        perf.load(man["perf_counters"])
        _emit(args, {"event": "resumed", "from": args.resume_from,
                     "next_step": start_step, "cursor": batcher.get_cursor(),
                     "consumption_count": cons.count, "learning_count": learn.count,
                     "perf_steps_restored": man["perf_counters"]["steps"]})
    elif args.fork_from:
        man = load_checkpoint(ckpt_dir, args.fork_from, model, opt)
        start_step = man["step"]
        batcher.set_cursor(man["cursor"])
        lineage = {
            "parent_run_id": man["run_id"],
            "parent_checkpoint": args.fork_from,
            "parent_step": man["step"],
            "parent_model_tensor_hash": man["model_tensor_hash"],
            "parent_consumption_head": man["consumption_offset"]["head"],
            "parent_cursor": man["cursor"],
        }
        write_json(os.path.join(ckpt_dir, f"{run_id}.lineage.json"),
                   {"run_id": run_id, **lineage})
        _emit(args, {"event": "forked", "from": args.fork_from,
                     "next_step": start_step, "lineage": lineage})

    end_step = args.until if args.until is not None else total_steps
    ckpt_every = cfg["checkpoint_every"]
    saved_tags: List[dict] = []

    for step in range(start_step, end_step):
        t0 = time.perf_counter()
        cursor_before = batcher.get_cursor()
        batch = batcher.build_step(step, advance=True)

        # ---- consumption ledger (before the optimizer touches anything) ----
        rec = batch_consumption_record(batch)
        rec["run_id"] = run_id
        rec["cursor_before"] = cursor_before
        rec["cursor_after"] = batcher.get_cursor()
        cons.append(rec)

        # ---- forward / backward -------------------------------------------
        input_ids, segment_ids, position_ids, loss_mask = to_tensors(batch)
        logits = model(input_ids, segment_ids, position_ids)
        loss, per_tok, flat_mask = compute_loss(logits, input_ids, loss_mask)
        lr = lr_at(step, total_steps, cfg["lr"])
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        dt = time.perf_counter() - t0

        lane_loss = per_lane_loss(per_tok, flat_mask, lane_positions(batch, seq_len))
        learn.append({
            "run_id": run_id,
            "step": step,
            "stage": batch["stage"],
            "batch_id": batch["batch_id"],       # links loss back to source data
            "loss": float(loss.item()),
            "lr": lr,
            "n_loss_tokens": batch["n_loss_tokens"],
            "per_lane_loss": lane_loss,
            "param_hash": param_hash(model),
            "wall_seconds": dt,
        })
        perf.add_step(batch["n_real_tokens"], batch["n_loss_tokens"],
                      batch["n_total_tokens"], dt)

        _emit(args, {"event": "step", "step": step, "stage": batch["stage"],
                     "batch_id": batch["batch_id"], "loss": round(float(loss.item()), 4),
                     "loss_tokens": batch["n_loss_tokens"]})

        # ---- checkpoint on cadence, and always at the end of the run -------
        is_last = (step + 1) == end_step
        if (step + 1) % ckpt_every == 0 or is_last:
            tag = f"{run_id}_step{step + 1}"
            man = save_checkpoint(
                ckpt_dir, tag, step + 1, batch["stage"], model, opt,
                batcher.get_cursor(),
                {"count": cons.count, "head": cons.head},
                {"count": learn.count, "head": learn.head},
                perf.to_counters(), run_id, lineage,
            )
            saved_tags.append({"tag": tag, "step": step + 1,
                               "model_tensor_hash": man["model_tensor_hash"]})
            _emit(args, {"event": "checkpoint_saved", "tag": tag, "step": step + 1,
                         "consumption_count": cons.count,
                         "model_tensor_hash": man["model_tensor_hash"]})

        # ---- deliberate crash ---------------------------------------------
        if args.crash_at is not None and step + 1 == args.crash_at:
            _emit(args, {"event": "crash", "at_step_completed": step,
                         "next_step_would_be": step + 1,
                         "consumption_count": cons.count,
                         "learning_count": learn.count})
            sys.stdout.flush()
            sys.stderr.flush()
            # A real process death: no exception, no finally blocks, no flush
            # hooks. Anything not already durable on disk is gone.
            os._exit(137)

    write_json(os.path.join(art, f"perf_counters_{run_id}.json"), perf.to_counters())
    _emit(args, {"event": "training_complete", "run_id": run_id,
                 "last_step": end_step - 1, "checkpoints": saved_tags,
                 "consumption_count": cons.count, "learning_count": learn.count})
    return 0


def _emit(args, obj: dict):
    """Emit a structured event line the orchestrator parses and logs."""
    print("EVENT " + json.dumps(obj, sort_keys=True), flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", required=True)
    p.add_argument("--run-id", dest="run_id", required=True)
    p.add_argument("--crash-at", dest="crash_at", type=int, default=None)
    p.add_argument("--resume-from", dest="resume_from", default=None)
    p.add_argument("--fork-from", dest="fork_from", default=None)
    p.add_argument("--until", type=int, default=None)
    args = p.parse_args(argv)
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
