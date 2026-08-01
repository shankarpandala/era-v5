"""Checkpoints tied to ledger offsets.

A checkpoint captures everything needed to resume the run as if it never stopped:

  * model + optimizer state (torch),
  * RNG states (torch, numpy, python) so stochastic ops replay identically,
  * the batcher cursor (per-lane stream position) -> the next batch is determined,
  * ledger offsets: committed entry count *and* chain-head hash for both the
    consumption and learning ledgers, so the checkpoint is bound to an exact
    ledger prefix,
  * step / stage and optional fork lineage.

A sidecar JSON manifest carries the offsets, cursor and a content hash of the
model tensors, so the audit can reason about checkpoints without loading torch.
"""

from __future__ import annotations

import io
import os
import random
from typing import Dict, Optional

import numpy as np
import torch

from .util import ensure_dir, read_json, sha256_bytes, write_json


def _model_tensor_hash(model) -> str:
    h = io.BytesIO()
    for name, p in sorted(model.state_dict().items()):
        h.write(name.encode("utf-8"))
        h.write(p.detach().cpu().contiguous().numpy().tobytes())
    return sha256_bytes(h.getvalue())


def save_checkpoint(
    ckpt_dir: str,
    tag: str,
    step: int,
    stage: str,
    model,
    optimizer,
    cursor: Dict[str, dict],
    consumption_offset: dict,
    learning_offset: dict,
    perf_counters: dict,
    run_id: str,
    lineage: Optional[dict] = None,
    data_binding: Optional[dict] = None,
) -> dict:
    ensure_dir(ckpt_dir)
    blob_path = os.path.join(ckpt_dir, f"{tag}.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
    }, blob_path)

    manifest = {
        "tag": tag,
        "run_id": run_id,
        "step": step,             # NEXT step to execute after resume
        "stage": stage,
        "cursor": cursor,
        "consumption_offset": consumption_offset,  # {count, head}
        "learning_offset": learning_offset,
        "perf_counters": perf_counters,
        "model_tensor_hash": _model_tensor_hash(model),
        "blob_file": os.path.basename(blob_path),
        "blob_hash": None,
        "lineage": lineage or {},
        # Bind the checkpoint to the immutable data identity so a resume against
        # a different tokenizer / schedule / inventory is detectable.
        "data_binding": data_binding or {},
    }
    with open(blob_path, "rb") as f:
        manifest["blob_hash"] = sha256_bytes(f.read())
    man_path = os.path.join(ckpt_dir, f"{tag}.manifest.json")
    manifest["manifest_hash"] = write_json(man_path, manifest)
    return manifest


def load_checkpoint(ckpt_dir: str, tag: str, model, optimizer=None) -> dict:
    man = read_json(os.path.join(ckpt_dir, f"{tag}.manifest.json"))
    blob_path = os.path.join(ckpt_dir, man["blob_file"])
    blob = torch.load(blob_path, weights_only=False)
    model.load_state_dict(blob["model"])
    if optimizer is not None and "optimizer" in blob:
        optimizer.load_state_dict(blob["optimizer"])
    torch.set_rng_state(blob["rng_torch"])
    np.random.set_state(blob["rng_numpy"])
    random.setstate(blob["rng_python"])
    return man


def verify_checkpoint(ckpt_dir: str, tag: str) -> Dict[str, object]:
    man = read_json(os.path.join(ckpt_dir, f"{tag}.manifest.json"))
    blob_path = os.path.join(ckpt_dir, man["blob_file"])
    ok = True
    err = None
    if not os.path.exists(blob_path):
        return {"ok": False, "error": "blob missing"}
    with open(blob_path, "rb") as f:
        if sha256_bytes(f.read()) != man["blob_hash"]:
            ok, err = False, "blob hash mismatch"
    return {"ok": ok, "error": err, "manifest": man}
