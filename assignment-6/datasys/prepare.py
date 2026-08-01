"""Data-preparation stage: documents -> shards -> firewall -> OPUS -> schedule.

Run once per demonstration, before any training. It materializes every artifact
the training / replay / audit stages read, so those stages share one immutable
view of the data:

  manifests/  tokenizer.json, root_manifest.json, <shard>.manifest.json + .bin,
              schedule.json, inventory.json, firewall.json, opus_summary.json
  ledgers/    opus_ledger.jsonl   (every admission decision)

The function returns a "prepared session" dict that ``train.py`` also rebuilds
from disk, guaranteeing the trainer consumes exactly what was prepared.
"""

from __future__ import annotations

import os
from typing import List

from . import firewall as fw_mod
from . import mixture as mix
from . import opus as opus_mod
from .ledger import Ledger
from .shards import ShardWriter, validate_shard
from .tokenizer import build_and_freeze
from .util import ensure_dir, sha256_text, write_json


def load_documents(corpus_path: str) -> List[dict]:
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(__import__("json").loads(line))
    return docs


def prepare(corpus_path: str, artifacts_dir: str, tokenizer_path: str,
            vocab_size: int, total_steps: int, seqs_per_step: int, seq_len: int,
            log=lambda *a, **kw: None) -> dict:
    manifests_dir = ensure_dir(os.path.join(artifacts_dir, "manifests"))
    ledgers_dir = ensure_dir(os.path.join(artifacts_dir, "ledgers"))

    documents = load_documents(corpus_path)
    log("documents_loaded", n=len(documents))

    # --- frozen tokenizer -------------------------------------------------
    train_texts = [d["text"] for d in documents if d["split"] == "train"]
    tok = build_and_freeze(train_texts, vocab_size, tokenizer_path)
    # copy the frozen tokenizer into the artifact manifests for the record
    tok_artifact = os.path.join(manifests_dir, "tokenizer.json")
    tok.save(tok_artifact)
    log("tokenizer_frozen", hash=tok.content_hash[:16], vocab=tok.vocab_size)

    # --- immutable shards + manifests -------------------------------------
    writer = ShardWriter(tok, manifests_dir)
    built = writer.build(documents)
    log("shards_created", n=built["root"]["n_shards"])

    # validate every shard by re-hashing on-disk bytes
    errors = []
    for m in built["shards"]:
        errors += validate_shard(manifests_dir, m, tok.content_hash)
    if errors:
        raise RuntimeError("shard validation failed: " + "; ".join(errors))
    log("manifests_validated", shards=len(built["shards"]))

    # --- firewall ---------------------------------------------------------
    firewall = fw_mod.Firewall.from_documents(documents, sha256_text)
    write_json(os.path.join(manifests_dir, "firewall.json"), firewall.to_obj())
    log("firewall_built", blocked=len(firewall.blocked_hashes))

    # doc_id -> shard span index (from shard manifests)
    shard_index = {}
    for m in built["shards"]:
        for dr in m["documents"]:
            shard_index[dr["doc_id"]] = {
                "shard_id": m["shard_id"],
                "token_start": dr["token_start"],
                "token_end": dr["token_end"],
                "n_tokens": dr["n_tokens"],
                "prompt_tokens": dr["prompt_tokens"],
            }

    # --- OPUS admission ---------------------------------------------------
    # A floored lane must be able to supply the tokens its floor will demand
    # across the whole run; when its budget cannot, OPUS overrides the budget.
    floor_demand = opus_mod.floor_token_demand(seq_len, seqs_per_step, total_steps)
    admission = opus_mod.admit(documents, shard_index, firewall, sha256_text,
                               floor_demand)
    # write opus decision ledger (append-only, hash-chained)
    opus_ledger = Ledger(os.path.join(ledgers_dir, "opus_ledger.jsonl"))
    for entry in admission["ledger"]:
        opus_ledger.append(entry)
    write_json(os.path.join(manifests_dir, "inventory.json"), admission["inventory"])
    write_json(os.path.join(manifests_dir, "opus_summary.json"), admission["summary"])
    log("opus_admission", **admission["summary"]["by_decision"])

    # Every lane the schedule can ask for must have admitted inventory, or the
    # mixture is unmeetable. Fail loudly here rather than mid-training.
    empty = [lane for lane in mix.LANES if not admission["inventory"].get(lane)]
    if empty:
        raise RuntimeError(f"lanes with no admitted inventory: {empty}")

    # --- mixture schedule -------------------------------------------------
    schedule = mix.compile_schedule(total_steps, seqs_per_step)
    write_json(os.path.join(manifests_dir, "schedule.json"), schedule)
    log("mixture_compiled", steps=total_steps, seqs_per_step=seqs_per_step)

    return {
        "tokenizer": tok,
        "manifests": built,
        "firewall": firewall,
        "inventory": admission["inventory"],
        "opus_summary": admission["summary"],
        "schedule": schedule,
        "documents": documents,
    }
