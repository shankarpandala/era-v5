"""Immutable tokenized shards + manifests.

A shard is a write-once binary file of uint32 token ids for the documents of one
(lane, split) group, plus a JSON manifest that pins down its identity:

  * the frozen tokenizer's content hash (so a shard is only valid under the
    tokenizer that produced it),
  * for every document: its byte hash, token span [start, end) inside the shard,
    token count, and per-document metadata (type, quality, prompt length),
  * the sha256 of the raw token bytes and of the manifest itself.

"Immutable" is enforced by never rewriting an existing shard, and *proven* every
run by ``validate_shard`` which re-hashes the on-disk bytes and compares to the
manifest. A root manifest hashes the set of shards so the whole dataset has one
identity.
"""

from __future__ import annotations

import os
import struct
from typing import Dict, List

from .tokenizer import EOS_ID, Tokenizer
from .util import (
    ensure_dir,
    read_json,
    sha256_bytes,
    sha256_json,
    sha256_text,
    write_json,
)

TOKEN_STRUCT = "<I"  # little-endian uint32
TOKEN_BYTES = 4


def _pack_tokens(tokens: List[int]) -> bytes:
    return struct.pack(f"<{len(tokens)}I", *tokens)


def _unpack_tokens(data: bytes) -> List[int]:
    n = len(data) // TOKEN_BYTES
    return list(struct.unpack(f"<{n}I", data))


class ShardWriter:
    """Builds all shards from documents. Immutable: refuses to overwrite."""

    def __init__(self, tokenizer: Tokenizer, out_dir: str):
        self.tok = tokenizer
        self.out_dir = ensure_dir(out_dir)

    def build(self, documents: List[dict]) -> dict:
        # group by (lane, split); deterministic ordering by doc_id
        groups: Dict[tuple, List[dict]] = {}
        for d in sorted(documents, key=lambda x: x["doc_id"]):
            groups.setdefault((d["lane"], d["split"]), []).append(d)

        shard_manifests = []
        for (lane, split), docs in sorted(groups.items()):
            shard_manifests.append(self._build_one(lane, split, docs))

        root = {
            "tokenizer_hash": self.tok.content_hash,
            "n_shards": len(shard_manifests),
            "shards": [
                {
                    "shard_id": m["shard_id"],
                    "lane": m["lane"],
                    "split": m["split"],
                    "shard_hash": m["shard_hash"],
                    "manifest_hash": m["manifest_hash"],
                    "n_tokens": m["n_tokens"],
                }
                for m in shard_manifests
            ],
        }
        root["root_hash"] = sha256_json(root)
        root_path = os.path.join(self.out_dir, "root_manifest.json")
        write_json(root_path, root)
        return {"root": root, "shards": shard_manifests}

    def _build_one(self, lane: str, split: str, docs: List[dict]) -> dict:
        shard_id = f"{lane}__{split}"
        bin_path = os.path.join(self.out_dir, f"{shard_id}.tokens.bin")
        man_path = os.path.join(self.out_dir, f"{shard_id}.manifest.json")

        all_tokens: List[int] = []
        doc_records = []
        for d in docs:
            ids = self.tok.encode(d["text"])
            # prompt length in *tokens* for loss masking (reasoning/agentic)
            prompt_tokens = 0
            if d.get("prompt_len_chars", 0) > 0:
                prompt_tokens = len(self.tok.encode(d["text"][: d["prompt_len_chars"]]))
            start = len(all_tokens)
            all_tokens.extend(ids)
            all_tokens.append(EOS_ID)  # explicit document boundary
            end = len(all_tokens)  # includes the EOS
            doc_records.append({
                "doc_id": d["doc_id"],
                "type": d["type"],
                "quality": d.get("quality", "ok"),
                "text_hash": sha256_text(d["text"]),
                "token_start": start,
                "token_end": end,      # [start, end); end-1 is the EOS
                "n_tokens": end - start,
                "prompt_tokens": prompt_tokens,
            })

        token_bytes = _pack_tokens(all_tokens)
        # immutability: never overwrite an existing shard
        if os.path.exists(bin_path):
            existing = open(bin_path, "rb").read()
            if existing != token_bytes:
                raise RuntimeError(f"shard {shard_id} is immutable but content changed")
        else:
            with open(bin_path, "wb") as f:
                f.write(token_bytes)

        manifest = {
            "shard_id": shard_id,
            "lane": lane,
            "split": split,
            "tokenizer_hash": self.tok.content_hash,
            "n_docs": len(doc_records),
            "n_tokens": len(all_tokens),
            "token_file": os.path.basename(bin_path),
            "shard_hash": sha256_bytes(token_bytes),
            "documents": doc_records,
        }
        manifest["manifest_hash"] = sha256_json(
            {k: v for k, v in manifest.items() if k != "manifest_hash"}
        )
        write_json(man_path, manifest)
        return manifest


def load_shard_tokens(out_dir: str, shard_id: str) -> List[int]:
    path = os.path.join(out_dir, f"{shard_id}.tokens.bin")
    with open(path, "rb") as f:
        return _unpack_tokens(f.read())


def validate_shard(out_dir: str, manifest: dict, tokenizer_hash: str) -> List[str]:
    """Re-derive the shard's identity from disk. Returns a list of errors (empty
    means the shard is intact and matches its manifest)."""
    errors: List[str] = []
    shard_id = manifest["shard_id"]
    if manifest["tokenizer_hash"] != tokenizer_hash:
        errors.append(f"{shard_id}: tokenizer hash mismatch")
    bin_path = os.path.join(out_dir, manifest["token_file"])
    if not os.path.exists(bin_path):
        errors.append(f"{shard_id}: token file missing")
        return errors
    data = open(bin_path, "rb").read()
    if sha256_bytes(data) != manifest["shard_hash"]:
        errors.append(f"{shard_id}: shard hash mismatch (mutated bytes)")
    if len(data) // TOKEN_BYTES != manifest["n_tokens"]:
        errors.append(f"{shard_id}: token count mismatch")
    recomputed = sha256_json({k: v for k, v in manifest.items() if k != "manifest_hash"})
    if recomputed != manifest["manifest_hash"]:
        errors.append(f"{shard_id}: manifest hash mismatch")
    return errors


def load_manifests(out_dir: str) -> dict:
    root = read_json(os.path.join(out_dir, "root_manifest.json"))
    shards = {}
    for s in root["shards"]:
        m = read_json(os.path.join(out_dir, f"{s['shard_id']}.manifest.json"))
        shards[s["shard_id"]] = m
    return {"root": root, "shards": shards}
