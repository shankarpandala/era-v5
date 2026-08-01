"""Append-only, hash-chained ledgers.

Both the consumption ledger (what was fed to the model) and the learning ledger
(what the model did with it) are append-only JSONL where each line carries the
hash of the previous line:

    entry.prev = <hash of previous entry>
    entry.hash = sha256(prev + canonical(payload))

This makes the ledger tamper-evident: altering or dropping any past line breaks
the chain from that point on, which ``verify_chain`` detects. The current chain
head hash and entry count are stored inside every checkpoint, so a checkpoint is
cryptographically tied to an exact ledger prefix -- the basis for proving that
resume neither skips nor repeats a batch.
"""

from __future__ import annotations

import json
import os
from typing import Iterator, List

from .util import canonical_json, ensure_dir, sha256_text

GENESIS = "0" * 64


class Ledger:
    def __init__(self, path: str):
        self.path = path
        ensure_dir(os.path.dirname(path))
        self.head = GENESIS
        self.count = 0
        if os.path.exists(path):
            self._reload()

    def _reload(self):
        head = GENESIS
        count = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                head = obj["hash"]
                count += 1
        self.head = head
        self.count = count

    def append(self, payload: dict) -> dict:
        prev = self.head
        h = sha256_text(prev + canonical_json(payload))
        entry = {"seq": self.count, "prev": prev, "hash": h, "payload": payload}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.head = h
        self.count += 1
        return entry

    def truncate_to(self, count: int) -> None:
        """Drop entries so exactly ``count`` remain, recomputing the head.

        Used by resume: a checkpoint pins the committed entry count; any ledger
        lines written after the last checkpoint (the crash window) are rolled
        back so the resumed run rewrites them identically instead of duplicating.
        """
        kept: List[str] = []
        head = GENESIS
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                if obj["seq"] < count:
                    kept.append(line if line.endswith("\n") else line + "\n")
                    head = obj["hash"]
        with open(self.path, "w", encoding="utf-8") as f:
            f.writelines(kept)
        self.head = head
        self.count = count

    def entries(self) -> Iterator[dict]:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    yield json.loads(s)


def verify_chain(path: str) -> dict:
    """Recompute the hash chain from scratch. Returns {ok, count, head, error}."""
    prev = GENESIS
    count = 0
    if not os.path.exists(path):
        return {"ok": True, "count": 0, "head": GENESIS, "error": None}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if obj["seq"] != count:
                return {"ok": False, "count": count, "head": prev,
                        "error": f"seq gap at line {i}"}
            if obj["prev"] != prev:
                return {"ok": False, "count": count, "head": prev,
                        "error": f"prev mismatch at seq {obj['seq']}"}
            recomputed = sha256_text(prev + canonical_json(obj["payload"]))
            if recomputed != obj["hash"]:
                return {"ok": False, "count": count, "head": prev,
                        "error": f"hash mismatch at seq {obj['seq']}"}
            prev = obj["hash"]
            count += 1
    return {"ok": True, "count": count, "head": prev, "error": None}
