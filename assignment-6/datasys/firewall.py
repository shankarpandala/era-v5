"""Evaluation / validation firewall.

The single most important safety invariant of a training-data system: text that
is used to *measure* the model must never be used to *train* it. This firewall
is content-hash based, so it catches a blocked document even if it is relabelled
or copied into a training lane (the "poisoned" decoy in the corpus).

The firewall exposes two things:
  * a registry of blocked content hashes (built from every eval/validation doc),
  * ``check(content_hash)`` returning a reason string if blocked, else None.

Both the shard builder and the OPUS admission controller consult it, and the
independent audit re-scans the whole consumption ledger against it.
"""

from __future__ import annotations

from typing import Dict, List, Optional

BLOCKED_SPLITS = {"eval", "validation"}


class Firewall:
    def __init__(self):
        # content_hash -> {"split":..., "doc_id":...}
        self._blocked: Dict[str, dict] = {}

    def register(self, doc_id: str, split: str, content_hash: str) -> None:
        if split in BLOCKED_SPLITS:
            self._blocked[content_hash] = {"split": split, "doc_id": doc_id}

    def check(self, content_hash: str) -> Optional[str]:
        """Return a reason code if this content is firewalled, else None."""
        rec = self._blocked.get(content_hash)
        if rec is None:
            return None
        return f"{rec['split']}_firewall"

    @property
    def blocked_hashes(self) -> List[str]:
        return sorted(self._blocked.keys())

    def to_obj(self) -> dict:
        return {
            "blocked_splits": sorted(BLOCKED_SPLITS),
            "count": len(self._blocked),
            "entries": [
                {"content_hash": h, **self._blocked[h]}
                for h in sorted(self._blocked.keys())
            ],
        }

    @classmethod
    def from_documents(cls, documents: List[dict], content_hash_fn) -> "Firewall":
        fw = cls()
        for d in documents:
            fw.register(d["doc_id"], d["split"], content_hash_fn(d["text"]))
        return fw
