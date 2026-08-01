"""A tiny, frozen, byte-level BPE tokenizer.

Design decisions:
  * Byte-level base alphabet (0..255) so *any* text is representable and
    ``decode(encode(x)) == x`` exactly -- no unknown tokens, ever.
  * A handful of special tokens with fixed ids (PAD/EOS/BOS/SEP) reserved
    before the byte range so their ids never move when merges change.
  * Training is deterministic (ties broken by pair value) and one-shot: once
    ``tokenizer.json`` exists it is *frozen*. The system re-hashes it at every
    run start and refuses to proceed if the hash drifts -- this is the
    "frozen tokenizer + content hash" guarantee.

The file format is a plain JSON: ``{version, specials, merges}``. Its content
hash (over canonical JSON) is the identity used in every shard manifest.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, List, Tuple

from .util import read_json, sha256_json, write_json

# Fixed special-token ids. Reserved region [0, N_SPECIAL) never overlaps bytes.
SPECIALS = ["<pad>", "<eos>", "<bos>", "<sep>"]
N_SPECIAL = len(SPECIALS)
BYTE_OFFSET = N_SPECIAL  # byte b -> token id (BYTE_OFFSET + b)

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2
SEP_ID = 3


class Tokenizer:
    def __init__(self, merges: List[Tuple[int, int]], vocab_size: int):
        self.merges = [tuple(m) for m in merges]
        self.vocab_size = vocab_size
        # rank of each merge pair -> new id
        self._rank: Dict[Tuple[int, int], int] = {}
        self._new_id: Dict[Tuple[int, int], int] = {}
        nid = BYTE_OFFSET + 256
        for i, pair in enumerate(self.merges):
            self._rank[pair] = i
            self._new_id[pair] = nid
            nid += 1

    # -- identity -----------------------------------------------------------
    def to_obj(self) -> dict:
        return {
            "version": 1,
            "specials": SPECIALS,
            "byte_offset": BYTE_OFFSET,
            "vocab_size": self.vocab_size,
            "merges": [list(m) for m in self.merges],
        }

    @property
    def content_hash(self) -> str:
        return sha256_json(self.to_obj())

    def save(self, path: str) -> str:
        return write_json(path, self.to_obj())

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        obj = read_json(path)
        return cls(obj["merges"], obj["vocab_size"])

    # -- encode / decode ----------------------------------------------------
    def encode(self, text: str) -> List[int]:
        ids = [BYTE_OFFSET + b for b in text.encode("utf-8")]
        if not self.merges:
            return ids
        while True:
            # find the lowest-rank adjacent pair present
            best_rank = None
            best_pos = -1
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                r = self._rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pos = i
            if best_pos < 0:
                break
            pair = (ids[best_pos], ids[best_pos + 1])
            ids[best_pos : best_pos + 2] = [self._new_id[pair]]
        return ids

    def decode(self, ids: List[int]) -> str:
        # Expand each token back to its constituent bytes (specials -> nothing),
        # then UTF-8 decode. Recursion depth is bounded by merge count.
        inv = {nid: pair for pair, nid in self._new_id.items()}

        def expand(tid: int, acc: List[int]):
            if tid in inv:
                a, b = inv[tid]
                expand(a, acc)
                expand(b, acc)
            elif tid >= BYTE_OFFSET:
                acc.append(tid - BYTE_OFFSET)
            # specials expand to nothing

        acc: List[int] = []
        for tid in ids:
            expand(tid, acc)
        return bytes(acc).decode("utf-8", errors="replace")


def train_bpe(texts: List[str], vocab_size: int) -> Tokenizer:
    """Deterministic byte-level BPE training.

    Repeatedly merge the most frequent adjacent pair; ties broken by the pair
    tuple so the outcome depends only on the corpus, never on iteration order.
    """
    base = BYTE_OFFSET + 256
    n_merges = max(0, vocab_size - base)
    corpus: List[List[int]] = [[BYTE_OFFSET + b for b in t.encode("utf-8")] for t in texts]
    merges: List[Tuple[int, int]] = []
    new_id = base
    for _ in range(n_merges):
        counts: Counter = Counter()
        for seq in corpus:
            for i in range(len(seq) - 1):
                counts[(seq[i], seq[i + 1])] += 1
        if not counts:
            break
        # most frequent, tie-broken deterministically by pair
        best = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        pair = best[0]
        if best[1] < 2:
            break
        merges.append(pair)
        # apply merge
        for si in range(len(corpus)):
            seq = corpus[si]
            out: List[int] = []
            j = 0
            while j < len(seq):
                if j < len(seq) - 1 and (seq[j], seq[j + 1]) == pair:
                    out.append(new_id)
                    j += 2
                else:
                    out.append(seq[j])
                    j += 1
            corpus[si] = out
        new_id += 1
    return Tokenizer(merges, base + len(merges))


def build_and_freeze(texts: List[str], vocab_size: int, path: str) -> Tokenizer:
    """Train once and write the frozen file, only if it doesn't already exist.

    Idempotent: on a repo that already ships ``tokenizer.json`` this just loads
    it, so the tokenizer identity is stable across machines.
    """
    if os.path.exists(path):
        return Tokenizer.load(path)
    tok = train_bpe(texts, vocab_size)
    tok.save(path)
    return tok
