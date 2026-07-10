"""Standalone BPE tokenizer loader — the entry point a grader imports.

    from tokenizer import BPETokenizer
    tok = BPETokenizer.load("../public/tokenizer/tokenizer.json")
    ids = tok.encode(open("../public/tokenizer/corpora/te.txt").read())

Only dependency: the `regex` module (via bpe.py). The encoding algorithm lives
in bpe.py and is mirrored byte-for-byte by the JS widget (src/tokenizer/lib/bpe.js).
"""

from __future__ import annotations

import json

import bpe


class BPETokenizer:
    def __init__(self, merges, pattern: str = bpe.SPLIT_PATTERN):
        if pattern != bpe.SPLIT_PATTERN:
            raise ValueError(
                "tokenizer.json pattern differs from bpe.SPLIT_PATTERN; "
                "encoder and artifact are out of sync."
            )
        self.merges = [tuple(p) for p in merges]
        self.ranks = bpe._ranks_from_merges(self.merges)
        self.pattern = pattern
        self._vocab = None

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["merges"], data.get("pattern", bpe.SPLIT_PATTERN))

    def encode(self, text: str) -> list[int]:
        return bpe.encode(text, self.ranks)

    @property
    def vocab(self) -> dict[int, bytes]:
        if self._vocab is None:
            self._vocab = bpe.build_vocab(self.merges)
        return self._vocab

    def decode(self, ids: list[int]) -> str:
        vocab = self.vocab
        return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)
