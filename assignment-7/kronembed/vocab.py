"""The task vocabulary — one flat, ordered, hashed token list.

Integers 0..999 are single tokens (the xVal / FoNE single-token setting), so
an "unseen number" is a real token whose embedding was simply never trained —
the honest out-of-distribution unit for comparing deterministic vs learned
embeddings. Operators exist in symbolic and word form so the word path of the
unified scheme is exercised by the same model.
"""

from __future__ import annotations

from .embedding import token_value
from .util import sha256_json

PAD, BOS, EOS, ANS = "<pad>", "<bos>", "<eos>", "<ans>"
SPECIALS = [PAD, BOS, EOS, ANS]
OPERATORS = ["+", "*", "=", "plus", "times"]
MAX_INT_TOKEN = 999


def build_vocab() -> list[str]:
    return SPECIALS + OPERATORS + [str(v) for v in range(MAX_INT_TOKEN + 1)]


class Vocab:
    def __init__(self):
        self.tokens = build_vocab()
        self.to_id = {t: i for i, t in enumerate(self.tokens)}
        self.hash = sha256_json(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def id(self, token: str) -> int:
        return self.to_id[token]

    def token(self, idx: int) -> str:
        return self.tokens[idx]

    def value_of_id(self, idx: int):
        """Integer value of a token id, or None (delegates to the embedding's
        own notion of numeric-ness so the two can never disagree)."""
        return token_value(self.tokens[idx])
