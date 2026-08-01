"""Shared fixtures: a miniature prepared session built in a temp directory.

Tests build a real (tiny) pipeline rather than mocking it, so an invariant test
exercises the same code path the demonstration does.
"""

from __future__ import annotations

import os

from datasys.prepare import prepare

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "corpus", "documents.jsonl")
TOKENIZER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "tokenizer", "tokenizer.json")

# Same shape as the demonstration run, so admission budgets, protected-floor
# demand and packing behaviour in tests match what run_demo.py exercises.
SEQ_LEN = 256
SEQS_PER_STEP = 8
TOTAL_STEPS = 24


def build_session(tmpdir: str, total_steps: int = TOTAL_STEPS,
                  seqs_per_step: int = SEQS_PER_STEP, seq_len: int = SEQ_LEN):
    """Prepare shards/manifests/firewall/OPUS/schedule inside ``tmpdir``."""
    session = prepare(CORPUS, tmpdir, TOKENIZER, 1024, total_steps, seqs_per_step,
                      seq_len)
    return session
