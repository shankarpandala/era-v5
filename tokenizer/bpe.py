"""From-scratch byte-level Byte-Pair-Encoding (BPE) tokenizer.

Karpathy `minbpe`-style, but built so the *encoder* can be re-implemented in
JavaScript byte-for-byte (see ../src/tokenizer/lib/bpe.js). The only
parity-critical piece is the pre-tokenization regex `SPLIT_PATTERN`: merges
never cross a pre-token boundary, so if Python and JS split identically and
agree on UTF-8 bytes (they always do), the token streams are identical.

No third-party ML libraries — the whole algorithm is here. The one external
dependency is the `regex` module, needed because Python's built-in `re` does
not support Unicode property escapes (`\\p{L}` etc.).
"""

from __future__ import annotations

import heapq
import json
from collections import defaultdict

import regex as re

# --- Canonical pre-tokenization pattern -------------------------------------
# Shipped verbatim inside tokenizer.json and reused, character-for-character,
# by the JavaScript encoder. Merges never cross a pre-token boundary, so the
# pre-token IS the unit that can collapse to a single token.
#
# We pre-tokenize at WORD granularity: each chunk is a maximal run of
# non-whitespace with (at most) its single preceding whitespace character
# attached (`\s?\S+`), plus a fallback branch for any leftover whitespace run.
# Rationale:
#   * The assignment's metric is X = tokens / WORDS, where a word is a
#     whitespace-delimited run (len(text.split())). Making the pre-token equal
#     that word means the fertility floor is 1.0 (every word CAN become one
#     token), which is what makes X <= 1.2 attainable for every language —
#     including Telugu, whose 3-byte agglutinative words floor at ~1.35 under a
#     GPT-style pattern that splits punctuation off each word.
#   * Attaching the single leading whitespace keeps encoding reversible
#     (spaces/newlines are preserved, not dropped).
#   * The corpora are whitespace-normalized to single space / newline
#     separators, so on them every word carries exactly one separator and
#     `#chunks == #words`. The `\s+` branch only matters for arbitrary pasted
#     text in the live playground.
#   * NO Unicode property escapes, possessive quantifiers, atomic groups, or
#     inline flags — every construct exists identically in Python `regex` and JS
#     `RegExp(..., 'gu')`, so Python and JS split (hence tokenize) identically.
SPLIT_PATTERN = r"\s?\S+|\s+"

_COMPILED = re.compile(SPLIT_PATTERN)


def pre_tokenize(text: str) -> list[str]:
    """Split text into pre-tokens (merges never cross these boundaries)."""
    return _COMPILED.findall(text)


# --- Word-count metrics (denominator of fertility X = tokens / words) -------
# PRIMARY: whitespace-delimited runs (len(text.split())) — standard fertility.
#   This equals the word-faithful `[\p{L}\p{N}\p{M}]+` count within 1-2% on the
#   India corpora, because keeping combining marks (\p{M}) attached preserves
#   real Indic words.
# SECONDARY: `[\p{L}\p{N}]+` runs — equals Python's built-in `\w+` EXACTLY on
#   these corpora (en 10363, hi 15709, te 7370, mr 12203) and is byte-for-byte
#   replicable in JS `/[\p{L}\p{N}]+/gu`. CAVEAT: \w excludes combining marks,
#   so it splits Indic words at matras/viramas (2-3x inflation) — reported for
#   comparability with common classroom usage, not as a true word count.
WORD_PATTERN = r"[\p{L}\p{N}]+"
_WORD_RE = re.compile(WORD_PATTERN)


def count_words_wplus(text: str) -> int:
    """Secondary word count: Unicode letter/number runs (== re.findall(r'\\w+'))."""
    return len(_WORD_RE.findall(text))


def count_words_split(text: str) -> int:
    """Primary word count: whitespace-delimited runs (len(text.split()))."""
    return len(text.split())


# --- GPT-2 byte<->unicode map (for human-readable token display only) -------
def bytes_to_unicode() -> dict[int, str]:
    """Reversible map from bytes (0-255) to printable Unicode code points.

    Identical to GPT-2's mapping. Used only to render raw bytes visibly in
    vocab.txt and the widget's token chips; it plays no role in encoding.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}


def render_token(token_bytes: bytes) -> str:
    """Render a token's raw bytes as a printable string via the GPT-2 map."""
    return "".join(BYTE_ENCODER[b] for b in token_bytes)


# --- Core merge primitive (identical logic in bpe.js) -----------------------
def merge_seq(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` with `new_id`."""
    out: list[int] = []
    i = 0
    a, b = pair
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# --- Training ---------------------------------------------------------------
def train(
    words_with_weights: list[tuple[list[int], int]],
    vocab_size: int,
    verbose: bool = False,
) -> list[tuple[int, int]]:
    """Train byte-level BPE and return the ordered merge list.

    `words_with_weights` is a list of (byte-id sequence, integer weight) — one
    entry per pre-token occurrence-type. The weight lets us up/down-sample a
    language without duplicating its text (integer weights keep pair counts
    exact so deletions on hitting zero are safe).

    Uses an incremental pair-count index so each merge touches only the words
    that contained the merged pair — fast enough to run many times inside the
    mixing-weight search.

    Returns `merges`: a rank-ordered list of (a, b) pairs. The token id minted
    for the merge at rank r is `256 + r`.
    """
    assert vocab_size >= 256
    num_merges = vocab_size - 256

    words = [list(ids) for ids, _ in words_with_weights]
    weights = [w for _, w in words_with_weights]

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_where: dict[tuple[int, int], set[int]] = defaultdict(set)
    for i, ids in enumerate(words):
        w = weights[i]
        for a, b in zip(ids, ids[1:]):
            pair_counts[(a, b)] += w
            pair_where[(a, b)].add(i)

    # Lazy max-heap keyed (-count, a, b): pops highest count, then lowest (a, b)
    # — exactly our tie-break. Entries go stale when counts change; we re-push
    # the current count of every touched pair each merge and skip stale pops.
    heap = [(-c, a, b) for (a, b), c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[tuple[int, int]] = []
    for k in range(num_merges):
        best = None
        while heap:
            neg_c, a, b = heapq.heappop(heap)
            cur = pair_counts.get((a, b))
            if cur is None or cur != -neg_c:
                continue  # stale entry
            best = (a, b)
            break
        if best is None:
            break

        new_id = 256 + k
        merges.append(best)
        touched: set[tuple[int, int]] = set()

        for i in list(pair_where[best]):
            ids = words[i]
            w = weights[i]
            # Remove this word's current pair contributions.
            for p in zip(ids, ids[1:]):
                pair_counts[p] -= w
                touched.add(p)
                if pair_counts[p] == 0:
                    del pair_counts[p]
                    pair_where.pop(p, None)
                else:
                    pair_where[p].discard(i)
            # Merge, then add the word's new pair contributions.
            new_ids = merge_seq(ids, best, new_id)
            words[i] = new_ids
            for p in zip(new_ids, new_ids[1:]):
                pair_counts[p] += w
                pair_where[p].add(i)
                touched.add(p)

        # Re-push the current count of every pair whose count changed, so the
        # heap always holds a valid entry for each live pair.
        for p in touched:
            c = pair_counts.get(p)
            if c is not None:
                heapq.heappush(heap, (-c, p[0], p[1]))

        if verbose and (k + 1) % 1000 == 0:
            print(f"  merge {k + 1}/{num_merges}")

    return merges


def build_vocab(merges: list[tuple[int, int]]) -> dict[int, bytes]:
    """Reconstruct id -> raw bytes for every token (base bytes + merges)."""
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for k, (a, b) in enumerate(merges):
        vocab[256 + k] = vocab[a] + vocab[b]
    return vocab


# --- Encoding (this exact algorithm is mirrored in bpe.js) ------------------
def _ranks_from_merges(merges: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {tuple(pair): i for i, pair in enumerate(merges)}


def encode_chunk(ids: list[int], ranks: dict[tuple[int, int], int]) -> list[int]:
    """Greedily apply the lowest-rank applicable merge until none remain."""
    while len(ids) >= 2:
        best_pair = None
        best_rank = None
        for pair in zip(ids, ids[1:]):
            r = ranks.get(pair)
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best_pair = pair
        if best_pair is None:
            break
        ids = merge_seq(ids, best_pair, 256 + best_rank)
    return ids


def encode(text: str, ranks: dict[tuple[int, int], int]) -> list[int]:
    """Encode text to token ids: pre-tokenize, then merge within each chunk."""
    out: list[int] = []
    for piece in pre_tokenize(text):
        out.extend(encode_chunk(list(piece.encode("utf-8")), ranks))
    return out


# --- Serialization ----------------------------------------------------------
def save_tokenizer(path: str, merges: list[tuple[int, int]], meta: dict) -> None:
    """Write tokenizer.json (pattern + ordered merges + metadata)."""
    payload = {
        "pattern": SPLIT_PATTERN,
        "word_pattern": WORD_PATTERN,
        "vocab_size": 256 + len(merges),
        "byte_encoder": "gpt2",
        "special_tokens": {},
        "merges": [[int(a), int(b)] for a, b in merges],
        **meta,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=0)


def load_merges(path: str) -> list[tuple[int, int]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [tuple(pair) for pair in data["merges"]]


def write_vocab_txt(path: str, merges: list[tuple[int, int]]) -> None:
    """Write the human-readable token list: `id<TAB>rendered_token` per line."""
    vocab = build_vocab(merges)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(len(vocab)):
            f.write(f"{i}\t{render_token(vocab[i])}\n")
