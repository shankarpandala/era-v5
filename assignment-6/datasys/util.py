"""Deterministic primitives shared across the system.

Everything that must be reproducible funnels through here: canonical JSON,
content hashing, and a counter-based (splittable, stateless) PRNG so that a
byte-identical stream can be regenerated from a seed + coordinate at any time,
without carrying mutable RNG state around.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from typing import Any

# ---------------------------------------------------------------------------
# Canonical serialization + hashing
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, UTF-8 safe.

    Two structurally-equal objects always serialize to the same bytes, which is
    what makes hashing of manifests / ledger entries meaningful.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def short(h: str, n: int = 12) -> str:
    return h[:n]


# ---------------------------------------------------------------------------
# Counter-based PRNG (stateless / splittable)
# ---------------------------------------------------------------------------


def _hash_to_u64(*parts: Any) -> int:
    """Map an arbitrary coordinate to a 64-bit integer via SHA-256.

    This is the whole trick behind reproducibility: randomness is a *pure
    function* of (seed, coordinate). Regenerating a batch means recomputing the
    same coordinates, never replaying a mutable RNG.
    """
    key = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return struct.unpack_from("<Q", digest, 0)[0]


def rand_u64(seed: int, *coord: Any) -> int:
    return _hash_to_u64(seed, *coord)


def rand_float(seed: int, *coord: Any) -> float:
    """Uniform float in [0, 1) that depends only on (seed, coord)."""
    return (rand_u64(seed, *coord) >> 11) / float(1 << 53)


def rand_int(seed: int, lo: int, hi: int, *coord: Any) -> int:
    """Uniform int in [lo, hi) depending only on (seed, coord)."""
    if hi <= lo:
        return lo
    return lo + (rand_u64(seed, *coord) % (hi - lo))


def deterministic_shuffle(items: list, seed: int, *coord: Any) -> list:
    """Return a new list shuffled purely as a function of (seed, coord).

    Uses a Fisher-Yates whose swaps are drawn from the counter PRNG, so the
    permutation is fully reconstructible.
    """
    out = list(items)
    n = len(out)
    for i in range(n - 1, 0, -1):
        j = rand_u64(seed, *coord, "shuffle", i) % (i + 1)
        out[i], out[j] = out[j], out[i]
    return out


# ---------------------------------------------------------------------------
# Small filesystem helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj: Any) -> str:
    """Write pretty JSON (human-inspectable) and return the content hash of the
    *canonical* form (so formatting never changes the hash)."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    return sha256_json(obj)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def largest_remainder(weights: dict, total: int) -> dict:
    """Apportion an integer `total` across keys in proportion to `weights`.

    Deterministic largest-remainder (Hamilton) method: floors first, then hand
    out the leftover units to the largest fractional remainders. Ties broken by
    key name so the result never depends on dict ordering. This is how a lane
    mixture (fractions) becomes an exact integer quota per step.
    """
    keys = sorted(weights.keys())
    s = sum(max(0.0, weights[k]) for k in keys)
    if s <= 0 or total <= 0:
        return {k: 0 for k in keys}
    exact = {k: max(0.0, weights[k]) / s * total for k in keys}
    floors = {k: int(exact[k]) for k in keys}
    used = sum(floors.values())
    remainder = total - used
    order = sorted(keys, key=lambda k: (-(exact[k] - floors[k]), k))
    for i in range(remainder):
        floors[order[i % len(order)]] += 1
    return floors
