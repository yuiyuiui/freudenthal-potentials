"""Types and relative rows for the all-degree construction."""

from functools import lru_cache
from itertools import permutations

import numpy as np

from . import data

PERMUTATIONS = tuple(permutations(range(3)))


def type_key(q, component: int, N: int, degree: int):
    """Return (boundary layers, ordering, clipped gaps, component) and cell.

    Ties in the coordinate ordering use increasing coordinate indices.
    A clipped gap equal to 3 means an unbounded integer at least 3.
    """
    q = tuple(int(x) for x in q)
    if len(q) != 3 or not all(0 <= x <= N * degree for x in q):
        raise ValueError("Coefficient position must lie in the coefficient cube")
    if component not in range(3) or N < 2 or degree < 7:
        raise ValueError("All-degree types require N>=2, d>=7 and component 0, 1 or 2")
    cell = tuple(min(x // degree, N - 1) for x in q)
    remainder = tuple(x - degree * a for x, a in zip(q, cell))
    order = tuple(sorted(range(3), key=lambda j: (-remainder[j], j)))
    r = [remainder[j] for j in order]
    gaps = (degree - r[0], r[0] - r[1], r[1] - r[2], r[2])
    clipped = tuple(min(g, 3) for g in gaps)
    boundary = tuple(0 if a == 0 else 2 if a == N - 1 else 1 for a in cell)
    return (boundary, order, clipped, component), np.asarray(cell, dtype=np.int64)


def encode(key) -> int:
    boundary, order, gaps, component = key
    b = boundary[0] * 9 + boundary[1] * 3 + boundary[2]
    g = gaps[0] * 64 + gaps[1] * 16 + gaps[2] * 4 + gaps[3]
    return ((b * 6 + PERMUTATIONS.index(tuple(order))) * 256 + g) * 3 + component


@lru_cache(maxsize=1)
def rule_table() -> dict:
    result = {}
    for rule in data.rules():
        b, p, g, c = rule["key"]
        key = (tuple(b), tuple(p), tuple(g), c)
        if key in result:
            raise ValueError("Duplicate rule type")
        result[key] = rule
    if len(result) != 59943:
        raise ValueError("Incomplete all-degree rule table")
    return result


@lru_cache(maxsize=1)
def owner_masks() -> dict[int, int]:
    lines = data.certificate_bytes("all_degree/results/universal_owners.txt").decode().splitlines()
    result = dict(tuple(map(int, line.split())) for line in lines[1:])
    if len(result) != int(lines[0]) or len(result) != 42612 or not all(result.values()):
        raise ValueError("Incomplete universal vertex-owner table")
    return result


def critical_key(q, component: int, N: int, radius: int = 5):
    signature = tuple(
        (int(x) % 6, min(int(x), radius + 1), min(6 * N - int(x), radius + 1)) for x in q
    )
    return signature, component
