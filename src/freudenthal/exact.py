"""Exact integer sparse operations with explicit accumulation bounds."""

import numpy as np
from scipy import sparse


def integer_product(left, right):
    """Multiply integer sparse matrices only when int64 accumulation is safe."""
    left = left.tocsr()
    right = right.tocsr()
    if left.dtype.kind not in "iu" or right.dtype.kind not in "iu":
        raise TypeError("Exact products require integer matrices")
    a = max((abs(int(x)) for x in left.data), default=0)
    b = max((abs(int(x)) for x in right.data), default=0)
    terms = int(np.max(np.diff(left.indptr), initial=0))
    if terms * a * b >= np.iinfo(np.int64).max:
        raise OverflowError("Sparse integer accumulation may overflow int64")
    return left.astype(np.int64) @ right.astype(np.int64)


def require_zero(matrix, message: str):
    matrix = matrix.tocsr()
    matrix.eliminate_zeros()
    if matrix.nnz:
        raise ValueError(message)


def canonical_kernel(normalform, pivots):
    n = normalform.shape[1]
    free = np.setdiff1d(np.arange(n, dtype=np.int64), pivots)
    cf = normalform[:, free].tocoo()
    rows = np.r_[np.arange(len(free)), cf.col]
    cols = np.r_[free, pivots[cf.row]]
    values = np.r_[np.ones(len(free), dtype=np.int64), -cf.data]
    return sparse.coo_matrix((values, (rows, cols)), shape=(len(free), n)).tocsr(), free
