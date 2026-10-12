"""Independent assembly checks for the fixed degree-six certificates."""

import numpy as np
from scipy import sparse

from . import data
from .exact import integer_product, require_zero
from .geometry import BernsteinSpace, Mesh


def verify_reference_assembly() -> dict:
    """Reassemble using analytical integer gradients, then check source identities.

    The compatibility checkers preserve the recovered implementation. This
    second assembly removes their floating inverse from the trust boundary.
    """
    for N in range(1, 5):
        prefix = "critical/results" if N in (1, 4) else "critical/aim633_experiments/results"
        space = BernsteinSpace.build(Mesh.build(N), 6)
        A, _ = space.curl_constraints()
        saved = data.certificate_sparse(f"{prefix}/N{N}_d6/constraints.npz")
        require_zero(A - saved, f"Small-grid constraint assembly mismatch at N={N}")
    space = BernsteinSpace.build(Mesh.build(6), 6)
    A, _ = space.curl_constraints()
    prefix = "critical/results/rref_N6_d6"
    C = data.certificate_sparse(prefix + "/C.npz")
    W = data.certificate_sparse(prefix + "/W.npz")
    pivots = data.certificate_array(prefix + "/pivots.npy")
    require_zero(integer_product(W, A) - C, "Reassembled forward identity failed")
    require_zero(A - integer_product(A[:, pivots], C), "Reassembled reverse identity failed")
    require_zero(
        C[:, pivots] - sparse.eye(len(pivots), dtype=np.int64, format="csr"),
        "Reassembled pivot block failed",
    )
    return {
        "passed": True,
        "analytical_integer_gradients": True,
        "small_grid_matrices_equal": [1, 2, 3, 4],
        "reference_N": 6,
        "forward_residual_nnz": 0,
        "reverse_residual_nnz": 0,
        "pivot_residual_nnz": 0,
    }
