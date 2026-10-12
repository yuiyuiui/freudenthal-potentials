"""Construct, verify, save, and evaluate vertex-star potential bases."""

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy import sparse

from . import data
from .algebra import coo_from_rows, lift_rows, sparse_mod_rank
from .exact import canonical_kernel, integer_product, require_zero
from .geometry import BernsteinSpace, Mesh, compositions
from .rules import critical_key, encode, owner_masks, rule_table, type_key


def _parameters(N, degree):
    for name, value in (("N", N), ("degree", degree)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    if N < 1 or degree < 6:
        raise ValueError("Construction requires N>=1 and potential degree d>=6")
    return int(N), int(degree)


def _index(value, size, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer index")
    if not 0 <= value < size:
        raise IndexError(f"{name} index must lie in [0, {size})")
    return int(value)


@lru_cache(maxsize=16)
def _evaluation_tables(degree):
    alpha = np.asarray(compositions(degree, 4), dtype=np.int64)
    beta = np.asarray(compositions(degree - 1, 4), dtype=np.int64)
    lookup = {tuple(a): i for i, a in enumerate(alpha)}
    raised = np.asarray(
        [[lookup[tuple(b + np.eye(4, dtype=np.int64)[i])] for i in range(4)] for b in beta]
    )

    def multipliers(indices):
        return np.asarray(
            [
                math.factorial(sum(map(int, a))) / math.prod(math.factorial(int(z)) for z in a)
                for a in indices
            ]
        )

    return alpha, beta, raised, multipliers(alpha), multipliers(beta)


def dimension(N: int, degree: int) -> int:
    """Return the potential-space dimension for N>=1, degree>=6.

    For N>=2, or degree==6, this evaluates the dimension polynomial.
    For N==1 and degree>6 it constructs and checks an integer kernel,
    which can be expensive or fail rational reconstruction.
    """
    N, degree = _parameters(N, degree)
    if N == 1 and degree > 6:
        return len(_one_cube_basis(degree).owners)
    d = degree
    return (
        3 * N**3 * (d - 1) ** 2 * (d - 2)
        + 3 * N**2 * (d - 1) * (5 * d - 4)
        + 9 * N * (2 * d - 1)
        + 6
    )


def _owners(space, coefficients):
    owners = []
    for row in range(coefficients.shape[0]):
        candidates = None
        for col in coefficients.indices[coefficients.indptr[row] : coefficients.indptr[row + 1]]:
            possible = space.node_owner_vertices[int(col) // 3]
            candidates = possible.copy() if candidates is None else candidates & possible
        if not candidates:
            raise ValueError(f"Basis row {row} has no single vertex-star support")
        owners.append(min(candidates))
    return np.asarray(owners, dtype=np.int64)


@dataclass
class PotentialBasis:
    """Rows are Bernstein coefficient vectors; owners index mesh.vertices."""

    space: BernsteinSpace
    coefficients: sparse.csr_matrix
    owners: np.ndarray
    free: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    pivots: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    normalform: sparse.csr_matrix | None = None
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        return self.coefficients.shape[0]

    def verify(self) -> dict:
        """Check all curl components, support, and finite-instance independence.

        Uniform completeness uses the fixed exact certificates and the
        analytical parameter-coverage proof. Run verify_all.py to replay
        their whole finite verification suite.
        """
        space = self.space
        if self.coefficients.shape[1] != 3 * space.nscalar:
            raise ValueError("Basis has the wrong ambient dimension")
        A, _ = space.curl_constraints(all_components=True)
        require_zero(integer_product(A, self.coefficients.T), "Nonzero curl-continuity residual")
        if len(self.owners) != len(self):
            raise ValueError("One owner vertex is required for each basis row")
        for i, owner in enumerate(self.owners):
            if not 0 <= owner < len(space.mesh.vertices):
                raise ValueError("Invalid owner vertex")
            cols = self.coefficients.indices[
                self.coefficients.indptr[i] : self.coefficients.indptr[i + 1]
            ]
            if not len(cols) or not all(
                int(owner) in space.node_owner_vertices[int(c) // 3] for c in cols
            ):
                raise ValueError("Basis row violates its recorded vertex-star support")
        if self.normalform is not None:
            C = self.normalform
            require_zero(
                C[:, self.pivots] - sparse.eye(len(self.pivots), dtype=np.int64, format="csr"),
                "Non-unit pivot block",
            )
            require_zero(
                A - integer_product(A[:, self.pivots], C), "Reverse normal-form identity failed"
            )
        projected = self.coefficients[:, self.free] if len(self.free) else self.coefficients
        if self.metadata.get("canonical_free_identity", False):
            require_zero(
                projected - sparse.eye(len(self), dtype=np.int64, format="csr"),
                "Canonical free-coordinate block is not identity",
            )
        elif sparse_mod_rank(projected, 1000003).rank != len(self):
            raise ValueError("Basis rows fail the exact modular independence check")
        if (space.mesh.N >= 2 or space.d == 6) and len(self) != dimension(space.mesh.N, space.d):
            raise ValueError("Basis count disagrees with the certified dimension")
        report = {
            "all_curl_components_residual_nnz": 0,
            "all_rows_vertex_local": True,
            "independent_rows": len(self),
            "exact_checks_passed": True,
        }
        self.metadata.update(report)
        return report

    def save(self, directory: str | Path):
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Saving requires an empty directory to preserve previous results")
        directory.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(directory / "basis.npz", self.coefficients)
        if self.normalform is not None:
            sparse.save_npz(directory / "normalform.npz", self.normalform)
        np.savez_compressed(
            directory / "metadata.npz",
            N=self.space.mesh.N,
            degree=self.space.d,
            owners=self.owners,
            free=self.free,
            pivots=self.pivots,
        )
        (directory / "summary.json").write_text(json.dumps(self.summary(), indent=2) + "\n")

    @classmethod
    def load(cls, directory: str | Path, *, verify=True):
        directory = Path(directory)
        with np.load(directory / "metadata.npz", allow_pickle=False) as arrays:
            N, d = _parameters(int(arrays["N"]), int(arrays["degree"]))
            space = BernsteinSpace.build(Mesh.build(N), d)
            out = cls(
                space,
                sparse.load_npz(directory / "basis.npz"),
                arrays["owners"].copy(),
                arrays["free"].copy(),
                arrays["pivots"].copy(),
            )
        if (directory / "normalform.npz").exists():
            out.normalform = sparse.load_npz(directory / "normalform.npz")
        out.metadata = json.loads((directory / "summary.json").read_text())
        out.metadata["exact_checks_passed"] = False
        if verify:
            out.verify()
        return out

    def summary(self) -> dict:
        return {
            **self.metadata,
            "N": self.space.mesh.N,
            "potential_degree": self.space.d,
            "k": self.space.d - 1,
            "dimension": len(self),
            "variables": self.coefficients.shape[1],
            "basis_nnz": self.coefficients.nnz,
            "max_abs_coefficient": max((abs(int(x)) for x in self.coefficients.data), default=0),
        }

    def evaluate_on_tetrahedron(self, basis_id: int, tetrahedron: int, lambdas):
        """Evaluate the physical potential and curl from either side of a face."""
        basis_id = _index(basis_id, len(self), "basis")
        tetrahedron = _index(tetrahedron, len(self.space.mesh.tets), "tetrahedron")
        lam = np.asarray(lambdas, dtype=float)
        if (
            lam.shape != (4,)
            or not np.all(np.isfinite(lam))
            or np.any(lam < -1e-12)
            or not np.isclose(lam.sum(), 1, atol=1e-12, rtol=0)
        ):
            raise ValueError("Barycentric coordinates must be nonnegative and sum to one")
        lam = np.maximum(lam, 0)
        d = self.space.d
        nodes = self.space.local_to_global[tetrahedron]
        cols = (3 * nodes[:, None] + np.arange(3)).ravel()
        coeff = self.coefficients.getrow(basis_id)[:, cols].toarray().reshape(-1, 3)
        alpha, beta, raised, alpha_multiplier, beta_multiplier = _evaluation_tables(d)
        values = alpha_multiplier * np.prod(lam**alpha, axis=1)
        derivatives = beta_multiplier * np.prod(lam**beta, axis=1)
        gradients = self.space.mesh.gradients[tetrahedron]
        curl_coeff = d * self.space.mesh.N * np.cross(gradients, coeff[raised]).sum(axis=1)
        value = values @ coeff
        curl = derivatives @ curl_coeff
        return value, curl

    def evaluate(self, basis_id: int, point):
        """Evaluate at a physical point in the closed unit cube."""
        basis_id = _index(basis_id, len(self), "basis")
        x = np.asarray(point, dtype=float)
        if x.shape != (3,) or not np.all(np.isfinite(x)) or np.any(x < 0) or np.any(x > 1):
            raise ValueError("Point must lie in [0,1]^3")
        N = self.space.mesh.N
        y = N * x
        cell = np.minimum(np.floor(y).astype(int), N - 1)
        remainder = y - cell
        order = tuple(sorted(range(3), key=lambda i: (-remainder[i], i)))
        tet = 6 * ((cell[0] * N + cell[1]) * N + cell[2]) + tuple(permutations(range(3))).index(
            order
        )
        r = remainder[list(order)]
        return self.evaluate_on_tetrahedron(
            basis_id, int(tet), [1 - r[0], r[0] - r[1], r[1] - r[2], r[2]]
        )

    def evaluate_many(self, basis_id: int, points):
        """Return physical potential/curl arrays of shape (m,3).

        Points must have shape (m,3) and lie in the closed unit cube.
        Evaluation uses binary64 direct Bernstein polynomials; this is
        not a guarantee of stable evaluation at arbitrarily high degree.
        """
        basis_id = _index(basis_id, len(self), "basis")
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must have shape (m, 3)")
        if not np.all(np.isfinite(points)) or np.any(points < 0) or np.any(points > 1):
            raise ValueError("All points must lie in [0,1]^3")
        value, curl = np.empty_like(points), np.empty_like(points)
        for i, point in enumerate(points):
            value[i], curl[i] = self.evaluate(basis_id, point)
        return value, curl

    def support_tetrahedra(self, basis_id: int) -> np.ndarray:
        """Return sorted tetrahedron IDs with a nonzero polynomial piece.

        This is exact coefficient support, independent of plotting samples.
        It is a subset of the recorded owner's closed vertex star.
        """
        basis_id = _index(basis_id, len(self), "basis")
        row = self.coefficients.getrow(basis_id)
        row.eliminate_zeros()
        support = set()
        for node in np.unique(row.indices // 3):
            support.update(self.space.node_tets[int(node)])
        return np.asarray(sorted(support), dtype=np.int64)


def _high_degree(space):
    table, owners = rule_table(), owner_masks()
    N, d = space.mesh.N, space.d
    lookup = {tuple(q): i for i, q in enumerate(space.nodes)}
    vertices = {tuple(v): i for i, v in enumerate(space.mesh.vertices)}
    row, col, values, pivots, free, vertex = [], [], [], [], [], []
    for i, q in enumerate(space.nodes):
        for component in range(3):
            key, cell = type_key(q, component, N, d)
            rule = table[key]
            index = 3 * i + component
            if rule["kind"] == "pivot":
                ri = len(pivots)
                pivots.append(index)
                for delta, c, value in rule["row"]:
                    row.append(ri)
                    col.append(3 * lookup[tuple(q + np.asarray(delta))] + c)
                    values.append(value)
            else:
                free.append(index)
                mask = owners[encode(key)]
                bit = (mask & -mask).bit_length() - 1
                v = cell + [bit // 4, (bit // 2) % 2, bit % 2]
                vertex.append(vertices[tuple(v)])
    C = sparse.coo_matrix(
        (np.asarray(values, dtype=np.int64), (row, col)), shape=(len(pivots), 3 * space.nscalar)
    ).tocsr()
    pivots = np.asarray(pivots, dtype=np.int64)
    G, canonical_free = canonical_kernel(C, pivots)
    if not np.array_equal(canonical_free, free):
        raise ValueError("Free-coordinate ordering mismatch")
    return PotentialBasis(
        space,
        G,
        np.asarray(vertex, dtype=np.int64),
        canonical_free,
        pivots,
        C,
        {"construction": "parametric rules", "canonical_free_identity": True},
    )


def _critical_degree(space):
    N = space.mesh.N
    if N < 5:
        prefix = "critical/results" if N in (1, 4) else "critical/aim633_experiments/results"
        B = data.certificate_sparse(f"{prefix}/N{N}_d6/selected_local_basis.npz")
        return PotentialBasis(
            space, B, _owners(space, B), metadata={"construction": "small-grid certificate"}
        )
    table = {}
    for rule in data.certificate_json("critical/results/degree6_normalform_templates.json"):
        key = tuple(tuple(axis) for axis in rule["key"][0]), rule["key"][1]
        table[key] = rule
    lookup = {tuple(q): i for i, q in enumerate(space.nodes)}
    ri, ci, va, pivots = [], [], [], []
    for i, q in enumerate(space.nodes):
        for component in range(3):
            rule = table[critical_key(q, component, N)]
            if rule["kind"] == "free":
                continue
            row = len(pivots)
            pivots.append(3 * i + component)
            for delta, c, z in rule["row"]:
                ri.append(row)
                ci.append(3 * lookup[tuple(q + np.asarray(delta))] + c)
                va.append(z)
    pivots = np.asarray(pivots, dtype=np.int64)
    C = sparse.coo_matrix(
        (np.asarray(va, dtype=np.int64), (ri, ci)), shape=(len(pivots), 3 * space.nscalar)
    ).tocsr()
    G, free = canonical_kernel(C, pivots)
    ref_space, ref_free, ref_owner, parts, repairs = _critical_reference()
    representatives = {}
    for i, f in enumerate(ref_free):
        representatives.setdefault(critical_key(ref_space.nodes[f // 3], int(f % 3), 6, 12), i)
    pool = []
    for i, f in enumerate(free):
        q = space.nodes[f // 3]
        ref = representatives[critical_key(q, int(f % 3), N, 12)]
        if ref_owner[ref] >= 0:
            pool.append(G.getrow(i))
        else:
            delta = q - ref_space.nodes[ref_free[ref] // 3]
            repair = repairs[ref]
            if repair["denominator"] != 1:
                raise ValueError("Expected integral critical-degree repairs")
            for part_id in repair["part_indices"]:
                part = parts.getrow(part_id)
                row = {
                    3 * lookup[tuple(ref_space.nodes[c // 3] + delta)] + int(c % 3): int(z)
                    for c, z in zip(part.indices, part.data)
                }
                pool.append(coo_from_rows([row], 3 * space.nscalar))
    generators = sparse.vstack(pool, format="csr")
    echelon = sparse_mod_rank(generators[:, free], 1000003)
    if echelon.rank != len(free):
        raise ValueError("Local critical-degree generators do not span the canonical kernel")
    B = generators[echelon.independent_inputs].tocsr()
    return PotentialBasis(
        space,
        B,
        _owners(space, B),
        free,
        pivots,
        C,
        {
            "construction": "window transfer, repairs, independent subset",
            "generator_pool_size": generators.shape[0],
            "canonical_free_identity": False,
        },
    )


@lru_cache(maxsize=1)
def _critical_reference():
    prefix = "critical/results/repairs_N6_d6"
    space = BernsteinSpace.build(Mesh.build(6), 6)
    free = data.certificate_array(prefix + "/free.npy")
    owner = data.certificate_array(prefix + "/local_owner.npy")
    parts = data.certificate_sparse(prefix + "/repair_parts.npz")
    repairs = {r["free_index"]: r for r in data.certificate_json(prefix + "/repairs.json")}
    return space, free, owner, parts, repairs


def _one_cube_basis(degree):
    space = BernsteinSpace.build(Mesh.build(1), degree)
    A, _ = space.curl_constraints()
    echelon = sparse_mod_rank(A, 1000003)
    lifted, _ = lift_rows(echelon.nullspace_rows(), 1000003)
    B = coo_from_rows(lifted, A.shape[1])
    # Integer membership and a nonzero free-coordinate diagonal, together
    # with the modular constraint rank, prove real-field completeness.
    free = np.asarray([i for i in range(A.shape[1]) if i not in echelon.rows], dtype=np.int64)
    require_zero(integer_product(A, B.T), "One-cube rational lift failed")
    if sparse_mod_rank(B[:, free], 1000003).rank + echelon.rank != A.shape[1]:
        raise ValueError("One-cube completeness check failed")
    return PotentialBasis(
        space,
        B,
        np.zeros(B.shape[0], dtype=np.int64),
        free,
        metadata={"construction": "verified one-cube integer kernel"},
    )


def construct_basis(N: int, degree: int, *, verify=True) -> PotentialBasis:
    """Construct a basis for every N>=1 and d>=6; verify exact finite checks."""
    N, degree = _parameters(N, degree)
    if N == 1 and degree > 6:
        basis = _one_cube_basis(degree)
    else:
        space = BernsteinSpace.build(Mesh.build(N), degree)
        basis = _critical_degree(space) if degree == 6 else _high_degree(space)
    basis.metadata["exact_checks_passed"] = False
    if verify:
        basis.verify()
    return basis
