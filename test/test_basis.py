import numpy as np
import pytest
from scipy import sparse

from freudenthal import construct_basis, dimension
from freudenthal.algebra import sparse_mod_rank
from freudenthal.exact import integer_product


@pytest.fixture(scope="module")
def high_basis():
    return construct_basis(2, 7)


def test_high_degree_against_independent_constraint_ranks(high_basis):
    A, _ = high_basis.space.curl_constraints(all_components=True)
    for prime in (1000003, 1000033):
        assert sparse_mod_rank(A, prime).rank + len(high_basis) == A.shape[1]
    assert len(high_basis) == dimension(2, 7) == 6792
    assert high_basis.summary()["max_abs_coefficient"] <= 3


@pytest.mark.parametrize("N", [1, 2, 3, 4, 5])
def test_critical_construction_and_support(N):
    basis = construct_basis(N, 6)
    assert len(basis) == 300 * N**3 + 390 * N**2 + 99 * N + 6
    assert basis.metadata["all_rows_vertex_local"]


def test_one_cube_higher_degree():
    basis = construct_basis(1, 7)
    assert basis.verify()["independent_rows"] == len(basis)
    assert np.all(basis.owners == 0)


def test_serialization_revalidates_actual_coefficients(high_basis, tmp_path):
    high_basis.save(tmp_path)
    loaded = type(high_basis).load(tmp_path)
    assert (loaded.coefficients - high_basis.coefficients).nnz == 0
    with pytest.raises(FileExistsError):
        high_basis.save(tmp_path)
    loaded.coefficients = loaded.coefficients.copy()
    loaded.coefficients.data[0] += 1
    with pytest.raises(ValueError):
        loaded.verify()


def test_false_owner_is_rejected(high_basis):
    old = high_basis.owners[0]
    high_basis.owners[0] = len(high_basis.space.mesh.vertices) - 1
    try:
        with pytest.raises(ValueError, match="support"):
            high_basis.verify()
    finally:
        high_basis.owners[0] = old


def test_exact_product_rejects_overflow():
    matrix = sparse.csr_matrix([[10**12]], dtype=np.int64)
    with pytest.raises(OverflowError):
        integer_product(matrix, matrix)


@pytest.mark.parametrize("N,d", [(0, 6), (2, 5), (True, 6), (2, 7.5)])
def test_invalid_constructor_inputs(N, d):
    with pytest.raises((TypeError, ValueError)):
        construct_basis(N, d)
