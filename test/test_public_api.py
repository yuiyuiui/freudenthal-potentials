"""Contract checks independent of the local-generation implementation."""

import numpy as np
import pytest
from test_evaluation import make_polynomial_field

from freudenthal import construct_basis


@pytest.fixture(scope="module")
def basis():
    return construct_basis(2, 7)


def test_batch_evaluation_against_analytic_curl():
    field = make_polynomial_field(7)
    points = np.array([[0.11, 0.23, 0.37], [0, 0, 0], [1, 1, 1]])
    value, curl = field.evaluate_many(0, points)
    np.testing.assert_allclose(value, points[:, [2, 0, 1]] ** 2, atol=2e-14, rtol=2e-13)
    np.testing.assert_allclose(curl, 2 * points[:, [1, 2, 0]], atol=2e-14, rtol=2e-13)
    assert field.evaluate_many(0, np.empty((0, 3)))[0].shape == (0, 3)


def test_exact_support_against_independent_local_polynomials(basis):
    for row_id in [0, len(basis) // 2, len(basis) - 1]:
        row = basis.coefficients.getrow(row_id)
        expected = []
        for t, nodes in enumerate(basis.space.local_to_global):
            cols = (3 * nodes[:, None] + np.arange(3)).ravel()
            if np.any(row[:, cols].toarray()):
                expected.append(t)
        support = basis.support_tetrahedra(row_id)
        assert support.tolist() == expected
        assert set(support) <= basis.space.mesh.stars[basis.owners[row_id]]
        outside = sorted(set(range(len(basis.space.mesh.tets))) - set(support))
        centers = np.mean(basis.space.mesh.vertices[basis.space.mesh.tets[outside]], axis=1) / 2
        value, curl = basis.evaluate_many(row_id, centers)
        assert np.count_nonzero(value) == np.count_nonzero(curl) == 0


def test_curl_of_constructed_basis_against_finite_differences(basis):
    point = np.array([0.17, 0.29, 0.41])  # away from cell/diagonal faces
    # Choose a row with a nonzero value on the containing tetrahedron.
    tet = 5
    columns = (3 * basis.space.local_to_global[tet, :, None] + np.arange(3)).ravel()
    touched = np.flatnonzero(basis.coefficients[:, columns].getnnz(axis=1))
    row_id = int(touched[len(touched) // 2])
    step = 1e-6
    gradient = np.column_stack(
        [
            (
                basis.evaluate(row_id, point + step * axis)[0]
                - basis.evaluate(row_id, point - step * axis)[0]
            )
            / (2 * step)
            for axis in np.eye(3)
        ]
    )
    expected = [
        gradient[2, 1] - gradient[1, 2],
        gradient[0, 2] - gradient[2, 0],
        gradient[1, 0] - gradient[0, 1],
    ]
    _, actual = basis.evaluate(row_id, point)
    assert np.linalg.norm(actual) > 1e-8
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize("index", [True, 0.5, -1, 1000000])
def test_index_errors_are_explicit(basis, index):
    with pytest.raises((TypeError, IndexError)):
        basis.evaluate(index, [0.1, 0.2, 0.3])
    with pytest.raises((TypeError, IndexError)):
        basis.support_tetrahedra(index)
    with pytest.raises((TypeError, IndexError)):
        basis.evaluate_on_tetrahedron(0, index, [0.25] * 4)


@pytest.mark.parametrize("points", [[0, 0, 0], [[0, 0]], [[np.inf, 0, 0]], [[1.1, 0, 0]]])
def test_batch_rejects_invalid_shape_and_domain(basis, points):
    with pytest.raises(ValueError):
        basis.evaluate_many(0, points)
