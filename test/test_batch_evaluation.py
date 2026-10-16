"""Independent scalar-path controls for bounded vectorized evaluation."""
import numpy as np
import pytest

from freudenthal import construct_basis


@pytest.mark.parametrize('N,degree', [(1, 6), (2, 7)])
def test_batch_matches_scalar_on_faces_ties_vertices_and_random_points(N, degree):
    basis = construct_basis(N, degree)
    rng = np.random.default_rng(633)
    points = np.vstack((rng.uniform(size=(259, 3)),
                        [[0, 0, 0], [1, 1, 1], [.5, .5, .5],
                         [.25, .25, .75], [1, 0, 0], [.5, 1, .5]]))
    original = points.copy()
    for index in (0, min(10, len(basis) - 1)):
        actual = basis.evaluate_many(index, points)
        expected = [basis.evaluate(index, p) for p in points]
        # Same direct binary64 Bernstein polynomials, with batched dot products.
        # Absolute/relative tolerances allow rounding from reduction ordering.
        np.testing.assert_allclose(actual[0], np.array([x[0] for x in expected]),
                                   atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(actual[1], np.array([x[1] for x in expected]),
                                   atol=2e-11, rtol=2e-12)
    np.testing.assert_array_equal(points, original)


def test_piece_data_is_not_stale_after_coefficient_changes():
    basis = construct_basis(1, 6)
    points = np.array([[.2, .3, .4], [.8, .7, .6]])
    value, curl = basis.evaluate_many(0, points)
    basis.coefficients.data *= 2
    after_value, after_curl = basis.evaluate_many(0, points)
    np.testing.assert_allclose(after_value, 2 * value)
    np.testing.assert_allclose(after_curl, 2 * curl)


def test_empty_points_keep_shape_and_invalid_inputs_fail():
    basis = construct_basis(1, 6)
    assert all(a.shape == (0, 3) for a in basis.evaluate_many(0, np.empty((0, 3))))
    for points in ([[np.nan, 0, 0]], [[-1, 0, 0]], [[1, 2]], [1, 2, 3]):
        with pytest.raises(ValueError):
            basis.evaluate_many(0, points)
