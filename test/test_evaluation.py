"""Evaluation is tested against an independent global polynomial field."""

import numpy as np
import pytest
from scipy import sparse

from freudenthal.basis import PotentialBasis
from freudenthal.geometry import BernsteinSpace, Mesh


def make_polynomial_field(degree):
    space = BernsteinSpace.build(Mesh.build(2), degree)
    values = np.zeros((space.nscalar, 3))
    for t, tet in enumerate(space.mesh.tets):
        vertices = space.mesh.vertices[tet] / 2
        for alpha, node in zip(space.alpha, space.local_to_global[t]):
            # Bernstein coefficients of x_j^2, obtained from falling factorials.
            sums = alpha @ vertices
            square = (sums**2 - alpha @ (vertices**2)) / (degree * (degree - 1))
            values[node] = square[[2, 0, 1]]
    return PotentialBasis(space, sparse.csr_matrix(values.ravel()[None, :]), np.array([0]))


@pytest.fixture(scope="module")
def polynomial_field():
    return make_polynomial_field(6)


@pytest.mark.parametrize("degree", [7, 20])
def test_evaluation_at_higher_degrees(degree):
    field = make_polynomial_field(degree)
    value, curl = field.evaluate(0, [0.31, 0.22, 0.18])
    np.testing.assert_allclose(value, [0.18**2, 0.31**2, 0.22**2], rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(curl, [0.44, 0.36, 0.62], rtol=2e-13, atol=2e-14)


@pytest.mark.parametrize(
    "point", [[0, 0, 0], [1, 1, 1], [0.31, 0.22, 0.18], [0.5, 0.5, 0.5], [0.25, 0.75, 1]]
)
def test_physical_polynomial_and_curl(polynomial_field, point):
    x, y, z = point
    value, curl = polynomial_field.evaluate(0, point)
    np.testing.assert_allclose(value, [z * z, x * x, y * y], rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(curl, [2 * y, 2 * z, 2 * x], rtol=2e-13, atol=2e-14)


def test_both_sides_of_every_internal_face(polynomial_field):
    mesh = polynomial_field.space.mesh
    for face, incidents in mesh.faces.items():
        if len(incidents) != 2:
            continue
        outputs = []
        for t, _ in incidents:
            lam = np.array([1 / 3 if int(v) in face else 0 for v in mesh.tets[t]])
            outputs.append(polynomial_field.evaluate_on_tetrahedron(0, t, lam))
        np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("point", [[-0.1, 0, 0], [1.1, 0, 0], [np.nan, 0, 0], [0, 0]])
def test_invalid_evaluation_point(polynomial_field, point):
    with pytest.raises(ValueError):
        polynomial_field.evaluate(0, point)
