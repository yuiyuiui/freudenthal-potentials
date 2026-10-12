"""Independent geometry and derivative invariants for exact assembly."""

import numpy as np
import pytest

from freudenthal.geometry import BernsteinSpace, Mesh


@pytest.mark.parametrize("N", [1, 2, 3])
def test_mesh_affine_gradients_and_incidence(N):
    mesh = Mesh.build(N)
    assert len(mesh.tets) == 6 * N**3
    for tet, gradients in zip(mesh.tets, mesh.gradients):
        vertices = mesh.vertices[tet]
        affine = np.vstack([np.ones(4, dtype=np.int64), vertices.T])
        constants = np.eye(4, dtype=np.int64)[:, 0] - gradients @ vertices[0]
        inverse = np.column_stack([constants, gradients])
        assert np.array_equal(inverse @ affine, np.eye(4, dtype=np.int64))
        assert abs(round(np.linalg.det((vertices[1:] - vertices[0]).T))) == 1
    assert all(len(incident) in (1, 2) for incident in mesh.faces.values())


def test_all_curl_components_and_omitted_normal_identity():
    space = BernsteinSpace.build(Mesh.build(2), 3)
    A, metadata = space.curl_constraints(all_components=True)
    for i in range(0, len(metadata), 3):
        face, _, _ = metadata[i]
        t, opposite = space.mesh.faces[face][0]
        normal = space.mesh.gradients[t, opposite]
        row = sum(int(normal[j]) * A.getrow(i + j) for j in range(3))
        row.eliminate_zeros()
        assert row.nnz == 0
    # Independently prescribed global affine vector field has continuous curl.
    c = space.nodes[:, [2, 0, 1]].ravel()
    assert np.array_equal(A @ c, np.zeros(A.shape[0], dtype=np.int64))


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_mesh_parameter(value):
    with pytest.raises(ValueError):
        Mesh.build(value)
