"""A small complete installed-package example; optional non-overwriting save."""

import argparse
import json

import numpy as np

from freudenthal import PotentialBasis, construct_basis, dimension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", help="Optional fresh output directory")
    args = parser.parse_args()
    basis = construct_basis(N=2, degree=7)
    assert len(basis) == dimension(2, 7) == 6792
    assert basis.coefficients.nnz == 35712
    report = basis.verify()
    points = np.array([[0.5, 0.5, 0.875], [0.1, 0.1, 0.9]])
    value, curl = basis.evaluate_many(1, points)
    assert value.shape == curl.shape == (2, 3)
    assert np.linalg.norm(value[0]) > 0 and np.linalg.norm(curl[0]) > 0
    assert np.array_equal(value[1], np.zeros(3)) and np.array_equal(curl[1], np.zeros(3))
    assert set(basis.support_tetrahedra(1)) <= basis.space.mesh.stars[basis.owners[1]]
    if args.save:
        basis.save(args.save)
        restored = PotentialBasis.load(args.save)
        assert (restored.coefficients - basis.coefficients).nnz == 0
    print(
        json.dumps(
            {
                "dimension": len(basis),
                "scalar_nonzeros": basis.coefficients.nnz,
                "validation": report,
                "potential": value.tolist(),
                "curl": curl.tolist(),
            }
        )
    )


if __name__ == "__main__":
    main()
