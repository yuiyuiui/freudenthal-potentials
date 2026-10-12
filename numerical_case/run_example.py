"""Compute one fixed local-basis geometry example into a fresh directory."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np

from freudenthal import __version__, construct_basis
from freudenthal.exact import integer_product
from freudenthal.provenance import implementation_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resolution", type=int, default=17, help="Slice samples per axis, 5..33; default 17"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            "Choose a fresh output directory; existing runs are never overwritten"
        )
    if not 5 <= args.resolution <= 33:
        raise ValueError("resolution must lie in [5,33]")
    args.output.mkdir(parents=True)
    basis = construct_basis(N=2, degree=7)
    mesh = basis.space.mesh
    owner = int(np.flatnonzero(np.all(mesh.vertices == [1, 1, 1], axis=1))[0])
    candidates = np.flatnonzero(basis.owners == owner)
    curl_map = mesh.N * basis.space.curl_map()
    curls = integer_product(basis.coefficients[candidates], curl_map.T)
    curls.eliminate_zeros()
    n_beta = len(basis.space.alpha) * basis.space.d // (basis.space.d + 3)
    # Pick the widest nonzero-curl support at the interior owner, breaking ties by ID.
    counts = [
        len(np.unique(curls.getrow(i).indices // (3 * n_beta))) for i in range(len(candidates))
    ]
    position = int(np.argmax(counts))
    if counts[position] == 0:
        raise ValueError("The fixed example has no nonzero-curl candidate")
    row_id = int(candidates[position])
    curl_row = curls.getrow(position)
    peak_col = int(curl_row.indices[np.argmax(np.abs(curl_row.data))])
    peak_tet = peak_col // (3 * n_beta)
    slice_z = float(np.mean(mesh.vertices[mesh.tets[peak_tet], 2]) / mesh.N)
    coordinate = np.linspace(0, 1, args.resolution)
    xx, yy = np.meshgrid(coordinate, coordinate)
    points = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, slice_z)])
    potential, curl = basis.evaluate_many(row_id, points)
    if not np.all(np.isfinite(potential)) or not np.all(np.isfinite(curl)):
        raise ValueError("Nonfinite slice evaluation")
    if np.max(np.linalg.norm(curl, axis=1)) <= 0:
        raise ValueError("Selected slice missed the nonzero curl")
    support = basis.support_tetrahedra(row_id)
    assert set(support) <= mesh.stars[owner]
    outside = sorted(set(range(len(mesh.tets))) - set(support))
    outside_centers = np.mean(mesh.vertices[mesh.tets[outside]], axis=1) / mesh.N
    outside_value, outside_curl = basis.evaluate_many(row_id, outside_centers)
    assert np.count_nonzero(outside_value) == np.count_nonzero(outside_curl) == 0
    geometry = {
        "N": 2,
        "potential_degree": 7,
        "basis_id": row_id,
        "owner_vertex": owner,
        "vertices": (mesh.vertices / mesh.N).tolist(),
        "tetrahedra": mesh.tets.tolist(),
        "support_tetrahedra": support.tolist(),
        "owner_star_tetrahedra": sorted(mesh.stars[owner]),
        "slice_z": slice_z,
        "resolution": args.resolution,
    }
    summary = {
        **basis.summary(),
        "selected_basis_id": row_id,
        "owner_vertex": owner,
        "owner_physical_point": (mesh.vertices[owner] / mesh.N).tolist(),
        "support_tetrahedra_count": len(support),
        "owner_star_tetrahedra_count": len(mesh.stars[owner]),
        "curl_support_tetrahedra_count": counts[position],
        "slice_z": slice_z,
        "resolution": args.resolution,
        "slice_max_potential_magnitude": float(np.max(np.linalg.norm(potential, axis=1))),
        "slice_max_curl_magnitude": float(np.max(np.linalg.norm(curl, axis=1))),
        "outside_support_centroid_nonzero_values": 0,
        "evidence_scope": "one finite basis instance and binary64 slice evaluation",
    }
    (args.output / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(args.output / "slice.npz", points=points, potential=potential, curl=curl)
    basis.save(args.output / "basis")
    files = {
        file.relative_to(args.output).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(args.output.rglob("*"))
        if file.is_file()
    }
    manifest = {
        "format_version": 1,
        "parameters": {"N": 2, "degree": 7, "resolution": args.resolution},
        "selection": "widest nonzero-curl tetrahedral support at center owner; lowest-ID tie",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_version": __version__,
        "implementation": implementation_manifest(),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy"]},
        "thread_limits": {
            name: os.environ.get(name)
            for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]
        },
        "randomness": "none",
        "files": files,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
