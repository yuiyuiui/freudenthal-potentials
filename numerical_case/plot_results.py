"""Render computed geometry and fixed sparse diagnostics; never overwrite inputs."""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_run(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != digest:
            raise ValueError(f"Run input changed or escaped its directory: {name}")
    return manifest


def boundary_triangles(tets, ids):
    counts = {}
    for index in ids:
        for face in combinations(tets[index], 3):
            key = tuple(sorted(map(int, face)))
            counts[key] = counts.get(key, 0) + 1
    return [face for face, count in sorted(counts.items()) if count == 1]


def save(fig, output, name):
    fig.savefig(output / (name + ".svg"), metadata={"Date": None})
    fig.savefig(output / (name + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    svg = output / (name + ".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            "Choose a fresh figure directory; existing figures are never overwritten"
        )
    run = validate_run(args.input)
    provenance = json.loads((args.reference / "PROVENANCE.json").read_text())
    for name, digest in provenance["files"].items():
        if sha256(args.reference / name) != digest:
            raise ValueError(f"Reference input changed: {name}")
    geometry = json.loads((args.input / "geometry.json").read_text())
    summary = json.loads((args.input / "summary.json").read_text())
    histograms = json.loads((args.reference / "sparsity_histograms.json").read_text())
    cases = json.loads((args.reference / "paper_cases.json").read_text())["cases"]
    vertices = np.asarray(geometry["vertices"])
    tets = np.asarray(geometry["tetrahedra"])
    args.output.mkdir(parents=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "svg.hashsalt": "freudenthal-potentials-v0.1.0",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(7.8, 5.1), layout="constrained")
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    edges = sorted({tuple(sorted(map(int, edge))) for tet in tets for edge in combinations(tet, 2)})
    ax.add_collection3d(
        Line3DCollection(
            vertices[np.asarray(edges)], colors="#b6bfc8", linewidths=0.45, alpha=0.4, zorder=1
        )
    )
    star = boundary_triangles(tets, geometry["owner_star_tetrahedra"])
    active = boundary_triangles(tets, geometry["support_tetrahedra"])
    ax.add_collection3d(
        Poly3DCollection(
            vertices[np.asarray(star)],
            facecolor="#b3d5ed",
            edgecolor="#6a9eba",
            linewidths=0.5,
            alpha=0.10,
            zorder=2,
        )
    )
    ax.add_collection3d(
        Poly3DCollection(
            vertices[np.asarray(active)],
            facecolor="#287cb1",
            edgecolor="#164f78",
            linewidths=0.9,
            alpha=0.60,
            zorder=3,
        )
    )
    owner = vertices[geometry["owner_vertex"]]
    # Keep the owner marker visible as an annotation through the star surface.
    ax.scatter(*owner, c="#d89125", s=85, edgecolor="white", depthshade=False, zorder=4)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        zlim=(0, 1),
        xlabel="x",
        ylabel="y",
        zlabel="z",
        title=f"Exact support inside one vertex star · N=2, d=7\nBasis row {geometry['basis_id']}",
    )
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=23, azim=-52)
    ax.set_xticks([0, 0.5, 1], labels=["0", "0.5", "1"])
    ax.set_yticks([0, 0.5, 1], labels=["0", "0.5", "1"])
    ax.set_zticks([0, 0.5, 1], labels=["0", "0.5", "1"])
    fig.legend(
        handles=[
            Patch(facecolor="#287cb1", label="Nonzero support"),
            Patch(facecolor="#b3d5ed", label="Owner star"),
            Line2D([], [], marker="o", linestyle="none", color="#d89125", label="Owner vertex"),
        ],
        loc="outside lower center",
        ncols=3,
        frameon=False,
    )
    save(fig, args.output, "vertex_star")
    with np.load(args.input / "slice.npz", allow_pickle=False) as arrays:
        points, value, curl = arrays["points"], arrays["potential"], arrays["curl"]
    resolution = geometry["resolution"]
    xx, yy = (points[:, i].reshape(resolution, resolution) for i in [0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), layout="constrained")
    for ax, field, cmap, title in zip(
        axes,
        [value, curl],
        ["viridis", "magma"],
        ["Potential magnitude |w|", "Physical curl magnitude |curl w|"],
    ):
        magnitude = np.linalg.norm(field, axis=1).reshape(resolution, resolution)
        graphic = ax.pcolormesh(xx, yy, magnitude, cmap=cmap, shading="nearest", rasterized=False)
        ax.set(xlabel="x", ylabel="y", title=title, aspect="equal", xlim=(0, 1), ylim=(0, 1))
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        fig.colorbar(graphic, ax=ax, shrink=0.81, pad=0.03)
    fig.suptitle(
        f"Same local basis row {geometry['basis_id']} · physical slice z={geometry['slice_z']:.3f}"
    )
    save(fig, args.output, "potential_curl")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), layout="constrained")
    metrics = {}
    for d, color in [(6, "#255f95"), (7, "#bd5d32")]:
        record = histograms[f"N4_d{d}"]
        counts = np.asarray(record["counts"], dtype=np.int64)
        assert counts.sum() == record["dimension"] and counts[0] == 0
        budget = np.arange(len(counts))
        axes[0].step(
            budget,
            np.cumsum(counts) / counts.sum(),
            where="post",
            color=color,
            lw=1.8,
            label=f"d={d}",
        )
        metrics[str(d)] = {
            "mean_scalar_nonzeros": float(np.dot(counts, budget) / counts.sum()),
            "max_scalar_nonzeros": len(counts) - 1,
            "rows": int(counts.sum()),
        }
    axes[0].set(
        xlim=(0, 70),
        ylim=(0, 1.02),
        xlabel="Nonzero scalar Bernstein entries per row",
        ylabel="Fraction of rows with at most this count",
        title="Complete sparsity distributions · N=4",
    )
    axes[0].legend(frameon=False)
    critical = [case for case in cases if case["potential_degree"] == 6]
    assert all(case["exact_checks_passed"] for case in critical)
    grids = [case["N"] for case in critical]
    axes[1].semilogy(
        grids, [case["dimension"] for case in critical], "o-", color="#255f95", label="Basis rows"
    )
    axes[1].semilogy(
        grids,
        [case["basis_nnz"] for case in critical],
        "s-",
        color="#bd5d32",
        label="Scalar nonzeros",
    )
    axes[1].set(
        xlabel="Cubes per axis N",
        ylabel="Count (log scale)",
        title="Recorded construction/storage counts · d=6",
        xticks=grids,
    )
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.18)
    save(fig, args.output, "basis_structure")
    files = {file.name: sha256(file) for file in sorted(args.output.iterdir()) if file.is_file()}
    manifest = {
        "source_sha256": sha256(Path(__file__)),
        "run_manifest_sha256": sha256(args.input / "manifest.json"),
        "reference_provenance_sha256": sha256(args.reference / "PROVENANCE.json"),
        "basis_summary": summary,
        "histogram_metrics": metrics,
        "figure_files": files,
        "mathematical_scope": "Exact support and finite construction diagnostics; no stability or speed claim",
        "run_parameters": run["parameters"],
    }
    (args.output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"figures": sorted(files), "histogram_metrics": metrics}))


if __name__ == "__main__":
    main()
