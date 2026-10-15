"""Installed-package policy checks against actual kernels and physical values."""

import json
import subprocess
import sys

import numpy as np
import pytest

from freudenthal import BernsteinSpace, Mesh, PotentialBasis, construct_basis, plan_construction
from freudenthal.algebra import sparse_mod_rank


@pytest.fixture(scope="module")
def small():
    return construct_basis(1, 6)


def test_default_and_explicit_modular_reference(small):
    other = construct_basis(
        1, 6, method="stored", independence="modular", rank_prime=1000033, max_variables=1029
    )
    assert (small.coefficients - other.coefficients).nnz == 0
    A, _ = other.space.curl_constraints(all_components=True)
    assert sparse_mod_rank(A, 1000003).rank + len(other) == A.shape[1] == 1029
    assert other.configuration["verification"]["rank_prime"] == 1000033
    assert len(other) == 795


def test_parametric_identity_and_forced_modular_have_same_physical_output():
    default = construct_basis(2, 7)
    override = construct_basis(2, 7, independence="modular", rank_prime=1000033)
    assert default.configuration["verification"] == {
        "status": "passed",
        "independence": "identity",
        "rank_prime": None,
    }
    assert override.configuration["verification"]["rank_prime"] == 1000033
    assert (default.coefficients - override.coefficients).nnz == 0
    value, curl = default.evaluate(1, [0.5, 0.5, 0.875])
    # Frozen v0.1.0 reference at a nonzero point; analytic/finite-difference
    # evaluations are tested independently in test_evaluation/test_public_api.
    np.testing.assert_array_equal(value, [0.01153564453125, 0.0013427734375, 0])
    np.testing.assert_array_equal(curl, [0.0615234375, -0.39990234375, -0.00341796875])


def test_one_cube_alternate_prime_is_a_complete_kernel():
    basis = construct_basis(1, 7, method="one-cube", rank_prime=1000033)
    A, _ = basis.space.curl_constraints(all_components=True)
    assert sparse_mod_rank(A, 1000003).rank + len(basis) == A.shape[1]
    assert basis.metadata["exact_checks_passed"] and np.all(basis.owners == 0)


def test_deferred_verification_can_be_completed():
    basis = construct_basis(1, 6, verify=False)
    assert not basis.metadata["exact_checks_passed"]
    assert basis.configuration["verification"]["status"] == "deferred"
    assert "independent_rows" not in basis.metadata
    assert basis.verify(rank_prime=1000033)["independent_rows"] == 795
    assert basis.configuration["verification"]["rank_prime"] == 1000033


def test_failed_reverification_clears_previous_success():
    basis = construct_basis(1, 6)
    basis.owners[0] = len(basis.space.mesh.vertices)
    with pytest.raises(ValueError, match="owner"):
        basis.verify()
    assert basis.metadata["exact_checks_passed"] is False
    assert "independent_rows" not in basis.metadata
    assert basis.configuration["verification"]["status"] == "failed"


def test_deferral_does_not_skip_mandatory_reconstruction_check(monkeypatch):
    import freudenthal.basis as implementation

    A, _ = BernsteinSpace.build(Mesh.build(1), 7).curl_constraints()
    constrained_column = int(A.indices[0])
    original = implementation.lift_rows

    def corrupted_lift(*args):
        rows, report = original(*args)
        rows[0][constrained_column] = rows[0].get(constrained_column, 0) + 1
        return rows, report

    monkeypatch.setattr(implementation, "lift_rows", corrupted_lift)
    with pytest.raises(ValueError, match="rational lift"):
        construct_basis(1, 7, verify=False)


def test_plan_and_guard_do_not_allocate_or_change_problem(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Mesh allocation must not happen")

    monkeypatch.setattr(Mesh, "build", forbidden)
    plan = plan_construction(1_000_000, 7)
    assert plan.variables == 3 * (7_000_001) ** 3
    assert plan.N == 1_000_000 and plan.degree == 7
    with pytest.raises(MemoryError, match="not changed"):
        construct_basis(1_000_000, 7, max_variables=1000)


@pytest.mark.parametrize(
    "N,d,options,error",
    [
        (1, 6, {"verify": "false"}, TypeError),
        (1, 6, {"verify": 0}, TypeError),
        (1, 6, {"rank_prime": True}, TypeError),
        (1, 6, {"rank_prime": 1000003.0}, TypeError),
        (1, 6, {"rank_prime": 1000000}, ValueError),
        (1, 6, {"method": "parametric"}, ValueError),
        (2, 7, {"method": "stored"}, ValueError),
        (1, 7, {"method": "window"}, ValueError),
        (1, 6, {"method": "cuda"}, ValueError),
        (1, 6, {"independence": "identity"}, ValueError),
        (1, 6, {"independence": "floating"}, ValueError),
        (1, 6, {"max_variables": False}, TypeError),
        (1, 6, {"max_variables": 0}, ValueError),
        (1, 6, {"max_variables": 1028}, MemoryError),
    ],
)
def test_invalid_and_incompatible_settings(N, d, options, error):
    with pytest.raises(error):
        construct_basis(N, d, **options)


def test_configuration_snapshot_is_detached(small):
    config = small.configuration
    config["construction"]["method"] = "invalid"
    assert small.configuration["construction"]["method"] == "stored"
    assert plan_construction(1, 6).to_dict() == plan_construction(np.int64(1), 6).to_dict()


def test_load_reuses_policy_and_never_trusts_saved_success(small, tmp_path):
    small.save(tmp_path)
    deferred = PotentialBasis.load(tmp_path, verify=False, max_variables=1029)
    assert deferred.metadata["exact_checks_passed"] is False
    assert "independent_rows" not in deferred.metadata
    assert deferred.configuration["load"]["max_variables"] == 1029
    assert deferred.verify(rank_prime=1000033)["independent_rows"] == 795
    summary = json.loads((tmp_path / "summary.json").read_text())
    summary.pop("configuration")  # v0.1.0 persistence compatibility
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert PotentialBasis.load(tmp_path).metadata["exact_checks_passed"]
    with pytest.raises(MemoryError):
        PotentialBasis.load(tmp_path, max_variables=1028)
    with pytest.raises(TypeError):
        PotentialBasis.load(tmp_path, verify="false")


def test_noninteger_saved_problem_input_is_not_silently_rounded(small, tmp_path):
    small.save(tmp_path)
    with np.load(tmp_path / "metadata.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    arrays["N"] = np.array(1.75)
    np.savez_compressed(tmp_path / "metadata.npz", **arrays)
    with pytest.raises(TypeError, match="N must be an integer"):
        PotentialBasis.load(tmp_path)


def test_installed_cli_uses_same_plan_and_validation_policy():
    result = subprocess.run(
        [sys.executable, "-m", "freudenthal", "2", "7", "--plan"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == plan_construction(2, 7).to_dict()
    result = subprocess.run(
        [sys.executable, "-m", "freudenthal", "1", "6", "--rank-prime", "1000033"],
        check=True,
        capture_output=True,
        text=True,
    )
    output = json.loads(result.stdout)
    assert output["dimension"] == 795 and output["exact_checks_passed"]
    assert output["configuration"]["verification"]["rank_prime"] == 1000033
    result = subprocess.run(
        [sys.executable, "-m", "freudenthal", "1", "6", "--max-variables", "1"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "not changed" in result.stderr


def test_low_level_public_parameters_are_checked():
    space = BernsteinSpace.build(Mesh.build(1), 2)
    np.testing.assert_array_equal(space.allowed_nodes(set(range(6))), np.arange(space.nscalar))
    assert len(space.vertex_nodes(0)) > 0
    for invalid in (True, 0.5):
        with pytest.raises(TypeError):
            space.vertex_nodes(invalid)
    with pytest.raises(IndexError):
        space.allowed_nodes({6})
    with pytest.raises(TypeError):
        space.allowed_nodes([0])
    with pytest.raises(TypeError, match="all_components"):
        space.curl_constraints(all_components="false")
    with pytest.raises(TypeError, match="mesh"):
        BernsteinSpace.build(None, 2)
