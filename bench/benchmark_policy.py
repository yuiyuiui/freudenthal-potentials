"""Matched cold-process comparisons of supported construction policies.

All published production times include policy selection and preprocessing.
Deferred checks are timed separately and then included in the validated total.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

from freudenthal import __version__, construct_basis, plan_construction
from freudenthal.provenance import implementation_manifest

CASES = ((1, 6), (1, 7), (2, 7), (5, 6))
CONFIGURATIONS = ("default", "explicit", "modular-reference", "deferred-then-verify")


def options(N, degree, configuration):
    if configuration == "default":
        return {}
    if configuration == "explicit":
        plan = plan_construction(N, degree)
        return {"method": plan.method, "max_variables": plan.variables}
    if configuration == "modular-reference":
        return {"independence": "modular", "rank_prime": 1000033}
    return {"verify": False}


def worker(N, degree, configuration):
    start = time.perf_counter()
    basis = construct_basis(N, degree, **options(N, degree, configuration))
    construction = time.perf_counter() - start
    deferred = time.perf_counter()
    if configuration == "deferred-then-verify":
        assert not basis.metadata["exact_checks_passed"]
        basis.verify()
    validation = time.perf_counter() - deferred
    assert basis.metadata["exact_checks_passed"]
    matrix = basis.coefficients.copy()
    matrix.sort_indices()
    digest = hashlib.sha256()
    for array in (matrix.indptr, matrix.indices, matrix.data, basis.owners):
        digest.update(array.astype("<i8").tobytes())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "construction_seconds": construction,
        "deferred_verification_seconds": validation,
        "validated_seconds": construction + validation,
        "process_peak_rss_bytes": int(peak if sys.platform == "darwin" else 1024 * peak),
        "coefficient_and_owner_sha256": digest.hexdigest(),
        "csr_storage_bytes": sum(a.nbytes for a in (matrix.data, matrix.indices, matrix.indptr)),
        "summary": basis.summary(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--smoke", action="store_true", help="Omit the N=5 window case")
    parser.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        N, degree = map(int, args.worker[:2])
        print(json.dumps(worker(N, degree, args.worker[2])))
        return
    if args.output.exists():
        raise FileExistsError("Benchmark output must be a fresh directory")
    args.output.mkdir(parents=True)
    env = os.environ.copy()
    env.update(
        {
            key: "1"
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        }
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cases = CASES[:-1] if args.smoke else CASES
    records = []
    for N, degree in cases:
        trials = {name: [] for name in CONFIGURATIONS}
        for repetition in range(args.repetitions):
            order = CONFIGURATIONS[repetition % 4 :] + CONFIGURATIONS[: repetition % 4]
            for name in order:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output",
                    str(args.output),
                    "--worker",
                    str(N),
                    str(degree),
                    name,
                ]
                begin = time.perf_counter()
                result = subprocess.run(
                    command, env=env, capture_output=True, text=True, check=True
                )
                trial = json.loads(result.stdout)
                trial["whole_process_seconds"] = time.perf_counter() - begin
                trials[name].append(trial)
        baseline = trials["default"][0]
        for name, samples in trials.items():
            equal = all(
                sample["coefficient_and_owner_sha256"] == baseline["coefficient_and_owner_sha256"]
                for sample in samples
            )
            assert all(
                sample["summary"]["dimension"] == baseline["summary"]["dimension"]
                and sample["summary"]["exact_checks_passed"]
                for sample in samples
            )
            if name != "modular-reference":
                assert equal, "Default, pinned and deferred policies must return the same basis"
            records.append(
                {
                    "N": N,
                    "degree": degree,
                    "configuration": name,
                    "same_basis_as_default": equal,
                    "dimension": baseline["summary"]["dimension"],
                    "basis_nnz": samples[0]["summary"]["basis_nnz"],
                    "csr_storage_bytes": samples[0]["csr_storage_bytes"],
                    "effective_configuration": samples[0]["summary"]["configuration"],
                    "validated_seconds_median": statistics.median(
                        s["validated_seconds"] for s in samples
                    ),
                    "peak_process_rss_bytes_max": max(s["process_peak_rss_bytes"] for s in samples),
                    "samples": [
                        {k: v for k, v in sample.items() if k != "summary"} for sample in samples
                    ],
                }
            )
        (args.output / "partial.json").write_text(json.dumps(records, indent=2) + "\n")
    root = Path(__file__).resolve().parents[1]
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=False
    )
    report = {
        "format_version": 1,
        "software_version": __version__,
        "source_commit": git.stdout.strip() if git.returncode == 0 else None,
        "working_tree_dirty": bool(dirty.stdout.strip()),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementation": implementation_manifest(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy")},
            "thread_limits": {
                key: env[key]
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        },
        "sampling": {
            "repetitions": args.repetitions,
            "smoke": args.smoke,
            "processes_at_once": 1,
            "cache": "fresh process for each observation",
            "warmup": "none: cold first requests are measured deliberately",
            "order": "rotate configurations across repetitions",
            "randomness": "none",
        },
        "timing_scope": "validated_seconds includes input-policy selection, preprocessing, "
        "construction and full checks; whole_process_seconds also includes imports "
        "and Python startup. Peak RSS is whole-process high water including imports.",
        "comparison_contract": "All configurations finish full checks. Deferred construction alone "
        "is not compared as an equally validated result. Alternative prime "
        "may choose a different basis of the same checked space.",
        "limitations": "One host, bounded cases, no optimality, conditioning or uniform stability claim.",
        "all_cases_passed": True,
        "records": records,
    }
    (args.output / "policy.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "all_cases_passed": True,
                "case_count": len(cases),
                "configurations": len(CONFIGURATIONS),
                "repetitions": args.repetitions,
            }
        )
    )


if __name__ == "__main__":
    main()
