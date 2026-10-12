"""Replay every finite certificate used by the all-N, all-degree theorem.

Extracted data, compiler products, logs and results stay in the chosen
output directory. All required fixed inputs ship with the installed package.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZipFile

from freudenthal.data import CERTIFICATES, verify_data_integrity
from freudenthal.provenance import implementation_manifest
from freudenthal.verification import verify_reference_assembly

ROOT = Path(__file__).resolve().parent


def main():
    if not __debug__:
        raise RuntimeError("Certificate replay requires assertions: do not use Python -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "numerical_case/results/certificate_replay"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory; refusing to overwrite {output}")
    output.mkdir(parents=True)
    start = time.perf_counter()
    verify_data_integrity()
    code = CERTIFICATES.parent / "_certificate_check"
    shutil.copytree(code / "all_degree", output / "all_degree")
    shutil.copytree(code / "critical", output / "critical")
    with ZipFile(CERTIFICATES / "verification_data.zip") as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        for name, record in manifest.items():
            blob = archive.read(name)
            path = output / name
            if not path.resolve().is_relative_to(output):
                raise ValueError("Unsafe certificate path")
            if hashlib.sha256(blob).hexdigest() != record["sha256"]:
                raise ValueError(f"Certificate member failed integrity: {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
    env = os.environ.copy()
    env.pop("PYTHONOPTIMIZE", None)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    commands = []

    def run(command, cwd, *, capture=False):
        commands.append({"argv": list(map(str, command)), "cwd": str(cwd)})
        print("RUN:", shlex.join(list(map(str, command))), flush=True)
        result = subprocess.run(
            list(map(str, command)),
            cwd=cwd,
            env=env,
            check=True,
            capture_output=capture,
            text=capture,
        )
        if capture:
            print(result.stdout, end="", flush=True)
            print(result.stderr, end="", file=sys.stderr, flush=True)
        return result.stdout

    high = output / "all_degree"
    low = output / "critical"
    original_owners = (high / "results/universal_owners.txt").read_bytes()
    run([sys.executable, "src/verify_symbolic_sources.py"], high)
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    compiler_version = run([*compiler, "--version"], high, capture=True).strip()
    executable = output / "exact_parameter_check"
    run(
        [
            *compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            "src/exact_parameter_check.cpp",
            "-o",
            executable,
        ],
        high,
    )
    run([executable, high / "results", "all", "5"], high)
    if (high / "results/universal_owners.txt").read_bytes() != original_owners:
        raise ValueError("Recomputed universal owner masks differ from the distributed table")
    run([sys.executable, "src/dimension_all_degrees.py"], high)
    critical_stdout = run([sys.executable, "verify_all.py"], low, capture=True)
    summaries = {"independent_integer_assembly": verify_reference_assembly()}
    for name in [
        "universal_sources_verified",
        "pivot_and_support_verified",
        "reverse_identity_verified",
        "all_degree_dimension",
    ]:
        report = json.loads((high / "results" / (name + ".json")).read_text())
        if not report.get("passed"):
            raise ValueError(f"Missing certificate pass: {name}")
        summaries[name] = report
    summaries["critical_dimension"] = json.loads(
        (low / "results/dimension_polynomial.json").read_text()
    )
    summaries["small_grid_certificates"] = {}
    for line in critical_stdout.splitlines():
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "case" in report:
            if not report.get("passed"):
                raise ValueError(f"Small-grid certificate failed: {report['case']}")
            summaries["small_grid_certificates"][report["case"]] = report
    if len(summaries["small_grid_certificates"]) != 4:
        raise ValueError("Missing small-grid verification reports")
    for name in [
        "uniform_normalform_verified",
        "independent_repairs_verified",
        "transport_N5_verified",
        "transport_N7_verified",
    ]:
        report = json.loads((low / "results" / (name + ".json")).read_text())
        if not report.get("passed"):
            raise ValueError(f"Missing critical-degree pass: {name}")
        summaries[name] = report
    result = {
        "certificate_suite_passed": True,
        "scope": "N>=1, k>=5; analytical coverage lemmas also required",
        "formal_verification": False,
        "seconds": time.perf_counter() - start,
        "commands": commands,
        "compiler": {"argv": compiler, "version": compiler_version},
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "sympy")},
        "implementation": implementation_manifest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summaries": summaries,
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    public = {
        "certificate_suite_passed": True,
        "scope": result["scope"],
        "versions": result["versions"],
        "summaries": summaries,
    }
    # The compact record also belongs to this execution, never to a shared summary path.
    (output / "compact_summary.json").write_text(json.dumps(public, indent=2) + "\n")
    print(json.dumps({"certificate_suite_passed": True, "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
