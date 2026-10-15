# Development and validation

Use Python 3.13 for the retained numerical/plotting dependency lock. Runtime
installation on Python 3.11 resolves its own compatible NumPy/SciPy versions.
The package has no compiled runtime extension. A C++17 compiler and SymPy
are needed only for the complete certificate replay.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install ".[validation]"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m pytest test
.venv/bin/python -m ruff check src/freudenthal test examples bench numerical_case/*.py verify_all.py
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
```

For editable development, use `pip install -e ".[validation]"`. Before a
release, separately install the **built wheel** in a fresh environment and
run the tests and examples from outside the checkout. The pytest configuration
does not prepend `src`; CI tests an installed wheel. `python -m build` also
checks that a wheel can be built from the source distribution.

The suite covers independent affine gradients and curl-constraint ranks,
two modular primes, degree-six transfer, one-cube reconstruction, manufactured
polynomials at degrees 6/7/20, both traces at internal faces, independent
finite-difference curl, exact support, batch domains and storage corruption.
Floating tolerances apply only to evaluation; integer membership residuals
must be exactly zero.

Replay the fixed certificate suite from a fresh output directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python verify_all.py --output numerical_case/results/certificates-1
```

`CXX` can select the compiler. The checker uses C++17, warnings as errors and
undefined-behavior sanitization. A Linux CI job replays the suite from the
installed package. Ordinary package use does not invoke a compiler.

New examples and plotting runs use separate output directories and manifests.
The distributed figures are selected compact outputs; complete basis arrays,
logs, virtual environments and build products remain untracked.

Policy tests exercise real default/explicit kernels, alternate-prime
independence, an analytical nonzero physical field, the installed CLI,
admission guards before allocation, deferred/failed validation state and
legacy persistence. The bounded [matched benchmark](../bench/README.md)
finishes full checks for every configuration. Run it with a fresh output:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python bench/benchmark_policy.py --repetitions 3 \
  --output numerical_case/results/policy-1
```

The implementation, API, tests, benchmarks and documentation were created
and maintained by AI agents under Kaiwen Jin's direction. Review the exact
limits and attribution in [NOTICE.md](../NOTICE.md); passing CI is not
independent human review or mathematical certification.
