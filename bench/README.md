# Paired core benchmark

Baseline `8e16007c3439875dadfdeb65fc458932b255fc95` and candidate source hashes are recorded in
[raw results](results/core_v0.2.1.json), including all samples, validation,
settings, dependency versions and profiles. macOS arm64, CPython 3.13.3, one
worker and one BLAS/OpenMP thread. 7 warm repetitions per case.

| Case | Baseline median s | Candidate median s | Ratio | Python peak B (old/new) | Process RSS B (old/new) |
|---|---:|---:|---:|---:|---:|
| small | 0.00117479 | 0.000425375 | 2.76x | 39232 / 55632 | 64782336 / 64700416 |
| medium | 0.0192355 | 0.00116329 | 16.54x | 43268 / 135160 | 64520192 / 65159168 |
| large | 0.0772793 | 0.00384658 | 20.09x | 80116 / 194976 | 64684032 / 65454080 |

Profiles identified per-point CSR extraction and polynomial evaluation overhead. Bounded point batches group by tetrahedron, reuse at most eight local coefficient pieces per call and contract Bernstein weights. Scalar evaluation is retained as an independent reference; input arrays are unchanged and no cache survives the call. Dot-product reassociation can change roundoff; value/curl tolerances are explicit in tests.

Setup/import, first call, warm calls, separate allocation and separate profiling
passes are distinguished. Validation is outside timed regions; process RSS
includes imports, input and validation. Python tracemalloc excludes untraced
native allocations. These finite cases do not prove an asymptotic or universal speed/memory gain.
Timing noise and hardware matter; small RSS changes are not interpreted as gains.
The remaining arithmetic/BLAS kernels dominate larger workloads.

```sh
git worktree add --detach ../freudenthal-potentials-baseline 8e16007c3439875dadfdeb65fc458932b255fc95
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bench/benchmark_core.py \
  --repeats 7 --baseline-source ../freudenthal-potentials-baseline/src \
  --output /tmp/freudenthal-potentials-core.json
```

The output path must be new. Install the package's development dependencies first;
SVD also needs Boost >=1.89 and C++17. Each variant imports the specified source
in a fresh worker. Performance thresholds are not enforced on shared CI.

Earlier policy benchmarks: [original guide](README.previous.md).
