# Matched policy benchmarks

These are finite software measurements, not stability or optimality claims.
Use the installed package on Linux or macOS. The driver records dependency
versions, implementation and fixed-witness hashes, source commit/dirty state,
effective configurations, selection reasons, timings, storage, peak process
RSS and exact validation results. No random input or seed is used.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python bench/benchmark_policy.py \
    --output numerical_case/results/policy-1 --repetitions 3
```

The output must be fresh. Selected results are retained in
[`results/policy_v0.2.0.json`](results/policy_v0.2.0.json); large generated
arrays and disposable execution outputs stay in ignored `numerical_case/results/`.
All existing mathematical figures and their reproduction commands remain in
[`numerical_case/`](../numerical_case/README.md).

The four fixed problems exercise every automatic construction branch:
`(N,d)=(1,6),(1,7),(2,7),(5,6)`. Each is run with:

1. Only the problem inputs, using the default policy.
2. An explicit compatible branch and a matching ambient-variable admission cap.
3. Modular verification at alternate supported prime `1000033`.
4. Deferred optional verification, followed by an explicit full check.

All configurations finish the same membership, locality and finite
independence requirements. Default, pinned and deferred configurations must
have identical coefficient/owner hashes. An alternate modulus can choose a
different valid basis; matching dimensions and successful full checks are
required and exact basis equality is reported separately. The unit suite also
checks independent constraint ranks and analytical physical fields.

Each observation uses a fresh child process, sequentially, with one native
thread. Constructor timings include parameter selection, preprocessing,
certificate/rule loading and checks. Deferred verification time is included
in the validated total; its constructor-only time is not advertised as an
equally validated speedup. A separate whole-process time includes imports and
Python startup. Peak RSS is process high water, including the interpreter and
imports, not incremental workspace or a RAM bound. CSR storage counts data,
indices and row pointers, excluding owners and Python objects.

No warm-up is discarded: the benchmark deliberately measures cold first
requests. Configuration order rotates across repetitions. The default three
samples per configuration are a bounded comparison on one host; scheduling,
cache and allocator variation still affect timing. These results do not show
that any policy is universally fastest. Repeating costs scales with the
number of repetitions. Defaults for correctness do not depend on these times.

The bounded CI command omits the window case and uses one observation per
configuration:

```bash
python bench/benchmark_policy.py --smoke --repetitions 1 \
  --output numerical_case/results/policy-smoke-1
```

`--repetitions` supports integers 1..5. `--smoke` changes the benchmark's case
set, not an API problem or verification guarantee. Resource admission for the
full benchmark should account for the `N=6` fixed reference used by window
transfer; API `max_variables` is not a whole-process memory cap.
