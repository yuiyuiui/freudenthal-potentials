# Parameter contract and selection advice

This contract describes the installed v0.2 API. Indices are zero-based; vector
components are `(x,y,z)`. The sole computational backend is CPU NumPy/SciPy
sparse arithmetic. Runtime construction needs no symbolic package or compiler.
There is no selectable GPU, floating rank tolerance or arbitrary-precision mode.
An unavailable/unsupported requested method fails; no fallback changes the problem.

## Problem inputs and shared construction policy

```python
construct_basis(N, degree, *, verify=True, method="auto",
                independence="auto", rank_prime=1000003, max_variables=None)
plan_construction(N, degree, *, verify=True, method="auto",
                  independence="auto", rank_prime=1000003, max_variables=None)
```

`plan_construction` validates and resolves the same policy without building a
mesh or matrix. Its `ConstructionPlan.to_dict()` returns detached JSON data:
required inputs, requested and selected methods/checks, selected prime and
whether it is used, exact ambient/tetrahedral/local-coefficient counts, the
resource guard and selection reasons. These counts are not peak-memory bounds.
Use the factory rather than manually instantiating the result dataclass.

| Parameter | Type, units, range and role | Default, rule and recommendation | Accuracy, stability and cost |
|---|---|---|---|
| `N` | Required Python/NumPy integer, excluding booleans, `N>=1`; cubes per axis of the unit cube | No default. Set from the scientific mesh requirement; never automatically changed | Defines the problem, not an error tolerance. `6*N**3` tetrahedra; work/storage grow strongly with `N` |
| `degree` | Required integer excluding booleans, `degree>=6`; potential polynomial degree `d=k+1` | No default. Set the requested approximation space; never replaced by an easier degree | Defines the problem. Ambient variables `3*(N*d+1)**3`, local coefficients `binomial(d+3,3)`; high-degree binary64 evaluation can lose accuracy or overflow |
| `verify` | `bool` or NumPy boolean; optional full finite verification | `True`. Recommended for ordinary use. `False` is advanced deferral only | Full checks establish concrete curl membership, owner support, independence and the applicable count. Deferral saves current work but reports `exact_checks_passed=False`; mandatory reconstruction/span checks still execute |
| `method` | String: `auto`, `stored`, `window`, `one-cube`, `parametric` | `auto`, using the table below. Pin the returned method for reproducibility | Branches have different validity ranges; they are not interchangeable accuracy modes. No search, tuning or fallback |
| `independence` | String: `auto`, `identity`, `modular` | `auto`: exact canonical free-coordinate identity for the parametric branch, modular rank otherwise. Use `modular` for an extra modular reference check | Both are exact finite independence checks. Explicit modular adds work and still checks any canonical identity. Explicit identity is rejected if that structure is unavailable |
| `rank_prime` | Integer excluding booleans; exactly `1000003` or `1000033` | `1000003` preserves v0.1.0. Use `1000033` with modular checking for a separate arithmetic cross-check | Used in window generator selection, one-cube elimination/reconstruction and modular verification; unused by pure identity verification. It is not a precision parameter. A bad reconstruction/rank fails explicitly; no retry at another prime |
| `max_variables` | `None` or positive integer excluding booleans; admission cap on ambient coefficient count | `None`, preserving the previous domain without guessing RAM. Set a cap appropriate to a workload you have measured | An explicit cap is enforced before mesh allocation. It never truncates output or reduces `N,d`. It is **not** a byte limit: sparse fill, Python objects, temporary constraints and the fixed `N=6` window reference also consume memory |

| Inputs | Auto method and reason |
|---|---|
| `d=6, 1<=N<=4` | `stored`: the shipped small-grid integer bases cover these grids |
| `d=6, N>=5` | `window`: transfer the normal form, repair local generators, select an independent subset |
| `N=1, d>6` | `one-cube`: checked modular elimination and rational reconstruction; reconstruction can fail |
| `N>=2, d>=7` | `parametric`: fixed coefficient-type rules and canonical free-coordinate identity |

These ranges are disjoint. For example, requesting `method="stored"` for
`N=2,d=7` raises `ValueError` before allocation. There is no norm-stability,
condition-number, PDE-error or all-input feasibility guarantee attached to a
default. Signed-64-bit products retain conservative overflow guards.

`basis.configuration` returns a detached dictionary with `construction` and
`verification` records. The former is the resolved policy at construction;
the latter reports the **actual last check**, including `status`, chosen
independence check and prime (`None` when no modular verification was used).
`basis.summary()` includes this information and storage counts. A deferred
result has no stale successful check report. Failed revalidation clears a
previous success before raising. Arrays remain mutable: a past report cannot
certify later in-place edits; call `verify()` again after changing raw data.

## Verification, persistence and dimension

| Callable parameter | Type/range/default and advice | Consequence |
|---|---|---|
| `basis.verify(independence=None)` | `None` reuses the construction policy, or `auto` for legacy/manual objects; explicit choices are those above | Always performs full membership/support checks. Identity requires an actual recorded canonical block; explicit modular also checks it |
| `basis.verify(rank_prime=None)` | `None` reuses the construction prime, or `1000003` for legacy objects; same two integer primes allowed | Applies only when the verification is modular. Use `independence="modular"` as well when a modular cross-check is intended |
| `basis.save(directory)` | Required path string or `Path`, empty or new directory | Saves compressed integer arrays, owners and configuration; refuses nonempty directories, never overwrites a prior run |
| `PotentialBasis.load(directory, verify=True)` | Required path to saved arrays; `verify` is a strict boolean with the same meaning as construction | Default rechecks actual arrays. Saved success is cleared first. `verify=False` reports deferral, even if the file contains an old passing report. Pickle is disabled |
| `load(independence=None, rank_prime=None)` | Same defaults, choices and recommendations as `basis.verify` | Overrides current load verification, not the historical construction record. Explicit incompatible identity settings fail |
| `load(max_variables=None)` | Same positive ambient-count guard or `None` | Enforced before mesh reconstruction. Effective load options are recorded separately in `configuration['load']` |
| `dimension(N, degree)` | The required problem inputs above; no method knobs | Uses the polynomial count for `N>=2` or `d=6`. For `N=1,d>6`, actually reconstructs a checked kernel using the standard prime; this can be expensive/fail. For a guarded or alternate-prime count, construct with those options and use `len(basis)` |

Valid v0.1.0 calls and saved bases remain supported. Legacy loads report their
new check; unknown historical option values are not invented. Previously
truthy invalid flags such as `verify="false"` now raise `TypeError`.
Noninteger saved `N`/degree values are rejected without rounding.

## Evaluation and exact support

| Parameter | Type, range, default and recommendation | Output/cost/accuracy |
|---|---|---|
| `basis_id` in evaluation/support methods | Required integer excluding booleans, `0<=i<len(basis)` | One basis row; `support_tetrahedra(i)` returns sorted exact nonzero tetrahedron IDs independently of samples |
| `point` in `evaluate(i, point)` | Required finite array-like `(3,)`, physical coordinates in `[0,1]^3` | Two binary64 `(3,)` arrays: potential and physical curl; includes factor `N` in derivatives |
| `points` in `evaluate_many(i, points)` | Required finite array-like `(m,3)` in `[0,1]^3`; empty `(0,3)` accepted | Two `(m,3)` arrays; repeated direct evaluation, not an adaptive or accuracy-controlled sampler |
| `tetrahedron` in `evaluate_on_tetrahedron(i, tetrahedron, lambdas)` | Required integer excluding booleans, valid mesh tetrahedron ID | Choose either incident tetrahedron to inspect a face trace |
| `lambdas` in that method | Required finite array-like `(4,)`, nonnegative, sum one; fixed input guard allows `1e-12` boundary roundoff | Two physical `(3,)` arrays. Tiny accepted negative entries are clamped to zero. The input guard is not an error tolerance or a high-degree stability guarantee |

Potential units are those of the chosen coefficients; curl has one additional
inverse-length factor, with the physical domain normalized to a unit cube.
Direct Bernstein arithmetic, coefficient growth and conditioning limit high
degrees. No precision choice or unchecked conversion to a lower degree occurs.

## Low-level assembly and raw containers

Use the builders rather than manually constructing mutable mesh/space records.
Lower degrees are allowed for assembly; the potential-basis constructor still
requires `d>=6`.

| Parameter | Contract/default and recommendation | Consequence |
|---|---|---|
| `Mesh.build(N)` | Required positive integer excluding booleans | Returns integer vertices in `[0,N]^3`; divide by `N` for physical coordinates. No resource guard in this low-level builder |
| `BernsteinSpace.build(mesh, d)` | Required `Mesh` and positive integer degree excluding booleans | Scalar continuous Bernstein coefficient space; integer labels are coefficients, not nodal values. Cost grows with tetrahedra times `binomial(d+3,3)` |
| `space.allowed_nodes(support)` | Required set/frozenset of valid tetrahedron integer IDs | Coefficients whose incident tetrahedra lie wholly in `support`; empty set is allowed |
| `space.vertex_nodes(v)` | Required valid integer vertex ID, excluding booleans | Coefficients whose owner intersection contains that vertex |
| `space.curl_constraints(all_components=False)` | Strict boolean; default omits the redundant normal component | Returns CSR integer jump matrix and face-mode metadata. `True` retains all Cartesian components, with additional rows/work. Both omit common nonzero `d*N`; no physical boundary constraints |
| `space.c1_constraints()` | No parameters | Scalar transverse derivative jumps divided by `d` in scaled coordinates, plus face-mode metadata |
| `space.curl_map()` | No parameters | Broken tetrahedral curl coefficients in `y=N*x`, retaining degree factor `d`; multiply by `N` for physical curl |

`PotentialBasis` is an advanced mutable data container. Its raw constructor
takes required `space` (a `BernsteinSpace`), `coefficients` (integer CSR of
shape `(rows,3*space.nscalar)`) and `owners` (integer `(rows,)` vertex IDs).
Optional `free` and `pivots` default to empty integer arrays; they are column
IDs for an actual normal form. `normalform=None` omits that matrix, otherwise
it has integer CSR shape `(len(pivots),3*space.nscalar)`. `metadata` defaults
to an empty dictionary. These are representation data, not tuning choices.
Manual initialization itself does not verify any invariant; use the factory
for ordinary construction. `len`, `summary` and `configuration` have no parameters.

## Installed CLI and benchmark parameters

`python -m freudenthal N degree` prints the same basis summary as the API.
`--method`, `--independence`, `--rank-prime`, `--max-variables` and `--no-verify`
map directly to the shared constructor policy; omission uses its defaults.
`--plan` prints only the plan and cannot be combined with `--output`.
`--output PATH` is optional and uses the same empty/new-directory rule as save.
Errors return exit code 2 with a diagnostic. No files are written by default.

`bench/benchmark_policy.py --output PATH` requires a fresh output directory.
`--repetitions` is an integer 1..5, default 3; more samples cost proportionally
more time and reduce sampling uncertainty, not mathematical error. `--smoke`
defaults false; true omits the larger window case. Four configurations are
fixed and finish the same full finite checks. See [the benchmark contract](../bench/README.md).

The retained figure driver uses one fixed `N=2,d=7` case. Its `--resolution`
is an integer 5..33 (default 17) for plotting samples, not basis construction
or PDE accuracy. Its `--output` and the plotter's `--input`, `--reference` and
`--output` are required paths; the plotter checks hashes and refuses existing
output directories. The independent certificate-replay script also requires
a fresh `--output`; `CXX` may select an available C++17 compiler. Missing
compiler/SymPy or a failed witness predicate is an explicit failure.

Types/ranges outside these contracts raise actionable type, value, index,
file or overflow errors. A `MemoryError` from an admission guard describes
the unchanged inputs and required ambient count. No silent validation downgrade,
backend substitution or prime retry is part of any supported path.
