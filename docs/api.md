# Public API

Import the five public objects from `freudenthal`. Indices are zero-based;
array component order is `(x,y,z)`.

| Object | Contract |
|---|---|
| `Mesh.build(N)` | Positive integer grid count, `(N+1)^3` integer vertices and `6*N^3` tetrahedra |
| `BernsteinSpace.build(mesh, d)` | Continuous Bernstein coefficient space, positive integer `d`; lower degrees are available for assembly only |
| `construct_basis(N, degree, verify=True)` | Construct a `PotentialBasis` for integers `N>=1, degree>=6`; finite checks run by default |
| `dimension(N, degree)` | Polynomial count for `N>=2` or `d=6`; checked full kernel construction for `N=1,d>6` |
| `PotentialBasis` | Basis rows, owner vertices, evaluation, support, validation and persistence |

## Mesh and coefficient conventions

`mesh.vertices` has shape `((N+1)^3,3)` in integer coordinates on `[0,N]^3`.
Divide by `N` for physical coordinates. `mesh.tets` indexes those vertices;
`mesh.stars[v]` contains incident tetrahedron IDs. `mesh.gradients` contains
integer barycentric gradients in the scaled coordinate `y=N*x`.

`space.nodes` has shape `((N*d+1)^3,3)`. These are integer **coefficient
labels**, not point evaluations. `space.alpha` contains degree-`d`
four-component multiindices, and `space.local_to_global[t]` identifies shared
coefficients. Vector columns are ordered `3*node_id + component`.

`space.curl_constraints(all_components=True)` returns the sparse integer curl
jump matrix and face-mode metadata. Continuity of the potential is built into
the shared columns. Constraints omit the nonzero physical factor `d*N`.
Physical boundary faces have no constraints. The default keeps two independent
curl components per face. `space.curl_map()` gives broken tetrahedral curl
coefficients in `y`; multiply the map by `N` for physical curl.

## Basis methods

`basis.coefficients` is a CSR matrix of signed integer Bernstein rows,
shape `(dimension, 3*(N*d+1)^3)`. `basis.owners[i]` indexes the owner vertex.
`basis.free`, `basis.pivots` and `basis.normalform` expose branch-dependent
normal-form information; small-grid stored bases may not have a normal form.

- `basis.verify()` checks all Cartesian curl jumps, owner-star support,
  independence and the applicable dimension formula, returning a report.
  Unsafe signed-64-bit multiplication raises `OverflowError`.
- `basis.support_tetrahedra(i)` returns sorted tetrahedron IDs with a nonzero
  polynomial piece, derived exactly from Bernstein coefficients.
- `basis.evaluate(i, point)` returns two `(3,)` floating arrays: physical
  potential and physical curl at a point of `[0,1]^3`.
- `basis.evaluate_many(i, points)` takes `(m,3)` points and returns two `(m,3)`
  arrays. Empty arrays of shape `(0,3)` are accepted.
- `basis.evaluate_on_tetrahedron(i, t, lambdas)` uses four nonnegative
  barycentric coordinates summing to one. This permits either trace at a face.
- `basis.summary()` returns dimensions, scalar sparse counts, coefficient
  bounds, construction branch and any completed finite checks.
- `basis.save(directory)` requires an empty or new directory.
  `PotentialBasis.load(directory, verify=True)` disables pickle loading and
  rechecks the saved coefficients by default.

```python
from freudenthal import PotentialBasis, construct_basis

basis = construct_basis(1, 6)
basis.save("my-fresh-basis")
restored = PotentialBasis.load("my-fresh-basis")
assert len(restored) == 795
```

Wrong construction types raise `TypeError`; unsupported ranges raise
`ValueError`. Evaluation rejects noninteger/boolean indices with `TypeError`,
out-of-range indices with `IndexError`, and invalid shapes, nonfinite points
or out-of-domain coordinates with `ValueError`. Kernel reconstruction can fail
with `ValueError`, rather than returning an unverified basis. No allocation
guard can predict every feasible `(N,d)` combination; start with the examples.
