# Space, construction and limitations

The physical domain is the closed unit cube. Each grid cell is divided into
the six coordinate-chain tetrahedra, one for each coordinate permutation:

`a + h*conv{0, e_sigma1, e_sigma1+e_sigma2, (1,1,1)}`, with `h=1/N`.

All cells have the same orientation. For potential degree `d=k+1`, the space
contains continuous real vector polynomials of total degree at most `d` on
each tetrahedron, with continuous piecewise curl. It contains gradients; no
boundary condition or quotient is added. Support means closed support.

Shared Bernstein coefficients encode potential continuity. Curl continuity
becomes an integer sparse matrix kernel. The implementation constructs local
generators of that kernel and assigns every accepted row to a single vertex
star. All-component exact residuals and finite-instance independence are
checked before the default constructor returns.

| Range | Implementation |
|---|---|
| `d=6, 1<=N<=4` | Fixed integer small-grid bases |
| `d=6, N>=5` | Transferred normal form and integral local repair parts; modular selection of independent local generators |
| `d>=7, N>=2` | Parametric coefficient-type rules, canonical free-coordinate identity and owner masks |
| `d>6, N=1` | Modular elimination, rational reconstruction and exact integer membership/completeness checks; reconstruction may fail |

For `N>=2,d>=6`, and also for `N=1,d=6`, the dimension count is

`3*N^3*(d-1)^2*(d-2) + 3*N^2*(d-1)*(5*d-4) + 9*N*(2*d-1) + 6`.

This polynomial is not used for `N=1,d>6`. Calling `dimension` there actually
constructs and checks a kernel.

Integer construction and binary64 evaluation are separate operations.
Signed-64-bit sparse products have conservative overflow checks. Modular
independence plus exact membership supports the concrete returned bases;
the entire parameter family also requires analytical transfer and coverage
arguments. Evaluation uses direct Bernstein polynomials and may lose accuracy
or overflow at very high degree. This release supplies neither a uniform
decomposition norm nor a conditioning estimate, multigrid solver, general
mesh extension or boundary-condition treatment.

## Mathematical sources

The full construction appears in Kaiwen Jin, *Local potential bases on
Freudenthal meshes*, dated 10 October 2026:
[paper](https://github.com/yuiyuiui/AIM/blob/f69a9853cae1afc0f82ee4f466c5a53b30709f9a/research/solutions/633-kaiwen-jin-freudenthal-potentials/paper/main.pdf).
It is a submitted natural-language computer-assisted argument, not a formal
proof or an assertion of independent human certification.

The original finite-element conjecture and consistently oriented subdivision
are in P. E. Farrell, L. Mitchell and L. R. Scott, *Two conjectures on the
Stokes complex in three dimensions on Freudenthal meshes*, Conjecture 2(b)
and Appendix A: [DOI](https://doi.org/10.1137/22M1533943),
[versioned preprint](https://arxiv.org/abs/2211.05494v2).
The implementation does not assume that conjecture as an established theorem.

The software includes the complete fixed integer witnesses and checking code
described in [CERTIFICATES.md](../src/freudenthal/CERTIFICATES.md). Successful
replay validates their finite predicates; the analytical arguments remain in
the paper. An open solution PR or a green CI workflow is not mathematical
acceptance.
