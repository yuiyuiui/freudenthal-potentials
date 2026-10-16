# Changelog

## 0.2.1 (release candidate)

Group point evaluation into bounded tetrahedron batches and reuse sparse local pieces per call. Retain scalar evaluation as an independent reference and test mutation, facets and empty inputs.
Public mathematical contracts are unchanged.


## 0.2.0 — 2026-10-15

- Shared inspectable `plan_construction` policy and installed CLI; existing
  valid construction calls keep the same problem, branch and verification default.
- Explicit branch pinning, alternate supported rank prime, forced modular
  reference checks, and a pre-allocation ambient-variable guard.
- Actual construction/verification configuration and selection reasons in
  results; strict verification flags and explicit failure without fallback.
- Failed revalidation clears stale success, deferred loads clear old check
  reports, and noninteger saved mesh/degree inputs are rejected without rounding.
- Parameter recommendations, matched cold-process default/override benchmarks,
  regressions and explicit AI-created software provenance.

## 0.1.0 — 2026-10-13

- Sparse integer construction of local polynomial potential bases, with owner
  vertices and exact curl-continuity, support and finite-instance rank checks.
- Physical-coordinate potential/curl evaluation, batch evaluation, exact
  tetrahedral support queries and checked persistence.
- A worked nonzero-curl example and three reproducible SVG/PDF figures showing
  locality, sampled fields and sparse storage structure.
- Installable source/wheel distributions, public API and method documentation,
  MIT licensing, research attribution and an explicit AI-assistance statement.
- Installed-package tests and fixed certificate replay in Linux/macOS CI.
