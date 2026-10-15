"""Shared input validation and deterministic construction policy.

Planning is allocation-free: counts are exact, but are not memory estimates.
"""

from dataclasses import asdict, dataclass

import numpy as np

METHODS = ("auto", "stored", "window", "parametric", "one-cube")
INDEPENDENCE_CHECKS = ("auto", "identity", "modular")
RANK_PRIMES = (1000003, 1000033)


@dataclass(frozen=True)
class _Defaults:
    verify: bool = True
    method: str = "auto"
    independence: str = "auto"
    rank_prime: int = RANK_PRIMES[0]
    max_variables: int | None = None


DEFAULTS = _Defaults()


def parameters(N, degree):
    for name, value in (("N", N), ("degree", degree)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    if N < 1 or degree < 6:
        raise ValueError("Construction requires N>=1 and potential degree d>=6")
    return int(N), int(degree)


def verification_flag(verify, name="verify"):
    if not isinstance(verify, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(verify)


def index(value, size, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer index")
    if not 0 <= value < size:
        raise IndexError(f"{name} index must lie in [0, {size})")
    return int(value)


def rank_modulus(prime):
    if isinstance(prime, (bool, np.bool_)) or not isinstance(prime, (int, np.integer)):
        raise TypeError("rank_prime must be an integer")
    if prime not in RANK_PRIMES:
        raise ValueError(f"rank_prime must be one of {RANK_PRIMES}; no automatic prime fallback")
    return int(prime)


def independence_check(requested, canonical):
    if not isinstance(requested, str) or requested not in INDEPENDENCE_CHECKS:
        raise ValueError(f"independence must be one of {INDEPENDENCE_CHECKS}")
    if requested == "identity" and not canonical:
        raise ValueError("independence='identity' requires a canonical free-coordinate identity")
    return ("identity" if canonical else "modular") if requested == "auto" else requested


def variable_guard(N, degree, max_variables):
    variables = 3 * (N * degree + 1) ** 3
    if max_variables is not None:
        if isinstance(max_variables, (bool, np.bool_)) or not isinstance(
            max_variables, (int, np.integer)
        ):
            raise TypeError("max_variables must be a positive integer or None")
        if max_variables < 1:
            raise ValueError("max_variables must be positive")
        if variables > max_variables:
            raise MemoryError(
                f"The requested N={N}, degree={degree} needs {variables} ambient variables; "
                f"max_variables={max_variables}. Increase the guard explicitly. "
                "The mesh and degree were not changed; this guard is not a RAM estimate."
            )
    return variables


@dataclass(frozen=True)
class ConstructionPlan:
    """An exact input/configuration snapshot; no matrix has been constructed."""

    N: int
    degree: int
    method: str
    verify: bool
    independence: str
    rank_prime: int
    max_variables: int | None
    variables: int
    tetrahedra: int
    local_coefficients: int
    requested_method: str
    requested_independence: str
    rank_prime_active: bool
    selection_reasons: tuple[str, ...]

    def to_dict(self):
        """Return a detached, JSON-compatible configuration."""
        result = asdict(self)
        result["selection_reasons"] = list(self.selection_reasons)
        return result


def plan_construction(
    N: int,
    degree: int,
    *,
    verify=DEFAULTS.verify,
    method=DEFAULTS.method,
    independence=DEFAULTS.independence,
    rank_prime=DEFAULTS.rank_prime,
    max_variables=DEFAULTS.max_variables,
) -> ConstructionPlan:
    """Resolve the same policy used by construction and the CLI, without allocation.

    Required inputs N and degree are never adapted. Methods have disjoint
    validity ranges. Explicit incompatible choices fail instead of falling back.
    """
    N, degree = parameters(N, degree)
    verify = verification_flag(verify)
    prime = rank_modulus(rank_prime)
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(
            f"method must be one of {METHODS}; only the CPU sparse backend is supplied"
        )
    if N == 1 and degree > 6:
        chosen, reason = "one-cube", "One cube at degree>6 requires checked kernel reconstruction."
    elif degree == 6:
        chosen = "stored" if N <= 4 else "window"
        reason = "Degree-six grids N<=4 use stored bases; larger grids use window transfer."
    else:
        chosen, reason = "parametric", "N>=2 and degree>=7 use the fixed parametric rules."
    if method != "auto" and method != chosen:
        raise ValueError(
            f"method={method!r} is incompatible with N={N}, degree={degree}; use {chosen!r} "
            "or 'auto'. The problem inputs were not changed."
        )
    check = independence_check(independence, chosen == "parametric")
    variables = variable_guard(N, degree, max_variables)
    if independence == "auto":
        check_reason = (
            "The canonical free-coordinate identity gives exact independence without modular work."
            if check == "identity"
            else "This branch has no canonical free-coordinate identity; modular rank checks independence."
        )
    else:
        check_reason = (
            f"Explicit {check} verification selected; any canonical identity is also checked."
        )
    reasons = (
        reason,
        check_reason,
        "The standard rank prime preserves v0.1.0; no prime retry is performed."
        if prime == DEFAULTS.rank_prime
        else "The supported alternate prime was requested for an exact arithmetic cross-check; no retry.",
        "Full finite verification will run after construction."
        if verify
        else "Optional full verification is deferred; mandatory constructor checks remain enabled.",
        "No ambient-variable cap is guessed; no peak-RAM estimate is claimed."
        if max_variables is None
        else f"Admission is capped at {max_variables} ambient variables; this is not a peak-RAM bound.",
    )
    return ConstructionPlan(
        N,
        degree,
        chosen,
        verify,
        check,
        prime,
        None if max_variables is None else int(max_variables),
        variables,
        6 * N**3,
        (degree + 1) * (degree + 2) * (degree + 3) // 6,
        method,
        independence,
        chosen in ("window", "one-cube") or (verify and check == "modular"),
        reasons,
    )
