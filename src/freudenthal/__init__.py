"""Continuous vector polynomial potentials with continuous curl.

The degree argument is the potential degree d=k+1. Coordinates used by
the assembler are integer cube coordinates; evaluation uses [0,1]^3.
"""

from .basis import PotentialBasis, construct_basis, dimension
from .geometry import BernsteinSpace, Mesh
from .policy import ConstructionPlan, plan_construction

__version__ = "0.2.1"

__all__ = [
    "BernsteinSpace",
    "ConstructionPlan",
    "Mesh",
    "PotentialBasis",
    "__version__",
    "construct_basis",
    "dimension",
    "plan_construction",
]
