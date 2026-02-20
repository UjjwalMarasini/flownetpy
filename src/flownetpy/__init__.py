"""
flownetpy
---------

Finite-difference flownet seepage solver for dam-foundation problems.
"""

__version__ = "0.1.0"

from .types import (
    Geometry,
    BoundaryConditions,
    CutoffConfig,
    SolverConfig,
    Result,
)

from .api import run_seepage
from .plotting import plot_flownet

__all__ = [
    "Geometry",
    "BoundaryConditions",
    "CutoffConfig",
    "SolverConfig",
    "Result",
    "run_seepage",
]