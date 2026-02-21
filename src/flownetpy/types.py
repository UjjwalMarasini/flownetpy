# src/flownetpy/types.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass(frozen=True)
class Geometry:
    """
    Geometry + domain + grid definition.
    All units are meters (m).
    """
    dam_height: float
    base_width: float
    top_width: float
    embed_depth: float

    left_domain: float
    right_domain: float
    bottom_domain: float

    grid_x: float
    grid_y: float

    def __post_init__(self) -> None:
        # Minimal sanity checks.
        if self.dam_height <= 0:
            raise ValueError("dam_height must be > 0.")
        if self.base_width <= 0:
            raise ValueError("base_width must be > 0.")
        if self.top_width <= 0:
            raise ValueError("top_width must be > 0.")
        if self.top_width > self.base_width:
            raise ValueError("top_width cannot be greater than base_width.")
        if self.embed_depth < 0:
            raise ValueError("embed_depth must be >= 0.")

        if self.left_domain <= 0 or self.right_domain <= 0 or self.bottom_domain <= 0:
            raise ValueError("left_domain, right_domain, bottom_domain must be > 0.")

        if self.grid_x <= 0 or self.grid_y <= 0:
            raise ValueError("grid_x and grid_y must be > 0.")

    @property
    def heel_x(self) -> float:
        return 0.0

    @property
    def toe_x(self) -> float:
        return float(self.base_width)

    def dam_polygon(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (x, y) polygon arrays for dam body.
        Useful for plotting.
        """
        heel = self.heel_x
        toe = self.toe_x
        crest_left = heel
        crest_right = heel + self.top_width

        x = np.array([heel, crest_left, crest_right, toe, toe, heel], dtype=float)
        y = np.array([0.0, self.dam_height, self.dam_height, 0.0, -self.embed_depth, -self.embed_depth], dtype=float)
        return x, y

    def domain_limits(self) -> Tuple[float, float, float, float]:
        """
        Returns (x_min, x_max, y_min, y_max) for the soil domain (below ground).
        """
        x_min = -self.left_domain
        x_max = self.toe_x + self.right_domain
        y_min = -self.bottom_domain
        y_max = 0.0
        return float(x_min), float(x_max), float(y_min), float(y_max)


@dataclass(frozen=True)
class BoundaryConditions:
    """
    Boundary conditions in meters of hydraulic head.
    """
    us_head: float
    ds_head: float

    def __post_init__(self) -> None:
        # Heads can be negative in some abstract problems
        # it usually makes sense to be >= 0. Keep permissive:
        if self.us_head == self.ds_head:
            # Not wrong, but likely not intended (no driving gradient)
            # Keep as ValueError or just allow; I’ll allow but warn later in core if needed.
            pass


@dataclass(frozen=True)
class CutoffConfig:
    """
    Cutoff walls. A cutoff is considered active if width > 0 AND depth > 0.
    All units are meters (m).
    """
    us_cutoff_width: float = 0.0
    us_cutoff_depth: float = 0.0
    ds_cutoff_width: float = 0.0
    ds_cutoff_depth: float = 0.0

    def __post_init__(self) -> None:
        for name, val in [
            ("us_cutoff_width", self.us_cutoff_width),
            ("us_cutoff_depth", self.us_cutoff_depth),
            ("ds_cutoff_width", self.ds_cutoff_width),
            ("ds_cutoff_depth", self.ds_cutoff_depth),
        ]:
            if val < 0:
                raise ValueError(f"{name} must be >= 0.")

    @property
    def use_upstream(self) -> bool:
        return (self.us_cutoff_width > 0.0) and (self.us_cutoff_depth > 0.0)

    @property
    def use_downstream(self) -> bool:
        return (self.ds_cutoff_width > 0.0) and (self.ds_cutoff_depth > 0.0)

    def upstream_polygon(self, heel_x: float = 0.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns (x, y) polygon arrays for upstream cutoff rectangle, or None if inactive.
        """
        if not self.use_upstream:
            return None
        x0 = heel_x
        x1 = heel_x + self.us_cutoff_width
        y0 = 0.0
        y1 = -self.us_cutoff_depth
        x = np.array([x0, x1, x1, x0], dtype=float)
        y = np.array([y0, y0, y1, y1], dtype=float)
        return x, y

    def downstream_polygon(self, toe_x: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns (x, y) polygon arrays for downstream cutoff rectangle, or None if inactive.
        """
        if not self.use_downstream:
            return None
        x1 = toe_x
        x0 = toe_x - self.ds_cutoff_width
        y0 = 0.0
        y1 = -self.ds_cutoff_depth
        x = np.array([x0, x1, x1, x0], dtype=float)
        y = np.array([y0, y0, y1, y1], dtype=float)
        return x, y


@dataclass(frozen=True)
class SolverConfig:
    """
    Physical + numerical controls.
    k: hydraulic conductivity (m/s)
    """
    k: float
    tol: float = 1e-4
    max_iter: int = 500

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be > 0.")
        if self.tol <= 0:
            raise ValueError("tol must be > 0.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be > 0.")


@dataclass
class Result:
    """
    Outputs from seepage solver.
    Q: seepage discharge per unit width (m^3/s/m)
    """
    x_nodes: np.ndarray
    y_nodes: np.ndarray
    h: np.ndarray

    Q: float

    converged: bool
    n_iter: int
    max_change: float

    barrier_x: Optional[np.ndarray] = None
    U: Optional[np.ndarray] = None
    V: Optional[np.ndarray] = None

    # Optional metadata for plotting / debugging
    case_label: Optional[str] = None