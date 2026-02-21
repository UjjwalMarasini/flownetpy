# src/flownetpy/api.py

from __future__ import annotations
from typing import Optional
import numpy as np
from .types import Geometry, BoundaryConditions, CutoffConfig, SolverConfig, Result
from . import core


def run_seepage(
    geom: Geometry,
    bc: BoundaryConditions,
    cutoffs: Optional[CutoffConfig] = None,
    solver: Optional[SolverConfig] = None,
    *,
    x_control: Optional[float] = None,
    compute_velocity: bool = True,
    case_label: Optional[str] = None,
) -> Result:
    if cutoffs is None:
        cutoffs = CutoffConfig()

    if solver is None:
        raise ValueError("solver must be provided (SolverConfig with k, tol, max_iter).")

    # 1) grid + initial head
    x_nodes, y_nodes, h0 = core.build_grid(geom, bc)

    # 2) barriers (cutoffs)
    barrier_x = core.build_barrier_x(geom, cutoffs, x_nodes, y_nodes)

    # 3) solve head
    h, converged, n_iter, max_change = core.solve_head(
        h0=h0,
        geom=geom,
        bc=bc,
        barrier_x=barrier_x,
        solver=solver,
        x_nodes=x_nodes,
    )

    # 4) velocity (optional)
    U = V = None
    if compute_velocity:
        U, V = core.compute_velocity(h=h, geom=geom, solver=solver, barrier_x=barrier_x)

    # 5) discharge at control section
    if x_control is None:
        # Default: a bit downstream of toe, but safely inside domain
        # (avoid boundary where central differences break)
        x_control = geom.toe_x + max(geom.grid_x * 2.0, min(5.0, 0.25 * geom.right_domain))

    Q = core.compute_discharge(
        h=h,
        geom=geom,
        solver=solver,
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        x_control=x_control,
    )

    return Result(
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        h=h,
        Q=Q,
        converged=converged,
        n_iter=n_iter,
        max_change=max_change,
        barrier_x=barrier_x,
        U=U,
        V=V,
        case_label=case_label,
    )