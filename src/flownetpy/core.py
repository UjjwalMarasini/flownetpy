from __future__ import annotations
from dataclasses import replace
from typing import Tuple
import numpy as np
from .types import Geometry, BoundaryConditions, CutoffConfig, SolverConfig


def build_grid(geom: Geometry, bc: BoundaryConditions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build x/y nodes and initial head array.
    Returns: x_nodes, y_nodes, h0 (Ny x Nx)
    """
    x_min, x_max, y_min, y_max = geom.domain_limits()
    dx, dy = geom.grid_x, geom.grid_y

    x_nodes = np.arange(x_min, x_max + dx, dx)
    y_nodes = np.arange(y_min, y_max + dy, dy)

    Nx = x_nodes.size
    Ny = y_nodes.size

    h0 = np.zeros((Ny, Nx), dtype=float)
    h0[:, :] = 0.5 * (bc.us_head + bc.ds_head)
    return x_nodes, y_nodes, h0


def build_barrier_x(
    geom: Geometry,
    cutoffs: CutoffConfig,
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
) -> np.ndarray:
    """
    Build internal barrier array barrier_x of shape (Ny, Nx-1),
    where True blocks flow across that vertical face.
    """
    Ny = y_nodes.size
    Nx = x_nodes.size

    barrier_x = np.zeros((Ny, Nx - 1), dtype=bool)

    # Upstream cutoff (thin barrier at heel_x face)
    if cutoffs.use_upstream:
        heel_x = geom.heel_x
        i_cut_us = int(np.argmin(np.abs(x_nodes - heel_x)))
        us_rows = y_nodes >= -cutoffs.us_cutoff_depth
        for j in range(Ny):
            if us_rows[j] and 0 <= i_cut_us < Nx - 1:
                barrier_x[j, i_cut_us] = True

    # Downstream cutoff (thin barrier at ds_cutoff_x_start face)
    if cutoffs.use_downstream:
        toe_x = geom.toe_x
        ds_x_start = toe_x - cutoffs.ds_cutoff_width
        i_cut_ds = int(np.argmin(np.abs(x_nodes - ds_x_start)))
        ds_rows = y_nodes >= -cutoffs.ds_cutoff_depth
        for j in range(Ny):
            if ds_rows[j] and 0 <= i_cut_ds < Nx - 1:
                barrier_x[j, i_cut_ds] = True

    return barrier_x


def apply_boundary_conditions(
    h: np.ndarray,
    geom: Geometry,
    bc: BoundaryConditions,
    x_nodes: np.ndarray,
) -> np.ndarray:
    """
    Apply boundary conditions in-place style and return h.
    Mirrors your notebook logic but avoids global variables.
    """
    Ny, Nx = h.shape
    j_bottom = 0
    j_top = Ny - 1

    heel_x = geom.heel_x
    toe_x = geom.toe_x

    # Left/right boundaries: constant head
    h[:, 0] = bc.us_head
    h[:, -1] = bc.ds_head

    # Bottom no-flow
    h[j_bottom, :] = h[j_bottom + 1, :]

    # Ground surface
    for i, x in enumerate(x_nodes):
        if x <= heel_x:
            h[j_top, i] = bc.us_head
        elif x >= toe_x:
            h[j_top, i] = bc.ds_head
        else:
            h[j_top, i] = h[j_top - 1, i]  # no-flow under dam body

    return h


def solve_head(
    h0: np.ndarray,
    geom: Geometry,
    bc: BoundaryConditions,
    barrier_x: np.ndarray,
    solver: SolverConfig,
    x_nodes: np.ndarray,
) -> Tuple[np.ndarray, bool, int, float]:
    """
    Solve for hydraulic head using your iterative scheme.
    Returns: h, converged, n_iter, max_change
    """
    h = h0.copy()
    Ny, Nx = h.shape
    tol = solver.tol
    max_iter = solver.max_iter

    converged = False
    n_iter = 0
    max_change = np.inf

    for it in range(max_iter):
        n_iter = it + 1
        max_diff = 0.0

        h = apply_boundary_conditions(h, geom, bc, x_nodes)

        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                h_old = h[j, i]

                h_above = h[j + 1, i]
                h_below = h[j - 1, i]

                # left neighbor
                if barrier_x[j, i - 1]:
                    h_left = h[j, i]
                else:
                    h_left = h[j, i - 1]

                # right neighbor
                if barrier_x[j, i]:
                    h_right = h[j, i]
                else:
                    h_right = h[j, i + 1]

                h_new = 0.25 * (h_left + h_right + h_above + h_below)
                h[j, i] = h_new

                diff = abs(h_new - h_old)
                if diff > max_diff:
                    max_diff = diff

        max_change = max_diff
        if max_diff < tol:
            converged = True
            break

    h = apply_boundary_conditions(h, geom, bc, x_nodes)
    return h, converged, n_iter, float(max_change)


def compute_velocity(
    h: np.ndarray,
    geom: Geometry,
    solver: SolverConfig,
    barrier_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Darcy flux components U, V at nodes using your barrier-aware method.
    """
    dx, dy = geom.grid_x, geom.grid_y
    k = solver.k

    Ny, Nx = h.shape

    qx_face = np.zeros((Ny, Nx - 1))
    for j in range(Ny):
        for i in range(Nx - 1):
            if barrier_x[j, i]:
                qx_face[j, i] = 0.0
            else:
                qx_face[j, i] = -k * (h[j, i + 1] - h[j, i]) / dx

    qy_face = np.zeros((Ny - 1, Nx))
    for j in range(Ny - 1):
        for i in range(Nx):
            qy_face[j, i] = -k * (h[j + 1, i] - h[j, i]) / dy

    U = np.zeros_like(h)
    V = np.zeros_like(h)

    for j in range(Ny):
        for i in range(Nx):
            if i == 0:
                U[j, i] = qx_face[j, 0]
            elif i == Nx - 1:
                U[j, i] = qx_face[j, Nx - 2]
            else:
                U[j, i] = 0.5 * (qx_face[j, i - 1] + qx_face[j, i])

    for j in range(Ny):
        for i in range(Nx):
            if j == 0:
                V[j, i] = qy_face[0, i]
            elif j == Ny - 1:
                V[j, i] = qy_face[Ny - 2, i]
            else:
                V[j, i] = 0.5 * (qy_face[j - 1, i] + qy_face[j, i])

    return U, V


def compute_discharge(
    h: np.ndarray,
    geom: Geometry,
    solver: SolverConfig,
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    x_control: float,
) -> float:
    """
    Compute seepage discharge per unit width by integrating qx across a vertical section.
    (This is the single discharge you decided to keep.)
    """
    dx = geom.grid_x
    k = solver.k

    i = int(np.argmin(np.abs(x_nodes - x_control)))

    # safe central difference only if interior
    if i <= 0 or i >= (x_nodes.size - 1):
        raise ValueError("x_control must be inside the domain away from boundaries.")

    dh_dx = (h[:, i + 1] - h[:, i - 1]) / (2.0 * dx)
    qx = -k * dh_dx
    Q = float(np.trapz(qx, y_nodes))
    return Q