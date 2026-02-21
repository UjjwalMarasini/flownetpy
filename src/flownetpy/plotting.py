# src/flownetpy/plotting.py

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from .types import Geometry, BoundaryConditions, CutoffConfig, Result


def plot_flownet(
    result: Result,
    geom: Geometry,
    bc: BoundaryConditions,
    cutoffs: Optional[CutoffConfig] = None,
    *,
    show: bool = True,
    savepath: Optional[str] = None,
    dpi: int = 300,
    title: Optional[str] = None,
    n_equipotential: int = 11,
    n_flow_seeds: int = 5,
    seed_x_span: Tuple[float, float] = (-6.0, -1.0),
    seed_y: float = -1e-3,
    stream_density: float = 1.2,
    stream_maxlength: float = 100.0,
    stream_minlength: float = 0.01,
):
    if cutoffs is None:
        cutoffs = CutoffConfig()

    x_nodes = result.x_nodes
    y_nodes = result.y_nodes
    h = result.h

    # Need velocity for streamlines
    if result.U is None or result.V is None:
        raise ValueError(
            "Result does not contain velocity fields (U, V). "
            "Run with compute_velocity=True in run_seepage(), or store U/V in Result."
        )

    U = result.U.copy()
    V = result.V.copy()

    # Meshgrid
    X, Y = np.meshgrid(x_nodes, y_nodes)

    heel_x = geom.heel_x
    toe_x = geom.toe_x
    left_domain = geom.left_domain
    right_domain = geom.right_domain
    bottom_domain = geom.bottom_domain

    # Dam polygon
    dam_x, dam_y = geom.dam_polygon()

    # Domain limits
    x_min, x_max, y_min, _y_max = geom.domain_limits()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Ground line (for visual)
    ax.plot([-left_domain, heel_x], [0, 0], color="k", linewidth=1.0)
    ax.plot([toe_x, toe_x + right_domain], [0, 0], color="k", linewidth=1.0)

    # Dam body
    ax.fill(dam_x, dam_y, color="0.85", edgecolor="k", linewidth=1.5)

    # Cutoff polygons (only if active)
    us_poly = cutoffs.upstream_polygon(heel_x=heel_x)
    ds_poly = cutoffs.downstream_polygon(toe_x=toe_x)

    if us_poly is not None:
        ux, uy = us_poly
        ax.fill(ux, uy, color="dimgray", edgecolor="k", linewidth=1.2)

    if ds_poly is not None:
        dxp, dyp = ds_poly
        ax.fill(dxp, dyp, color="dimgray", edgecolor="k", linewidth=1.2)

    # Water levels (same style as your script)
    # Downstream water line drawn up to intersection on downstream slope
    intercept = bc.ds_head * (geom.base_width - geom.top_width) / geom.dam_height
    ax.hlines(bc.us_head, -left_domain, heel_x, colors="blue", linewidth=2)
    ax.hlines(bc.ds_head, toe_x + right_domain, toe_x - intercept, colors="blue", linewidth=2)

    # Equipotential lines
    levels_h = np.linspace(np.min(h), np.max(h), n_equipotential)
    cs = ax.contour(X, Y, h, levels=levels_h, linestyles="dashed", linewidths=1.0)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

    # Zero velocity inside cutoff rectangles (matches your script behavior)
    if cutoffs.use_upstream:
        us_x0 = heel_x
        us_x1 = heel_x + cutoffs.us_cutoff_width
        us_d = cutoffs.us_cutoff_depth
        inside_us = (X >= us_x0) & (X <= us_x1) & (Y <= 0.0) & (Y >= -us_d)
        U[inside_us] = 0.0
        V[inside_us] = 0.0

    if cutoffs.use_downstream:
        ds_x1 = toe_x
        ds_x0 = toe_x - cutoffs.ds_cutoff_width
        ds_d = cutoffs.ds_cutoff_depth
        inside_ds = (X >= ds_x0) & (X <= ds_x1) & (Y <= 0.0) & (Y >= -ds_d)
        U[inside_ds] = 0.0
        V[inside_ds] = 0.0

    # Seed points along upstream water surface, just inside soil
    sx0, sx1 = seed_x_span
    seed_x = np.linspace(heel_x + sx0, heel_x + sx1, n_flow_seeds)
    seed_y_arr = np.full_like(seed_x, seed_y, dtype=float)
    start_pts = np.column_stack((seed_x, seed_y_arr))

    ax.streamplot(
        X, Y, U, V,
        start_points=start_pts,
        color="k",
        linewidth=1.2,
        arrowsize=1.5,
        density=stream_density,
        integration_direction="forward",
        maxlength=stream_maxlength,
        minlength=stream_minlength,
    )

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Elevation (m)")

    if title is None:
        title = "Flownet from ∇·(K∇h) = 0"
        if cutoffs.use_upstream and cutoffs.use_downstream:
            title += " with Upstream & Downstream Cutoffs"
        elif cutoffs.use_upstream:
            title += " with Upstream Cutoff"
        elif cutoffs.use_downstream:
            title += " with Downstream Cutoff"
    ax.set_title(title)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-bottom_domain, geom.dam_height)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax