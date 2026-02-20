# flownetpy

Finite-difference flownet seepage solver for dam–foundation problems.

`flownetpy` solves the steady-state groundwater flow equation:

\[
\nabla \cdot (K \nabla h) = 0
\]

using a 2D finite-difference scheme with optional upstream and downstream cutoff walls.

---

## Features

- 2D finite-difference solver
- Upstream and/or downstream vertical cutoffs
- Darcy velocity field computation
- Seepage discharge calculation
- Equipotential contours and streamline plotting
- Clean object-oriented API

---

## Installation (development mode)

From the project root:

```bash
pip install -e .


## Quick Example
The example below solves a dam foundation problem with both upstream and downstream cutoffs and generates a flownet plot.

```python
import flownetpy as fn

geom = fn.Geometry(
    dam_height=5.0,
    base_width=10.0,
    top_width=4.0,
    embed_depth=0.5,
    left_domain=20.0,
    right_domain=20.0,
    bottom_domain=10.0,
    grid_x=1.0,
    grid_y=1.0,
)

bc = fn.BoundaryConditions(
    us_head=4.0,
    ds_head=1.0,
)

cutoffs = fn.CutoffConfig(
    us_cutoff_width=1.0,
    us_cutoff_depth=5.0,
    ds_cutoff_width=1.0,
    ds_cutoff_depth=5.0,
)

solver = fn.SolverConfig(
    k=1e-5,
    tol=1e-4,
    max_iter=500,
)

result = fn.run_seepage(
    geom,
    bc,
    cutoffs,
    solver,
    compute_velocity=True,
)

print("Seepage discharge Q' =", result.Q, "m²/s")

fn.plot_flownet(
    result,
    geom,
    bc,
    cutoffs,
    savepath="flow_net.png",
)
```