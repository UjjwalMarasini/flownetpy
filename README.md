# flownetpy

Finite-difference flownet seepage solver for dam–foundation problems.

`flownetpy` solves the steady-state groundwater flow equation:

∇ · (K ∇h) = 0

using a structured 2D finite-difference formulation with optional upstream and downstream cutoff walls.

---

## Features

- 2D finite-difference solver for steady-state seepage
- Upstream and/or downstream vertical cutoff walls
- Darcy velocity field computation
- Seepage discharge calculation
- Equipotential contours and streamline plotting
- Clean object-oriented API

---

## Installation

Install from PyPI:

```bash
pip install flownetpy
```

Development install (from project root):

```bash
pip install -e .
```

---

## Quick Example

The example below solves a dam foundation seepage problem with both upstream and downstream cutoffs and generates a flownet plot.

```python
import flownetpy as fn

# Geometry definition
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

# Boundary conditions
bc = fn.BoundaryConditions(
    us_head=4.0,
    ds_head=1.0,
)

# Cutoff wall configuration
cutoffs = fn.CutoffConfig(
    us_cutoff_width=1.0,
    us_cutoff_depth=5.0,
    ds_cutoff_width=1.0,
    ds_cutoff_depth=5.0,
)

# Solver configuration
solver = fn.SolverConfig(
    k=1e-5,
    tol=1e-4,
    max_iter=500,
)

# Run seepage analysis
result = fn.run_seepage(
    geom,
    bc,
    cutoffs,
    solver,
    compute_velocity=True,
)

print("Seepage discharge Q' =", result.Q, "m²/s")

# Plot flownet
fn.plot_flownet(
    result,
    geom,
    bc,
    cutoffs,
    savepath="flow_net.png",
)
```

---

## Package Structure

- `types.py` — Core data structures (Geometry, BoundaryConditions, CutoffConfig, SolverConfig, Result)
- `solver.py` — Finite-difference numerical solver
- `api.py` — Public API functions
- `plotting.py` — Visualization utilities

---

## Mathematical Model

The solver computes hydraulic head distribution under steady-state conditions:

∇ · (K ∇h) = 0

For homogeneous hydraulic conductivity, this reduces to the Laplace equation:

∇²h = 0

---

## Roadmap

- Transient seepage solver
- Spatially variable hydraulic conductivity
- Grid refinement tools
- Benchmark validation cases

---

## Author

Ujjwal Marasini  
Department of Civil Engineering  
New Mexico State University