import flownetpy as fn


def test_smoke_run():
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
        compute_velocity=False,
    )

    # Basic sanity checks
    assert result.converged is True
    assert result.h.shape[0] > 0
    assert result.h.shape[1] > 0
    assert result.Q != 0.0