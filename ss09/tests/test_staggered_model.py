"""Model-level tests for the staggered-v (C-grid) integration.

The load-bearing property is exact mirror parity: started from a
mirror-symmetric state (u, theta symmetric about the equator; v antisymmetric)
under the production gate-on + MC-stencil physics, the leapfrog integration
must keep u and theta exactly symmetric and v exactly antisymmetric, to the
last floating-point bit, over many steps. The collocated model has this
invariant; the symmetric-association face Laplacian and the parity-safe face
coordinate make it hold on the staggered grid too. A parity drift is the
signature of an asymmetric floating-point association somewhere in the
operators, which is exactly the class of bug the staggered refactor had to
avoid.
"""

import numpy as np
import pytest

from ss09.sw_model import SWModel, StaggeredSWModel
from ss09.sw_config import SWConfig
from ss09.theta_e import ThetaEConfig, Sin2Profile


def _leapfrog_steps(model, n_steps):
    """Advance the model n_steps with the same leapfrog + Asselin core as
    run_sim, without the daily-averaging / diagnostics / I/O machinery."""
    model.vars_prev_step = type(model.vars_prev_step)(*model.init_prev_step_vars())
    for i in range(n_steps):
        model.state = model.state._replace(t=i * model.config.dt)
        model.vars_next_step = model.vars_next_step._replace(
            u=model.leapfrog_step(model.vars_prev_step.u, model.du_dt),
            v=model.leapfrog_step(model.vars_prev_step.v, model.dv_dt),
            theta=model.leapfrog_step(model.vars_prev_step.theta, model.dtheta_dt),
        )
        model.vars_prev_step = model.vars_prev_step._replace(
            u=model.asselin_filt(
                model.vars_prev_step.u, model.vars_next_step.u, model.state.u
            ),
            v=model.asselin_filt(
                model.vars_prev_step.v, model.vars_next_step.v, model.state.v
            ),
            theta=model.asselin_filt(
                model.vars_prev_step.theta, model.vars_next_step.theta, model.state.theta
            ),
        )
        model.state = model.state._replace(
            u=model.vars_next_step.u,
            v=model.vars_next_step.v,
            theta=model.vars_next_step.theta,
        )
        model.enforce_boundary_conditions()


def test_swmodel_dispatches_to_staggered_on_grid():
    """SWModel(config) returns a StaggeredSWModel when config.grid is
    staggered, and a plain SWModel when it is collocated, so the config alone
    drives the dynamics."""
    stag = SWModel(
        SWConfig(total_integration_days=1, ny=51, dt=30, grid="staggered"),
        Sin2Profile(ThetaEConfig()),
    )
    assert isinstance(stag, StaggeredSWModel)
    assert stag.state.v.shape == (50,)  # ny-1 faces

    collo = SWModel(
        SWConfig(total_integration_days=1, ny=51, dt=30, grid="collocated"),
        Sin2Profile(ThetaEConfig()),
    )
    assert type(collo) is SWModel
    assert collo.state.v.shape == (51,)  # ny centers


@pytest.mark.parametrize("grid", ["collocated", "staggered"])
def test_parity_bitexact_200_steps_gate_on_mc(grid):
    """200 leapfrog steps of the production physics (gate-on + MC stencil)
    from a mirror-symmetric state keep u exactly symmetric and v exactly
    antisymmetric, on both grids."""
    config = SWConfig(
        total_integration_days=1, ny=201, dt=30, v_d=2.5,
        emfd_heaviside_gate=True, emfd_stencil="mc", grid=grid,
    )
    model = SWModel(config, Sin2Profile(ThetaEConfig()))
    # initial state is mirror-symmetric: theta from the y_0=0 profile is
    # symmetric, u is zero (symmetric), v is zero (trivially antisymmetric)
    _leapfrog_steps(model, 200)

    u = model.state.u
    v = model.state.v
    theta = model.state.theta
    assert np.max(np.abs(u - u[::-1])) == 0.0, "u lost exact mirror symmetry"
    assert np.max(np.abs(theta - theta[::-1])) == 0.0, "theta lost exact symmetry"
    # v is antisymmetric on both grids (faces reflect index j -> nv-1-j)
    assert np.max(np.abs(v + v[::-1])) == 0.0, "v lost exact antisymmetry"


def test_staggered_model_requires_staggered_config():
    """Constructing StaggeredSWModel with a collocated config fails loudly,
    so a v-shape mismatch cannot slip through."""
    with pytest.raises(ValueError, match="staggered"):
        StaggeredSWModel(
            SWConfig(total_integration_days=1, ny=51, dt=30, grid="collocated"),
            Sin2Profile(ThetaEConfig()),
        )
