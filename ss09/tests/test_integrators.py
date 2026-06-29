"""Cross-integrator validation of the hand-rolled RK4 (ss09/integrators.py).

Anchor 5 of the rewrite: the fixed-step RK4 must agree with scipy's adaptive
DOP853 on the identical RHS, and converge at 4th order under step refinement.
DOP853 (with tight tolerances) serves as the independent reference solution.
"""

import numpy as np
from scipy.integrate import solve_ivp

from ss09 import rhs
from ss09.integrators import RK4Integrator
from ss09.model_state import ModelState
from ss09.sw_config import SWConfig
from ss09.theta_e import ThetaEConfig, Sin2Profile


def _setup():
    config = SWConfig(ny=50, total_integration_days=1)
    profile = Sin2Profile(ThetaEConfig(theta_e_type="sin2", y_0=0.0))
    u0 = 5.0 * np.exp(-(config.y / 3e6) ** 2)
    v0 = np.zeros(config.ny + 1)
    v0[1:-1] = 0.5 * np.sin(2 * np.pi * config.yf[1:-1] / config.domain_size)
    theta_e = profile(ModelState(0.0, u0, v0, np.zeros(config.ny), config.y))
    theta0 = theta_e + 2.0 * np.cos(np.pi * config.y / config.domain_size)
    init = ModelState(0.0, u0, v0, theta0, config.y)
    return config, profile, init


def _rk4_integrate(init, config, profile, dt, nsteps):
    integ = RK4Integrator()
    state = init
    for _ in range(nsteps):
        state = integ.step(state, dt, lambda s: rhs.rhs(s, config, profile))
    return rhs.pack(state)


def _scipy_truth(init, config, profile, T):
    sol = solve_ivp(
        rhs.rhs_flat,
        [0.0, T],
        rhs.pack(init),
        args=(config, profile),
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
    )
    return sol.y[:, -1]


def test_rk4_matches_scipy_dop853():
    config, profile, init = _setup()
    T = 7200.0  # 2 h
    dt = 60.0
    rk4 = _rk4_integrate(init, config, profile, dt, int(T / dt))
    truth = _scipy_truth(init, config, profile, T)
    np.testing.assert_allclose(rk4, truth, rtol=1e-6, atol=1e-8)


def test_rk4_is_fourth_order():
    config, profile, init = _setup()
    T = 7200.0
    truth = _scipy_truth(init, config, profile, T)
    err = []
    for dt in (480.0, 240.0):
        approx = _rk4_integrate(init, config, profile, dt, int(T / dt))
        err.append(np.max(np.abs(approx - truth)))
    # halving dt should cut the error by ~16x for a 4th-order scheme
    ratio = err[0] / err[1]
    assert 8.0 < ratio < 40.0, f"convergence ratio {ratio} not ~16"


def test_rk4_keeps_v_walls_zero():
    config, profile, init = _setup()
    integ = RK4Integrator()
    state = integ.step(init, 3600.0, lambda s: rhs.rhs(s, config, profile))
    assert state.v[0] == 0.0
    assert state.v[-1] == 0.0
