"""Tests for the staggered-grid RHS assembly (ss09/rhs.py).

Covers state shapes, the v-wall boundary condition, and the symmetric-parity
anchor: from symmetric data (u even, v odd, theta even) the tendencies must
preserve parity exactly (du even, dv odd, dtheta even).
"""

import numpy as np

from ss09 import rhs
from ss09 import grid as ops
from ss09.model_state import ModelState
from ss09.sw_config import SWConfig
from ss09.theta_e import ThetaEConfig, Sin2Profile


def _setup(n=50):
    config = SWConfig(ny=n, total_integration_days=1)
    profile = Sin2Profile(ThetaEConfig(theta_e_type="sin2", y_0=0.0))
    return config, profile


def test_rhs_shapes_are_staggered():
    config, profile = _setup()
    n = config.ny
    u = np.zeros(n)
    v = np.zeros(n + 1)
    theta = profile(ModelState(0.0, u, v, np.zeros(n), config.y))
    du, dv, dtheta = rhs.rhs(ModelState(0.0, u, v, theta, config.y), config, profile)
    assert du.shape == (n,)
    assert dv.shape == (n + 1,)
    assert dtheta.shape == (n,)


def test_rhs_v_wall_tendency_is_zero():
    config, profile = _setup()
    n = config.ny
    rng = np.random.default_rng(1)
    u = rng.standard_normal(n)
    v = rng.standard_normal(n + 1)
    v[0] = 0.0
    v[-1] = 0.0
    theta = profile(ModelState(0.0, u, v, np.zeros(n), config.y)) + rng.standard_normal(n)
    _, dv, _ = rhs.rhs(ModelState(0.0, u, v, theta, config.y), config, profile)
    # boundary faces do not evolve -> v stays exactly 0 at the walls
    assert dv[0] == 0.0
    assert dv[-1] == 0.0


def test_rhs_preserves_symmetric_parity():
    config, profile = _setup(n=50)
    yc, yf = config.y, config.yf
    L = config.domain_size

    u = np.cos(np.pi * yc / L)               # even about the equator
    theta = profile(ModelState(0.0, u, np.zeros(config.ny + 1), np.zeros(config.ny), yc))
    theta = theta + 3.0 * np.cos(np.pi * yc / L)  # still even
    v = np.sin(2 * np.pi * yf / L)           # odd, vanishes on the walls

    du, dv, dtheta = rhs.rhs(ModelState(0.0, u, v, theta, yc), config, profile)

    # du, dtheta even on centers: f[i] == f[N-1-i]
    np.testing.assert_allclose(du, du[::-1], atol=1e-12)
    np.testing.assert_allclose(dtheta, dtheta[::-1], atol=1e-12)
    # dv odd on faces: g[j] == -g[N-j]
    np.testing.assert_allclose(dv, -dv[::-1], atol=1e-12)


def test_rhs_holds_u_walls_fixed():
    """The full RHS zeros the u tendency on the two boundary centers (u=0 wall
    Dirichlet BC), so u stays at its initial value there."""
    config, profile = _setup()
    n = config.ny
    rng = np.random.default_rng(2)
    u = rng.standard_normal(n)
    v = rng.standard_normal(n + 1)
    v[0] = v[-1] = 0.0
    theta = profile(ModelState(0.0, u, v, np.zeros(n), config.y)) + rng.standard_normal(n)
    du, _, _ = rhs.rhs(ModelState(0.0, u, v, theta, config.y), config, profile)
    assert du[0] == 0.0
    assert du[-1] == 0.0


def test_momentum_diffusion_scales_with_k_u():
    """k_u d^2u/dy^2 is the explicit eddy viscosity on u; off when k_u=0."""
    profile = Sin2Profile(ThetaEConfig(theta_e_type="sin2", y_0=0.0))
    n = 50
    rng = np.random.default_rng(3)
    u = rng.standard_normal(n)
    cfg_off = SWConfig(ny=n, total_integration_days=1, k_u=0.0)
    cfg_on = SWConfig(ny=n, total_integration_days=1, k_u=1e5)
    assert np.all(rhs.momentum_diffusion_u(u, cfg_off) == 0.0)
    diff = rhs.momentum_diffusion_u(u, cfg_on)
    assert np.any(diff != 0.0)
    # linear in k_u
    np.testing.assert_allclose(
        rhs.momentum_diffusion_u(u, SWConfig(ny=n, total_integration_days=1, k_u=2e5)),
        2.0 * diff,
    )


# --------------------------------------------------------------------------
# Vertical-momentum-exchange gating H(theta_E - theta) (vert term in isolation)
# --------------------------------------------------------------------------
def _vert_only_config():
    return SWConfig(ny=50, total_integration_days=1, include_merid_advec_u=False)


def test_vert_advec_half_strength_at_equilibrium():
    config = _vert_only_config()
    profile = Sin2Profile(ThetaEConfig(theta_e_type="sin2", y_0=0.0))
    u = np.full(config.ny, 10.0)
    v = np.sin(np.pi * config.yf / (config.domain_size / 2))  # divergent, v=0 at walls
    theta_e = profile(ModelState(0.0, u, v, np.zeros(config.ny), config.y))
    dvdy = ops.div_f2c(v, config.dy)
    # theta == theta_E -> H(0) = 0.5 -> -0.5 u dv/dy
    adv = rhs.u_advection(u, v, theta_e.copy(), theta_e, config)
    np.testing.assert_allclose(adv, -0.5 * u * dvdy)


def test_vert_advec_active_when_cooler_than_equilibrium():
    config = _vert_only_config()
    profile = Sin2Profile(ThetaEConfig(theta_e_type="sin2", y_0=0.0))
    u = np.full(config.ny, 10.0)
    v = np.sin(np.pi * config.yf / (config.domain_size / 2))
    theta_e = profile(ModelState(0.0, u, v, np.zeros(config.ny), config.y))
    dvdy = ops.div_f2c(v, config.dy)
    # theta < theta_E -> H = 1 -> full strength -u dv/dy
    adv = rhs.u_advection(u, v, theta_e - 5.0, theta_e, config)
    np.testing.assert_allclose(adv, -u * dvdy)


def test_vert_advec_inactive_when_warmer_than_equilibrium():
    config = _vert_only_config()
    profile = Sin2Profile(ThetaEConfig(theta_e_type="sin2", y_0=0.0))
    u = np.full(config.ny, 10.0)
    v = np.sin(np.pi * config.yf / (config.domain_size / 2))
    theta_e = profile(ModelState(0.0, u, v, np.zeros(config.ny), config.y))
    # theta > theta_E -> H = 0 -> inactive
    adv = rhs.u_advection(u, v, theta_e + 5.0, theta_e, config)
    np.testing.assert_allclose(adv, 0.0, atol=1e-15)


# --------------------------------------------------------------------------
# Eddy-momentum flux divergence S = v_d H(u) sgn(y) du/dy
# --------------------------------------------------------------------------
def test_emfd_only_acts_on_westerlies():
    config, _ = _setup()
    ny, y = config.ny, config.y
    eq = ny // 2
    u = np.zeros(ny)
    u[eq - 5:eq + 5] = -5.0                       # easterlies (u < 0)
    u[:eq - 5] = np.linspace(10.0, 2.0, eq - 5)   # SH westerlies
    u[eq + 5:] = np.linspace(2.0, 10.0, ny - (eq + 5))  # NH westerlies
    emfd = rhs.emfd_u(u, config)
    assert np.all(emfd[u < 0] == 0.0)
    # nonzero somewhere on the westerly flanks with shear
    assert np.any(emfd[u > 0] != 0.0)


def test_emfd_heaviside_half_at_zero_u():
    config, _ = _setup()
    ny, y = config.ny, config.y
    k = ny // 2 + 5  # a non-equatorial NH center
    u = (np.arange(ny) - k) * 1.0
    assert u[k] == 0.0
    emfd = rhs.emfd_u(u, config)
    dudy = ops.ddy_center(u, config.dy)
    expected = config.v_d * 0.5 * np.sign(y[k]) * dudy[k]
    assert np.isclose(emfd[k], expected)
    assert not np.isclose(emfd[k], 0.0)  # would be 0 under the old H(0)=0 convention


# --------------------------------------------------------------------------
# Tanh-smoothed EMFD gate (lever A): g(u) = 0.5(1 + tanh(u/u_w)); u_w=0 -> hard H(u)
# --------------------------------------------------------------------------
def test_emfd_gate_width_zero_matches_hard_heaviside():
    """emfd_gate_width=0 (default) reproduces the hard-Heaviside gate exactly."""
    config = SWConfig(ny=50, total_integration_days=1)  # emfd_gate_width defaults to 0
    assert config.emfd_gate_width == 0.0
    rng = np.random.default_rng(21)
    u = rng.standard_normal(config.ny) * 10.0
    expected = (
        config.v_d * rhs._heaviside(u) * np.sign(config.y)
        * ops.ddy_center(u, config.dy)
    )
    np.testing.assert_allclose(rhs.emfd_u(u, config), expected, atol=1e-15)


def test_emfd_tanh_gate_smooths_around_u_zero():
    """width>0 applies g(u)=0.5(1+tanh(u/width)); differs from hard near u=0."""
    width = 5.0
    cfg = SWConfig(ny=50, total_integration_days=1, emfd_gate_width=width)
    y = cfg.y
    u = 1e-5 * y  # linear, crosses zero at the equator, constant slope
    dudy = ops.ddy_center(u, cfg.dy)
    gate = rhs.emfd_u(u, cfg) / (cfg.v_d * np.sign(y) * dudy)
    np.testing.assert_allclose(gate, 0.5 * (1 + np.tanh(u / width)), atol=1e-12)
    # saturating limits: strongly westerly -> 1, strongly easterly -> 0
    assert gate[-1] > 0.99 and gate[0] < 0.01
    # differs from the hard gate only near u=0
    emfd_hard = rhs.emfd_u(u, SWConfig(ny=50, total_integration_days=1))
    near = np.abs(u) < width
    far = np.abs(u) > 10 * width
    assert np.any(np.abs(rhs.emfd_u(u, cfg) - emfd_hard)[near] > 1e-9)
    np.testing.assert_allclose(rhs.emfd_u(u, cfg)[far], emfd_hard[far], atol=1e-9)


def test_emfd_tanh_gate_keyed_on_u_not_theta():
    """The smoothed gate responds to the sign/value of u (not theta): an
    all-westerly column gets near-full strength, an all-easterly column ~none."""
    width = 5.0
    cfg = SWConfig(ny=50, total_integration_days=1, emfd_gate_width=width)
    shape = np.abs(cfg.y) / np.abs(cfg.y).max() * 20.0
    e_west = rhs.emfd_u(shape + 10.0, cfg)   # u in [10, 30] -> gate ~1
    e_east = rhs.emfd_u(-(shape + 10.0), cfg)  # u in [-30, -10] -> gate ~0
    assert np.max(np.abs(e_east)) < 0.05 * np.max(np.abs(e_west))


# --------------------------------------------------------------------------
# Biharmonic hyperdiffusion on u (lever B): -k_u4 d^4u/dy^4
# --------------------------------------------------------------------------
def test_hyperdiffusion_zero_when_k_u4_zero():
    cfg = SWConfig(ny=50, total_integration_days=1)  # k_u4 defaults to 0
    assert cfg.k_u4 == 0.0
    rng = np.random.default_rng(31)
    u = rng.standard_normal(cfg.ny)
    assert np.all(rhs.momentum_hyperdiffusion_u(u, cfg) == 0.0)


def test_hyperdiffusion_linear_in_k_u4():
    n = 50
    rng = np.random.default_rng(32)
    u = rng.standard_normal(n)
    h1 = rhs.momentum_hyperdiffusion_u(u, SWConfig(ny=n, total_integration_days=1, k_u4=1e17))
    assert np.any(h1 != 0.0)
    np.testing.assert_allclose(
        rhs.momentum_hyperdiffusion_u(u, SWConfig(ny=n, total_integration_days=1, k_u4=2e17)),
        2.0 * h1,
    )


def test_hyperdiffusion_damps_2dy_mode():
    """-k_u4 d^4u restores a 2Δy mode (tendency opposes u) and barely touches a
    smooth (quadratic) field."""
    n = 50
    cfg = SWConfig(ny=n, total_integration_days=1, k_u4=1e17)
    interior = slice(2, n - 2)  # away from the Neumann walls
    u_mode = (-1.0) ** np.arange(n)
    h_mode = rhs.momentum_hyperdiffusion_u(u_mode, cfg)
    assert np.all(h_mode[interior] * u_mode[interior] < 0)  # restoring
    u_smooth = (cfg.y / (cfg.domain_size / 2)) ** 2  # quadratic -> d^4 = 0
    h_smooth = rhs.momentum_hyperdiffusion_u(u_smooth, cfg)
    assert np.max(np.abs(h_smooth[interior])) < 1e-3 * np.max(np.abs(h_mode[interior]))


# --------------------------------------------------------------------------
# Optional eddy heat diffusion (rhs_im)
# --------------------------------------------------------------------------
def _state_with_theta_curvature(config, profile):
    u = np.zeros(config.ny)
    v = np.zeros(config.ny + 1)
    theta = profile(ModelState(0.0, u, v, np.zeros(config.ny), config.y))
    return ModelState(0.0, u, v, theta, config.y)


def test_eddy_heat_diffusion_inactive_by_default():
    config, profile = _setup()  # coeff_eddy_heat_diff defaults to 0
    state = _state_with_theta_curvature(config, profile)
    _, _, dtheta = rhs.rhs_im(state, config)
    assert np.all(dtheta == 0.0)


def test_eddy_heat_diffusion_active_when_set():
    config = SWConfig(ny=50, total_integration_days=1, coeff_eddy_heat_diff=1e4)
    profile = Sin2Profile(ThetaEConfig(theta_e_type="sin2", y_0=0.0))
    state = _state_with_theta_curvature(config, profile)
    _, _, dtheta = rhs.rhs_im(state, config)
    assert np.any(dtheta != 0.0)


def test_advection_toggles_change_tendency():
    profile = Sin2Profile(ThetaEConfig(theta_e_type="sin2", y_0=0.0))
    rng = np.random.default_rng(7)
    n = 50
    u = rng.standard_normal(n)
    v = rng.standard_normal(n + 1)
    v[0] = v[-1] = 0.0
    cfg_both = SWConfig(ny=n, total_integration_days=1)
    theta_e = profile(ModelState(0.0, u, v, np.zeros(n), cfg_both.y))
    theta = theta_e + rng.standard_normal(n)
    adv_both = rhs.u_advection(u, v, theta, theta_e, cfg_both)
    adv_none = rhs.u_advection(
        u, v, theta, theta_e,
        SWConfig(ny=n, total_integration_days=1,
                 include_merid_advec_u=False, include_vert_advec_u=False),
    )
    np.testing.assert_allclose(adv_none, 0.0, atol=1e-15)
    assert not np.allclose(adv_both, adv_none)
