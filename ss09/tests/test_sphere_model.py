"""Tests for the spherical model: forcing, operators, identities, physics
properties, integration invariants, restart, output and the command line.

The spherical model works in the coordinate y = a sin(lat) on a grid uniform
in y from pole to pole, with c = cos(lat) = (1 - y^2/a^2)^(1/2). Its operators
(SCIENCE.md, "Spherical geometry"):

    u:     du/dt - v [f - d(c u)/dy] = -G u d(c v)/dy - eps_u u - S
    v:     2 dv/dt + (f + u tan(lat)/a) u = -(g H / T_0) c dT/dy + k_v c d/dy[d(c v)/dy]
    theta: dtheta/dt + (delta Delta_z / H) d(c v)/dy = (theta_E - theta)/tau

with f = (2 Omega / a) y and S = (v_d H(u) sgn(y) / c) d(c^2 u)/dy.

Regular axisymmetric test fields: a vector component is c F(s) with F smooth
in s = sin(lat), so c u = (1 - s^2) F is smooth in y; a scalar is any smooth
function of s. Their exact derivatives below are written out by hand and were
checked against sympy when the tests were written.
"""

import os
import subprocess

import numpy as np
import pytest
import xarray as xr

from ss09.output_path_utils import generate_descriptive_path
from ss09.restart_state import RestartState
from ss09.sw_config import SWConfig
from ss09.sw_model import (
    THETA_TO_TEMP,
    AuxiliaryVars,
    SWModel,
    cwv_integral,
    mc_limited_slope,
    moisture_transport_tendency,
    muscl_mc_du_dy,
    v_divergence_at_centers,
    v_face_laplacian,
    v_faces_to_centers,
)
from ss09.theta_e import SB08Profile, SphericalProfile, Sin2Profile, ThetaEConfig

A = 6.371e6
OMEGA = 7.292115e-5


def sphere_model(ny=81, dt=120, ndays=1, tmp_path=None, theta_e_kwargs=None, **kw):
    out = {}
    if tmp_path is not None:
        out = dict(output_path=str(tmp_path / "out.nc"),
                   restart_output_dir=str(tmp_path))
    cfg = SWConfig(geometry="sphere", ny=ny, dt=dt, total_integration_days=ndays,
                   **out, **kw)
    te = dict(theta_e_type="sphere")
    te.update(theta_e_kwargs or {})
    return SWModel(cfg, SphericalProfile(ThetaEConfig(**te)))


def bits(x):
    return np.ascontiguousarray(x, dtype=np.float64).view(np.uint64)


# --- the spherical forcing profile ------------------------------------------

@pytest.mark.parametrize("lat_0", [0.0, 10.0, -6.5])
def test_spherical_profile_formula(lat_0):
    """theta_E = theta_00 - Delta_y (sin^2(lat) - 2 sin(lat_0) sin(lat)), the
    Schneider and Bordoni (2008) profile, evaluated from the latitude."""
    m = sphere_model(ny=325, theta_e_kwargs=dict(lat_0=lat_0))
    lat = np.arcsin(m.config.y / A)
    phi0 = np.deg2rad(lat_0)
    expected = 330.0 - 50.0 * (np.sin(lat) ** 2 - 2 * np.sin(phi0) * np.sin(lat))
    np.testing.assert_allclose(m.current_theta_e(), expected, rtol=0, atol=1e-11)
    # the equator-to-pole drop is Delta_y for lat_0 = 0
    if lat_0 == 0.0:
        theta_e = m.current_theta_e()
        assert theta_e[162] == 330.0 and theta_e[0] == 280.0 == theta_e[-1]
        assert np.array_equal(theta_e, theta_e[::-1])


@pytest.mark.parametrize(
    "te_kwargs, match",
    [
        (dict(y_0=5e5), "y_0"),
        (dict(y_one=8e6), "y_one"),
        (dict(y_0_seasonal_amp=7e5), "Seasonal"),
    ],
)
def test_spherical_profile_refusals(te_kwargs, match):
    with pytest.raises(ValueError, match=match):
        SphericalProfile(ThetaEConfig(theta_e_type="sphere", **te_kwargs))


def test_spherical_profile_requires_its_type_label():
    with pytest.raises(ValueError, match="theta_e_type='sphere'"):
        SphericalProfile(ThetaEConfig())


@pytest.mark.parametrize("te_kwargs", [dict(lat_0=5.0), dict(radius=3e6)])
def test_beta_plane_profiles_refuse_spherical_parameters(te_kwargs):
    for cls in (Sin2Profile, SB08Profile):
        with pytest.raises(ValueError, match="spherical profile"):
            cls(ThetaEConfig(**te_kwargs))


def test_model_pairs_forcing_with_geometry():
    with pytest.raises(ValueError, match="spherical forcing profile"):
        SWModel(SWConfig(geometry="sphere", ny=41), Sin2Profile(ThetaEConfig()))
    with pytest.raises(ValueError, match="requires geometry='sphere'"):
        SWModel(SWConfig(ny=41),
                SphericalProfile(ThetaEConfig(theta_e_type="sphere")))
    with pytest.raises(ValueError, match="radius"):
        SWModel(SWConfig(geometry="sphere", ny=41),
                SphericalProfile(ThetaEConfig(theta_e_type="sphere", radius=3e6)))


# --- the beta plane is unchanged, bit for bit --------------------------------

def _beta_plane_reference(model, u, v, theta, theta_e):
    """The staggered-grid tendencies as written on main before the spherical
    option (commit 074aa56), transcribed operation for operation."""
    cfg = model.config
    dy = cfg.dy
    vc = v_faces_to_centers(v)
    grad = np.zeros_like(u)
    pos, neg = vc > 0, vc < 0
    grad[1:][pos[1:]] = (u[1:][pos[1:]] - u[:-1][pos[1:]]) / dy
    grad[:-1][neg[:-1]] = (u[1:][neg[:-1]] - u[:-1][neg[:-1]]) / dy
    merid = vc * grad if cfg.include_merid_advec_u else 0
    dv_dy = v_divergence_at_centers(v, dy)
    if cfg.vert_advec_gate == "kinematic":
        gate_v = np.heaviside(dv_dy, 0.5)
    else:
        gate_v = np.heaviside(theta_e - theta, 0.5)
    vert = u * dv_dy * gate_v if cfg.include_vert_advec_u else 0
    gate = np.heaviside(u, 0.5) if cfg.emfd_heaviside_gate else 1.0
    if cfg.emfd_stencil == "upwind":
        diff = (u[1:] - u[:-1]) / dy
        backward = np.zeros_like(u)
        backward[1:] = diff
        forward = np.zeros_like(u)
        forward[:-1] = diff
        du_dy = np.where(cfg.y > 0, backward, forward)
    elif cfg.emfd_stencil == "mc":
        du_dy = muscl_mc_du_dy(u, dy, cfg.y)
    else:
        du_dy = np.gradient(u, dy)
    emfd = cfg.v_d * gate * np.sign(cfg.y) * du_dy
    du = cfg.beta * cfg.y * vc - merid - vert - u * cfg.epsilon_u - emfd
    uf = 0.5 * (u[:-1] + u[1:])
    T = theta * THETA_TO_TEMP
    dt_dy = (T[1:] - T[:-1]) / dy
    dp = cfg.gravity * cfg.height * dt_dy / cfg.t_ref
    coriolis = cfg.beta * cfg.y_v * uf
    dv = (-coriolis - dp + v_face_laplacian(v, dy) * cfg.k_v) / 2
    newt = (theta_e - theta) / cfg.tau
    vadv = -cfg.delta * cfg.delta_z * dv_dy / cfg.height
    if cfg.coeff_eddy_heat_diff == 0.0:
        eddy = np.zeros_like(theta)
    else:
        eddy = cfg.coeff_eddy_heat_diff * np.gradient(np.gradient(theta, dy), dy)
    dth = newt + vadv + eddy
    return du, dv, dth


FLAT_VARIANTS = [
    dict(),
    dict(vert_advec_gate="kinematic", coeff_eddy_heat_diff=1.0e5),
    dict(emfd_stencil="upwind"),
    dict(emfd_stencil="centered", emfd_heaviside_gate=False),
    dict(include_merid_advec_u=False, include_vert_advec_u=False),
]


@pytest.mark.parametrize("variant", FLAT_VARIANTS, ids=range(len(FLAT_VARIANTS)))
def test_beta_plane_tendencies_unchanged_bit_for_bit(variant):
    """With the geometry arrays at their flat values, every tendency equals
    the pre-sphere beta-plane formula bit for bit, on random states with
    exact and signed zeros planted (the H(0) = 0.5 branches)."""
    rng = np.random.default_rng(20261108)
    cfg = SWConfig(ny=101, **variant)
    model = SWModel(cfg, Sin2Profile(ThetaEConfig()))
    theta_e = model.current_theta_e()
    for _ in range(20):
        u = 30.0 * rng.standard_normal(cfg.ny)
        u[[0, -1]] = 0.0
        u[rng.integers(1, cfg.ny - 1, 5)] = 0.0
        u[rng.integers(1, cfg.ny - 1, 3)] = -0.0
        v = rng.standard_normal(cfg.nv)
        v[rng.integers(0, cfg.nv, 4)] = 0.0
        v[rng.integers(0, cfg.nv, 2)] = -0.0
        theta = theta_e + 3.0 * rng.standard_normal(cfg.ny)
        theta[rng.integers(0, cfg.ny, 3)] = theta_e[rng.integers(0, cfg.ny, 3)]
        model.state = model.state._replace(u=u, v=v, theta=theta)
        du, dv, dth = _beta_plane_reference(model, u, v, theta, theta_e)
        assert np.array_equal(bits(model.du_dt()), bits(du))
        assert np.array_equal(bits(model.dv_dt()), bits(dv))
        assert np.array_equal(bits(model.dtheta_dt()), bits(dth))


def test_beta_plane_moisture_flux_unchanged_bit_for_bit():
    rng = np.random.default_rng(3)
    dy = 39377.5
    for _ in range(20):
        ny = int(rng.integers(5, 200))
        w = 45.0 + 10.0 * rng.standard_normal(ny)
        w_lag = w + 0.1 * rng.standard_normal(ny)
        v = rng.standard_normal(ny - 1)
        v[rng.integers(0, ny - 1, 2)] = 0.0
        c_f = -(2.0 * 0.79 - 1.0) * v
        sigma = mc_limited_slope(w, dy)
        w_face = np.where(c_f > 0, w[:-1] + 0.5 * dy * sigma[:-1],
                          w[1:] - 0.5 * dy * sigma[1:])
        flux = c_f * w_face - 1.0e6 * (w_lag[1:] - w_lag[:-1]) / dy
        expected = -v_divergence_at_centers(flux, dy)
        for cos_face in (1.0, np.ones(ny - 1)):
            got = moisture_transport_tendency(w, w_lag, v, 0.79, 1.0e6, dy,
                                              cos_face=cos_face)
            assert np.array_equal(bits(got), bits(expected))


# --- accuracy of the spherical operators -------------------------------------

U0, V0, DTH = 30.0, 2.0, 40.0


def _test_fields(model):
    """u = c U s e^s, v = c V s e^(-s/2), theta = 300 - DTH s^2 e^(s/2), and
    the exact value of each operator on them."""
    cfg, g = model.config, model.config.geom
    s, sv = cfg.y / A, cfg.y_v / A
    c, cv = g.cos_lat, g.cos_lat_face
    u = c * U0 * s * np.exp(s)
    v = cv * V0 * sv * np.exp(-sv / 2)
    theta = 300.0 - DTH * s**2 * np.exp(s / 2)
    vc = c * V0 * s * np.exp(-s / 2)
    exact = {
        "divergence": V0 * np.exp(-s / 2) * (1 - s / 2 - 3 * s**2 + s**3 / 2) / A,
        "v at centers": vc,
        "meridional advection": vc * U0 * (1 - 3 * s**2 + s - s**3) * np.exp(s) / A,
        "smoothing": cv * V0 * np.exp(-sv / 2)
        * (-1 - 5.75 * sv + 3 * sv**2 - sv**3 / 4) / A**2,
        "pressure gradient": cfg.gravity * cfg.height / cfg.t_ref * cv * THETA_TO_TEMP
        * (-DTH * np.exp(sv / 2) * (2 * sv + sv**2 / 2)) / A,
    }
    return u, v, theta, exact


def _numerical(model, name):
    if name == "divergence":
        return model.dv_dy(), model.config.geom.lat
    if name == "v at centers":
        return model._v_at_centers(), model.config.geom.lat
    if name == "meridional advection":
        return model.merid_advec_u(), model.config.geom.lat
    if name == "smoothing":
        return model.diffusion_v() / model.config.k_v, model.config.geom.lat_face
    # the pressure gradient: dv_dt with u = 0 and k_v = 0 is -(pressure term)/2
    model.state = model.state._replace(u=np.zeros_like(model.state.u))
    model.config.k_v = 0.0
    return -2.0 * model.dv_dt(), model.config.geom.lat_face


# Measured orders on 2026-11-08 at ny = 161, 321, 641 (the two successive
# halvings of dy), the maximum error scaled by max|exact|:
#   divergence           whole domain 1.00, 1.00   within 60 deg 2.00, 1.99
#   v at centers         0.54, 0.52                2.00, 1.96
#   meridional advection 0.85, 0.72                1.00, 1.00
#   smoothing            0.50, 0.50                2.00, 2.00
#   pressure gradient    2.00, 2.00                2.00, 2.00
# Within 60 degrees each operator converges at its design order (second, and
# first for the upwind advection). Over the whole domain the half-cell
# divergence at the poles is first order, and the operators acting on odd
# fields (the v average) or on the pole half-cell divergence (the smoothing)
# lose accuracy in the outermost cells, as the plan's polar-cells section
# anticipates. The thresholds sit just below the measured orders.
ORDER_BOUNDS = {
    "divergence": (0.95, 1.9),
    "v at centers": (0.45, 1.9),
    "meridional advection": (0.65, 0.95),
    "smoothing": (0.45, 1.9),
    "pressure gradient": (1.9, 1.9),
}


@pytest.mark.parametrize("name", list(ORDER_BOUNDS))
def test_spherical_operator_convergence(name):
    errs_all, errs_60 = [], []
    for ny in (161, 321, 641):
        model = sphere_model(ny=ny)
        u, v, theta, exact = _test_fields(model)
        model.state = model.state._replace(u=u, v=v, theta=theta)
        num, lat = _numerical(model, name)
        err = np.abs(num - exact[name]) / np.max(np.abs(exact[name]))
        errs_all.append(np.max(err))
        errs_60.append(np.max(err[np.abs(lat) <= 60.0]))
    orders_all = np.log2(np.array(errs_all[:-1]) / np.array(errs_all[1:]))
    orders_60 = np.log2(np.array(errs_60[:-1]) / np.array(errs_60[1:]))
    lo_all, lo_60 = ORDER_BOUNDS[name]
    assert np.all(orders_all >= lo_all), (name, orders_all)
    assert np.all(orders_60 >= lo_60), (name, orders_60)


def _heat_and_moisture(model, name):
    """Numerical and exact values of the eddy heat diffusion, kappa
    d/dy(c^2 d(theta)/dy), on theta = 300 - DTH s^2 e^(s/2), and of the
    moisture transport tendency, -d/dy{c [c_W W - D c dW/dy]} with
    c_W = -(2 a_W - 1) v, on W = 40 + 10 e^s and v = c V s e^(-s/2)."""
    cfg, g = model.config, model.config.geom
    s = cfg.y / A
    if name == "eddy heat diffusion":
        model.state = model.state._replace(theta=300.0 - DTH * s**2 * np.exp(s / 2))
        num = model.eddy_heat_flux() / cfg.coeff_eddy_heat_diff
        exact = (DTH / 4 * (s**4 + 12 * s**3 + 23 * s**2 - 8 * s - 8)
                 * np.exp(s / 2) / A**2)
        return num, exact
    a_w, d_w = 0.79, 1.0e6
    w = 40.0 + 10.0 * np.exp(s)
    v = g.cos_lat_face * V0 * (cfg.y_v / A) * np.exp(-cfg.y_v / A / 2)
    num = moisture_transport_tendency(w, w, v, a_w, d_w, cfg.dy,
                                      cos_face=g.cos_lat_face)
    k = -(2 * a_w - 1)
    W, dW, d2W = w, 10.0 * np.exp(s), 10.0 * np.exp(s)
    F = V0 * s * np.exp(-s / 2)
    dF = V0 * np.exp(-s / 2) * (1 - s / 2)
    bracket = k * F * W - d_w / A * dW
    d_bracket = k * (dF * W + F * dW) - d_w / A * d2W
    exact = -(-2 * s * bracket + (1 - s**2) * d_bracket) / A
    return num, exact


@pytest.mark.parametrize("name", ["eddy heat diffusion", "moisture transport"])
def test_spherical_heat_and_moisture_convergence(name):
    """Measured 2026-11-08 at ny = 161, 321, 641: second order within 60
    degrees (2.00, 1.99 for both) and first order over the whole domain
    (0.99, 1.00 and 1.00, 1.00), where the poles' one-sided forms set the
    error. The moisture operator is tested directly, since moist runs on the
    sphere are not yet enabled."""
    errs_all, errs_60 = [], []
    for ny in (161, 321, 641):
        model = sphere_model(ny=ny, coeff_eddy_heat_diff=1.0e5)
        num, exact = _heat_and_moisture(model, name)
        err = np.abs(num - exact) / np.max(np.abs(exact))
        errs_all.append(np.max(err))
        errs_60.append(np.max(err[np.abs(model.config.geom.lat) <= 60.0]))
    orders_all = np.log2(np.array(errs_all[:-1]) / np.array(errs_all[1:]))
    orders_60 = np.log2(np.array(errs_60[:-1]) / np.array(errs_60[1:]))
    assert np.all(orders_all >= 0.95), (name, orders_all)
    assert np.all(orders_60 >= 1.9), (name, orders_60)


def test_spherical_eddy_term_accuracy():
    """Form (c), gate off so it acts in both hemispheres, on u = c U s e^s,
    whose exact value is v_d sgn(y) (U/a) e^s (1 + s - 4 s^2 - s^3).

    Measured 2026-11-08 at ny = 161, 321, 641. Within 60 degrees the mean
    error falls as 6.3e-5, 1.3e-5, 4.9e-6 of max|S|. The maximum there does
    not converge regularly (2.0e-3, 9.5e-4, 9.2e-4), because the MC limiter
    reverts toward first order at the extremum of c^2 u near 36 degrees and
    its error there depends on where the extremum falls within a cell; the
    plain beta-plane stencil on the same q shows the same error at the same
    point. Over the whole domain the maximum error sits in the outermost
    interior cells at 0.83%, 0.85%, 0.87% of max|S| and does not shrink: c^2 u
    is odd in colatitude, and 1/c magnifies the stencil's error there.
    """
    means, maxes = [], []
    for ny in (161, 321, 641):
        model = sphere_model(ny=ny, emfd_heaviside_gate=False)
        cfg, g = model.config, model.config.geom
        s = cfg.y / A
        model.state = model.state._replace(u=g.cos_lat * U0 * s * np.exp(s))
        exact = (cfg.v_d * np.sign(cfg.y) * U0 * np.exp(s)
                 * (1 + s - 4 * s**2 - s**3) / A)
        exact[[0, -1]] = 0.0
        err = np.abs(model.edd_mom_flux_div_u() - exact) / np.max(np.abs(exact))
        means.append(np.mean(err[np.abs(g.lat) <= 60.0]))
        maxes.append(np.max(err))
    orders = np.log2(np.array(means[:-1]) / np.array(means[1:]))
    assert np.all(orders >= 1.2), orders
    assert max(maxes) < 0.012, maxes


def test_metric_term_and_coriolis_in_v_equation():
    """With theta uniform and v = 0 (no pressure gradient, no smoothing),
    2 dv/dt = -(f + u tan(lat)/a) u at the faces, with u the face average."""
    model = sphere_model(ny=81)
    g = model.config.geom
    rng = np.random.default_rng(5)
    u = 20.0 * rng.standard_normal(model.config.ny)
    u[[0, -1]] = 0.0
    model.state = model.state._replace(
        u=u, v=np.zeros(model.config.nv), theta=np.full(model.config.ny, 300.0)
    )
    uf = 0.5 * (u[:-1] + u[1:])
    lat_face = np.arcsin(model.config.y_v / A)
    expected = -(2 * OMEGA * np.sin(lat_face) + uf * np.tan(lat_face) / A) * uf / 2
    np.testing.assert_allclose(model.dv_dt(), expected, rtol=1e-12, atol=1e-20)


# --- exact discrete identities -----------------------------------------------

def test_smoothing_is_the_gradient_of_the_divergence_and_dissipative():
    """c L[c v] = c (div[c v]_{k+1} - div[c v]_k)/dy at every face, and the
    sum over faces of v times it is minus the equal-area sum of div^2 (half
    weights at the pole half cells): the smoothing never raises sum(v^2)."""
    model = sphere_model(ny=325)
    cfg, g = model.config, model.config.geom
    rng = np.random.default_rng(11)
    v = rng.standard_normal(cfg.nv)
    model.state = model.state._replace(v=v)
    K = model.diffusion_v() / cfg.k_v
    D = model.dv_dy()
    np.testing.assert_allclose(
        K, g.cos_lat_face * (D[1:] - D[:-1]) / cfg.dy, rtol=0,
        atol=1e-12 * np.max(np.abs(K)),
    )
    lhs = np.sum(v * K)
    rhs = -(np.sum(D[1:-1] ** 2) + 0.5 * D[0] ** 2 + 0.5 * D[-1] ** 2)
    assert lhs < 0.0
    assert abs(lhs - rhs) <= 1e-13 * abs(rhs)


def test_spherical_moisture_transport_conserves_water():
    model = sphere_model(ny=325)
    cfg, g = model.config, model.config.geom
    rng = np.random.default_rng(12)
    w = 45.0 + 10.0 * rng.random(cfg.ny)
    tend = moisture_transport_tendency(
        w, w + 0.1 * rng.standard_normal(cfg.ny), 5.0 * rng.standard_normal(cfg.nv),
        0.79, 1.0e6, cfg.dy, cos_face=g.cos_lat_face,
    )
    assert abs(cwv_integral(tend, cfg.dy)) <= 1e-14 * cwv_integral(np.abs(tend), cfg.dy)
    # the spherical flux really differs from the beta-plane one here
    flat = moisture_transport_tendency(
        w, w + 0.1 * rng.standard_normal(cfg.ny), 5.0 * rng.standard_normal(cfg.nv),
        0.79, 1.0e6, cfg.dy,
    )
    assert not np.allclose(tend, flat)


def test_eddy_torque_reduces_to_boundary_face_values():
    """Form (c) conserves angular momentum: over the northern westerlies of
    u = U s c, the equal-area sum of c S equals v_d times the difference of
    the outermost and innermost left-reconstructed face values of c^2 u. The
    literal transcription, v_d sgn(y) c du/dy, fails the same identity."""
    model = sphere_model(ny=325)
    cfg, g = model.config, model.config.geom
    j0 = cfg.ny // 2
    s = cfg.y / A
    u = 40.0 * s * g.cos_lat
    model.state = model.state._replace(u=u)
    S = model.edd_mom_flux_div_u()
    q = g.cos2_lat * u
    q_left = q + 0.5 * cfg.dy * mc_limited_slope(q, cfg.dy)
    nh = slice(j0 + 1, cfg.ny - 1)
    torque = np.sum(g.cos_lat[nh] * S[nh]) * cfg.dy
    expected = cfg.v_d * (q_left[cfg.ny - 2] - q_left[j0])
    scale = np.sum(np.abs(g.cos_lat[nh] * S[nh])) * cfg.dy
    assert abs(torque - expected) <= 1e-12 * scale
    # positive control: the literal form
    S_lit = cfg.v_d * np.sign(cfg.y) * g.cos_lat * muscl_mc_du_dy(u, cfg.dy, cfg.y)
    torque_lit = np.sum(g.cos_lat[nh] * S_lit[nh]) * cfg.dy
    u_left = u + 0.5 * cfg.dy * mc_limited_slope(u, cfg.dy)
    expected_lit = cfg.v_d * (u_left[cfg.ny - 2] - u_left[j0])
    assert abs(torque_lit - expected_lit) > 1e-3 * scale


# --- physics properties -------------------------------------------------------

def test_amc_wind_is_steady_up_to_the_upwind_truncation():
    """The angular-momentum-conserving wind u = Omega a sin^2/cos makes
    f + zeta = 0. Discretely c u = Omega y^2/a, whose first-order upwind
    difference is f - (Omega/a) dy sgn(v) exactly, so with v_d = 0, no
    vertical advection and no drag the u tendency is |v| Omega dy / a, a
    fraction dy/(2|y|) of the Coriolis term. The beta-plane advection of u
    rather than c u leaves a residual comparable to the Coriolis term."""
    model = sphere_model(ny=81, v_d=0.0, include_vert_advec_u=False, epsilon_u=0.0)
    cfg, g = model.config, model.config.geom
    s = cfg.y / A
    u = np.zeros(cfg.ny)
    u[1:-1] = OMEGA * A * s[1:-1] ** 2 / g.cos_lat[1:-1]
    v = 3.0 * np.sin(2 * np.pi * cfg.y_v / A)  # both signs
    model.state = model.state._replace(u=u, v=v)
    du = model.du_dt()
    vc = v_faces_to_centers(v)
    inner = slice(2, cfg.ny - 2)
    np.testing.assert_allclose(
        du[inner], np.abs(vc[inner]) * OMEGA * cfg.dy / A, rtol=1e-8, atol=1e-20
    )
    # positive control: advecting u itself, as on the beta plane
    grad = np.zeros_like(u)
    pos, neg = vc > 0, vc < 0
    grad[1:][pos[1:]] = (u[1:][pos[1:]] - u[:-1][pos[1:]]) / cfg.dy
    grad[:-1][neg[:-1]] = (u[1:][neg[:-1]] - u[:-1][neg[:-1]]) / cfg.dy
    residual_flat = g.f * vc - vc * grad
    mid = np.abs(g.lat) < 60
    assert np.max(np.abs(residual_flat[mid])) > 30 * np.max(np.abs(du[mid]))


def _gradient_wind_state(model, Ro=0.3):
    """A uniform-Ro wind u = Ro Omega a sin^2/cos and the temperature in
    full-angle gradient-wind balance with it (the balance written in
    puffins.grad_bal.pot_temp_avg_unif_ro), cut off poleward of 80 degrees
    where both are singular."""
    cfg, g = model.config, model.config.geom
    c = g.cos_lat
    ok = c > np.cos(np.deg2rad(80.0))
    s = cfg.y / A
    u = np.where(ok, Ro * OMEGA * A * s**2 / np.where(ok, c, 1.0), 0.0)
    cc = np.where(ok, c, 1.0)
    G = (2 - Ro) * cc**2 + 4 * (1 - Ro) * np.log(1 / cc) + Ro / cc**2 - 2
    T = -cfg.t_ref * Ro * OMEGA**2 * A**2 / (2 * cfg.gravity * cfg.height) * G
    theta = 300.0 + np.where(ok, T, 0.0) / THETA_TO_TEMP
    return u, theta


def test_steady_v_equation_is_full_angle_gradient_wind_balance():
    """For the balanced uniform-Ro state the v tendency vanishes to second
    order within 60 degrees, relative to the Coriolis-plus-metric term;
    without the metric term it does not vanish.

    Measured 2026-11-08 at ny = 161, 321, 641: the residual is 1.7e-4,
    5.2e-5, 1.3e-5 of the Coriolis-plus-metric term, orders 1.75 and 2.04
    (the first halving is not yet asymptotic)."""
    rel, rel_nometric = [], []
    for ny in (161, 321, 641):
        model = sphere_model(ny=ny)
        g = model.config.geom
        u, theta = _gradient_wind_state(model)
        model.state = model.state._replace(
            u=u, v=np.zeros(model.config.nv), theta=theta
        )
        dv = model.dv_dt()
        uf = 0.5 * (u[:-1] + u[1:])
        coriolis = (g.f_face * uf + g.metric_face * uf * uf) / 2
        mid = np.abs(g.lat_face) <= 60
        scale = np.max(np.abs(coriolis[mid]))
        rel.append(np.max(np.abs(dv[mid])) / scale)
        rel_nometric.append(
            np.max(np.abs((dv + g.metric_face * uf * uf / 2)[mid])) / scale
        )
    orders = np.log2(np.array(rel[:-1]) / np.array(rel[1:]))
    assert np.all(orders >= 1.6) and orders[-1] >= 1.9, (rel, orders)
    assert min(rel_nometric) > 100 * max(rel), (rel_nometric, rel)


def test_rossby_number_of_a_uniform_ro_wind():
    """The diagnosed local Rossby number, -zeta/f with the spherical zeta,
    returns the Ro of a uniform-Ro wind within 60 degrees."""
    model = sphere_model(ny=325)
    g = model.config.geom
    u, _ = _gradient_wind_state(model, Ro=0.4)
    ro = model.hadley_diagnostics.compute_rossby_number(
        u, model.config.y, model.config.dy, model.config.beta,
        f=g.f, cos_lat=g.cos_lat,
    )
    mid = (np.abs(g.lat) <= 60) & np.isfinite(ro)
    assert np.sum(mid) > 200
    np.testing.assert_allclose(ro[mid], 0.4, rtol=0, atol=1e-3)


# --- integration invariants ---------------------------------------------------

def _leapfrog_steps(model, n_steps):
    model.vars_prev_step = AuxiliaryVars(*model.init_prev_step_vars())
    for i in range(n_steps):
        model.state = model.state._replace(t=i * model.config.dt)
        nu = model.leapfrog_step(model.vars_prev_step.u, model.du_dt)
        nv = model.leapfrog_step(model.vars_prev_step.v, model.dv_dt)
        nth = model.leapfrog_step(model.vars_prev_step.theta, model.dtheta_dt)
        model.vars_prev_step = AuxiliaryVars(
            model.asselin_filt(model.vars_prev_step.u, nu, model.state.u),
            model.asselin_filt(model.vars_prev_step.v, nv, model.state.v),
            model.asselin_filt(model.vars_prev_step.theta, nth, model.state.theta),
        )
        model.state = model.state._replace(u=nu, v=nv, theta=nth)
        model.enforce_boundary_conditions()


def test_sphere_mirror_parity_bitexact():
    """From rest under the hemispherically symmetric forcing (lat_0 = 0), at
    the production physics, two days of steps keep u and theta exactly
    symmetric and v exactly antisymmetric; lat_0 = 5 breaks the symmetry
    (the positive control)."""
    model = sphere_model(ny=81, dt=120)
    _leapfrog_steps(model, 1440)
    u, v, theta = model.state.u, model.state.v, model.state.theta
    assert np.all(np.isfinite(u)) and np.max(np.abs(v)) > 1.0
    assert np.max(np.abs(u - u[::-1])) == 0.0
    assert np.max(np.abs(theta - theta[::-1])) == 0.0
    assert np.max(np.abs(v + v[::-1])) == 0.0
    off = sphere_model(ny=81, dt=120, theta_e_kwargs=dict(lat_0=5.0))
    _leapfrog_steps(off, 100)
    assert np.max(np.abs(off.state.u - off.state.u[::-1])) > 0.0


def test_sphere_restart_continuation_bitwise(tmp_path):
    straight = sphere_model(ny=41, dt=300, ndays=4, tmp_path=tmp_path / "straight")
    straight.run_sim()
    first = sphere_model(ny=41, dt=300, ndays=2, tmp_path=tmp_path / "split")
    first.run_sim()
    cont = sphere_model(ny=41, dt=300, ndays=4, tmp_path=tmp_path / "split")
    cont.restart_day = cont.load_from_restart(str(tmp_path / "split" / "restart_day0002.nc"))
    cont.run_sim()
    assert np.count_nonzero(straight.results.time) == 4
    for day in (2, 3):
        for name in ("u", "v", "theta"):
            assert np.array_equal(
                getattr(straight.results, name)[day], getattr(cont.results, name)[day]
            ), (name, day)
    assert np.array_equal(straight.state.u, cont.state.u)
    assert np.array_equal(straight.vars_prev_step.v, cont.vars_prev_step.v)


def test_restart_refuses_a_geometry_mismatch(tmp_path):
    flat = SWModel(
        SWConfig(ny=41, dt=1800, total_integration_days=1,
                 output_path=str(tmp_path / "flat" / "out.nc"),
                 restart_output_dir=str(tmp_path / "flat")),
        Sin2Profile(ThetaEConfig()),
    )
    flat.run_sim()
    flat_file = str(tmp_path / "flat" / "restart_day0001.nc")
    sphere = sphere_model(ny=41, dt=300, ndays=2, tmp_path=tmp_path / "sphere")
    with pytest.raises(ValueError, match="Geometry mismatch"):
        sphere.load_from_restart(flat_file)
    # and a sphere restart at another radius
    sphere.run_sim()
    sphere_file = str(tmp_path / "sphere" / "restart_day0002.nc")
    other = SWModel(
        SWConfig(geometry="sphere", ny=41, dt=300, total_integration_days=3,
                 radius=3.0e6),
        SphericalProfile(ThetaEConfig(theta_e_type="sphere", radius=3.0e6)),
    )
    with pytest.raises(ValueError, match="radius"):
        other.load_from_restart(sphere_file)
    state = RestartState.from_netcdf(sphere_file)
    assert state.config_snapshot["geometry"] == "sphere"


# --- output -------------------------------------------------------------------

def test_sphere_output_has_latitudes(tmp_path):
    model = sphere_model(ny=41, dt=300, ndays=2, tmp_path=tmp_path)
    model.run_sim()
    model.save_results()
    ds = xr.open_dataset(model.config.output_path, decode_times=False)
    assert "sin(latitude)" in ds.y.attrs["long_name"]
    np.testing.assert_allclose(ds.lat.values, np.rad2deg(np.arcsin(ds.y.values / A)),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        ds.lat_face.values, np.rad2deg(np.arcsin(ds.y_face.values / A)),
        rtol=0, atol=1e-12,
    )
    for name in ("north_jet_lat", "south_jet_lat", "north_cell_center_lat",
                 "ascending_edge_lat"):
        y_pos = ds[name].values
        np.testing.assert_allclose(
            ds[f"{name}_deg"].values,
            np.rad2deg(np.arcsin(np.clip(y_pos / A, -1, 1))), rtol=0, atol=1e-12,
        )
        assert "not distance" in ds[name].attrs["description"]
    assert ds.attrs["geometry"] == "sphere"
    assert ds.attrs["theta_e_theta_e_type"] == "sphere"
    ds.close()


def test_beta_plane_output_has_no_sphere_additions(tmp_path):
    """Written after a spherical output in the same process, a beta-plane
    output carries no latitude variables and its diagnostics' descriptions
    are untouched (the module-level attribute dicts are not mutated)."""
    sphere = sphere_model(ny=41, dt=300, ndays=1, tmp_path=tmp_path / "s")
    sphere.run_sim()
    sphere.save_results()
    flat = SWModel(
        SWConfig(ny=41, dt=1800, total_integration_days=1,
                 output_path=str(tmp_path / "f" / "out.nc"),
                 restart_output_dir=str(tmp_path / "f")),
        Sin2Profile(ThetaEConfig()),
    )
    flat.run_sim()
    flat.save_results()
    ds = xr.open_dataset(flat.config.output_path, decode_times=False)
    assert "lat" not in ds.coords and "north_jet_lat_deg" not in ds
    assert "not distance" not in ds["north_jet_lat"].attrs["description"]
    assert ds.y.attrs["long_name"] == "meridional distance from equator"
    ds.close()


def test_sphere_descriptive_output_path():
    cfg = SWConfig(geometry="sphere", ny=325, total_integration_days=100)
    te = ThetaEConfig(theta_e_type="sphere", lat_0=6.5)
    out, rdir = generate_descriptive_path(cfg, te, base_dir="mo", timestamp="T")
    assert out == os.path.join("mo", "sphere", "run_T_noseas_lat0p06.5_ny325_100days_output.nc")
    te = ThetaEConfig(theta_e_type="sphere", lat_0=-10.0)
    out, _ = generate_descriptive_path(cfg, te, base_dir="mo", timestamp="T")
    assert "lat0n10.0" in out


# --- command line -------------------------------------------------------------

def test_cli_sphere_run(tmp_path):
    out = str(tmp_path / "sphere.nc")
    proc = subprocess.run(
        ["run-sw-model", "--geometry", "sphere", "--ny", "41", "--dt", "300",
         "--ndays", "1", "--lat-0", "4", "--output-path", out],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    ds = xr.open_dataset(out, decode_times=False)
    assert ds.attrs["geometry"] == "sphere"
    assert ds.attrs["theta_e_lat_0"] == 4.0
    assert ds.sizes["y"] == 41 and "lat" in ds.coords
    ds.close()


@pytest.mark.parametrize(
    "args, message",
    [
        (["--geometry", "sphere", "--beta", "2e-11"], "--beta"),
        (["--geometry", "sphere", "--domain-size", "1e7"], "--domain-size"),
        (["--geometry", "sphere", "--y0", "5e5"], "--y0"),
        (["--geometry", "sphere", "--y1", "8e6"], "--y1"),
        (["--geometry", "sphere", "--theta-e-type", "sin2"], "sin2"),
        (["--lat-0", "5"], "--lat-0"),
        (["--radius", "3e6"], "--radius"),
        (["--theta-e-type", "sphere"], "--theta-e-type sphere"),
    ],
)
def test_cli_geometry_refusals(args, message):
    proc = subprocess.run(
        ["run-sw-model", "--ny", "41", "--ndays", "1"] + args,
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert message in proc.stderr + proc.stdout
