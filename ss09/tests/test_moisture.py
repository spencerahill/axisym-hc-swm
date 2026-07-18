"""Tests for the moist V1 passive column-water-vapor extension.

V1 (guides/moist_axisymmetric_model_spec.pdf Eq. Q1) adds a prognostic
column water vapor W(y, t) riding passively on the dry circulation:

    dW/dt + d/dy[-(2a-1) v W - D dW/dy] = E_0 - P,   P = (W - W_c)^+ / tau_c

W lives on the ny centers; the transport is finite-volume in flux form with
fluxes on the ny-1 interior faces where the staggered v already lives. Two
hard invariants define success: (1) with enable_moisture on, the dry fields
u, v, theta are bit-for-bit identical to the same run with it off (W is
one-way coupled); (2) with it off, the dry code path is untouched (guarded
by the existing regression baselines).
"""

from typing import Any

import numpy as np
import pytest

from ss09.cli import parse_arguments, setup_sw_config, setup_theta_e_config
from ss09.sw_config import SWConfig
from ss09.sw_model import (
    StaggeredSWModel,
    SWModel,
    cwv_integral,
    mc_face_values,
    moisture_transport_tendency,
    precipitation,
)
from ss09.theta_e import Sin2Profile, ThetaEConfig


def _sin2_profile(y_0=0.0):
    return Sin2Profile(
        ThetaEConfig(
            theta_00=330.0, y_0=y_0, y_one=9439e3, delta_y=50.0,
            theta_e_type="sin2",
        )
    )


def _even_centers(ny, seed):
    """A center array w (length ny) with w[i] == w[ny-1-i] exactly."""
    rng = np.random.default_rng(seed)
    half = rng.uniform(30.0, 60.0, ny // 2)
    if ny % 2 == 0:
        return np.concatenate([half, half[::-1]])
    return np.concatenate([half, [45.0], half[::-1]])


def _odd_faces(nf, seed):
    """A face array f (length nf) with f[j] == -f[nf-1-j] exactly."""
    rng = np.random.default_rng(seed)
    half = rng.normal(0.0, 5.0, nf // 2)
    if nf % 2 == 0:
        return np.concatenate([half, -half[::-1]])
    return np.concatenate([half, [0.0], -half[::-1]])


# --- Config validation -----------------------------------------------------

def _moist_config(**overrides):
    """A valid moist configuration on the production (staggered) grid."""
    kwargs: dict[str, Any] = dict(
        total_integration_days=2,
        ny=51,
        dt=1800,
        enable_moisture=True,
    )
    kwargs.update(overrides)
    return SWConfig(**kwargs)


def test_config_enable_moisture_accepted_on_defaults():
    config = _moist_config()
    assert config.enable_moisture is True
    # placeholder defaults from the V1 plan (ERA5-calibrated values pending)
    assert config.cwv_frac == 0.85
    assert config.d_w == 1.0e6
    assert config.w_crit == 50.0
    assert config.tau_c == 14400.0
    assert config.evap == 4.6e-5
    assert config.w_init is None  # None means "use w_crit"


@pytest.mark.parametrize(
    "param, value",
    [
        ("cwv_frac", 0.9),
        ("d_w", 2.0e6),
        ("w_crit", 40.0),
        ("tau_c", 7200.0),
        ("evap", 5.0e-5),
        ("w_init", 45.0),
    ],
)
def test_config_moist_param_without_enable_moisture_raises(param, value):
    """Repo style: passing a moist parameter without enable_moisture is a
    hard error, never a silent no-op."""
    with pytest.raises(ValueError, match="enable_moisture"):
        SWConfig(total_integration_days=2, ny=51, dt=1800, **{param: value})


def test_config_enable_moisture_rejects_numba_backend():
    """The numba kernel does not mirror the moisture step yet (deferred until
    the V1 physics is frozen)."""
    with pytest.raises(ValueError, match="numba"):
        _moist_config(backend="numba")


def test_config_enable_moisture_rejects_collocated_grid():
    """The collocated layout is the frozen Zhang et al. (2025) reproduction
    path; moisture assumes face-v fluxes."""
    with pytest.raises(ValueError, match="collocated"):
        _moist_config(grid="collocated")


# --- CLI -------------------------------------------------------------------

def test_cli_enable_moisture_flag(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["run-sw-model", "--enable-moisture", "--dt", "1800"],
    )
    args = parse_arguments()
    config = setup_sw_config(args, setup_theta_e_config(args))
    assert config.enable_moisture is True
    assert config.w_crit == 50.0


def test_cli_moist_values_reach_config(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run-sw-model", "--enable-moisture", "--dt", "1800",
            "--cwv-frac", "0.9", "--dw", "2e6", "--w-crit", "45",
            "--tau-c", "7200", "--evap", "5e-5", "--w-init", "40",
        ],
    )
    args = parse_arguments()
    config = setup_sw_config(args, setup_theta_e_config(args))
    assert config.cwv_frac == 0.9
    assert config.d_w == 2.0e6
    assert config.w_crit == 45.0
    assert config.tau_c == 7200.0
    assert config.evap == 5.0e-5
    assert config.w_init == 40.0


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--cwv-frac", "0.9"),
        ("--dw", "2e6"),
        ("--w-crit", "45"),
        ("--tau-c", "7200"),
        ("--evap", "5e-5"),
        ("--w-init", "40"),
    ],
)
def test_cli_moist_flag_without_enable_moisture_exits(monkeypatch, flag, value):
    monkeypatch.setattr(
        "sys.argv", ["run-sw-model", "--dt", "1800", flag, value]
    )
    with pytest.raises(SystemExit) as exc_info:
        parse_arguments()
    # our explicit error, not argparse's unknown-argument exit (code 2)
    assert "--enable-moisture" in str(exc_info.value.code)


def test_cli_dry_default_has_moisture_off(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run-sw-model", "--dt", "1800"])
    args = parse_arguments()
    config = setup_sw_config(args, setup_theta_e_config(args))
    assert config.enable_moisture is False


# --- mc_face_values (MUSCL-MC face reconstruction) -------------------------

def test_mc_face_values_constant_field_exact():
    """A constant field reconstructs to the constant on every face,
    whichever way each face upwinds."""
    w = np.full(9, 47.25)
    c_f = np.array([1.0, -1.0, 0.5, 0.0, -0.5, 2.0, -2.0, 1.0])
    np.testing.assert_array_equal(mc_face_values(w, 1.0, c_f), np.full(8, 47.25))


def test_mc_face_values_linear_field_exact_interior():
    """On a linear field the MC slope is the exact slope, so interior faces
    reconstruct the exact midpoint value from either side. The outermost
    cells have a zero-padded one-sided slope (limiter clips to 0), so the
    boundary face served by an endpoint cell reverts to the cell value."""
    w = np.array([0.0, 2.0, 4.0, 6.0, 8.0])  # dy = 1 -> exact arithmetic
    # upwind from the left everywhere: face j reconstructs from cell j
    left = mc_face_values(w, 1.0, np.full(4, 1.0))
    np.testing.assert_array_equal(left[1:], [3.0, 5.0, 7.0])
    assert left[0] == w[0]  # endpoint cell: limited slope = 0
    # upwind from the right everywhere: face j reconstructs from cell j+1
    right = mc_face_values(w, 1.0, np.full(4, -1.0))
    np.testing.assert_array_equal(right[:-1], [1.0, 3.0, 5.0])
    assert right[-1] == w[-1]


def test_mc_face_values_extremum_reverts_to_cell_value():
    """At a local extremum the limited slope is zero, so the face value is
    exactly the upwind cell value (no overshoot at the sharp ITCZ front)."""
    w = np.array([1.0, 2.0, 7.0, 2.0, 1.0])  # max in cell 2
    up_left = mc_face_values(w, 1.0, np.ones(4))
    assert up_left[2] == 7.0  # face 2 upwinds from the extremum cell 2
    up_right = mc_face_values(w, 1.0, -np.ones(4))
    assert up_right[1] == 7.0  # face 1 upwinds from cell 2


def test_mc_face_values_bounded_by_adjacent_cells():
    """Every face value lies between the two adjacent cell values (the MC
    limiter's non-oscillatory guarantee), for random rough fields and mixed
    upwind directions."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        w = rng.uniform(0.0, 60.0, 41)
        c_f = rng.normal(0.0, 2.0, 40)
        w_face = mc_face_values(w, 39377.5, c_f)
        lo = np.minimum(w[:-1], w[1:])
        hi = np.maximum(w[:-1], w[1:])
        assert np.all(w_face >= lo - 1e-12)
        assert np.all(w_face <= hi + 1e-12)


# --- moisture_transport_tendency -------------------------------------------

def test_moisture_transport_tendency_conserves_mass():
    """Interior fluxes telescope and the wall fluxes are zero, so the
    cell-weighted sum of the transport tendency vanishes to roundoff:
    transport alone moves no total water."""
    rng = np.random.default_rng(11)
    dy = 39377.5
    for ny in (21, 51, 52):
        w_adv = rng.uniform(20.0, 60.0, ny)
        w_diff = rng.uniform(20.0, 60.0, ny)
        v_f = rng.normal(0.0, 3.0, ny - 1)
        tend = moisture_transport_tendency(
            w_adv, w_diff, v_f, cwv_frac=0.85, d_w=1.0e6, dy=dy
        )
        total = cwv_integral(tend, dy)
        # scale: the largest single flux crossing a face
        scale = np.max(np.abs(v_f)) * np.max(np.abs(w_adv))
        assert abs(total) < 1e-12 * scale


def test_moisture_transport_tendency_uniform_w_zero_v_is_zero():
    """No gradients and no flow: transport contributes exactly nothing."""
    ny = 31
    w = np.full(ny, 50.0)
    v_f = np.zeros(ny - 1)
    tend = moisture_transport_tendency(
        w, w, v_f, cwv_frac=0.85, d_w=1.0e6, dy=1000.0
    )
    np.testing.assert_array_equal(np.abs(tend), np.zeros(ny))


def test_moisture_transport_tendency_parity_bitexact():
    """Even W and odd face-v give an exactly even tendency, bit-for-bit:
    the discrete parity that lets a symmetric moist run hold
    max|W(y) - W(-y)| == 0.0 through the full integration."""
    for ny in (20, 21, 51, 52):
        w_adv = _even_centers(ny, seed=ny)
        w_diff = _even_centers(ny, seed=1000 + ny)
        v_f = _odd_faces(ny - 1, seed=2000 + ny)
        tend = moisture_transport_tendency(
            w_adv, w_diff, v_f, cwv_frac=0.85, d_w=1.0e6, dy=39377.5
        )
        assert np.max(np.abs(tend - tend[::-1])) == 0.0


def test_moisture_transport_tendency_pure_diffusion_of_parabola():
    """With v_f = 0 and a parabolic W, the interior tendency is the exact
    constant D * d2W/dy2 (the compact face-difference divergence is exact on
    quadratics away from the zero-flux walls)."""
    ny, dy, d_w = 41, 100.0, 2.0e5
    y = np.arange(ny) * dy
    w = 1e-6 * (y - y[ny // 2]) ** 2
    tend = moisture_transport_tendency(
        w, w, np.zeros(ny - 1), cwv_frac=0.85, d_w=d_w, dy=dy
    )
    np.testing.assert_allclose(
        tend[1:-1], np.full(ny - 2, d_w * 2.0e-6), rtol=1e-10
    )


# --- cwv_integral ----------------------------------------------------------

def test_cwv_integral_trapezoid_weights():
    """The discrete integral uses the FV cell widths: dy/2 at the wall
    centers (half cells), dy in the interior."""
    w = np.array([2.0, 3.0, 5.0, 7.0])
    assert cwv_integral(w, 10.0) == 10.0 * (1.0 + 3.0 + 5.0 + 3.5)


# --- precipitation ----------------------------------------------------------

def test_precipitation_zero_at_or_below_w_crit():
    w = np.array([0.0, 30.0, 49.999, 50.0])
    np.testing.assert_array_equal(
        precipitation(w, w_crit=50.0, tau_c=14400.0), np.zeros(4)
    )


def test_precipitation_relaxation_rate_above_w_crit():
    # exactly representable values, so (w - w_crit) / tau_c is exact
    w = np.array([57.25, 194.0])
    np.testing.assert_array_equal(
        precipitation(w, w_crit=50.0, tau_c=14400.0),
        np.array([7.25 / 14400.0, 144.0 / 14400.0]),
    )


# --- W state and time stepping ----------------------------------------------

def test_moist_model_initializes_w_uniform_at_w_crit():
    config = _moist_config()
    model = SWModel(config, _sin2_profile())
    np.testing.assert_array_equal(model.w, np.full(config.ny, 50.0))


def test_moist_model_initializes_w_uniform_at_explicit_w_init():
    config = _moist_config(w_init=42.0)
    model = SWModel(config, _sin2_profile())
    np.testing.assert_array_equal(model.w, np.full(config.ny, 42.0))


def test_dry_model_has_no_w_state():
    config = SWConfig(total_integration_days=2, ny=51, dt=1800)
    model = SWModel(config, _sin2_profile())
    assert model.w is None
    assert model.w_prev is None


def test_moisture_conserved_without_sources():
    """With E_0 = 0 and W held below W_c (so P = 0 identically), transport is
    the only W tendency, and it moves no total water: the cell-weighted
    integral of W is conserved to roundoff across a multi-day run with the
    circulation spinning up through it."""
    config = _moist_config(
        total_integration_days=4, ny=21, evap=0.0, w_init=20.0
    )
    model = SWModel(config, _sin2_profile())
    i0 = cwv_integral(model.w, config.dy)
    model.run_sim()
    assert np.all(model.w < config.w_crit)  # P stayed off, as constructed
    i1 = cwv_integral(model.w, config.dy)
    assert abs(i1 - i0) / i0 < 1e-11


class _BudgetRecorder(StaggeredSWModel):
    """Mirrors the model's own leapfrog + Asselin cycle on the scalar total
    water, feeding it only the applied sources (E_0 - P at the lagged level).
    Any transport contribution to total W would make the mirrored chain and
    the model diverge at O(flux * dt) per step; agreement to roundoff proves
    dInt(W) = IntInt(E_0 - P) exactly, the plan's budget identity."""

    chain = None

    def _step_moisture(self):
        cfg = self.config
        if self.chain is None:
            # first call: both live levels exist (w_prev was seeded)
            self.chain = (
                cwv_integral(self.w_prev, cfg.dy),
                cwv_integral(self.w, cfg.dy),
            )
        i_prev, i_now = self.chain
        source = cwv_integral(
            cfg.evap - precipitation(self.w_prev, cfg.w_crit, cfg.tau_c),
            cfg.dy,
        )
        i_next = i_prev + 2.0 * cfg.dt * source
        i_prev = i_now + cfg.asselin_filt_coef * (i_next + i_prev - 2.0 * i_now)
        self.chain = (i_prev, i_next)
        super()._step_moisture()


def test_moisture_budget_matches_applied_sources():
    """Multi-day budget: the change in total W equals the accumulated applied
    source integral (E_0 - P), to roundoff, with precipitation active."""
    config = _moist_config(total_integration_days=3, ny=21)
    model = _BudgetRecorder(config, _sin2_profile())
    model.run_sim()
    assert np.any(model.w > config.w_crit)  # P actually fired
    predicted = model.chain[1]
    actual = cwv_integral(model.w, config.dy)
    assert abs(actual - predicted) / actual < 1e-12


# --- quiescent equilibrium, dry invariance, parity --------------------------

def test_quiescent_equilibrium_relaxes_to_analytic_value():
    """With a flat theta_e (delta_y = 0) the dry circulation stays exactly
    zero, and W reduces to the pointwise ODE dW/dt = E_0 - (W - W_c)^+/tau_c,
    whose fixed point is W_c + tau_c * E_0 exactly (the discrete scheme's
    fixed point coincides with the analytical one). 4 days = 24 tau_c."""
    config = _moist_config(total_integration_days=4, ny=21)
    profile = Sin2Profile(
        ThetaEConfig(
            theta_00=330.0, y_0=0.0, y_one=9439e3, delta_y=0.0,
            theta_e_type="sin2",
        )
    )
    model = SWModel(config, profile)
    model.run_sim()
    assert np.max(np.abs(model.state.v)) == 0.0  # truly quiescent
    assert np.max(model.w) == np.min(model.w)  # W stayed exactly uniform
    w_eq = config.w_crit + config.tau_c * config.evap
    np.testing.assert_allclose(
        model.w, np.full(config.ny, w_eq), rtol=1e-10
    )


def test_moist_run_dry_fields_bitwise_identical_to_dry_twin():
    """The V1 hard invariant: W is one-way coupled, so switching moisture on
    leaves every dry field bit-for-bit unchanged, in the daily averages, the
    final instantaneous state, and the filtered leapfrog history."""
    dry_config = SWConfig(total_integration_days=5, ny=51, dt=1800)
    moist_config = _moist_config(total_integration_days=5)
    dry = SWModel(dry_config, _sin2_profile())
    moist = SWModel(moist_config, _sin2_profile())
    dry.run_sim()
    moist.run_sim()
    for name in ("u", "v", "theta"):
        daily_dry = getattr(dry.results, name)
        daily_moist = getattr(moist.results, name)
        assert np.max(np.abs(daily_moist - daily_dry)) == 0.0, (
            f"daily {name} differs between moist run and dry twin"
        )
        assert np.max(
            np.abs(getattr(moist.state, name) - getattr(dry.state, name))
        ) == 0.0, f"final {name} differs between moist run and dry twin"
        assert np.max(
            np.abs(
                getattr(moist.vars_prev_step, name)
                - getattr(dry.vars_prev_step, name)
            )
        ) == 0.0, f"filtered prev {name} differs between moist run and dry twin"


def test_moist_symmetric_run_holds_w_parity_bitexact():
    """From a symmetric state (y_0 = 0), the whole moist integration holds
    W exactly even, bit-for-bit, alongside the dry mirror-parity invariant:
    max|W(y) - W(-y)| == 0.0 on both leapfrog levels after 5 days."""
    config = _moist_config(total_integration_days=5)
    model = SWModel(config, _sin2_profile(y_0=0.0))
    model.run_sim()
    u = model.state.u
    assert np.max(np.abs(u - u[::-1])) == 0.0  # dry invariant still holds
    assert np.max(np.abs(model.w - model.w[::-1])) == 0.0
    assert np.max(np.abs(model.w_prev - model.w_prev[::-1])) == 0.0
    # and the field is nontrivial: the circulation has structured W by day 5
    assert np.max(model.w) - np.min(model.w) > 0.1
