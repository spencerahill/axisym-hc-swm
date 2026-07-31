"""Tests for moist V2: the latent-heating feedback.

V2 (guides/moist_axisymmetric_model_spec.pdf Eq. T2/Q2) makes exactly one
structural change to V1: the thermodynamic relaxation is retargeted from the
warm radiative-convective profile to a steeper, purely radiative one, and
explicit latent heating is added,

    dtheta/dt + (d Delta_z/H) dv/dy = (theta_rad - theta)/tau + Lambda P,

with Lambda = L_v/C pinned to the model's own column accounting (C Lambda =
L_v). The moisture equation is unchanged. Three invariants define success:

1. The same P array heats theta and drains W, so L_v P cancels exactly in the
   column MSE budget d<h>/dt + d/dy[v Hhat - L_v D dW/dy] = radiation + L_v E_0.
2. With Lambda = 0 the run reduces, bit-for-bit, to a dry run against the
   radiative target (the V2_0 bridge that separates the retarget from the
   latent feedback).
3. The vertical-advection gate switches to SS09's kinematic H(dv/dy): the
   V1 gate H(theta_E - theta) reads the *current* relaxation target, which in
   V2 is theta_rad, and the column runs warmer than theta_rad wherever it
   precipitates, so that gate would switch off exactly where convection is
   strongest (spec item 11).
"""

import numpy as np
import pytest

from ss09.cli import parse_arguments, setup_sw_config, setup_theta_e_config
from ss09.moist_constants import (
    C_P,
    DELTA_P,
    EXNER,
    L_V,
    P_TOP,
    column_heat_capacity,
    latent_heating_coeff,
)
from ss09.sw_config import SWConfig
from ss09.sw_model import THETA_TO_TEMP, SWModel, cwv_integral
from ss09.theta_e import Sin2Profile, ThetaEConfig


def _sin2_profile(y_0=0.0, delta_y=50.0):
    return Sin2Profile(
        ThetaEConfig(
            theta_00=330.0, y_0=y_0, y_one=9439e3, delta_y=delta_y,
            theta_e_type="sin2",
        )
    )


# dt=900, not the dry suite's 1800: the radiative target's steeper contrast
# (75 K vs 50 K) drives a stronger circulation (max|u| 46 vs 27 m/s at ny=51)
# and tightens the advective CFL, so dt=1800 diverges by day 5 at ny=51 under
# EITHER gate, dry or moist (measured 2026-07-30). dt=900 is stable and
# resolution-converged there (max|u| within 1.3% of dt=450).
V2_DT = 900


def _v2_config(**overrides):
    """A valid V2 configuration: moisture on, latent heating on."""
    kwargs: dict = dict(
        total_integration_days=2,
        ny=51,
        dt=V2_DT,
        enable_moisture=True,
        enable_latent_heating=True,
    )
    kwargs.update(overrides)
    return SWConfig(**kwargs)


# --- column thermodynamic constants ----------------------------------------

def test_exner_matches_the_model_theta_to_temp():
    """The column constants must use the SAME Exner factor the dynamics do,
    or C (and hence Lambda) would describe a different column than the one
    the pressure-gradient term integrates."""
    assert EXNER == THETA_TO_TEMP


def test_tropopause_pressure_matches_spec():
    """p_t = p_s Pi^(c_p/R) = 193 hPa at the SS13 Exner factor."""
    assert P_TOP / 100.0 == pytest.approx(193.0, abs=0.5)
    assert DELTA_P / 100.0 == pytest.approx(807.0, abs=0.5)


def test_column_heat_capacity_matches_spec():
    """C = c_p Pi Delta_p / g ~ 5.2e6 J m^-2 K^-1 (spec Eq. Cdef)."""
    assert column_heat_capacity(9.81) == pytest.approx(5.2e6, rel=0.01)


def test_latent_heating_coeff_matches_spec():
    """Lambda = L_v/C ~ 0.48 K per unit precipitation rate: 10 mm/day of
    precipitation heats the column ~4.8 K/day (spec Eq. Lconv)."""
    lam = latent_heating_coeff(9.81)
    assert lam == pytest.approx(0.48, rel=0.02)
    mm_per_day = 10.0 / 86400.0  # kg m^-2 s^-1
    assert lam * mm_per_day * 86400.0 == pytest.approx(4.8, rel=0.02)


def test_latent_heating_coeff_closes_the_energy_conversion():
    """C * Lambda == L_v is the consistency the MSE budget's P cancellation
    rests on; anything else double-counts or loses latent energy."""
    for gravity in (9.81, 9.0, 10.5):
        assert (
            column_heat_capacity(gravity) * latent_heating_coeff(gravity)
            == pytest.approx(L_V, rel=1e-15)
        )
    assert C_P == 1004.0


# --- config validation ------------------------------------------------------

def test_config_latent_heating_requires_moisture():
    """Latent heating is Lambda * P, and P is a function of the prognostic W;
    without moisture there is no P to release."""
    with pytest.raises(ValueError, match="enable_moisture"):
        SWConfig(
            total_integration_days=2, ny=51, dt=1800,
            enable_latent_heating=True,
        )


@pytest.mark.parametrize(
    "param, value", [("delta_y_rad", 80.0), ("lambda_conv", 0.3)]
)
def test_config_latent_param_without_latent_heating_raises(param, value):
    """Repo style: a parameter that would be silently inert is a hard error."""
    with pytest.raises(ValueError, match="enable_latent_heating"):
        SWConfig(
            total_integration_days=2, ny=51, dt=1800,
            enable_moisture=True, **{param: value},
        )


def test_config_latent_defaults_resolve():
    config = _v2_config()
    assert config.delta_y_rad == 75.0  # spec's provisional 75-100 K, low end
    assert config.lambda_conv == latent_heating_coeff(config.gravity)


def test_config_latent_params_stay_none_when_dry():
    """Off, the V2 parameters read as not-applicable rather than carrying a
    number that never acted."""
    config = SWConfig(total_integration_days=2, ny=51, dt=1800)
    assert config.enable_latent_heating is False
    assert config.delta_y_rad is None
    assert config.lambda_conv is None


def test_config_explicit_latent_values_are_kept():
    config = _v2_config(delta_y_rad=100.0, lambda_conv=0.0)
    assert config.delta_y_rad == 100.0
    assert config.lambda_conv == 0.0  # the V2_0 bridge


def test_config_vert_advec_gate_defaults_by_version():
    """V1 and dry runs keep the production H(theta_E - theta) gate; V2
    defaults to SS09's kinematic H(dv/dy), which survives the retarget."""
    assert SWConfig(total_integration_days=2, ny=51, dt=1800).vert_advec_gate == "theta"
    assert _v2_config().vert_advec_gate == "kinematic"


def test_config_vert_advec_gate_explicit_override():
    dry = SWConfig(
        total_integration_days=2, ny=51, dt=1800, vert_advec_gate="kinematic"
    )
    assert dry.vert_advec_gate == "kinematic"
    # the spec-item-11 pathology stays reachable, but only on explicit request
    with pytest.warns(UserWarning, match="kinematic"):
        v2 = _v2_config(vert_advec_gate="theta")
    assert v2.vert_advec_gate == "theta"


def test_config_rejects_unknown_vert_advec_gate():
    with pytest.raises(ValueError, match="vert_advec_gate"):
        SWConfig(
            total_integration_days=2, ny=51, dt=1800, vert_advec_gate="precip"
        )


# --- the radiative relaxation target ---------------------------------------

def test_v2_relaxes_to_the_radiative_target():
    """The model rebuilds its relaxation target at delta_y_rad, so theta_E in
    every term (relaxation, output, diagnostics) is theta_rad."""
    model = SWModel(_v2_config(delta_y_rad=80.0), _sin2_profile(delta_y=50.0))
    assert model.theta_e_profile.config.delta_y == 80.0
    theta_rad = model.current_theta_e()
    assert np.max(theta_rad) == pytest.approx(330.0)  # background unchanged
    assert np.max(theta_rad) - np.min(theta_rad) == pytest.approx(80.0)
    # the initial state is the radiative target, as in V1 it is theta_E
    np.testing.assert_array_equal(model.state.theta, theta_rad)


def test_v1_target_untouched_without_latent_heating():
    model = SWModel(
        SWConfig(total_integration_days=2, ny=51, dt=1800, enable_moisture=True),
        _sin2_profile(delta_y=50.0),
    )
    assert model.theta_e_profile.config.delta_y == 50.0


def test_v2_warns_when_radiative_target_is_not_steeper():
    """Delta_theta^rad <= Delta_theta means the relaxation still carries
    convective heating that Lambda P now adds a second time."""
    with pytest.warns(UserWarning, match="radiative"):
        SWModel(_v2_config(delta_y_rad=40.0), _sin2_profile(delta_y=50.0))


# --- the kinematic vertical-advection gate ---------------------------------

def _seeded_model(config):
    """A model with a nontrivial, non-symmetric (u, v) state so the two gates
    disagree at many points."""
    model = SWModel(config, _sin2_profile())
    rng = np.random.default_rng(3)
    model.state.u[:] = rng.normal(0.0, 8.0, config.ny)
    model.state.v[:] = rng.normal(0.0, 0.4, config.nv)
    return model


def test_kinematic_gate_follows_the_sign_of_dv_dy():
    model = _seeded_model(_v2_config())
    dv_dy = model.dv_dy()
    expected = model.state.u * dv_dy * np.heaviside(dv_dy, 0.5)
    np.testing.assert_array_equal(model.vert_advec_u(), expected)


def test_theta_gate_unchanged_when_selected():
    model = _seeded_model(SWConfig(total_integration_days=2, ny=51, dt=1800))
    expected = (
        model.state.u
        * model.dv_dy()
        * np.heaviside(model.current_theta_e() - model.state.theta, 0.5)
    )
    np.testing.assert_array_equal(model.vert_advec_u(), expected)


# --- latent heating in the thermodynamic equation ---------------------------

def test_latent_heating_uses_the_precipitation_the_moisture_step_applies():
    """The heating and the moisture sink must be the SAME array (the lagged
    n-1 level), or L_v P would not cancel in the MSE budget."""
    model = SWModel(_v2_config(total_integration_days=2, ny=21), _sin2_profile())
    model.run_sim()
    _, p_applied = model._moisture_rhs(model.w, model.w_prev)
    np.testing.assert_array_equal(
        model.latent_heating(), model.config.lambda_conv * p_applied
    )


def test_latent_heating_absent_from_the_dry_tendency():
    model = SWModel(
        SWConfig(total_integration_days=2, ny=21, dt=1800, enable_moisture=True),
        _sin2_profile(),
    )
    tendency = model.dtheta_dt()
    expected = (
        model.newt_cool_term() + model.vert_advec_theta() + model.eddy_heat_flux()
    )
    np.testing.assert_array_equal(tendency, expected)


def test_discrete_mse_budget_closes():
    """The column MSE budget, evaluated on the discrete operators at a spun-up
    state: integrating C * (theta tendency) + L_v * (W tendency) over the
    domain, precipitation cancels exactly and both flux divergences telescope
    to zero, leaving radiation + L_v E_0."""
    config = _v2_config(total_integration_days=3, ny=21)
    model = SWModel(config, _sin2_profile())
    model.run_sim()

    dy = config.dy
    c_col = column_heat_capacity(config.gravity)
    theta_tend = model.dtheta_dt()
    w_tend, p_applied = model._moisture_rhs(model.w, model.w_prev)
    assert np.any(p_applied > 0.0), "P never fired; the test would be vacuous"
    assert np.max(np.abs(model.state.v)) > 0.0, "no circulation; test vacuous"
    assert np.all(np.isfinite(theta_tend)) and np.all(np.isfinite(w_tend))

    domain = cwv_integral(np.ones(config.ny), dy)
    lhs = c_col * cwv_integral(theta_tend, dy) + L_V * cwv_integral(w_tend, dy)
    rhs = (
        c_col * cwv_integral(model.newt_cool_term(), dy) + L_V * config.evap * domain
    )
    scale = max(
        abs(c_col * cwv_integral(model.newt_cool_term(), dy)),
        abs(L_V * cwv_integral(p_applied, dy)),
        abs(L_V * config.evap * domain),
        abs(c_col * cwv_integral(model.vert_advec_theta(), dy)),
    )
    assert abs(lhs - rhs) < 1e-12 * scale


# --- integration-level invariants -------------------------------------------

def test_zero_lambda_reproduces_the_dry_radiative_twin_bitwise():
    """The V2_0 bridge: with Lambda = 0 the coupling is severed and the run is
    bit-for-bit a dry run against theta_rad under the same kinematic gate, so
    any V1-to-V2 difference splits cleanly into retarget and latent parts."""
    v2 = SWModel(
        _v2_config(total_integration_days=5, delta_y_rad=75.0, lambda_conv=0.0),
        _sin2_profile(delta_y=50.0),
    )
    dry = SWModel(
        SWConfig(
            total_integration_days=5, ny=51, dt=V2_DT,
            vert_advec_gate="kinematic",
        ),
        _sin2_profile(delta_y=75.0),
    )
    v2.run_sim()
    dry.run_sim()
    assert np.all(np.isfinite(v2.state.u)), "run diverged; comparison is vacuous"
    for name in ("u", "v", "theta"):
        assert np.max(
            np.abs(getattr(v2.results, name) - getattr(dry.results, name))
        ) == 0.0, f"daily {name} differs from the Lambda=0 dry twin"
        assert np.max(
            np.abs(getattr(v2.state, name) - getattr(dry.state, name))
        ) == 0.0, f"final {name} differs from the Lambda=0 dry twin"


def test_latent_heating_warms_the_column_and_moves_the_circulation():
    """The V2 feedback is live: against its own Lambda=0 twin (identical
    target, identical gate), latent heating warms theta and changes v."""
    coupled = SWModel(_v2_config(total_integration_days=5), _sin2_profile())
    bridge = SWModel(
        _v2_config(total_integration_days=5, lambda_conv=0.0), _sin2_profile()
    )
    coupled.run_sim()
    bridge.run_sim()
    assert np.all(np.isfinite(coupled.state.theta))
    assert np.all(np.isfinite(bridge.state.theta)), "twin diverged; test vacuous"
    # Lambda * E_0 ~ 1.9 K/day of heating once the column reaches W_c
    assert np.max(coupled.state.theta - bridge.state.theta) > 1.0
    assert np.max(np.abs(coupled.state.v - bridge.state.v)) > 0.0


def test_v2_symmetric_run_holds_parity_bitexact():
    """The added coupling preserves the mirror-parity invariant: u even, W
    even, bit-for-bit, from symmetric forcing on the odd-ny grid."""
    model = SWModel(_v2_config(total_integration_days=5), _sin2_profile(y_0=0.0))
    model.run_sim()
    u = model.state.u
    assert np.max(np.abs(u - u[::-1])) == 0.0
    assert np.max(np.abs(model.w - model.w[::-1])) == 0.0
    assert np.max(np.abs(model.state.theta - model.state.theta[::-1])) == 0.0


def test_v2_restart_continuation_bit_identical(tmp_path):
    """A V2 run checkpointed and resumed reproduces the uninterrupted
    trajectory bit-for-bit across the coupled theta-W pair."""
    total_days, split_day = 4, 2

    def _model(directory, ndays, name):
        directory.mkdir(exist_ok=True)
        return SWModel(
            _v2_config(
                total_integration_days=ndays,
                ny=21,
                output_path=str(directory / f"{name}.nc"),
                restart_output_dir=str(directory),
            ),
            _sin2_profile(),
        )

    full = _model(tmp_path / "full", total_days, "out")
    full.run_sim()

    part = _model(tmp_path / "part", split_day, "out")
    part.run_sim()
    from ss09.output_path_utils import generate_restart_filename

    restart_file = generate_restart_filename(part.config.output_path, split_day)

    cont = _model(tmp_path / "part", total_days, "cont")
    cont.restart_day = cont.load_from_restart(restart_file)
    cont.run_sim()

    for day in range(split_day, total_days):
        for name in ("u", "v", "theta", "w", "precip"):
            assert np.max(
                np.abs(
                    getattr(cont.results, name)[day]
                    - getattr(full.results, name)[day]
                )
            ) == 0.0, f"{name} differs on resumed day {day}"


# --- output and CLI ---------------------------------------------------------

def test_v2_parameters_land_in_the_output_attrs(tmp_path):
    import xarray as xr

    config = _v2_config(
        total_integration_days=2,
        ny=21,
        output_path=str(tmp_path / "v2.nc"),
        restart_output_dir=str(tmp_path),
    )
    model = SWModel(config, _sin2_profile())
    model.run_sim()
    model.save_results()
    ds = xr.open_dataset(config.output_path)
    assert ds.attrs["enable_latent_heating"] == "True"
    assert ds.attrs["delta_y_rad"] == 75.0
    assert ds.attrs["lambda_conv"] == pytest.approx(latent_heating_coeff(9.81))
    assert ds.attrs["vert_advec_gate"] == "kinematic"
    # the relaxation target written to file is theta_rad
    assert float(ds["theta_e"].max()) - float(ds["theta_e"].min()) == pytest.approx(
        75.0
    )
    ds.close()


def test_cli_latent_heating_flags_reach_config(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run-sw-model", "--dt", "1800", "--enable-moisture",
            "--enable-latent-heating", "--delta-y-rad", "90",
            "--lambda-conv", "0.5",
        ],
    )
    args = parse_arguments()
    config = setup_sw_config(args, setup_theta_e_config(args))
    assert config.enable_latent_heating is True
    assert config.delta_y_rad == 90.0
    assert config.lambda_conv == 0.5
    assert config.vert_advec_gate == "kinematic"


def test_cli_vert_advec_gate_flag(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["run-sw-model", "--dt", "1800", "--vert-advec-gate", "kinematic"],
    )
    args = parse_arguments()
    config = setup_sw_config(args, setup_theta_e_config(args))
    assert config.vert_advec_gate == "kinematic"
    assert config.enable_latent_heating is False


@pytest.mark.parametrize(
    "flag, value", [("--delta-y-rad", "90"), ("--lambda-conv", "0.5")]
)
def test_cli_latent_param_without_flag_exits(monkeypatch, flag, value):
    monkeypatch.setattr(
        "sys.argv",
        ["run-sw-model", "--dt", "1800", "--enable-moisture", flag, value],
    )
    with pytest.raises(SystemExit) as exc_info:
        parse_arguments()
    assert "--enable-latent-heating" in str(exc_info.value.code)


def test_cli_latent_heating_without_moisture_exits(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["run-sw-model", "--dt", "1800", "--enable-latent-heating"],
    )
    with pytest.raises(SystemExit) as exc_info:
        parse_arguments()
    assert "--enable-moisture" in str(exc_info.value.code)
