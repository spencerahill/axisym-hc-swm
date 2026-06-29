import pytest
import numpy as np
from ss09.sw_model import SWModel
from ss09.sw_config import SWConfig
from ss09.theta_e import ThetaEConfig, SS09Profile, Sin2Profile, SB08Profile


@pytest.fixture
def sw_config():
    return SWConfig(
        total_integration_days=100,
        gravity=9.81,
        height=16000,
        beta=2e-11,
        t_ref=300.0,
        output_path="./output.nc",
        ny=50,
        dt=3600,
        coeff_eddy_heat_diff=0.0,
    )


@pytest.fixture
def theta_e_config():
    return ThetaEConfig(theta_00=310.0, y_0=1000e3, y_one=9000e3, delta_y=45)


@pytest.fixture
def model(sw_config, theta_e_config):
    theta_e_profile = SS09Profile(theta_e_config)
    return SWModel(sw_config, theta_e_profile)


@pytest.mark.parametrize("profile_class", [SS09Profile, Sin2Profile, SB08Profile])
def test_sw_model_initialization(sw_config, theta_e_config, profile_class):
    theta_e_profile = profile_class(theta_e_config)
    model = SWModel(sw_config, theta_e_profile)

    assert model.config == sw_config
    assert model.theta_e_profile == theta_e_profile
    # Staggered C-grid: u/theta on ny centers, v on ny+1 faces.
    assert model.state.u.shape == (sw_config.ny,)
    assert model.state.v.shape == (sw_config.ny + 1,)
    assert model.state.theta.shape == (sw_config.ny,)
    assert model.state.y.shape == (sw_config.ny,)


def test_rhs_shapes(model):
    du, dv, dtheta = model.rhs(model.state)
    assert du.shape == (model.config.ny,)
    assert dv.shape == (model.config.ny + 1,)
    assert dtheta.shape == (model.config.ny,)


def test_initial_theta_equals_theta_e(model):
    """The model is initialized at the equilibrium profile on the cell centers."""
    np.testing.assert_allclose(model.state.theta, model.theta_e_profile(model.state))


def test_v_walls_stay_zero_during_run(sw_config, theta_e_config):
    """The boundary faces carry v=0 throughout the integration."""
    sw_config.total_integration_days = 5
    model = SWModel(sw_config, SS09Profile(theta_e_config))
    model.run_sim()
    assert model.state.v[0] == 0.0
    assert model.state.v[-1] == 0.0


# --------------------------------------------------------------------------
# Stability anchor: the off-equatorial bug the rewrite fixes
# --------------------------------------------------------------------------
def test_off_equatorial_stable_at_default_dt():
    """Constant off-equatorial SB08 forcing from rest is stable at dt=3600.

    The old collocated/leapfrog scheme went to NaN within ~2 steps here; the
    staggered RK4 scheme with explicit momentum diffusion integrates it cleanly
    far past the day-~100 point where the centered/no-diffusion scheme blew up.
    """
    config = SWConfig(total_integration_days=400, ny=50, dt=3600,
                      output_path="./output.nc")
    profile = SB08Profile(ThetaEConfig(theta_e_type="SB08", y_0=1000e3))
    model = SWModel(config, profile)
    model.run_sim()
    assert np.all(np.isfinite(model.state.u))
    assert np.all(np.isfinite(model.state.v))
    assert np.all(np.isfinite(model.state.theta))
    # a physical winter jet has spun up at subtropical latitudes, not run away
    jet = np.max(np.abs(model.results.u[-1]))
    assert 10.0 < jet < 80.0


def test_symmetric_climate_is_steady_and_physical():
    """Symmetric forcing reaches a steady ~28 m/s subtropical jet with no
    equatorial superrotation (the explicit momentum diffusion suppresses the
    EMFD-driven runaway the clean scheme would otherwise expose)."""
    config = SWConfig(total_integration_days=400, ny=50, dt=3600,
                      output_path="./output.nc")
    profile = Sin2Profile(ThetaEConfig(theta_e_type="sin2", y_0=0.0))
    model = SWModel(config, profile)
    model.run_sim()
    u = model.results.u[-1]
    y = config.y
    assert np.all(np.isfinite(u))
    # subtropical jet of the right magnitude, not a runaway
    assert 20.0 < np.max(u) < 40.0
    # near-zero equatorial wind (no superrotation)
    assert abs(u[np.argmin(np.abs(y))]) < 5.0


# --------------------------------------------------------------------------
# Symmetric-parity anchor (exact to roundoff on an even grid)
# --------------------------------------------------------------------------
def test_symmetric_run_preserves_parity():
    """Symmetric forcing (y0=0, even N) keeps u/theta even and v odd."""
    config = SWConfig(total_integration_days=10, ny=50, dt=3600,
                      output_path="./output.nc")
    profile = Sin2Profile(ThetaEConfig(theta_e_type="sin2", y_0=0.0))
    model = SWModel(config, profile)
    model.run_sim()
    u = model.results.u[-1]
    v = model.results.v[-1]  # daily-mean v already interpolated to centers
    theta = model.results.theta[-1]
    np.testing.assert_allclose(u, u[::-1], atol=1e-10)
    np.testing.assert_allclose(theta, theta[::-1], atol=1e-10)
    np.testing.assert_allclose(v, -v[::-1], atol=1e-10)


def test_steady_state_integration(sw_config, theta_e_config):
    """Test that steady-state detection integrates with model"""
    sw_config.total_integration_days = 50
    sw_config.enable_steady_state = True
    sw_config.steady_state_window_size = 5
    sw_config.steady_state_threshold = 0.001

    model = SWModel(sw_config, SS09Profile(theta_e_config))
    assert model.steady_state_detector.enabled

    # Run simulation (may or may not converge, just test it doesn't crash)
    model.run_sim()

    # Check that results are valid
    assert model.results.time is not None
    # If converged, check metadata
    if model.steady_state_detector.is_converged:
        assert model.steady_state_detector.convergence_day >= 0
        assert len(model.steady_state_detector.kinetic_energy_history) > 0


def test_steady_state_disabled_by_default(sw_config, theta_e_config):
    """Test that steady-state detection is disabled by default"""
    model = SWModel(sw_config, SS09Profile(theta_e_config))
    assert not model.steady_state_detector.enabled
    assert not model.steady_state_detector.is_converged


def test_steady_state_warning_when_not_converged(sw_config, theta_e_config, caplog):
    """Test that a warning is logged when steady-state is enabled but not reached"""
    import logging

    # Configure a very short simulation that won't converge
    sw_config.total_integration_days = 5
    sw_config.enable_steady_state = True
    sw_config.steady_state_window_size = 3
    sw_config.steady_state_threshold = 1e-10  # Very strict threshold to ensure no convergence

    model = SWModel(sw_config, SS09Profile(theta_e_config))

    # Run simulation with caplog to capture warnings
    with caplog.at_level(logging.WARNING):
        model.run_sim()

    # Check that we did NOT converge
    assert not model.steady_state_detector.is_converged

    # Check that the warning was logged
    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    convergence_warnings = [m for m in warning_messages if "convergence" in m.lower()]
    assert len(convergence_warnings) >= 1, "Expected a warning about convergence not being reached"
    assert "without reaching convergence" in convergence_warnings[0].lower() or "not reached" in convergence_warnings[0].lower()


def test_hadley_diagnostics_integration(sw_config, theta_e_config):
    """Test that Hadley diagnostics are computed and saved correctly"""
    sw_config.total_integration_days = 10

    model = SWModel(sw_config, SS09Profile(theta_e_config))

    # Check diagnostics initialized
    assert model.hadley_diagnostics is not None
    assert model.hadley_diagnostics.days_recorded == 0

    # Run short simulation
    model.run_sim()

    # Check that diagnostics were recorded
    assert model.hadley_diagnostics.days_recorded > 0

    # Check that diagnostics are included in xarray output
    ds = model.results.to_xarray(
        model.config,
        model.theta_e_profile,
        model.steady_state_detector,
        model.hadley_diagnostics,
    )

    # Verify all expected variables present
    assert "rossby_number" in ds.data_vars
    assert "north_jet_lat" in ds.data_vars
    assert "north_jet_magnitude" in ds.data_vars
    assert "south_jet_lat" in ds.data_vars
    assert "south_jet_magnitude" in ds.data_vars

    # Check dimensions
    assert ds["rossby_number"].dims == ("time", "y")
    assert ds["north_jet_lat"].dims == ("time",)

    # Check units and metadata
    assert ds["rossby_number"].attrs["units"] == "dimensionless"
    assert ds["north_jet_lat"].attrs["units"] == "m"
    assert ds["north_jet_magnitude"].attrs["units"] == "m/s"

    # Check that values are reasonable (not all NaN)
    assert not np.all(np.isnan(ds["north_jet_lat"].values))
    # Rossby number will have some NaN near equator, but not all
    assert not np.all(np.isnan(ds["rossby_number"].values))


def test_hadley_diagnostics_backward_compatibility(sw_config, theta_e_config):
    """Test that code works without passing hadley_diagnostics"""
    model = SWModel(sw_config, SS09Profile(theta_e_config))

    # Should work without hadley_diagnostics argument (backward compat)
    ds = model.results.to_xarray(
        model.config, model.theta_e_profile, model.steady_state_detector
    )

    # Should not crash, but won't have Hadley diagnostics
    assert "u" in ds.data_vars
    assert "v" in ds.data_vars


def test_seasonal_sb08_integration(sw_config):
    """Test that model runs successfully with seasonal SB08 profile"""
    from ss09.theta_e import SB08Profile, ThetaEConfig

    theta_e_config = ThetaEConfig(
        theta_00=330.0,
        y_0=0.0,  # Start at equator
        y_one=9439e3,
        delta_y=50.0,
        theta_e_type="SB08",
        y_0_seasonal_amp=500e3,  # 500 km migration (more conservative)
        seasonal_period_days=360.0,
        seasonal_phase_days=0.0
    )

    sw_config.total_integration_days = 20  # Shorter test run

    # Create model with seasonal SB08 profile
    model = SWModel(sw_config, SB08Profile(theta_e_config))

    # Run simulation
    model.run_sim()

    # Check that simulation completed without NaN values
    assert not np.any(np.isnan(model.results.u))
    assert not np.any(np.isnan(model.results.v))
    assert not np.any(np.isnan(model.results.theta))

    # Check that we have results for all requested days
    assert model.results.time[19] > 0  # Day 19 should have data


def test_seasonal_convergence_enabled(sw_config):
    """Test that seasonal convergence detection stops simulation early when enabled"""
    from ss09.theta_e import SB08Profile, ThetaEConfig

    theta_e_config = ThetaEConfig(
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        theta_e_type="SB08",
        y_0_seasonal_amp=500e3,  # 500 km migration
        seasonal_period_days=360.0,
        seasonal_phase_days=0.0
    )

    # Enable seasonal convergence
    sw_config.total_integration_days = 1000  # Long run
    sw_config.seasonal_convergence_enabled = True
    sw_config.seasonal_convergence_window = 30
    sw_config.seasonal_convergence_threshold = 0.01

    model = SWModel(sw_config, SB08Profile(theta_e_config))
    model.run_sim()

    # Should stop before 1000 days if converged
    # (Might or might not converge depending on dynamics, but test should not crash)
    if model.steady_state_detector.is_converged:
        assert model.steady_state_detector.convergence_day > 360  # At least 1 year
        assert model.steady_state_detector.convergence_day < 1000


def test_seasonal_convergence_disabled_runs_full_duration(sw_config):
    """Test that with seasonal forcing but convergence disabled, simulation runs full duration"""
    from ss09.theta_e import SB08Profile, ThetaEConfig

    theta_e_config = ThetaEConfig(
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        theta_e_type="SB08",
        y_0_seasonal_amp=500e3,
        seasonal_period_days=360.0,
        seasonal_phase_days=0.0
    )

    # Disable seasonal convergence (default behavior)
    sw_config.total_integration_days = 100
    sw_config.seasonal_convergence_enabled = False  # Explicit disable

    model = SWModel(sw_config, SB08Profile(theta_e_config))
    model.run_sim()

    # Should run full 100 days (no early stopping)
    # Check last day has data
    last_day_idx = np.where(model.results.time > 0)[0][-1]
    assert last_day_idx >= 99  # Should be day 99 or close to it


def test_has_seasonal_forcing_detection(sw_config, theta_e_config):
    """Test that has_seasonal_forcing() correctly detects seasonal forcing"""
    from ss09.theta_e import SS09Profile, SB08Profile, ThetaEConfig

    # Non-seasonal profile
    model_no_seasonal = SWModel(sw_config, SS09Profile(theta_e_config))
    assert not model_no_seasonal.has_seasonal_forcing()

    # Seasonal profile (y_0_seasonal_amp > 0)
    theta_e_config_seasonal = ThetaEConfig(
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        theta_e_type="SB08",
        y_0_seasonal_amp=500e3,  # Seasonal forcing enabled
        seasonal_period_days=360.0,
        seasonal_phase_days=0.0
    )
    model_seasonal = SWModel(sw_config, SB08Profile(theta_e_config_seasonal))
    assert model_seasonal.has_seasonal_forcing()

    # SB08 but with zero amplitude (no seasonal forcing)
    theta_e_config_no_amp = ThetaEConfig(
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        theta_e_type="SB08",
        y_0_seasonal_amp=0.0,  # No seasonal forcing
        seasonal_period_days=360.0,
        seasonal_phase_days=0.0
    )
    model_no_amp = SWModel(sw_config, SB08Profile(theta_e_config_no_amp))
    assert not model_no_amp.has_seasonal_forcing()
