import pytest
import numpy as np
from ss09.sw_model import SWModel
from ss09.sw_config import SWConfig
from ss09.theta_e import ThetaEConfig, SS09Profile, Sin2Profile, SB08Profile
from ss09.model_state import ModelState


@pytest.fixture
def sw_config():
    return SWConfig(
        total_integration_days=100,
        gravity=9.81,
        height=16000,
        beta=2e-11,
        t_ref=300.0,
        output_path="./output.nc",
        ny=51,
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
    assert model.state.u.shape == (sw_config.ny,)
    assert model.state.v.shape == (sw_config.ny,)
    assert model.state.theta.shape == (sw_config.ny,)
    assert model.state.y.shape == (sw_config.ny,)


def test_du_dt(model):
    du_dt_result = model.du_dt()
    assert du_dt_result.shape == (model.config.ny,)
    # Add more specific assertions based on expected behavior


def test_dv_dt(model):
    dv_dt_result = model.dv_dt()
    assert dv_dt_result.shape == (model.config.ny,)
    # Add more specific assertions based on expected behavior


def test_dtheta_dt(model):
    dtheta_dt_result = model.dtheta_dt()
    assert dtheta_dt_result.shape == (model.config.ny,)
    # Add more specific assertions based on expected behavior


def test_eddy_heat_flux_inactive(model):
    eddy_heat_flux_result = model.eddy_heat_flux()
    assert np.all(eddy_heat_flux_result == 0)
    assert eddy_heat_flux_result.shape == (model.config.ny,)


def test_eddy_heat_flux_active(sw_config, theta_e_config):
    # Activate eddy heat flux by setting kappa_theta
    sw_config.coeff_eddy_heat_diff = 1.0
    model = SWModel(sw_config, SS09Profile(theta_e_config))
    eddy_heat_flux_result = model.eddy_heat_flux()
    assert eddy_heat_flux_result.shape == (model.config.ny,)
    # Add more specific assertions based on expected behavior


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


def test_merid_advec_u_toggle(sw_config, theta_e_config):
    """Test that meridional advection of u can be toggled on/off"""
    # Test with advection enabled (default)
    sw_config.include_merid_advec_u = True
    model_with = SWModel(sw_config, SS09Profile(theta_e_config))
    model_with.state.u[:] = np.linspace(1.0, 10.0, sw_config.ny)
    model_with.state.v[:] = 2.0
    du_dt_with = model_with.du_dt()

    # Test with advection disabled
    sw_config.include_merid_advec_u = False
    model_without = SWModel(sw_config, SS09Profile(theta_e_config))
    model_without.state.u[:] = np.linspace(1.0, 10.0, sw_config.ny)
    model_without.state.v[:] = 2.0
    du_dt_without = model_without.du_dt()

    # Results should be different
    assert not np.allclose(du_dt_with, du_dt_without)

    # Verify shape is correct in both cases
    assert du_dt_with.shape == (sw_config.ny,)
    assert du_dt_without.shape == (sw_config.ny,)


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
    # Create a fresh ThetaEConfig specifically for SB08 with reasonable parameters
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


def test_emfd_only_acts_on_westerlies(sw_config, theta_e_config):
    """Test that EMFD only acts where u > 0 (westerlies), per SCIENCE.md eq 3.1.

    The Heaviside function H(u) = 1 if u > 0, else 0 ensures that eddies
    only extract momentum from westerly flow.
    """
    model = SWModel(sw_config, SS09Profile(theta_e_config))

    ny = model.config.ny
    equator_idx = ny // 2

    # Create u profile: easterlies near equator, westerlies in subtropics
    u_profile = np.zeros(ny)
    u_profile[equator_idx - 5:equator_idx + 6] = -5.0  # easterlies (u < 0)
    u_profile[:equator_idx - 5] = np.linspace(10.0, 2.0, equator_idx - 5)  # SH westerlies
    u_profile[equator_idx + 6:] = np.linspace(2.0, 10.0, ny - equator_idx - 6)  # NH westerlies

    model.state = model.state._replace(u=u_profile)
    emfd = model.edd_mom_flux_div_u()

    # EMFD must be exactly zero where u <= 0 (binary Heaviside)
    easterly_mask = u_profile <= 0
    assert np.all(emfd[easterly_mask] == 0.0), \
        f"EMFD must be exactly zero in easterly regions, got: {emfd[easterly_mask]}"

    # EMFD should be non-zero where u > 0 and du/dy != 0
    westerly_mask = u_profile > 0
    du_dy = np.gradient(u_profile, model.config.dy)
    active_mask = westerly_mask & (np.abs(du_dy) > 1e-10)
    assert not np.allclose(emfd[active_mask], 0.0), \
        "EMFD should be non-zero in westerly regions with shear"


def test_vert_advec_u_heaviside_at_equilibrium(sw_config, theta_e_config):
    """Test that vertical advection is inactive when theta equals theta_e.

    Per physics: H(theta_e - theta) should return 0 when theta_e = theta exactly,
    meaning vertical momentum exchange is inactive at radiative-convective equilibrium.
    """
    model = SWModel(sw_config, SS09Profile(theta_e_config))

    # Set u to non-zero values so vert_advec_u term would be nonzero if H != 0
    model.state = model.state._replace(u=np.ones(model.config.ny) * 10.0)

    # Set v to have divergence (so dv/dy != 0)
    v_profile = np.sin(np.pi * model.config.y / model.config.y.max())
    model.state = model.state._replace(v=v_profile)

    # Crucially: set theta = theta_e exactly (at equilibrium)
    theta_e = model.theta_e_profile(model.state)
    model.state = model.state._replace(theta=theta_e.copy())

    # At equilibrium (theta_e - theta = 0), the Heaviside function should be 0,
    # so vertical advection should be exactly zero
    vert_advec = model.vert_advec_u()

    assert np.allclose(vert_advec, 0.0), (
        f"Vertical advection should be zero at equilibrium (theta = theta_e). "
        f"Max value: {np.abs(vert_advec).max()}"
    )


def test_vert_advec_u_active_when_cooler_than_equilibrium(sw_config, theta_e_config):
    """Test that vertical advection is active when theta < theta_e.

    Per physics: H(theta_e - theta) = 1 when theta_e > theta (atmosphere is
    cooler than equilibrium, indicating convective tendency).
    """
    model = SWModel(sw_config, SS09Profile(theta_e_config))

    # Set u to non-zero values
    model.state = model.state._replace(u=np.ones(model.config.ny) * 10.0)

    # Set v to have divergence (so dv/dy != 0)
    v_profile = np.sin(np.pi * model.config.y / model.config.y.max())
    model.state = model.state._replace(v=v_profile)

    # Set theta cooler than theta_e
    theta_e = model.theta_e_profile(model.state)
    model.state = model.state._replace(theta=theta_e - 5.0)  # 5 K cooler

    # Now H(theta_e - theta) = H(5) = 1, so vertical advection should be active
    vert_advec = model.vert_advec_u()

    # Check that at least some points have non-zero vertical advection
    # (where dv/dy != 0)
    dv_dy = np.gradient(v_profile, model.config.dy)
    active_mask = np.abs(dv_dy) > 1e-10

    assert not np.allclose(vert_advec[active_mask], 0.0), (
        "Vertical advection should be active when theta < theta_e"
    )


def test_vert_advec_u_inactive_when_warmer_than_equilibrium(sw_config, theta_e_config):
    """Test that vertical advection is inactive when theta > theta_e.

    Per physics: H(theta_e - theta) = 0 when theta_e < theta (atmosphere is
    warmer than equilibrium, no convective tendency).
    """
    model = SWModel(sw_config, SS09Profile(theta_e_config))

    # Set u to non-zero values
    model.state = model.state._replace(u=np.ones(model.config.ny) * 10.0)

    # Set v to have divergence (so dv/dy != 0)
    v_profile = np.sin(np.pi * model.config.y / model.config.y.max())
    model.state = model.state._replace(v=v_profile)

    # Set theta warmer than theta_e
    theta_e = model.theta_e_profile(model.state)
    model.state = model.state._replace(theta=theta_e + 5.0)  # 5 K warmer

    # Now H(theta_e - theta) = H(-5) = 0, so vertical advection should be zero
    vert_advec = model.vert_advec_u()

    assert np.allclose(vert_advec, 0.0), (
        f"Vertical advection should be zero when theta > theta_e. "
        f"Max value: {np.abs(vert_advec).max()}"
    )
