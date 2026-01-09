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
