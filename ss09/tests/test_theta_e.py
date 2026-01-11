import pytest
import numpy as np
from ss09.theta_e import ThetaEConfig, SB08Profile
from ss09.model_state import ModelState


@pytest.fixture
def theta_e_config():
    """Basic ThetaEConfig for testing"""
    return ThetaEConfig(
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        theta_e_type="SB08"
    )


@pytest.fixture
def model_state():
    """Basic ModelState for testing"""
    y = np.linspace(-1e7, 1e7, 51)
    return ModelState(
        t=0.0,
        u=np.zeros(51),
        v=np.zeros(51),
        theta=np.zeros(51),
        y=y
    )


def test_sb08_no_seasonal_cycle(theta_e_config, model_state):
    """Test that SB08Profile with y_0_seasonal_amp=0 produces constant output over time"""
    profile = SB08Profile(theta_e_config)

    # Compute theta_e at different times
    theta_e_t0 = profile(model_state)

    model_state_t1 = model_state._replace(t=86400)  # 1 day later
    theta_e_t1 = profile(model_state_t1)

    model_state_t2 = model_state._replace(t=10*86400)  # 10 days later
    theta_e_t2 = profile(model_state_t2)

    # Should be identical at all times when no seasonal cycle
    assert np.allclose(theta_e_t0, theta_e_t1)
    assert np.allclose(theta_e_t0, theta_e_t2)


def test_sb08_seasonal_cycle_phases(theta_e_config, model_state):
    """Test that seasonal cycle varies correctly through phases"""
    # Enable seasonal cycle
    theta_e_config.y_0_seasonal_amp = 1000e3  # 1000 km amplitude
    theta_e_config.seasonal_period_days = 360.0
    theta_e_config.seasonal_phase_days = 0.0

    profile = SB08Profile(theta_e_config)

    # At t=0: y_0 should be at baseline (0)
    model_state_t0 = model_state._replace(t=0)
    theta_e_t0 = profile(model_state_t0)

    # At t=T/4 (90 days): y_0 should be at maximum (baseline + amplitude = 1000e3)
    model_state_t90 = model_state._replace(t=90*86400)
    theta_e_t90 = profile(model_state_t90)

    # At t=T/2 (180 days): y_0 should return to baseline (0)
    model_state_t180 = model_state._replace(t=180*86400)
    theta_e_t180 = profile(model_state_t180)

    # At t=3T/4 (270 days): y_0 should be at minimum (baseline - amplitude = -1000e3)
    model_state_t270 = model_state._replace(t=270*86400)
    theta_e_t270 = profile(model_state_t270)

    # At t=T (360 days): should complete cycle back to baseline
    model_state_t360 = model_state._replace(t=360*86400)
    theta_e_t360 = profile(model_state_t360)

    # t=0 and t=180 should be similar (both at baseline)
    assert np.allclose(theta_e_t0, theta_e_t180, rtol=1e-5)

    # t=0 and t=360 should be identical (full cycle)
    assert np.allclose(theta_e_t0, theta_e_t360, rtol=1e-5)

    # t=90 and t=270 should be different (max vs min)
    assert not np.allclose(theta_e_t90, theta_e_t270)

    # Verify that profiles actually change with time
    assert not np.allclose(theta_e_t0, theta_e_t90)


def test_sb08_seasonal_phase_offset(theta_e_config, model_state):
    """Test that phase offset shifts the seasonal cycle correctly"""
    # Setup with 90 day phase offset
    theta_e_config.y_0_seasonal_amp = 1000e3
    theta_e_config.seasonal_period_days = 360.0
    theta_e_config.seasonal_phase_days = 90.0  # Quarter period offset

    profile = SB08Profile(theta_e_config)

    # With 90 day phase offset:
    # At t=0: phase = -π/2, so sin(phase) = -1, y_0 at minimum
    model_state_t0 = model_state._replace(t=0)
    theta_e_t0 = profile(model_state_t0)

    # At t=90: phase = 0, so sin(phase) = 0, y_0 at baseline
    model_state_t90 = model_state._replace(t=90*86400)
    theta_e_t90 = profile(model_state_t90)

    # At t=180: phase = π/2, so sin(phase) = 1, y_0 at maximum
    model_state_t180 = model_state._replace(t=180*86400)
    theta_e_t180 = profile(model_state_t180)

    # Now create profile without phase offset for comparison
    theta_e_config_no_phase = ThetaEConfig(
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        theta_e_type="SB08",
        y_0_seasonal_amp=1000e3,
        seasonal_period_days=360.0,
        seasonal_phase_days=0.0
    )
    profile_no_phase = SB08Profile(theta_e_config_no_phase)

    # With no phase offset, t=0 has sin(0)=0 (baseline)
    # With 90 day offset, t=0 has sin(-π/2)=-1 (minimum)
    # So with phase, t=0 should match no-phase at t=270 (minimum)
    model_state_t270_no_phase = model_state._replace(t=270*86400)
    theta_e_t270_no_phase = profile_no_phase(model_state_t270_no_phase)

    assert np.allclose(theta_e_t0, theta_e_t270_no_phase, rtol=1e-5)


def test_sb08_zero_amplitude_equals_no_cycle(theta_e_config, model_state):
    """Test that y_0_seasonal_amp=0 behaves exactly like no seasonal parameters"""
    # Profile with explicit zero amplitude
    theta_e_config.y_0_seasonal_amp = 0.0
    theta_e_config.seasonal_period_days = 360.0
    profile_with_zero = SB08Profile(theta_e_config)

    # Profile with defaults (amplitude defaults to 0)
    theta_e_config_default = ThetaEConfig(
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        theta_e_type="SB08"
    )
    profile_default = SB08Profile(theta_e_config_default)

    # Should produce identical results
    theta_e_zero = profile_with_zero(model_state)
    theta_e_default = profile_default(model_state)

    assert np.allclose(theta_e_zero, theta_e_default)
