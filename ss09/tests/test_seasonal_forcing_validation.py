"""Tests for seasonal forcing validation in ThetaE profiles."""
import pytest
import numpy as np
from ss09.theta_e import (
    ThetaEConfig,
    SS09Profile,
    Sin2Profile,
    SB08Profile,
)
from ss09.model_state import ModelState


def test_ss09_rejects_seasonal_forcing():
    """SS09Profile should raise ValueError when seasonal forcing is enabled"""
    config = ThetaEConfig(
        theta_e_type="SS09",
        y_0_seasonal_amp=700e3,  # 700 km seasonal migration
    )

    with pytest.raises(ValueError) as exc_info:
        SS09Profile(config)

    assert "Seasonal forcing" in str(exc_info.value)
    assert "SS09Profile" in str(exc_info.value)
    assert "not supported" in str(exc_info.value)
    assert "SB08" in str(exc_info.value)


def test_sin2_rejects_seasonal_forcing():
    """Sin2Profile should raise ValueError when seasonal forcing is enabled"""
    config = ThetaEConfig(
        theta_e_type="sin2",
        y_0_seasonal_amp=500e3,  # 500 km seasonal migration
    )

    with pytest.raises(ValueError) as exc_info:
        Sin2Profile(config)

    assert "Seasonal forcing" in str(exc_info.value)
    assert "Sin2Profile" in str(exc_info.value)
    assert "not supported" in str(exc_info.value)


def test_sb08_allows_seasonal_forcing():
    """SB08Profile should allow seasonal forcing without errors"""
    config = ThetaEConfig(
        theta_e_type="SB08",
        y_0_seasonal_amp=700e3,  # 700 km seasonal migration
        seasonal_period_days=360.0,
    )

    # Should not raise
    profile = SB08Profile(config)

    # Verify it actually uses the seasonal parameters
    state = ModelState(
        t=0.0,
        u=np.zeros(5),
        v=np.zeros(5),
        theta=np.zeros(5),
        y=np.linspace(-1e6, 1e6, 5),
    )

    theta_e_t0 = profile(state)

    # At t = 90 days (quarter period), y_0 should be at maximum
    state_t90 = state._replace(t=90 * 86400)
    theta_e_t90 = profile(state_t90)

    # Results should differ because y_0 changed
    assert not np.array_equal(theta_e_t0, theta_e_t90)


def test_all_profiles_allow_zero_seasonal_amp():
    """All profiles should work fine when seasonal forcing is disabled"""
    config = ThetaEConfig(
        theta_e_type="SS09",
        y_0_seasonal_amp=0.0,  # No seasonal forcing
    )

    # All should instantiate without error
    ss09 = SS09Profile(config)

    config_sin2 = ThetaEConfig(theta_e_type="sin2", y_0_seasonal_amp=0.0)
    sin2 = Sin2Profile(config_sin2)

    config_sb08 = ThetaEConfig(theta_e_type="SB08", y_0_seasonal_amp=0.0)
    sb08 = SB08Profile(config_sb08)

    # All should be callable
    state = ModelState(
        t=0.0,
        u=np.zeros(5),
        v=np.zeros(5),
        theta=np.zeros(5),
        y=np.linspace(-1e6, 1e6, 5),
    )

    ss09(state)
    sin2(state)
    sb08(state)


def test_edge_case_small_nonzero_seasonal_amp():
    """Even very small nonzero seasonal amplitude should be rejected for SS09/sin2"""
    config = ThetaEConfig(
        theta_e_type="SS09",
        y_0_seasonal_amp=1.0,  # 1 meter - tiny but nonzero
    )

    with pytest.raises(ValueError):
        SS09Profile(config)


def test_error_message_contains_fix_instructions():
    """Error message should provide clear instructions on how to fix"""
    config = ThetaEConfig(
        theta_e_type="sin2",
        y_0_seasonal_amp=1000e3,
    )

    with pytest.raises(ValueError) as exc_info:
        Sin2Profile(config)

    error_msg = str(exc_info.value)
    # Check that error message includes both fix options
    assert "--theta_e_type SB08" in error_msg or "SB08" in error_msg
    assert "y0-seasonal-amp 0" in error_msg or "y_0_seasonal_amp" in error_msg
