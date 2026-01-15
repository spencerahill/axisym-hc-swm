import pytest
import numpy as np
from ss09.theta_e import (
    ThetaEProfile,
    ThetaEConfig,
    SS09Profile,
    Sin2Profile,
    SB08Profile,
)
from ss09.model_state import ModelState


# Mock subclass for testing
class MockThetaEProfile(ThetaEProfile):
    def __call__(self, state: ModelState) -> np.ndarray:
        return np.full_like(state.y, self.config.theta_00)


def test_theta_e_profile_initialization():
    config = ThetaEConfig(theta_00=320.0)
    profile = MockThetaEProfile(config)

    assert profile.config.theta_00 == 320.0


def test_theta_e_profile_call():
    config = ThetaEConfig(theta_00=320.0)
    profile = MockThetaEProfile(config)
    state = ModelState(
        t=0.0, u=np.zeros(5), v=np.zeros(5), theta=np.zeros(5), y=np.linspace(-1, 1, 5)
    )

    result = profile(state)
    expected = np.full(5, 320.0)

    assert np.array_equal(result, expected)


def test_ss09_profile_call():
    config = ThetaEConfig(theta_00=310.0, y_0=1000e3, y_one=9000e3, delta_y=45)
    profile = SS09Profile(config)
    state = ModelState(
        t=0.0, u=np.zeros(5), v=np.zeros(5), theta=np.zeros(5), y=np.linspace(-1, 1, 5)
    )

    result = profile(state)
    expected = np.where(
        np.abs(state.y) < config.y_one,
        config.theta_00 - config.delta_y * (state.y / config.y_one) ** 2,
        config.theta_00 - config.delta_y,
    )

    assert np.array_equal(result, expected)


def test_sin2_profile_call():
    config = ThetaEConfig(theta_00=310.0, y_0=1000e3, y_one=9000e3, delta_y=45)
    profile = Sin2Profile(config)
    state = ModelState(
        t=0.0, u=np.zeros(5), v=np.zeros(5), theta=np.zeros(5), y=np.linspace(-1, 1, 5)
    )

    result = profile(state)
    expected = np.where(
        np.abs(state.y - config.y_0) < config.y_one,
        config.theta_00
        - config.delta_y
        * (np.sin(0.5 * np.pi * (state.y - config.y_0) / config.y_one) ** 2),
        config.theta_00 - config.delta_y,
    )

    assert np.array_equal(result, expected)


def test_sb08_profile_call():
    """Test SB08 profile in the tropics where floor is not active."""
    config = ThetaEConfig(theta_00=310.0, y_0=1000e3, y_one=9000e3, delta_y=45)
    profile = SB08Profile(config)
    # Use small y values where floor won't be active
    state = ModelState(
        t=0.0, u=np.zeros(5), v=np.zeros(5), theta=np.zeros(5), y=np.linspace(-1, 1, 5)
    )

    term1 = np.sin(np.pi * state.y / (2 * config.y_one)) ** 2
    term2 = (
        2
        * np.sin(np.pi * config.y_0 / (2 * config.y_one))
        * np.sin(np.pi * state.y / (2 * config.y_one))
    )
    raw_profile = config.theta_00 - config.delta_y * (term1 - term2)
    theta_e_min = config.theta_00 - config.delta_y
    expected = np.maximum(theta_e_min, raw_profile)

    result = profile(state)

    assert np.array_equal(result, expected)


def test_sb08_profile_floor_at_high_latitudes():
    """Test that SB08 profile applies minimum temperature floor at high latitudes.

    With nonzero y_0, the cross-term can push raw profile below theta_00 - delta_y
    at certain high latitudes. The floor prevents this.
    """
    config = ThetaEConfig(theta_00=330.0, y_0=500e3, y_one=9439e3, delta_y=50)
    profile = SB08Profile(config)

    # At y = 3*y_one with y_0 != 0, the raw profile goes below floor
    # Raw value ~271.7 K should be clipped to floor of 280 K
    y_high_lat = np.array([3.0 * config.y_one])
    state = ModelState(
        t=0.0,
        u=np.zeros_like(y_high_lat),
        v=np.zeros_like(y_high_lat),
        theta=np.zeros_like(y_high_lat),
        y=y_high_lat,
    )

    result = profile(state)
    theta_e_min = config.theta_00 - config.delta_y  # 280 K

    # Compute what raw value would be (without floor)
    term1 = np.sin(np.pi * y_high_lat / (2 * config.y_one)) ** 2
    term2 = 2 * np.sin(np.pi * config.y_0 / (2 * config.y_one)) * np.sin(
        np.pi * y_high_lat / (2 * config.y_one)
    )
    raw_value = config.theta_00 - config.delta_y * (term1 - term2)

    # Raw value should be below floor
    assert raw_value[0] < theta_e_min, "Test setup: raw value should be below floor"

    # Result should be at the floor (not the raw value)
    np.testing.assert_array_almost_equal(result, theta_e_min)


def test_sb08_profile_floor_with_seasonal_forcing():
    """Test that floor works correctly with seasonal ITCZ migration.

    At peak seasonal offset (y_0 = 700 km), the cross-term at y = 3*y_one
    pushes the raw profile below the floor.
    """
    config = ThetaEConfig(
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50,
        y_0_seasonal_amp=700e3,
        seasonal_period_days=360.0,
    )
    profile = SB08Profile(config)

    # Test at y = 3*y_one with peak seasonal offset (t = 90 days)
    # At this time, y_0_t = 700 km, which pushes profile below floor
    y_high_lat = np.array([3.0 * config.y_one])
    state = ModelState(
        t=90 * 86400,  # 90 days - peak seasonal offset (y_0 = +700 km)
        u=np.zeros_like(y_high_lat),
        v=np.zeros_like(y_high_lat),
        theta=np.zeros_like(y_high_lat),
        y=y_high_lat,
    )

    result = profile(state)
    theta_e_min = config.theta_00 - config.delta_y

    # Compute expected y_0_t at t = 90 days
    period_seconds = config.seasonal_period_days * 86400
    phase = 2 * np.pi * (90 * 86400) / period_seconds
    y_0_t = config.y_0 + config.y_0_seasonal_amp * np.sin(phase)

    # Compute raw value (without floor)
    term1 = np.sin(np.pi * y_high_lat / (2 * config.y_one)) ** 2
    term2 = 2 * np.sin(np.pi * y_0_t / (2 * config.y_one)) * np.sin(
        np.pi * y_high_lat / (2 * config.y_one)
    )
    raw_value = config.theta_00 - config.delta_y * (term1 - term2)

    # Raw value should be below floor when seasonal offset is active
    assert raw_value[0] < theta_e_min, "Test setup: raw value should be below floor"

    # Result should be clipped to floor
    np.testing.assert_array_almost_equal(result, theta_e_min)
