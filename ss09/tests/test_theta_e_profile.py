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
    """Test SB08 profile in the tropics where clamping is not active."""
    config = ThetaEConfig(theta_00=310.0, y_0=1000e3, y_one=9000e3, delta_y=45)
    profile = SB08Profile(config)
    # Use small y values inside [-y_one, +y_one] where clamping won't apply
    state = ModelState(
        t=0.0, u=np.zeros(5), v=np.zeros(5), theta=np.zeros(5), y=np.linspace(-1, 1, 5)
    )

    # Expected is just the raw SB08 formula (no clamping for |y| < y_one)
    term1 = np.sin(np.pi * state.y / (2 * config.y_one)) ** 2
    term2 = (
        2
        * np.sin(np.pi * config.y_0 / (2 * config.y_one))
        * np.sin(np.pi * state.y / (2 * config.y_one))
    )
    expected = config.theta_00 - config.delta_y * (term1 - term2)

    result = profile(state)

    np.testing.assert_array_almost_equal(result, expected)


def test_sb08_profile_clamps_at_high_latitudes():
    """Test that SB08 profile clamps to boundary values beyond ±y₁.

    For y > y₁, use θₑ(+y₁). For y < -y₁, use θₑ(-y₁).
    """
    config = ThetaEConfig(theta_00=330.0, y_0=500e3, y_one=9439e3, delta_y=50)
    profile = SB08Profile(config)

    # Test points: inside, at boundary, and outside in both hemispheres
    y_vals = np.array([
        -2.0 * config.y_one,  # far south (should clamp to value at -y₁)
        -config.y_one,         # at -y₁
        0.0,                   # equator
        config.y_one,          # at +y₁
        2.0 * config.y_one,   # far north (should clamp to value at +y₁)
    ])
    state = ModelState(
        t=0.0,
        u=np.zeros_like(y_vals),
        v=np.zeros_like(y_vals),
        theta=np.zeros_like(y_vals),
        y=y_vals,
    )

    result = profile(state)

    # Compute expected boundary values
    def sb08_raw(y_val):
        term1 = np.sin(np.pi * y_val / (2 * config.y_one)) ** 2
        term2 = 2 * np.sin(np.pi * config.y_0 / (2 * config.y_one)) * np.sin(
            np.pi * y_val / (2 * config.y_one)
        )
        return config.theta_00 - config.delta_y * (term1 - term2)

    theta_at_minus_y1 = sb08_raw(-config.y_one)
    theta_at_plus_y1 = sb08_raw(config.y_one)

    # Check: far south should equal value at -y₁
    np.testing.assert_almost_equal(result[0], theta_at_minus_y1)
    # Check: at -y₁ should equal value at -y₁
    np.testing.assert_almost_equal(result[1], theta_at_minus_y1)
    # Check: equator uses raw formula
    np.testing.assert_almost_equal(result[2], sb08_raw(0.0))
    # Check: at +y₁ should equal value at +y₁
    np.testing.assert_almost_equal(result[3], theta_at_plus_y1)
    # Check: far north should equal value at +y₁
    np.testing.assert_almost_equal(result[4], theta_at_plus_y1)


def test_sb08_profile_clamping_with_seasonal_forcing():
    """Test that clamping works correctly with seasonal ITCZ migration.

    With y₀ ≠ 0 (due to seasonal forcing), the values at +y₁ and -y₁ differ,
    so each hemisphere should clamp to its respective boundary value.
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

    # At t = 90 days, y_0_t = +700 km (peak seasonal offset)
    t = 90 * 86400
    period_seconds = config.seasonal_period_days * 86400
    phase = 2 * np.pi * t / period_seconds
    y_0_t = config.y_0 + config.y_0_seasonal_amp * np.sin(phase)

    # Test at high latitudes in both hemispheres
    y_vals = np.array([-1.5 * config.y_one, 1.5 * config.y_one])
    state = ModelState(
        t=t,
        u=np.zeros_like(y_vals),
        v=np.zeros_like(y_vals),
        theta=np.zeros_like(y_vals),
        y=y_vals,
    )

    result = profile(state)

    # Compute expected boundary values with seasonal y_0_t
    def sb08_raw(y_val):
        term1 = np.sin(np.pi * y_val / (2 * config.y_one)) ** 2
        term2 = 2 * np.sin(np.pi * y_0_t / (2 * config.y_one)) * np.sin(
            np.pi * y_val / (2 * config.y_one)
        )
        return config.theta_00 - config.delta_y * (term1 - term2)

    theta_at_minus_y1 = sb08_raw(-config.y_one)
    theta_at_plus_y1 = sb08_raw(config.y_one)

    # With y_0_t > 0, the two boundary values should differ
    assert theta_at_minus_y1 != theta_at_plus_y1, "Test setup: boundaries should differ"

    # Check clamping in each hemisphere
    np.testing.assert_almost_equal(result[0], theta_at_minus_y1)  # south
    np.testing.assert_almost_equal(result[1], theta_at_plus_y1)   # north
