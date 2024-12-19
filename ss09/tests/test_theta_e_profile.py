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
    config = ThetaEConfig(theta_00=310.0, y_0=1000e3, y_one=9000e3, delta_y=45)
    profile = SB08Profile(config)
    state = ModelState(
        t=0.0, u=np.zeros(5), v=np.zeros(5), theta=np.zeros(5), y=np.linspace(-1, 1, 5)
    )

    term1 = np.sin(np.pi * state.y / (2 * config.y_one)) ** 2
    term2 = (
        2
        * np.sin(np.pi * config.y_0 / (2 * config.y_one))
        * np.sin(np.pi * state.y / (2 * config.y_one))
    )
    expected = config.theta_00 - config.delta_y * (term1 + term2)

    result = profile(state)

    assert np.array_equal(result, expected)
