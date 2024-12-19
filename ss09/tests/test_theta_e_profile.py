import pytest
import numpy as np
from ss09.theta_e import ThetaEProfile, ThetaEConfig
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
