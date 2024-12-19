import pytest
from ss09.theta_e import ThetaEConfig


def test_theta_e_config_initialization():
    config = ThetaEConfig(
        theta_00=320.0, y_0=1.0, y_one=9000e3, delta_y=45.0, theta_e_type="SB08"
    )

    assert config.theta_00 == 320.0
    assert config.y_0 == 1.0
    assert config.y_one == 9000e3
    assert config.delta_y == 45.0
    assert config.theta_e_type == "SB08"
