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
        ny=51,
        dt=3600,
        kappa_theta=0.0,
    )


@pytest.fixture
def theta_e_config():
    return ThetaEConfig(theta_00=310.0, y_0=1000e3, y_one=9000e3, delta_y=45)


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
