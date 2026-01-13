import pytest
import subprocess
import json
from ss09.cli import setup_sw_config, setup_theta_e_config
from ss09.sw_config import SWConfig
from ss09.theta_e import ThetaEConfig
import argparse


@pytest.fixture
def default_args():
    return {
        "--total_integration_days": 1,
        "--gravity": 9.81,
        "--height": 16e3,
        "--beta": 2e-11,
        "--t_ref": 300.0,
        "--output_path": "./model_output/output.nc",
        "--ny": 51,
        "--dt": 3600,
        "--theta_e_type": "sin2",
        "--y_0": 0.0,
        "--delta_y": 50.0,
        "--theta_00": 330.0,
        "--y_one": 9439e3,
        "--coeff_eddy_heat_diff": 0.0,
        "--k_v": 778600,
        "--epsilon_u": 1e-8,
        "--delta_z": 60,
        "--delta": 4e3,
        "--tau": 37.0 * 86400,
        "--v_d": 2.5,
        "--domain_size": 15751e3 * 2,
        "--asselin_filt_coef": 0.04,
        "--include_vert_advec_u": True,
        "--include_merid_advec_u": True,
        "--enable_steady_state": False,
        "--steady_state_window_size": 10,
        "--steady_state_threshold": 0.001,
        "--steady_state_check_both": True,
        "--smoothness_threshold": 0.5,
    }


@pytest.mark.parametrize(
    "param, value",
    [
        ("--total_integration_days", 3),
        ("--gravity", 9.8),
        ("--height", 15000),
        ("--beta", 2.1e-11),
        ("--t_ref", 290.0),
        ("--output_path", "./output/test.nc"),
        ("--ny", 60),
        ("--dt", 1800),
        ("--theta_e_type", "SS09"),
        ("--y_0", 100.0),
        ("--delta_y", 60.0),
        ("--theta_00", 340.0),
        ("--y_one", 9500e3),
        ("--coeff_eddy_heat_diff", 0.1),
        ("--k_v", 800000),
        ("--epsilon_u", 1e-7),
        ("--delta_z", 70),
        ("--delta", 5000),
        ("--tau", 40.0 * 86400),
        ("--v_d", 3.0),
        ("--domain_size", 16000e3 * 2),
        ("--asselin_filt_coef", 0.05),
        ("--no-vert-advec-u", False),
        ("--no-merid-advec-u", False),
    ],
)
def test_cli_arguments(default_args, param, value):
    args = default_args.copy()
    if param == "--no-vert-advec-u":
        args["--include_vert_advec_u"] = False
    elif param == "--no-merid-advec-u":
        args["--include_merid_advec_u"] = False
    else:
        args[param] = value

    # Convert arguments to a list of strings
    cli_args = [str(item) for sublist in args.items() for item in sublist]

    # Simulate argument parsing
    parsed_args = argparse.Namespace(**{k.lstrip("--"): v for k, v in args.items()})

    # Setup configurations
    sw_config = setup_sw_config(parsed_args)
    theta_e_config = setup_theta_e_config(parsed_args)

    # Verify ThetaEConfig if applicable
    if param in ["--theta_00", "--y_0", "--y_one", "--delta_y", "--theta_e_type"]:
        assert getattr(theta_e_config, param.lstrip("--")) == value
    elif param == "--no-vert-advec-u":
        assert sw_config.include_vert_advec_u == value
    elif param == "--no-merid-advec-u":
        assert sw_config.include_merid_advec_u == value
    else:
        assert getattr(sw_config, param.lstrip("--")) == value


def test_cli_steady_state_args():
    """Test that CLI properly parses steady-state arguments"""
    args = argparse.Namespace(
        total_integration_days=250,
        gravity=9.81,
        height=16e3,
        beta=2e-11,
        t_ref=300.0,
        output_path="./model_output/output.nc",
        ny=51,
        dt=3600,
        coeff_eddy_heat_diff=0.0,
        k_v=778600,
        epsilon_u=1e-8,
        delta_z=60,
        delta=4e3,
        tau=37.0 * 86400,
        v_d=2.5,
        domain_size=15751e3 * 2,
        asselin_filt_coef=0.04,
        include_vert_advec_u=True,
        include_merid_advec_u=True,
        enable_steady_state=True,
        steady_state_window_size=15,
        steady_state_threshold=0.0005,
        steady_state_check_both=False,
        smoothness_threshold=0.7,
    )

    sw_config = setup_sw_config(args)

    assert sw_config.enable_steady_state is True
    assert sw_config.steady_state_window_size == 15
    assert sw_config.steady_state_threshold == 0.0005
    assert sw_config.steady_state_check_both is False
    assert sw_config.smoothness_threshold == 0.7


def test_cli_steady_state_defaults():
    """Test that steady-state parameters have correct defaults"""
    args = argparse.Namespace(
        total_integration_days=250,
        gravity=9.81,
        height=16e3,
        beta=2e-11,
        t_ref=300.0,
        output_path="./model_output/output.nc",
        ny=51,
        dt=3600,
        coeff_eddy_heat_diff=0.0,
        k_v=778600,
        epsilon_u=1e-8,
        delta_z=60,
        delta=4e3,
        tau=37.0 * 86400,
        v_d=2.5,
        domain_size=15751e3 * 2,
        asselin_filt_coef=0.04,
        include_vert_advec_u=True,
        include_merid_advec_u=True,
        enable_steady_state=False,
        steady_state_window_size=10,
        steady_state_threshold=0.001,
        steady_state_check_both=True,
        smoothness_threshold=0.5,
    )

    sw_config = setup_sw_config(args)

    assert sw_config.enable_steady_state is False
    assert sw_config.steady_state_window_size == 10
    assert sw_config.steady_state_threshold == 0.001
    assert sw_config.steady_state_check_both is True
    assert sw_config.smoothness_threshold == 0.5


def test_cli_rejects_seasonal_forcing_with_ss09():
    """CLI should fail when user tries seasonal forcing with SS09 profile"""
    from ss09.theta_e import SS09Profile

    args = argparse.Namespace(
        theta_e_type="SS09",
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        y_0_seasonal_amp=700e3,  # Enable seasonal forcing
        seasonal_period_days=360.0,
        seasonal_phase_days=0.0,
    )

    theta_e_config = setup_theta_e_config(args)

    # Attempting to instantiate profile should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        SS09Profile(theta_e_config)

    assert "Seasonal forcing" in str(exc_info.value)
    assert "SS09Profile" in str(exc_info.value)
    assert "not supported" in str(exc_info.value)


def test_cli_rejects_seasonal_forcing_with_sin2():
    """CLI should fail when user tries seasonal forcing with sin2 profile"""
    from ss09.theta_e import Sin2Profile

    args = argparse.Namespace(
        theta_e_type="sin2",
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        y_0_seasonal_amp=500e3,  # Enable seasonal forcing
        seasonal_period_days=360.0,
        seasonal_phase_days=0.0,
    )

    theta_e_config = setup_theta_e_config(args)

    # Attempting to instantiate profile should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        Sin2Profile(theta_e_config)

    assert "Seasonal forcing" in str(exc_info.value)
    assert "Sin2Profile" in str(exc_info.value)


def test_cli_accepts_seasonal_forcing_with_sb08():
    """CLI should succeed when user uses seasonal forcing with SB08 profile"""
    from ss09.theta_e import SB08Profile

    args = argparse.Namespace(
        theta_e_type="SB08",
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        y_0_seasonal_amp=700e3,  # Enable seasonal forcing
        seasonal_period_days=360.0,
        seasonal_phase_days=0.0,
    )

    theta_e_config = setup_theta_e_config(args)

    # Profile instantiation should succeed
    profile = SB08Profile(theta_e_config)

    assert profile.config.y_0_seasonal_amp == 700e3
    assert profile.config.seasonal_period_days == 360.0


def test_cli_accepts_all_profiles_without_seasonal_forcing():
    """All profiles should work when seasonal forcing is disabled"""
    from ss09.theta_e import SS09Profile, Sin2Profile, SB08Profile

    for theta_e_type, profile_class in [
        ("SS09", SS09Profile),
        ("sin2", Sin2Profile),
        ("SB08", SB08Profile),
    ]:
        args = argparse.Namespace(
            theta_e_type=theta_e_type,
            theta_00=330.0,
            y_0=0.0,
            y_one=9439e3,
            delta_y=50.0,
            y_0_seasonal_amp=0.0,  # No seasonal forcing
            seasonal_period_days=360.0,
            seasonal_phase_days=0.0,
        )

        theta_e_config = setup_theta_e_config(args)

        # All profiles should instantiate without error
        profile = profile_class(theta_e_config)
        assert profile.config.y_0_seasonal_amp == 0.0
