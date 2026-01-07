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
    ],
)
def test_cli_arguments(default_args, param, value):
    args = default_args.copy()
    if param == "--no-vert-advec-u":
        args["--include_vert_advec_u"] = False
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
