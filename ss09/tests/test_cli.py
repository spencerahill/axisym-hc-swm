import pytest
import subprocess
import json
from ss09.cli import setup_sw_config, setup_theta_e_config
from ss09.sw_config import SWConfig
from ss09.theta_e import ThetaEConfig
import argparse


@pytest.fixture
def default_args():
    """Default argument values using internal attribute names (not CLI flags)."""
    return {
        "total_integration_days": 1,
        "gravity": 9.81,
        "height": 16e3,
        "beta": 2e-11,
        "t_ref": 300.0,
        "output_path": "./model_output/output.nc",
        "ny": 51,
        "dt": 3600,
        "theta_e_type": "sin2",
        "y_0": 0.0,
        "delta_y": 50.0,
        "theta_00": 330.0,
        "y_one": 9439e3,
        "coeff_eddy_heat_diff": 0.0,
        "k_v": 778600,
        "epsilon_u": 1e-8,
        "delta_z": 60,
        "delta": 4e3,
        "tau": 37.0 * 86400,
        "v_d": 2.5,
        "domain_size": 15751e3 * 2,
        "k_u": 1e5,
        "include_vert_advec_u": True,
        "include_merid_advec_u": True,
        "enable_steady_state": False,
        "steady_state_window_size": 10,
        "steady_state_threshold": 0.001,
        "steady_state_check_both": True,
        "smoothness_threshold": 0.5,
        "restart_output_dir": "./model_output",
    }


@pytest.mark.parametrize(
    "attr, value",
    [
        ("total_integration_days", 3),
        ("gravity", 9.8),
        ("height", 15000),
        ("beta", 2.1e-11),
        ("t_ref", 290.0),
        ("output_path", "./output/test.nc"),
        ("ny", 60),
        ("dt", 1800),
        ("theta_e_type", "SS09"),
        ("y_0", 100.0),
        ("delta_y", 60.0),
        ("theta_00", 340.0),
        ("y_one", 9500e3),
        ("coeff_eddy_heat_diff", 0.1),
        ("k_v", 800000),
        ("epsilon_u", 1e-7),
        ("delta_z", 70),
        ("delta", 5000),
        ("tau", 40.0 * 86400),
        ("v_d", 3.0),
        ("domain_size", 16000e3 * 2),
        ("k_u", 5e4),
        ("include_vert_advec_u", False),
        ("include_merid_advec_u", False),
    ],
)
def test_cli_arguments(default_args, attr, value):
    """Test that config setup functions correctly map argument values."""
    args = default_args.copy()
    args[attr] = value

    # Create Namespace directly from attribute names
    parsed_args = argparse.Namespace(**args)

    # Setup configurations
    theta_e_config = setup_theta_e_config(parsed_args)
    sw_config = setup_sw_config(parsed_args, theta_e_config)

    # Verify ThetaEConfig if applicable
    if attr in ["theta_00", "y_0", "y_one", "delta_y", "theta_e_type"]:
        assert getattr(theta_e_config, attr) == value
    else:
        assert getattr(sw_config, attr) == value


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
        theta_e_type="sin2",
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        coeff_eddy_heat_diff=0.0,
        k_v=778600,
        epsilon_u=1e-8,
        delta_z=60,
        delta=4e3,
        tau=37.0 * 86400,
        v_d=2.5,
        domain_size=15751e3 * 2,
        k_u=1e5,
        include_vert_advec_u=True,
        include_merid_advec_u=True,
        enable_steady_state=True,
        steady_state_window_size=15,
        steady_state_threshold=0.0005,
        steady_state_check_both=False,
        smoothness_threshold=0.7,
        restart_output_dir="./model_output",
    )

    theta_e_config = setup_theta_e_config(args)
    sw_config = setup_sw_config(args, theta_e_config)

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
        theta_e_type="sin2",
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        coeff_eddy_heat_diff=0.0,
        k_v=778600,
        epsilon_u=1e-8,
        delta_z=60,
        delta=4e3,
        tau=37.0 * 86400,
        v_d=2.5,
        domain_size=15751e3 * 2,
        k_u=1e5,
        include_vert_advec_u=True,
        include_merid_advec_u=True,
        enable_steady_state=False,
        steady_state_window_size=10,
        steady_state_threshold=0.001,
        steady_state_check_both=True,
        smoothness_threshold=0.5,
        restart_output_dir="./model_output",
    )

    theta_e_config = setup_theta_e_config(args)
    sw_config = setup_sw_config(args, theta_e_config)

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


def test_cli_seasonal_cycle_type_passed_to_config():
    """Test that --seas-cycle-type is passed to ThetaEConfig"""
    # Test square wave
    args_square = argparse.Namespace(
        theta_e_type="SB08",
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        y_0_seasonal_amp=700e3,
        seasonal_period_days=360.0,
        seasonal_phase_days=0.0,
        seasonal_cycle_type="square",
    )

    theta_e_config_square = setup_theta_e_config(args_square)
    assert theta_e_config_square.seasonal_cycle_type == "square"

    # Test sin (default)
    args_sin = argparse.Namespace(
        theta_e_type="SB08",
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        y_0_seasonal_amp=700e3,
        seasonal_period_days=360.0,
        seasonal_phase_days=0.0,
        seasonal_cycle_type="sin",
    )

    theta_e_config_sin = setup_theta_e_config(args_sin)
    assert theta_e_config_sin.seasonal_cycle_type == "sin"


def test_cli_seasonal_cycle_type_default():
    """Test that seasonal_cycle_type defaults to 'sin' when not provided"""
    # Namespace without seasonal_cycle_type attribute
    args = argparse.Namespace(
        theta_e_type="SB08",
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        y_0_seasonal_amp=700e3,
        seasonal_period_days=360.0,
        seasonal_phase_days=0.0,
        # Note: seasonal_cycle_type NOT included
    )

    theta_e_config = setup_theta_e_config(args)
    assert theta_e_config.seasonal_cycle_type == "sin"


def test_cli_tanh_seasonal_cycle_type():
    """Test that --seas-cycle-type tanh is passed to ThetaEConfig"""
    args = argparse.Namespace(
        theta_e_type="SB08",
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        y_0_seasonal_amp=700e3,
        seasonal_period_days=360.0,
        seasonal_phase_days=0.0,
        seasonal_cycle_type="tanh",
        tanh_steepness=4.0,
    )

    theta_e_config = setup_theta_e_config(args)
    assert theta_e_config.seasonal_cycle_type == "tanh"
    assert theta_e_config.tanh_steepness == 4.0


def test_cli_tanh_steepness_custom():
    """Test that tanh_steepness is passed to ThetaEConfig"""
    args = argparse.Namespace(
        theta_e_type="SB08",
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        y_0_seasonal_amp=700e3,
        seasonal_period_days=360.0,
        seasonal_phase_days=0.0,
        seasonal_cycle_type="tanh",
        tanh_steepness=8.0,
    )

    theta_e_config = setup_theta_e_config(args)
    assert theta_e_config.tanh_steepness == 8.0


def test_cli_tanh_steepness_default():
    """Test that tanh_steepness defaults to 4.0 when not provided"""
    args = argparse.Namespace(
        theta_e_type="SB08",
        theta_00=330.0,
        y_0=0.0,
        y_one=9439e3,
        delta_y=50.0,
        y_0_seasonal_amp=700e3,
        seasonal_period_days=360.0,
        seasonal_phase_days=0.0,
        seasonal_cycle_type="tanh",
        # Note: tanh_steepness NOT included
    )

    theta_e_config = setup_theta_e_config(args)
    assert theta_e_config.tanh_steepness == 4.0


def test_cli_rejects_ndays_with_steady_state():
    """CLI should error when both --ndays and --stop-at-steady-state are provided"""
    from ss09.cli import parse_arguments
    import sys

    # Simulate CLI args with both flags
    test_args = ["run-sw-model", "--ndays", "100", "--stop-at-steady-state"]
    original_argv = sys.argv
    sys.argv = test_args

    try:
        with pytest.raises(SystemExit) as exc_info:
            parse_arguments()
        # Should exit with error (non-zero exit code or error message)
        assert exc_info.value.code != 0 or "Cannot specify both" in str(exc_info.value)
    finally:
        sys.argv = original_argv


def test_cli_default_ndays_without_steady_state():
    """When --stop-at-steady-state is NOT used, ndays should default to 250"""
    from ss09.cli import parse_arguments
    import sys

    # Simulate CLI args without --ndays and without --stop-at-steady-state
    test_args = ["run-sw-model"]
    original_argv = sys.argv
    sys.argv = test_args

    try:
        args = parse_arguments()
        assert args.total_integration_days == 250
    finally:
        sys.argv = original_argv


def test_cli_default_ndays_with_steady_state():
    """When --stop-at-steady-state is used without --ndays, default to 200000 days"""
    from ss09.cli import parse_arguments
    import sys

    # Simulate CLI args with --stop-at-steady-state but no --ndays
    test_args = ["run-sw-model", "--stop-at-steady-state"]
    original_argv = sys.argv
    sys.argv = test_args

    try:
        args = parse_arguments()
        assert args.total_integration_days == 200000
    finally:
        sys.argv = original_argv
