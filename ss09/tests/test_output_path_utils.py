"""Tests for output path generation utilities."""

import pytest
from ss09.output_path_utils import generate_descriptive_path, generate_restart_filename
from ss09.sw_config import SWConfig
from ss09.theta_e import ThetaEConfig


class TestGenerateDescriptivePath:
    """Tests for generate_descriptive_path()"""

    def test_seasonal_sb08_basic(self):
        """Test descriptive path generation for seasonal SB08 run."""
        config = SWConfig(
            total_integration_days=3600,
            ny=51,
            gravity=9.81,
            height=16e3,
            beta=2e-11,
            t_ref=300.0,
            output_path="",  # placeholder
            dt=3600,
            coeff_eddy_heat_diff=0.0,
            k_v=7786 * 100,
            epsilon_u=1e-8,
            delta_z=60,
            delta=4e3,
            tau=37.0 * 86400,
            v_d=2.5,
            domain_size=15751e3 * 2,
            include_vert_advec_u=True,
            include_merid_advec_u=True,
            enable_steady_state=False,
            steady_state_window_size=10,
            steady_state_threshold=0.001,
            steady_state_check_both=True,
            smoothness_threshold=0.5,
            seasonal_convergence_enabled=False,
            seasonal_convergence_window=30,
            seasonal_convergence_threshold=0.01,
            save_restart_every=0,
            restart_output_dir="",
        )

        theta_e_config = ThetaEConfig(
            theta_00=330.0,
            y_0=0.0,
            y_one=9439e3,
            delta_y=50.0,
            theta_e_type="SB08",
            y_0_seasonal_amp=700e3,  # 700 km seasonal migration
            seasonal_period_days=360.0,
            seasonal_phase_days=0.0,
        )

        timestamp = "20260111_134530"
        output_path, restart_dir = generate_descriptive_path(
            config, theta_e_config, base_dir="./model_output", timestamp=timestamp
        )

        # Check directory structure
        assert "SB08" in output_path
        assert restart_dir.endswith("SB08")

        # Check filename components
        assert "run_20260111_134530" in output_path
        assert "seas" in output_path  # seasonal indicator
        assert "y0p0000" in output_path  # y_0 = 0 km
        assert "ny051" in output_path
        assert "3600days" in output_path
        assert output_path.endswith("_output.nc")

    def test_nonseasonal_sin2(self):
        """Test descriptive path generation for non-seasonal sin2 run."""
        config = SWConfig(
            total_integration_days=250,
            ny=101,
            gravity=9.81,
            height=16e3,
            beta=2e-11,
            t_ref=300.0,
            output_path="",
            dt=3600,
            coeff_eddy_heat_diff=0.0,
            k_v=7786 * 100,
            epsilon_u=1e-8,
            delta_z=60,
            delta=4e3,
            tau=37.0 * 86400,
            v_d=2.5,
            domain_size=15751e3 * 2,
            include_vert_advec_u=True,
            include_merid_advec_u=True,
            enable_steady_state=False,
            steady_state_window_size=10,
            steady_state_threshold=0.001,
            steady_state_check_both=True,
            smoothness_threshold=0.5,
            seasonal_convergence_enabled=False,
            seasonal_convergence_window=30,
            seasonal_convergence_threshold=0.01,
            save_restart_every=0,
            restart_output_dir="",
        )

        theta_e_config = ThetaEConfig(
            theta_00=330.0,
            y_0=0.0,
            y_one=9439e3,
            delta_y=50.0,
            theta_e_type="sin2",
            y_0_seasonal_amp=0.0,  # No seasonal cycle
            seasonal_period_days=360.0,
            seasonal_phase_days=0.0,
        )

        timestamp = "20260111_120000"
        output_path, restart_dir = generate_descriptive_path(
            config, theta_e_config, base_dir="./model_output", timestamp=timestamp
        )

        # Check filename components
        assert "sin2" in output_path
        assert "noseas" in output_path  # non-seasonal
        assert "y0p0000" in output_path
        assert "ny101" in output_path
        assert "250days" in output_path
        assert output_path.endswith("_output.nc")

    def test_positive_y0(self):
        """Test y0 formatting for positive values."""
        config = SWConfig(
            total_integration_days=100,
            ny=51,
            gravity=9.81,
            height=16e3,
            beta=2e-11,
            t_ref=300.0,
            output_path="",
            dt=3600,
            coeff_eddy_heat_diff=0.0,
            k_v=7786 * 100,
            epsilon_u=1e-8,
            delta_z=60,
            delta=4e3,
            tau=37.0 * 86400,
            v_d=2.5,
            domain_size=15751e3 * 2,
            include_vert_advec_u=True,
            include_merid_advec_u=True,
            enable_steady_state=False,
            steady_state_window_size=10,
            steady_state_threshold=0.001,
            steady_state_check_both=True,
            smoothness_threshold=0.5,
            seasonal_convergence_enabled=False,
            seasonal_convergence_window=30,
            seasonal_convergence_threshold=0.01,
            save_restart_every=0,
            restart_output_dir="",
        )

        theta_e_config = ThetaEConfig(
            theta_00=330.0,
            y_0=1500e3,  # 1500 km north
            y_one=9439e3,
            delta_y=50.0,
            theta_e_type="SS09",
            y_0_seasonal_amp=0.0,
            seasonal_period_days=360.0,
            seasonal_phase_days=0.0,
        )

        timestamp = "20260111_120000"
        output_path, restart_dir = generate_descriptive_path(
            config, theta_e_config, timestamp=timestamp
        )

        assert "y0p1500" in output_path  # positive 1500 km

    def test_negative_y0(self):
        """Test y0 formatting for negative values."""
        config = SWConfig(
            total_integration_days=100,
            ny=51,
            gravity=9.81,
            height=16e3,
            beta=2e-11,
            t_ref=300.0,
            output_path="",
            dt=3600,
            coeff_eddy_heat_diff=0.0,
            k_v=7786 * 100,
            epsilon_u=1e-8,
            delta_z=60,
            delta=4e3,
            tau=37.0 * 86400,
            v_d=2.5,
            domain_size=15751e3 * 2,
            include_vert_advec_u=True,
            include_merid_advec_u=True,
            enable_steady_state=False,
            steady_state_window_size=10,
            steady_state_threshold=0.001,
            steady_state_check_both=True,
            smoothness_threshold=0.5,
            seasonal_convergence_enabled=False,
            seasonal_convergence_window=30,
            seasonal_convergence_threshold=0.01,
            save_restart_every=0,
            restart_output_dir="",
        )

        theta_e_config = ThetaEConfig(
            theta_00=330.0,
            y_0=-500e3,  # 500 km south
            y_one=9439e3,
            delta_y=50.0,
            theta_e_type="SS09",
            y_0_seasonal_amp=0.0,
            seasonal_period_days=360.0,
            seasonal_phase_days=0.0,
        )

        timestamp = "20260111_120000"
        output_path, restart_dir = generate_descriptive_path(
            config, theta_e_config, timestamp=timestamp
        )

        assert "y0n0500" in output_path  # negative 500 km

    def test_auto_timestamp(self):
        """Test that timestamp is auto-generated if not provided."""
        config = SWConfig(
            total_integration_days=10,
            ny=51,
            gravity=9.81,
            height=16e3,
            beta=2e-11,
            t_ref=300.0,
            output_path="",
            dt=3600,
            coeff_eddy_heat_diff=0.0,
            k_v=7786 * 100,
            epsilon_u=1e-8,
            delta_z=60,
            delta=4e3,
            tau=37.0 * 86400,
            v_d=2.5,
            domain_size=15751e3 * 2,
            include_vert_advec_u=True,
            include_merid_advec_u=True,
            enable_steady_state=False,
            steady_state_window_size=10,
            steady_state_threshold=0.001,
            steady_state_check_both=True,
            smoothness_threshold=0.5,
            seasonal_convergence_enabled=False,
            seasonal_convergence_window=30,
            seasonal_convergence_threshold=0.01,
            save_restart_every=0,
            restart_output_dir="",
        )

        theta_e_config = ThetaEConfig(
            theta_00=330.0,
            y_0=0.0,
            y_one=9439e3,
            delta_y=50.0,
            theta_e_type="sin2",
            y_0_seasonal_amp=0.0,
            seasonal_period_days=360.0,
            seasonal_phase_days=0.0,
        )

        output_path, restart_dir = generate_descriptive_path(config, theta_e_config)

        # Should have a timestamp in YYYYMMDD_HHMMSS format
        assert "run_202" in output_path  # Year starts with 202x


class TestGenerateRestartFilename:
    """Tests for generate_restart_filename()"""

    def test_restart_filename_from_descriptive_output(self):
        """Test restart filename generation from descriptive output path."""
        output_path = (
            "./model_output/SB08/run_20260111_134530_seas_y0p0000_ny051_3600days_output.nc"
        )
        restart_path = generate_restart_filename(output_path, 100)

        assert restart_path == (
            "./model_output/SB08/run_20260111_134530_seas_y0p0000_ny051_3600days_restart_day0100.nc"
        )

    def test_restart_filename_different_days(self):
        """Test restart filename with different day numbers."""
        output_path = "./model_output/sin2/run_20260111_120000_noseas_y0p0000_ny101_250days_output.nc"

        restart_1 = generate_restart_filename(output_path, 1)
        assert "restart_day0001.nc" in restart_1

        restart_50 = generate_restart_filename(output_path, 50)
        assert "restart_day0050.nc" in restart_50

        restart_1000 = generate_restart_filename(output_path, 1000)
        assert "restart_day1000.nc" in restart_1000

    def test_restart_filename_custom_path_fallback(self):
        """Test that custom paths fall back to simple restart naming."""
        custom_output_path = "/custom/path/my_run.nc"
        restart_path = generate_restart_filename(custom_output_path, 100)

        # Should use parent directory and simple restart filename
        assert "/custom/path/restart_day0100.nc" in restart_path
        assert restart_path.endswith("restart_day0100.nc")

    def test_restart_preserves_directory_structure(self):
        """Test that restart file is in same directory as output file."""
        output_path = "./model_output/SB08/run_20260111_134530_seas_y0p0000_ny051_3600days_output.nc"
        restart_path = generate_restart_filename(output_path, 250)

        # Extract directories
        import os

        output_dir = os.path.dirname(output_path)
        restart_dir = os.path.dirname(restart_path)

        assert output_dir == restart_dir
