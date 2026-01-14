import pytest
import numpy as np
from dataclasses import replace
from ss09.daily_results import DailyResults
from ss09.sw_config import SWConfig
from ss09.theta_e import ThetaEConfig, Sin2Profile
from ss09.hadley_diagnostics import HadleyDiagnostics
from ss09.steady_state import SteadyStateDetector


class TestDailyResults:

    @pytest.fixture
    def basic_config(self):
        """Create a basic SWConfig for testing"""
        return SWConfig(
            total_integration_days=100,
            ny=51,
            gravity=9.81,
            height=16e3,
            beta=2e-11,
            t_ref=300.0,
            output_path="./test_output.nc",
            dt=3600,
            coeff_eddy_heat_diff=0.0,
            k_v=778600.0,
            epsilon_u=1e-8,
            delta_z=60.0,
            delta=4000.0,
            tau=37.0 * 86400,
            v_d=2.5,
            domain_size=31502000.0,
            asselin_filt_coef=0.04,
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
            restart_output_dir="./",
        )

    @pytest.fixture
    def basic_theta_e_config(self):
        """Create a basic ThetaEConfig for testing"""
        return ThetaEConfig(
            theta_00=330.0,
            y_0=0.0,
            y_one=9439e3,
            delta_y=50.0,
            theta_e_type="sin2",
            y_0_seasonal_amp=0.0,
            seasonal_period_days=360.0,
            seasonal_phase_days=0.0,
        )

    def test_to_xarray_with_early_termination(self, basic_config, basic_theta_e_config):
        """Test that to_xarray works correctly when simulation stops early"""
        # Setup: Configure for 100 days but only record 30 days (simulating early termination)
        total_days = 100
        recorded_days = 30
        ny = basic_config.ny

        results = DailyResults(total_days, ny)
        theta_e_profile = Sin2Profile(basic_theta_e_config)

        # Record only first 30 days (time starts at 1, not 0, to avoid filtering)
        for day in range(recorded_days):
            time = float(day + 1)
            u = np.random.randn(ny) * 10.0 + 20.0
            v = np.random.randn(ny) * 2.0
            theta = np.random.randn(ny) * 5.0 + 300.0
            results.store_day(day, time, u, v, theta)

        # Create Hadley diagnostics with only recorded days
        hadley_diags = HadleyDiagnostics(ny=ny, total_days=total_days)
        y = basic_config.y
        for day in range(recorded_days):
            u = np.random.randn(ny) * 10.0 + 20.0
            v = np.zeros_like(y)  # Simple v for testing
            hadley_diags.record_day(day, u, v, y, basic_config.dy, basic_config.beta)

        # This should NOT raise IndexError (the bug that was fixed)
        ds = results.to_xarray(basic_config, theta_e_profile, None, hadley_diags)

        # Verify output has correct dimensions (recorded days, not total days)
        assert len(ds.time) == recorded_days
        assert ds.u.shape == (recorded_days, ny)
        assert ds.rossby_number.shape == (recorded_days, ny)
        assert ds.north_jet_lat.shape == (recorded_days,)
        assert ds.south_jet_lat.shape == (recorded_days,)

    def test_to_xarray_with_early_termination_and_steady_state(
        self, basic_config, basic_theta_e_config
    ):
        """Test early termination with steady-state detector enabled"""
        # Setup: 100 days configured, 40 days recorded
        total_days = 100
        recorded_days = 40
        ny = basic_config.ny

        # Enable steady-state detection
        config = replace(basic_config, enable_steady_state=True)

        results = DailyResults(total_days, ny)
        theta_e_profile = Sin2Profile(basic_theta_e_config)

        # Record data (time starts at 1 to avoid filtering)
        for day in range(recorded_days):
            time = float(day + 1)
            u = np.random.randn(ny) * 10.0 + 20.0
            v = np.random.randn(ny) * 2.0
            theta = np.random.randn(ny) * 5.0 + 300.0
            results.store_day(day, time, u, v, theta)

        # Create steady-state detector and record history
        steady_state = SteadyStateDetector(
            enabled=True,
            window_size=10,
            threshold=0.001,
            check_both_metrics=True,
            smoothness_threshold=0.5,
        )
        for day in range(recorded_days):
            u = np.random.randn(ny) * 10.0 + 20.0
            v = np.random.randn(ny) * 2.0
            theta = np.random.randn(ny) * 5.0 + 300.0
            steady_state.record_day(day, u, v, theta, config.dy)

        # Create Hadley diagnostics
        hadley_diags = HadleyDiagnostics(ny=ny, total_days=total_days)
        y = config.y
        for day in range(recorded_days):
            u = np.random.randn(ny) * 10.0 + 20.0
            v = np.zeros_like(y)  # Simple v for testing
            hadley_diags.record_day(day, u, v, y, config.dy, config.beta)

        # Should not raise IndexError
        ds = results.to_xarray(config, theta_e_profile, steady_state, hadley_diags)

        # Verify all diagnostics have correct length
        assert len(ds.time) == recorded_days
        assert len(ds.steady_state_kinetic_energy) == recorded_days
        assert len(ds.steady_state_temp_variance) == recorded_days
        assert len(ds.north_jet_lat) == recorded_days
        assert ds.rossby_number.shape[0] == recorded_days

    def test_to_xarray_without_hadley_diagnostics(self, basic_config, basic_theta_e_config):
        """Test that to_xarray works when hadley_diagnostics is None"""
        total_days = 50
        recorded_days = 50
        ny = basic_config.ny

        results = DailyResults(total_days, ny)
        theta_e_profile = Sin2Profile(basic_theta_e_config)

        for day in range(recorded_days):
            time = float(day + 1)
            u = np.random.randn(ny) * 10.0 + 20.0
            v = np.random.randn(ny) * 2.0
            theta = np.random.randn(ny) * 5.0 + 300.0
            results.store_day(day, time, u, v, theta)

        # Pass None for hadley_diagnostics
        ds = results.to_xarray(basic_config, theta_e_profile, None, None)

        # Should not have Hadley diagnostics variables
        assert 'rossby_number' not in ds
        assert 'north_jet_lat' not in ds
        assert 'south_jet_lat' not in ds

        # But should have basic variables
        assert 'u' in ds
        assert 'v' in ds
        assert 'T' in ds

    def test_to_xarray_hadley_diagnostics_array_sizes(self, basic_config, basic_theta_e_config):
        """Test that Hadley diagnostics arrays are correctly sized after early termination"""
        # This is the specific bug case: total_days != recorded_days
        total_days = 500
        recorded_days = 238  # Example from actual model run
        ny = basic_config.ny

        # Update config to match test case
        config = replace(basic_config, total_integration_days=total_days)

        results = DailyResults(total_days, ny)
        theta_e_profile = Sin2Profile(basic_theta_e_config)

        # Simulate early termination by only recording some days (time starts at 1)
        for day in range(recorded_days):
            time = float(day + 1)
            u = np.random.randn(ny) * 10.0 + 20.0
            v = np.random.randn(ny) * 2.0
            theta = np.random.randn(ny) * 5.0 + 300.0
            results.store_day(day, time, u, v, theta)

        # Create Hadley diagnostics that only records the same days
        hadley_diags = HadleyDiagnostics(ny=ny, total_days=total_days)
        y = config.y
        for day in range(recorded_days):
            u = np.random.randn(ny) * 10.0 + 20.0
            v = np.zeros_like(y)  # Simple v for testing
            hadley_diags.record_day(day, u, v, y, config.dy, config.beta)

        # The bug was: trying to apply mask[500 elements] to hadley_diags[238 elements]
        # This should NOT raise IndexError
        ds = results.to_xarray(config, theta_e_profile, None, hadley_diags)

        # All arrays should have recorded_days length, not total_days
        assert len(ds.time) == recorded_days
        assert ds.u.shape == (recorded_days, ny)
        assert ds.rossby_number.shape == (recorded_days, ny)
        assert len(ds.north_jet_lat) == recorded_days
        assert len(ds.north_jet_magnitude) == recorded_days
        assert len(ds.south_jet_lat) == recorded_days
        assert len(ds.south_jet_magnitude) == recorded_days

        # Verify no NaN padding (all data should be real)
        assert not np.any(np.isnan(ds.time.values))
        assert not np.all(np.isnan(ds.north_jet_lat.values))  # Some may be NaN if no jet
        assert not np.all(np.isnan(ds.south_jet_lat.values))

    def test_filtering_removes_zero_time_entries(self, basic_config, basic_theta_e_config):
        """Test that entries with time=0 are filtered out"""
        total_days = 100
        recorded_days = 25
        ny = basic_config.ny

        results = DailyResults(total_days, ny)
        theta_e_profile = Sin2Profile(basic_theta_e_config)

        # Record only first 25 days (rest will have time=0)
        for day in range(recorded_days):
            time = float(day + 1)  # Start from 1, not 0
            u = np.random.randn(ny) * 10.0 + 20.0
            v = np.random.randn(ny) * 2.0
            theta = np.random.randn(ny) * 5.0 + 300.0
            results.store_day(day, time, u, v, theta)

        # Convert to xarray (no Hadley diagnostics for simplicity)
        ds = results.to_xarray(basic_config, theta_e_profile, None, None)

        # Should only have recorded_days entries
        assert len(ds.time) == recorded_days
        # Time values should all be > 0
        assert np.all(ds.time.values > 0)
        # Should not include the zeros from unrecorded days
        assert len(ds.time) < total_days
