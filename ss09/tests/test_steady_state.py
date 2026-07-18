import pytest
import numpy as np
from ss09.steady_state import SteadyStateDetector


class TestSteadyStateDetector:

    def test_disabled_by_default(self):
        """Test that detector is disabled by default"""
        detector = SteadyStateDetector()
        assert not detector.enabled
        assert not detector.is_converged
        assert detector.convergence_day is None

    def test_compute_v_smoothness_constant_field_no_nan(self):
        """A constant (e.g. all-zero) v field is perfectly smooth, not NaN.

        np.corrcoef of a constant array is NaN (zero variance); that NaN would
        compare False against the threshold and trip a spurious grid-noise
        warning on, e.g., an early day with a near-uniform v field.
        """
        detector = SteadyStateDetector(enabled=True)
        result = detector.compute_v_smoothness(np.zeros(51), dy=1000.0)
        assert result["neighbor_correlation"] == 1.0
        assert result["is_smooth"] is True
        assert not np.isnan(result["neighbor_correlation"])

    def test_kinetic_energy_computation(self):
        """Test KE computation"""
        detector = SteadyStateDetector(enabled=True)
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([1.0, 1.0, 1.0])
        ke = detector.compute_kinetic_energy(u, v)
        expected = np.mean([2, 5, 10])  # u^2 + v^2 for each point
        assert np.isclose(ke, expected)

    def test_temp_variance_computation(self):
        """Test temperature variance computation"""
        detector = SteadyStateDetector(enabled=True)
        theta = np.array([300.0, 310.0, 320.0])
        tvar = detector.compute_temp_variance(theta)
        assert np.isclose(tvar, np.std(theta))

    def test_convergence_not_enough_data(self):
        """Test that convergence doesn't trigger with insufficient data"""
        detector = SteadyStateDetector(enabled=True, window_size=5)
        u = np.ones(10)
        v = np.ones(10)
        theta = np.ones(10) * 300

        for day in range(4):
            detector.record_day(day, u, v, theta)
            assert not detector.check_convergence(day)

    def test_convergence_stable_values(self):
        """Test convergence with perfectly stable values"""
        detector = SteadyStateDetector(
            enabled=True,
            window_size=5,
            threshold=0.001
        )
        u = np.ones(10) * 5.0
        v = np.ones(10) * 3.0
        theta = np.ones(10) * 300.0

        # Record enough days with identical values
        for day in range(10):
            detector.record_day(day, u, v, theta)

        # Should converge (relative change = 0)
        assert detector.check_convergence(9)
        assert detector.is_converged
        assert detector.convergence_day == 9

    def test_convergence_both_metrics_required(self):
        """Test that both metrics must converge when check_both_metrics=True"""
        detector = SteadyStateDetector(
            enabled=True,
            window_size=3,
            threshold=0.01,
            check_both_metrics=True
        )

        # KE stable, Tvar varying (variance itself changes with day)
        for day in range(5):
            u = np.ones(10) * 5.0  # Stable KE
            v = np.ones(10) * 3.0  # Stable KE
            # Scale the range to change variance with day
            theta = np.linspace(300.0, 300.0 + (day + 1) * 5, 10)  # Tvar increases with day
            detector.record_day(day, u, v, theta)

        assert not detector.check_convergence(4)
        assert not detector.is_converged

    def test_convergence_either_metric_sufficient(self):
        """Test that either metric converging is sufficient when check_both_metrics=False"""
        detector = SteadyStateDetector(
            enabled=True,
            window_size=3,
            threshold=0.01,
            check_both_metrics=False
        )

        # KE stable, Tvar varying (variance itself changes with day)
        for day in range(5):
            u = np.ones(10) * 5.0  # Stable KE
            v = np.ones(10) * 3.0  # Stable KE
            # Scale the range to change variance with day
            theta = np.linspace(300.0, 300.0 + (day + 1) * 5, 10)  # Tvar increases with day
            detector.record_day(day, u, v, theta)

        assert detector.check_convergence(4)
        assert detector.is_converged
        assert detector.ke_converged
        assert not detector.tvar_converged

    def test_history_storage(self):
        """Test that history is correctly stored"""
        detector = SteadyStateDetector(enabled=True)

        for day in range(3):
            u = np.ones(10) * (day + 1)
            v = np.ones(10) * (day + 1)
            theta = np.ones(10) * (300 + day)
            detector.record_day(day, u, v, theta)

        history = detector.get_history_dict()
        assert len(history['convergence_days']) == 3
        assert len(history['kinetic_energy']) == 3
        assert len(history['temp_variance']) == 3
        assert np.array_equal(history['convergence_days'], [0, 1, 2])

    def test_disabled_no_recording(self):
        """Test that disabled detector doesn't record data"""
        detector = SteadyStateDetector(enabled=False)
        u = np.ones(10)
        v = np.ones(10)
        theta = np.ones(10) * 300

        detector.record_day(0, u, v, theta)
        assert len(detector.kinetic_energy_history) == 0
        assert detector.get_history_dict() == {}

    def test_get_convergence_info(self):
        """Test convergence info dictionary"""
        detector = SteadyStateDetector(
            enabled=True,
            window_size=5,
            threshold=0.001,
            check_both_metrics=True
        )

        info = detector.get_convergence_info()
        # Booleans are converted to ints for NetCDF compatibility
        assert info['steady_state_enabled'] == 1
        assert info['steady_state_window_size'] == 5
        assert info['steady_state_threshold'] == 0.001
        assert info['steady_state_check_both_metrics'] == 1
        assert info['steady_state_converged'] == 0
        assert info['steady_state_convergence_day'] == -1

    def test_relative_change_computation(self):
        """Test relative change calculation"""
        detector = SteadyStateDetector(enabled=True)

        # Test with varying values
        values = [10.0, 12.0, 11.0, 10.5]
        rel_change = detector._compute_relative_change(values)
        expected = (12.0 - 10.0) / np.mean(values)
        assert np.isclose(rel_change, expected)

        # Test with all zeros (range = 0, so perfectly stable)
        values_zero = [0.0, 0.0, 0.0]
        rel_change_zero = detector._compute_relative_change(values_zero)
        assert rel_change_zero == 0.0

        # Test with varying values around zero mean (mean=0 but range>0)
        values_vary_zero_mean = [-1.0, 0.0, 1.0]
        rel_change_vary = detector._compute_relative_change(values_vary_zero_mean)
        assert rel_change_vary == np.inf

    def test_once_converged_stays_converged(self):
        """Test that once converged, the flag remains True"""
        detector = SteadyStateDetector(
            enabled=True,
            window_size=3,
            threshold=0.001
        )

        # Converge with stable values
        for day in range(5):
            u = np.ones(10) * 5.0
            v = np.ones(10) * 3.0
            theta = np.ones(10) * 300.0
            detector.record_day(day, u, v, theta)

        # Should converge
        assert detector.check_convergence(4)
        assert detector.is_converged
        convergence_day = detector.convergence_day

        # Even if values change afterwards, stays converged
        u_new = np.ones(10) * 10.0
        detector.record_day(5, u_new, v, theta)
        assert detector.check_convergence(5)  # Still returns True
        assert detector.convergence_day == convergence_day  # Day doesn't change

    def test_smoothness_detection_smooth_field(self):
        """Test that smooth v field is correctly identified"""
        detector = SteadyStateDetector(enabled=True, smoothness_threshold=0.5)

        # Create smooth v field (sinusoidal)
        v_smooth = np.sin(np.linspace(0, np.pi, 51))
        dy = 630040.0  # Grid spacing in meters

        smoothness = detector.compute_v_smoothness(v_smooth, dy)

        assert smoothness['neighbor_correlation'] > 0.8
        assert smoothness['is_smooth']
        assert 'grid_variance' in smoothness

    def test_smoothness_detection_noisy_field(self):
        """Test that noisy v field (2Δy oscillations) is correctly identified"""
        detector = SteadyStateDetector(enabled=True, smoothness_threshold=0.5)

        # Create oscillating v field (alternating pattern)
        v_noisy = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
        dy = 630040.0

        smoothness = detector.compute_v_smoothness(v_noisy, dy)

        assert smoothness['neighbor_correlation'] < 0.5
        assert not smoothness['is_smooth']
        assert smoothness['grid_variance'] > 0

    def test_smoothness_tracking_in_history(self):
        """Test that smoothness metrics are tracked when dy is provided"""
        detector = SteadyStateDetector(enabled=True)

        u = np.ones(10) * 5.0
        theta = np.ones(10) * 300.0
        v_smooth = np.sin(np.linspace(0, np.pi, 10))
        dy = 630040.0

        # Record with dy parameter
        for day in range(3):
            detector.record_day(day, u, v_smooth, theta, dy=dy)

        # Check that smoothness history was tracked
        assert len(detector.v_smoothness_history) == 3
        assert len(detector.v_grid_variance_history) == 3

        # Check it's included in history dict
        history = detector.get_history_dict()
        assert 'v_smoothness' in history
        assert 'v_grid_variance' in history
        assert len(history['v_smoothness']) == 3

    def test_smoothness_not_tracked_without_dy(self):
        """Test that smoothness is not tracked when dy is not provided"""
        detector = SteadyStateDetector(enabled=True)

        u = np.ones(10) * 5.0
        v = np.ones(10) * 3.0
        theta = np.ones(10) * 300.0

        # Record without dy parameter
        for day in range(3):
            detector.record_day(day, u, v, theta)

        # Smoothness should not be tracked
        assert len(detector.v_smoothness_history) == 0
        assert len(detector.v_grid_variance_history) == 0

        # Should not be in history dict
        history = detector.get_history_dict()
        assert 'v_smoothness' not in history
        assert 'v_grid_variance' not in history

    def test_smoothness_warning_issued_once(self):
        """Test that smoothness warning is only issued once"""
        detector = SteadyStateDetector(enabled=True, smoothness_threshold=0.8)

        u = np.ones(10) * 5.0
        theta = np.ones(10) * 300.0
        v_noisy = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
        dy = 630040.0

        # Initially no warning issued
        assert not detector.smoothness_warning_issued

        # Record noisy field multiple times
        for day in range(3):
            detector.record_day(day, u, v_noisy, theta, dy=dy)

        # Warning should be flagged
        assert detector.smoothness_warning_issued

        # Check convergence info includes warning
        info = detector.get_convergence_info()
        assert 'v_smoothness_warnings' in info
        assert info['v_smoothness_warnings'] == 1

    def test_smoothness_edge_case_short_array(self):
        """Test smoothness computation with very short arrays"""
        detector = SteadyStateDetector(enabled=True)

        # Single point
        v_single = np.array([1.0])
        dy = 630040.0
        smoothness = detector.compute_v_smoothness(v_single, dy)
        assert smoothness['is_smooth']
        assert smoothness['neighbor_correlation'] == 1.0
        assert smoothness['grid_variance'] == 0.0

    # Seasonal convergence tests
    def test_seasonal_convergence_not_enough_cycles(self):
        """Test that seasonal convergence requires at least 2 full cycles"""
        detector = SteadyStateDetector(enabled=True, window_size=10)
        seasonal_period = 360.0
        window_size = 30

        # Create synthetic data for < 2 years
        u = np.ones(10) * 5.0
        v = np.ones(10) * 3.0
        theta = np.ones(10) * 300.0

        for day in range(700):  # Less than 2 * 360 + 30
            detector.record_day(day, u, v, theta)

        # Should not converge (not enough data)
        assert not detector.check_seasonal_convergence(
            day, seasonal_period, window_size=window_size, threshold=0.01
        )

    def test_seasonal_convergence_matching_years(self):
        """Test that seasonal convergence triggers when Year 2 matches Year 1"""
        detector = SteadyStateDetector(enabled=True)
        seasonal_period = 360.0
        window_size = 30

        u = np.ones(10) * 5.0
        v = np.ones(10) * 3.0

        # Create repeating seasonal cycle: Year 1 = Year 2
        for day in range(800):
            # Theta varies sinusoidally with season
            phase = 2 * np.pi * day / seasonal_period
            theta_day = 300.0 + 10.0 * np.sin(phase) * np.ones(10)
            detector.record_day(day, u, v, theta_day)

        # After 2+ years with matching pattern, should converge
        assert detector.check_seasonal_convergence(
            750, seasonal_period, window_size=window_size, threshold=0.01
        )
        assert detector.is_converged
        assert detector.convergence_day == 750

    def test_seasonal_convergence_drifting_cycle(self):
        """Test that seasonal convergence doesn't trigger when amplitude drifts"""
        detector = SteadyStateDetector(enabled=True)
        seasonal_period = 360.0
        window_size = 30

        # Create drifting seasonal cycle: winds also drift so KE changes too
        for day in range(800):
            year_factor = 1.0 + 0.1 * (day / seasonal_period)  # 10% increase per year
            u = np.ones(10) * 5.0 * year_factor  # Drift in winds
            v = np.ones(10) * 3.0 * year_factor  # Drift in winds
            phase = 2 * np.pi * day / seasonal_period
            theta_day = 300.0 + 10.0 * year_factor * np.sin(phase) * np.ones(10)
            detector.record_day(day, u, v, theta_day)

        # Should NOT converge (cycle is drifting)
        assert not detector.check_seasonal_convergence(
            750, seasonal_period, window_size=window_size, threshold=0.01
        )

    def test_seasonal_convergence_threshold_sensitivity(self):
        """Test seasonal convergence sensitivity to threshold parameter"""
        detector = SteadyStateDetector(enabled=True)
        seasonal_period = 360.0
        window_size = 30

        # Create nearly-repeating cycle with small drift (0.5% per year)
        for day in range(800):
            year_factor = 1.0 + 0.005 * (day / seasonal_period)
            u = np.ones(10) * 5.0 * year_factor  # Small drift in winds
            v = np.ones(10) * 3.0 * year_factor  # Small drift in winds
            phase = 2 * np.pi * day / seasonal_period
            theta_day = 300.0 + 10.0 * year_factor * np.sin(phase) * np.ones(10)
            detector.record_day(day, u, v, theta_day)

        # With strict threshold (0.1%), should NOT converge
        assert not detector.check_seasonal_convergence(
            750, seasonal_period, window_size=window_size, threshold=0.001
        )

        # With relaxed threshold (1%), should converge
        assert detector.check_seasonal_convergence(
            750, seasonal_period, window_size=window_size, threshold=0.01
        )


class TestSlowDriftGate:
    """The slow-drift gate: an opt-in additional convergence criterion
    requiring the slow diagnostics (jet latitude, max |v|, equatorial
    depression) to have a relative range below slow_threshold over a
    trailing slow_window days. Motivated by runs 1a-1b (2026-07-17): the
    KE/Tvar criteria fired at day ~960 with the jet still moving -0.69%
    per 60 d on an oscillatory ~400-d tail."""

    def _detector(self, slow_window=10, slow_threshold=0.002):
        return SteadyStateDetector(
            enabled=True,
            window_size=5,
            threshold=0.01,
            slow_gate=True,
            slow_window=slow_window,
            slow_threshold=slow_threshold,
        )

    @staticmethod
    def _feed(detector, ndays, jet_fn, v_fn=None, depr_fn=None):
        u = np.ones(10) * 5.0
        v = np.ones(10) * 3.0
        theta = np.ones(10) * 300.0
        for day in range(ndays):
            detector.record_day(
                day, u, v, theta,
                jet_lat=jet_fn(day),
                v_absmax=v_fn(day) if v_fn else 9e-3,
                depression=depr_fn(day) if depr_fn else 0.6,
            )

    def test_slow_gate_blocks_convergence_when_jet_drifts(self):
        det = self._detector(slow_window=10, slow_threshold=0.002)
        # KE/Tvar constant (classic criteria pass); jet drifting 0.05%/day.
        self._feed(det, 20, jet_fn=lambda d: 2.3e6 * (1 + 5e-4 * d))
        assert not det.check_convergence(19)
        assert det.ke_converged and det.tvar_converged
        assert not det.slow_converged

    def test_slow_gate_allows_convergence_when_all_flat(self):
        det = self._detector(slow_window=10)
        self._feed(det, 20, jet_fn=lambda d: 2.3e6)
        assert det.check_convergence(19)
        assert det.is_converged
        assert det.slow_converged
        info = det.get_convergence_info()
        assert info["steady_state_slow_gate"] == 1
        assert info["steady_state_slow_window"] == 10
        assert info["steady_state_slow_converged"] == 1

    def test_slow_gate_requires_full_window(self):
        det = self._detector(slow_window=30)
        self._feed(det, 20, jet_fn=lambda d: 2.3e6)  # flat but short
        assert not det.check_convergence(19)
        assert not det.slow_converged

    def test_slow_gate_oscillation_at_turning_point_not_converged(self):
        # The failure mode a trend criterion misses: an oscillating jet
        # sampled near its crest has near-zero local slope but a range over
        # the window far above threshold.
        det = self._detector(slow_window=150, slow_threshold=0.002)
        self._feed(
            det, 300,
            jet_fn=lambda d: 2.3e6 * (1 + 0.01 * np.cos(2 * np.pi * d / 300)),
        )
        assert not det.check_convergence(299)
        assert not det.slow_converged

    def test_slow_gate_v_absmax_drift_blocks(self):
        det = self._detector(slow_window=10)
        self._feed(det, 20, jet_fn=lambda d: 2.3e6,
                   v_fn=lambda d: 9e-3 * (1 + 5e-4 * d))
        assert not det.check_convergence(19)

    def test_slow_gate_depression_drift_blocks(self):
        det = self._detector(slow_window=10)
        self._feed(det, 20, jet_fn=lambda d: 2.3e6,
                   depr_fn=lambda d: 0.6 * (1 + 5e-4 * d))
        assert not det.check_convergence(19)

    def test_slow_gate_disabled_preserves_classic_behavior(self):
        det = SteadyStateDetector(enabled=True, window_size=5, threshold=0.01)
        u = np.ones(10) * 5.0
        v = np.ones(10) * 3.0
        theta = np.ones(10) * 300.0
        for day in range(10):
            det.record_day(day, u, v, theta)
        assert det.check_convergence(9)


class TestSlowDriftGateConfig:

    def test_swconfig_slow_gate_requires_steady_state(self):
        from ss09.sw_config import SWConfig
        with pytest.raises(ValueError, match="slow_drift_gate"):
            SWConfig(slow_drift_gate=True)

    def test_swconfig_slow_gate_auto_window_from_epsilon_u(self):
        from ss09.sw_config import SWConfig
        config = SWConfig(
            enable_steady_state=True, slow_drift_gate=True, epsilon_u=1e-8
        )
        # ceil(1 / (1e-8 * 86400)) = ceil(1157.4) = 1158 days
        assert config.slow_drift_window == 1158

    def test_swconfig_slow_gate_explicit_window_kept(self):
        from ss09.sw_config import SWConfig
        config = SWConfig(
            enable_steady_state=True, slow_drift_gate=True,
            slow_drift_window=500,
        )
        assert config.slow_drift_window == 500

    def test_swconfig_slow_gate_auto_window_epsilon_zero_raises(self):
        from ss09.sw_config import SWConfig
        with pytest.raises(ValueError, match="slow-drift-window"):
            SWConfig(
                enable_steady_state=True, slow_drift_gate=True, epsilon_u=0.0
            )

    def test_swmodel_slow_gate_seasonal_raises(self):
        from ss09.sw_config import SWConfig
        from ss09.sw_model import SWModel
        from ss09.theta_e import ThetaEConfig, SB08Profile
        config = SWConfig(
            enable_steady_state=True, slow_drift_gate=True, ny=51, dt=3600
        )
        te_config = ThetaEConfig(
            theta_e_type="SB08", y_0_seasonal_amp=700e3
        )
        with pytest.raises(ValueError, match="seasonal"):
            SWModel(config, SB08Profile(te_config))

    def test_slow_gate_end_to_end_delays_stop(self):
        """Classic criteria converge almost immediately for a near-rest
        start with loose thresholds; the slow gate must hold the run until
        its window fills."""
        from ss09.sw_config import SWConfig
        from ss09.sw_model import SWModel
        from ss09.theta_e import ThetaEConfig, Sin2Profile

        def run(with_gate):
            # v_d = 0 and a small dt: a configuration that integrates
            # quietly for the full 25 days (the ny=51/dt=3600 default NaNs
            # out around day 5 from a cold start).
            kwargs = dict(
                total_integration_days=25,
                ny=51,
                dt=600,
                v_d=0.0,
                enable_steady_state=True,
                steady_state_window_size=3,
                steady_state_threshold=1e9,
            )
            if with_gate:
                kwargs.update(slow_drift_gate=True, slow_drift_window=15,
                              slow_drift_thresh=1e9)
            config = SWConfig(**kwargs)
            model = SWModel(config, Sin2Profile(ThetaEConfig()))
            model.run_sim()
            return model.steady_state_detector.convergence_day

        assert run(with_gate=False) == 2
        # With the gate, the stop must wait at least until the slow window
        # holds 15 finite samples (>= day 14; later if early-day jet
        # latitudes are NaN and must age out).
        day_gated = run(with_gate=True)
        assert day_gated is not None
        assert day_gated >= 14
