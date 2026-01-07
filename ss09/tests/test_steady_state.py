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
