"""
Steady-state detection for the shallow water model.

This module provides functionality to detect when a simulation has reached
a quasi-steady state based on convergence of multiple physical metrics.

Convergence Metrics:
    - Kinetic Energy (KE): domain-averaged u^2 + v^2
    - Temperature Variance (Tvar): spatial standard deviation of theta

Convergence Criterion:
    Relative change over a sliding window of N days must fall below a threshold.

    rel_change = (max(window) - min(window)) / mean(window) < threshold

Example:
    >>> detector = SteadyStateDetector(
    ...     enabled=True,
    ...     window_size=10,
    ...     threshold=0.001  # 0.1% relative change
    ... )
    >>> detector.record_day(day=5, u=u_avg, v=v_avg, theta=theta_avg)
    >>> if detector.check_convergence(day=5):
    ...     print(f"Converged at day {detector.convergence_day}")
"""

import numpy as np
from typing import Optional


class SteadyStateDetector:
    """
    Detects when the model reaches a steady state based on multiple convergence metrics.

    Monitors:
    - Kinetic energy: KE = mean(u^2 + v^2)
    - Temperature variance: Tvar = std(theta)

    Checks convergence using relative change over a sliding window.
    """

    def __init__(
        self,
        enabled: bool = False,
        window_size: int = 10,
        threshold: float = 0.001,
        check_both_metrics: bool = True,
        smoothness_threshold: float = 0.5,
        slow_gate: bool = False,
        slow_window: int = 0,
        slow_threshold: float = 0.002,
    ):
        """
        Initialize the steady-state detector.

        Args:
            enabled: Whether steady-state detection is active
            window_size: Number of days to use for convergence check
            threshold: Relative change threshold (e.g., 0.001 = 0.1%)
            check_both_metrics: If True, both KE and Tvar must converge;
                                if False, either metric converging is sufficient
            smoothness_threshold: Neighbor correlation threshold for v field smoothness
            slow_gate: Additionally require the slow diagnostics (jet
                latitude, max |v|, equatorial depression) to converge. The
                KE/Tvar metrics are nearly blind to the slow jet-position
                mode, which approaches equilibrium as a decaying
                oscillation, so a range criterion over a long window is
                used: near a turning point of the ringing the local trend
                vanishes while the amplitude does not, and only a window
                long enough to span a lobe of the oscillation sees it.
            slow_window: Trailing window (days) for the slow-gate range
                criterion; choose comparable to the slowest damping
                timescale (the drag time 1/epsilon_u).
            slow_threshold: Relative range threshold for the slow gate
                (default 0.002 = 0.2%).
        """
        self.enabled = enabled
        self.window_size = window_size
        self.threshold = threshold
        self.check_both_metrics = check_both_metrics
        self.smoothness_threshold = smoothness_threshold
        self.slow_gate = slow_gate
        self.slow_window = slow_window
        self.slow_threshold = slow_threshold

        # History storage
        self.kinetic_energy_history = []
        self.temp_variance_history = []
        self.days_recorded = []

        # Slow-diagnostic histories (populated only when slow_gate is on;
        # deliberately NOT persisted in restart files, so a restarted run
        # re-fills the window before it can stop -- conservative by design)
        self.jet_lat_history = []
        self.v_absmax_history = []
        self.depression_history = []

        # Smoothness tracking
        self.v_smoothness_history = []
        self.v_grid_variance_history = []
        self.smoothness_warning_issued = False

        # Convergence tracking
        self.is_converged = False
        self.convergence_day: Optional[int] = None
        self.ke_converged = False
        self.tvar_converged = False
        self.slow_converged = False

    def compute_kinetic_energy(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Compute domain-averaged kinetic energy: mean(u^2 + v^2).

        Args:
            u: Zonal wind field
            v: Meridional wind field

        Returns:
            Domain-averaged kinetic energy
        """
        return np.mean(u**2 + v**2)

    def compute_temp_variance(self, theta: np.ndarray) -> float:
        """
        Compute spatial standard deviation of theta.

        Args:
            theta: Potential temperature field

        Returns:
            Standard deviation of theta
        """
        return np.std(theta)

    def compute_v_smoothness(self, v: np.ndarray, dy: float) -> dict:
        """
        Compute v field smoothness metrics to detect grid-scale oscillations.

        Grid-scale oscillations (2Δy computational mode) manifest as:
        - Negative or low correlation between neighboring points
        - High variance in d²v/dy²

        Args:
            v: Meridional wind field
            dy: Grid spacing in meridional direction

        Returns:
            Dictionary with:
                - neighbor_correlation: correlation between v[i] and v[i+1]
                - grid_variance: variance of d²v/dy²
                - is_smooth: bool (True if correlation > threshold)
        """
        # Neighbor correlation (detects alternating pattern)
        if len(v) < 2:
            return {'neighbor_correlation': 1.0, 'grid_variance': 0.0, 'is_smooth': True}

        # A constant field has zero variance, for which np.corrcoef returns NaN;
        # treat it as perfectly smooth (correlation 1.0) instead.
        if np.std(v[:-1]) == 0 or np.std(v[1:]) == 0:
            neighbor_corr = 1.0
        else:
            neighbor_corr = np.corrcoef(v[:-1], v[1:])[0, 1]

        # Grid-scale variance (what k_v damps)
        if len(v) >= 3:
            second_diff = np.diff(v, n=2) / (dy**2)
            grid_var = np.var(second_diff)
        else:
            grid_var = 0.0

        is_smooth = neighbor_corr >= self.smoothness_threshold

        return {
            'neighbor_correlation': neighbor_corr,
            'grid_variance': grid_var,
            'is_smooth': is_smooth
        }

    def _warn_noisy_v_field(self, correlation: float):
        """Warn user about grid-scale oscillations in v field."""
        import logging
        logging.warning(
            f"Grid-scale oscillations detected in v field. "
            f"Neighbor correlation = {correlation:.3f} < {self.smoothness_threshold:.2f} threshold. "
            f"This indicates 2Δy computational mode from leapfrog scheme. "
            f"Consider increasing k_v for smoother fields."
        )
        self.smoothness_warning_issued = True

    def record_day(
        self,
        day: int,
        u: np.ndarray,
        v: np.ndarray,
        theta: np.ndarray,
        dy: Optional[float] = None,
        v_faces: Optional[np.ndarray] = None,
        jet_lat: Optional[float] = None,
        v_absmax: Optional[float] = None,
        depression: Optional[float] = None,
    ):
        """
        Record metrics for a given day.

        Args:
            day: Day number
            u: Daily-averaged zonal wind
            v: Daily-averaged meridional wind, on the u/theta grid (the
                center-reconstructed field for a staggered run, so kinetic
                energy is the u^2 + v^2 that u actually feels)
            theta: Daily-averaged potential temperature
            dy: Grid spacing in meridional direction (optional, for smoothness checks)
            v_faces: Daily-averaged v on its native grid, used for the
                grid-scale smoothness monitor. Defaults to ``v`` (the
                collocated case, where the two grids coincide); for a staggered
                run this is the face field, whose grid-scale noise the monitor
                exists to detect.
            jet_lat: Northern jet latitude for the day (slow gate only)
            v_absmax: Max |v| for the day (slow gate only)
            depression: Equatorial theta_E - theta for the day (slow gate only)
        """
        if not self.enabled:
            return

        ke = self.compute_kinetic_energy(u, v)
        tvar = self.compute_temp_variance(theta)

        self.days_recorded.append(day)
        self.kinetic_energy_history.append(ke)
        self.temp_variance_history.append(tvar)

        if self.slow_gate:
            if jet_lat is not None:
                self.jet_lat_history.append(jet_lat)
            if v_absmax is not None:
                self.v_absmax_history.append(v_absmax)
            if depression is not None:
                self.depression_history.append(depression)

        # Track v field smoothness if dy is provided
        if dy is not None:
            v_for_smoothness = v if v_faces is None else v_faces
            smoothness = self.compute_v_smoothness(v_for_smoothness, dy)
            self.v_smoothness_history.append(smoothness['neighbor_correlation'])
            self.v_grid_variance_history.append(smoothness['grid_variance'])

            # Issue warning if grid-scale oscillations detected
            if not smoothness['is_smooth'] and not self.smoothness_warning_issued:
                self._warn_noisy_v_field(smoothness['neighbor_correlation'])

    def check_convergence(self, current_day: int) -> bool:
        """
        Check if model has reached steady state.

        Args:
            current_day: Current simulation day

        Returns:
            True if converged, False otherwise.
            Updates self.is_converged, self.convergence_day, etc.
        """
        if not self.enabled or len(self.kinetic_energy_history) < self.window_size:
            return False

        # Already converged
        if self.is_converged:
            return True

        # Get last window_size values
        ke_window = self.kinetic_energy_history[-self.window_size:]
        tvar_window = self.temp_variance_history[-self.window_size:]

        # Compute relative change
        ke_rel_change = self._compute_relative_change(ke_window)
        tvar_rel_change = self._compute_relative_change(tvar_window)

        # Check convergence for each metric
        self.ke_converged = ke_rel_change < self.threshold
        self.tvar_converged = tvar_rel_change < self.threshold

        # Determine overall convergence
        if self.check_both_metrics:
            converged = self.ke_converged and self.tvar_converged
        else:
            converged = self.ke_converged or self.tvar_converged

        if self.slow_gate:
            self.slow_converged = self._check_slow_gate()
            converged = converged and self.slow_converged

        if converged and not self.is_converged:
            self.is_converged = True
            self.convergence_day = current_day
            return True

        return False

    def check_seasonal_convergence(
        self,
        current_day: int,
        seasonal_period_days: float,
        window_size: int = 30,
        threshold: float = 0.01
    ) -> bool:
        """
        Check if current seasonal cycle matches previous year (year-to-year convergence).

        For seasonally-varying forcing, the model never reaches a single steady state,
        but the seasonal cycle itself can converge. This method compares metrics at
        the same phase of consecutive seasonal cycles.

        Args:
            current_day: Current simulation day
            seasonal_period_days: Length of one full seasonal cycle (e.g., 360 days)
            window_size: Number of consecutive days that must match year-to-year
            threshold: Relative change threshold for year-to-year comparison

        Returns:
            True if last window_size days match corresponding days from previous year
        """
        # Need at least 2 full cycles + window for meaningful comparison
        min_days_needed = int(2 * seasonal_period_days + window_size)
        if current_day < min_days_needed:
            return False

        # Already converged? Stay converged
        if self.is_converged:
            return True

        # Need enough data in history
        if len(self.kinetic_energy_history) < min_days_needed:
            return False

        # Compare last window_size days to same window from last year
        converged_days = 0
        period_int = int(seasonal_period_days)

        for offset in range(window_size):
            day_now = current_day - offset
            day_last_year = day_now - period_int

            # Check bounds
            if day_now < 0 or day_last_year < 0:
                continue
            if day_now >= len(self.kinetic_energy_history):
                continue
            if day_last_year >= len(self.kinetic_energy_history):
                continue

            # Get metrics at current position and same position last year
            ke_now = self.kinetic_energy_history[day_now]
            ke_last = self.kinetic_energy_history[day_last_year]

            tvar_now = self.temp_variance_history[day_now]
            tvar_last = self.temp_variance_history[day_last_year]

            # Compute relative changes
            rel_ke = abs(ke_now - ke_last) / (abs(ke_last) + 1e-10)
            rel_tvar = abs(tvar_now - tvar_last) / (abs(tvar_last) + 1e-10)

            # Check convergence based on metric requirements
            if self.check_both_metrics:
                if rel_ke < threshold and rel_tvar < threshold:
                    converged_days += 1
            else:
                if rel_ke < threshold or rel_tvar < threshold:
                    converged_days += 1

        # All days in window must match
        if converged_days == window_size:
            self.is_converged = True
            self.convergence_day = current_day
            self.ke_converged = True  # Mark as converged
            self.tvar_converged = True
            return True

        return False

    def _check_slow_gate(self) -> bool:
        """Range criterion on the slow diagnostics: each history's relative
        range over the trailing slow_window days must be below
        slow_threshold. A window containing non-finite values (e.g. a jet
        latitude the spline finder could not determine) blocks convergence
        until those days age out of the window."""
        histories = (
            self.jet_lat_history,
            self.v_absmax_history,
            self.depression_history,
        )
        for hist in histories:
            if len(hist) < self.slow_window:
                return False
            window = np.asarray(hist[-self.slow_window:], dtype=float)
            if not np.all(np.isfinite(window)):
                return False
            if self._compute_relative_change(list(window)) >= self.slow_threshold:
                return False
        return True

    def _compute_relative_change(self, values: list) -> float:
        """
        Compute relative change over window.

        rel_change = (max - min) / mean

        Args:
            values: List of metric values over the window

        Returns:
            Relative change (dimensionless)
        """
        values_array = np.array(values)
        mean_val = np.mean(values_array)
        range_val = np.max(values_array) - np.min(values_array)

        # If values are perfectly stable (range = 0), relative change is 0
        if range_val == 0:
            return 0.0
        # If mean is zero but values vary, relative change is infinite
        if mean_val == 0:
            return np.inf
        return range_val / np.abs(mean_val)

    def get_history_dict(self) -> dict:
        """
        Return convergence history as dict for NetCDF storage.

        Returns:
            Dictionary with convergence history arrays, or empty dict if disabled/no data
        """
        if not self.enabled or len(self.days_recorded) == 0:
            return {}

        history = {
            'convergence_days': np.array(self.days_recorded),
            'kinetic_energy': np.array(self.kinetic_energy_history),
            'temp_variance': np.array(self.temp_variance_history),
        }

        # Add smoothness history if available
        if len(self.v_smoothness_history) > 0:
            history['v_smoothness'] = np.array(self.v_smoothness_history)
            history['v_grid_variance'] = np.array(self.v_grid_variance_history)

        return history

    def get_convergence_info(self) -> dict:
        """
        Return convergence information for logging/attributes.

        Returns:
            Dictionary with convergence configuration and results
        """
        info = {
            'steady_state_enabled': int(self.enabled),
            'steady_state_window_size': self.window_size,
            'steady_state_threshold': self.threshold,
            'steady_state_check_both_metrics': int(self.check_both_metrics),
            'steady_state_converged': int(self.is_converged),
            'steady_state_convergence_day': self.convergence_day if self.is_converged else -1,
            'steady_state_ke_converged': int(self.ke_converged),
            'steady_state_tvar_converged': int(self.tvar_converged),
            'steady_state_slow_gate': int(self.slow_gate),
            'steady_state_slow_window': self.slow_window,
            'steady_state_slow_threshold': self.slow_threshold,
            'steady_state_slow_converged': int(self.slow_converged),
        }

        # Add smoothness summary statistics if available
        if len(self.v_smoothness_history) > 0:
            info['v_smoothness_mean'] = float(np.mean(self.v_smoothness_history))
            info['v_smoothness_min'] = float(np.min(self.v_smoothness_history))
            info['v_smoothness_warnings'] = int(self.smoothness_warning_issued)

        return info
