"""
Hadley cell diagnostics for the shallow water model.

This module computes diagnostics relevant to Hadley cell dynamics:
- Local Rossby number: Ro = (du/dy) / (β*y)
- Subtropical jet positions and magnitudes (both hemispheres)
"""

import numpy as np


class HadleyDiagnostics:
    """
    Compute Hadley cell diagnostics from daily-averaged fields.

    Diagnostics:
        - Local Rossby number: Ro = (du/dy) / (β*y)
        - Subtropical jet positions and magnitudes (both hemispheres)
    """

    def __init__(self, ny: int, total_days: int):
        """
        Initialize Hadley diagnostics storage.

        Args:
            ny: Number of meridional grid points
            total_days: Maximum number of days to store
        """
        # 2D fields (time, y)
        self.rossby_number = np.full([total_days, ny], np.nan)

        # 1D fields (time) - separate for each hemisphere
        self.north_jet_lat = np.full(total_days, np.nan)
        self.north_jet_magnitude = np.full(total_days, np.nan)
        self.south_jet_lat = np.full(total_days, np.nan)
        self.south_jet_magnitude = np.full(total_days, np.nan)

        self.days_recorded = 0
        self.start_day = None  # Track first recorded day for restart support

    def compute_rossby_number(
        self, u: np.ndarray, y: np.ndarray, dy: float, beta: float
    ) -> np.ndarray:
        """
        Compute local Rossby number: Ro = (du/dy) / (β*y).

        The Rossby number measures the ratio of relative vorticity to
        planetary vorticity. Values O(1) indicate nonlinear dynamics.

        Args:
            u: Zonal wind field (ny,)
            y: Meridional coordinate (ny,) in meters from equator
            dy: Grid spacing in meters
            beta: Beta parameter (df/dy) in m^-1 s^-1

        Returns:
            Rossby number field (ny,) with NaN near equator
        """
        # Compute relative vorticity: du/dy
        du_dy = np.gradient(u, dy)

        # Compute planetary vorticity: β*y
        planetary_vorticity = beta * y

        # Compute Rossby number
        rossby = du_dy / planetary_vorticity

        # Set to NaN where |y| is small (equator singularity)
        # Threshold: 1 grid point from equator
        equator_mask = np.abs(y) < dy
        rossby[equator_mask] = np.nan

        return rossby

    def find_jet_position(
        self, u: np.ndarray, y: np.ndarray, hemisphere: str
    ) -> tuple[float, float]:
        """
        Find subtropical jet position and magnitude in specified hemisphere.

        Uses cubic spline interpolation locally around grid-point maximum to
        achieve sub-grid accuracy by finding where du/dy = 0. Only interpolates
        in the neighborhood of the grid maximum (max point ± 1 neighbor).

        Args:
            u: Zonal wind field (ny,)
            y: Meridional coordinate (ny,) in meters from equator
            hemisphere: 'north' or 'south'

        Returns:
            (jet_latitude, jet_magnitude) in meters and m/s
            - latitude: Interpolated position where du/dy = 0 (sub-grid accuracy)
            - magnitude: Grid-point maximum value (no interpolation)

        Raises:
            ValueError: If hemisphere is not 'north' or 'south'
        """
        # Validate hemisphere and extract data
        if hemisphere == "north":
            mask = y > 0
        elif hemisphere == "south":
            mask = y < 0
        else:
            raise ValueError(
                f"hemisphere must be 'north' or 'south', got {hemisphere}"
            )

        u_hem = u[mask]
        y_hem = y[mask]

        if len(u_hem) == 0:
            return np.nan, np.nan

        # Find grid-point maximum
        max_idx = np.argmax(u_hem)
        grid_jet_lat = y_hem[max_idx]
        jet_mag = u_hem[max_idx]  # Always use grid-point magnitude

        # Check if max is at boundary (can't interpolate)
        if max_idx == 0 or max_idx == len(u_hem) - 1:
            return grid_jet_lat, jet_mag

        # Extract local neighborhood: max_idx and its neighbors
        # Need at least 3 points for cubic spline (use 4 if available)
        if max_idx == 1:
            # Near left boundary: use indices [0, 1, 2, 3]
            idx_start = 0
            idx_end = min(4, len(u_hem))
        elif max_idx == len(u_hem) - 2:
            # Near right boundary: use last 4 points
            idx_start = max(0, len(u_hem) - 4)
            idx_end = len(u_hem)
        else:
            # Interior: use [max_idx-1, max_idx, max_idx+1, max_idx+2]
            idx_start = max_idx - 1
            idx_end = min(max_idx + 3, len(u_hem))

        y_local = y_hem[idx_start:idx_end]
        u_local = u_hem[idx_start:idx_end]

        # Need at least 3 points for interpolation
        if len(y_local) < 3:
            return grid_jet_lat, jet_mag

        # Build cubic spline over local neighborhood
        from scipy.interpolate import CubicSpline
        from scipy.optimize import brentq

        try:
            spline = CubicSpline(y_local, u_local, bc_type='natural')
            spline_deriv = spline.derivative(nu=1)
        except Exception:
            # Spline construction failed
            return grid_jet_lat, jet_mag

        # Find zero of du/dy in the local interval
        # Search between first and last point of local neighborhood
        y_left = y_local[0]
        y_right = y_local[-1]

        try:
            # Check if there's a sign change in the derivative
            dudy_left = spline_deriv(y_left)
            dudy_right = spline_deriv(y_right)

            if dudy_left * dudy_right < 0:
                # Sign change exists - find exact zero
                refined_jet_lat = brentq(spline_deriv, y_left, y_right)
            else:
                # No sign change - evaluate at multiple points to find minimum |du/dy|
                y_eval = np.linspace(y_left, y_right, 20)
                dudy_eval = np.abs(spline_deriv(y_eval))
                min_deriv_idx = np.argmin(dudy_eval)
                refined_jet_lat = y_eval[min_deriv_idx]

        except Exception:
            # Root finding or evaluation failed
            refined_jet_lat = grid_jet_lat

        return refined_jet_lat, jet_mag

    def record_day(
        self, day: int, u: np.ndarray, y: np.ndarray, dy: float, beta: float
    ):
        """
        Compute and store diagnostics for a given day.

        Args:
            day: Day number (0-indexed)
            u: Daily-averaged zonal wind (ny,)
            y: Meridional coordinate (ny,) in meters from equator
            dy: Grid spacing in meters
            beta: Beta parameter in m^-1 s^-1
        """
        # Track start day for restart support
        if self.start_day is None:
            self.start_day = day

        # Compute Rossby number
        self.rossby_number[day] = self.compute_rossby_number(u, y, dy, beta)

        # Find jet positions
        north_lat, north_mag = self.find_jet_position(u, y, "north")
        south_lat, south_mag = self.find_jet_position(u, y, "south")

        self.north_jet_lat[day] = north_lat
        self.north_jet_magnitude[day] = north_mag
        self.south_jet_lat[day] = south_lat
        self.south_jet_magnitude[day] = south_mag

        self.days_recorded = day + 1

    def get_diagnostics_dict(self) -> dict:
        """
        Return diagnostics as dict for xarray Dataset construction.

        Returns:
            Dictionary with all diagnostic arrays (filtered to recorded days)
        """
        if self.days_recorded == 0:
            return {}

        # Filter to only recorded days (handles restart case where start_day > 0)
        start = self.start_day if self.start_day is not None else 0
        mask = slice(start, self.days_recorded)

        return {
            "rossby_number": self.rossby_number[mask],
            "north_jet_lat": self.north_jet_lat[mask],
            "north_jet_magnitude": self.north_jet_magnitude[mask],
            "south_jet_lat": self.south_jet_lat[mask],
            "south_jet_magnitude": self.south_jet_magnitude[mask],
        }
