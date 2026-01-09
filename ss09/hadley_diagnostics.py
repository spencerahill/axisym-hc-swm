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

        Args:
            u: Zonal wind field (ny,)
            y: Meridional coordinate (ny,) in meters from equator
            hemisphere: 'north' or 'south'

        Returns:
            (jet_latitude, jet_magnitude) in meters and m/s

        Raises:
            ValueError: If hemisphere is not 'north' or 'south'
        """
        if hemisphere == "north":
            mask = y > 0
        elif hemisphere == "south":
            mask = y < 0
        else:
            raise ValueError(
                f"hemisphere must be 'north' or 'south', got {hemisphere}"
            )

        # Extract hemisphere data
        u_hem = u[mask]
        y_hem = y[mask]

        if len(u_hem) == 0:
            return np.nan, np.nan

        # Find maximum
        max_idx = np.argmax(u_hem)
        jet_lat = y_hem[max_idx]
        jet_mag = u_hem[max_idx]

        return jet_lat, jet_mag

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

        # Filter to only recorded days
        mask = slice(0, self.days_recorded)

        return {
            "rossby_number": self.rossby_number[mask],
            "north_jet_lat": self.north_jet_lat[mask],
            "north_jet_magnitude": self.north_jet_magnitude[mask],
            "south_jet_lat": self.south_jet_lat[mask],
            "south_jet_magnitude": self.south_jet_magnitude[mask],
        }
