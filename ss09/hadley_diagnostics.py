"""
Hadley cell diagnostics for the shallow water model.

This module computes diagnostics relevant to Hadley cell dynamics:
- Local Rossby number: Ro = (du/dy) / (β*y)
- Subtropical jet positions and magnitudes (both hemispheres)
- Hadley cell center latitudes and strengths (where v extremizes)
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

        # Hadley cell center (where v extremizes in each hemisphere)
        # Upper branch: poleward flow, so north has max v, south has min v
        self.north_cell_center_lat = np.full(total_days, np.nan)
        self.north_cell_strength = np.full(total_days, np.nan)
        self.south_cell_center_lat = np.full(total_days, np.nan)
        self.south_cell_strength = np.full(total_days, np.nan)

        # Hadley cell edge latitudes (1D, time)
        self.ascending_edge_lat = np.full(total_days, np.nan)
        self.north_descending_edge_lat = np.full(total_days, np.nan)
        self.south_descending_edge_lat = np.full(total_days, np.nan)

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

    def find_zero_crossings(self, v: np.ndarray, y: np.ndarray) -> list[float]:
        """
        Find all latitudes where v=0 using linear interpolation.

        Args:
            v: Meridional wind field (ny,)
            y: Meridional coordinate (ny,) in meters from equator

        Returns:
            List of latitudes where v crosses zero, sorted south to north.
        """
        crossings = []
        for i in range(len(v) - 1):
            # Check for exact zero at grid point (handle v[i] == 0 case)
            if v[i] == 0:
                # Only count if it's a true crossing (signs differ on either side)
                # or if we're at the start and next point is non-zero
                if i > 0 and v[i - 1] * v[i + 1] < 0:
                    crossings.append(y[i])
                elif i == 0 and v[i + 1] != 0:
                    crossings.append(y[i])
            elif v[i] * v[i + 1] < 0:  # Sign change between grid points
                # Linear interpolation
                y_zero = y[i] - v[i] * (y[i + 1] - y[i]) / (v[i + 1] - v[i])
                crossings.append(y_zero)
        # Check last point for exact zero
        if v[-1] == 0 and len(v) > 1 and v[-2] != 0:
            crossings.append(y[-1])
        return crossings

    def compute_cell_edges(
        self, v: np.ndarray, y: np.ndarray, north_jet_lat: float, south_jet_lat: float
    ) -> tuple[float, float, float]:
        """
        Compute Hadley cell edge latitudes from meridional wind field.

        Uses jet positions to anchor the descending edges, then finds the
        ascending edge as the v=0 crossing between them.

        Args:
            v: Meridional wind field (ny,)
            y: Meridional coordinate (ny,) in meters from equator
            north_jet_lat: Northern hemisphere jet latitude (m)
            south_jet_lat: Southern hemisphere jet latitude (m)

        Returns:
            (ascending_edge_lat, north_descending_edge_lat, south_descending_edge_lat)
            All in meters. NaN if edge not found.
        """
        crossings = self.find_zero_crossings(v, y)

        if not crossings:
            return np.nan, np.nan, np.nan

        # North descending edge: v=0 crossing closest to northern jet
        if np.isnan(north_jet_lat):
            north_descending = np.nan
        else:
            north_descending = min(crossings, key=lambda x: abs(x - north_jet_lat))

        # South descending edge: v=0 crossing closest to southern jet
        if np.isnan(south_jet_lat):
            south_descending = np.nan
        else:
            south_descending = min(crossings, key=lambda x: abs(x - south_jet_lat))

        # Ascending edge: v=0 crossing between the two descending edges
        if np.isnan(north_descending) or np.isnan(south_descending):
            ascending = np.nan
        else:
            # Find crossings between south_descending and north_descending
            inner_crossings = [
                c
                for c in crossings
                if south_descending < c < north_descending
                and c != south_descending
                and c != north_descending
            ]
            if len(inner_crossings) == 1:
                ascending = inner_crossings[0]
            elif len(inner_crossings) == 0:
                ascending = np.nan
            else:
                # Multiple crossings between edges — pick closest to equator
                ascending = min(inner_crossings, key=abs)

        return ascending, north_descending, south_descending

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

    def find_cell_center(
        self, v: np.ndarray, y: np.ndarray, dy: float, hemisphere: str
    ) -> tuple[float, float]:
        """
        Find Hadley cell center (v extremum) using linear interpolation of dv/dy.

        For upper-branch circulation (poleward flow):
        - Northern cell: maximum v (northward flow)
        - Southern cell: minimum v (southward flow)

        Uses linear interpolation between grid points where dv/dy changes sign
        to achieve sub-grid accuracy.

        Args:
            v: Meridional wind field (ny,)
            y: Meridional coordinate (ny,) in meters from equator
            dy: Grid spacing in meters
            hemisphere: 'north' or 'south'

        Returns:
            (cell_center_lat, cell_strength) in meters and m/s
            - latitude: Interpolated position where dv/dy = 0 (sub-grid accuracy)
            - strength: Interpolated v value at cell center

        Raises:
            ValueError: If hemisphere is not 'north' or 'south'
        """
        # Validate hemisphere and extract data
        if hemisphere == "north":
            mask = y > 0
            find_extremum = np.argmax  # max v for northward flow
        elif hemisphere == "south":
            mask = y < 0
            find_extremum = np.argmin  # min v for southward flow
        else:
            raise ValueError(
                f"hemisphere must be 'north' or 'south', got {hemisphere}"
            )

        v_hem = v[mask]
        y_hem = y[mask]

        if len(v_hem) == 0:
            return np.nan, np.nan

        # Find grid-point extremum
        ext_idx = find_extremum(v_hem)
        grid_center_lat = y_hem[ext_idx]
        grid_strength = v_hem[ext_idx]

        # Check if extremum is at boundary (can't interpolate)
        if ext_idx == 0 or ext_idx == len(v_hem) - 1:
            return grid_center_lat, grid_strength

        # Compute dv/dy using centered differences at grid points
        # dv/dy[i] ≈ (v[i+1] - v[i-1]) / (2*dy)
        dvdy = np.zeros(len(v_hem))
        dvdy[1:-1] = (v_hem[2:] - v_hem[:-2]) / (2 * dy)
        # One-sided differences at boundaries
        dvdy[0] = (v_hem[1] - v_hem[0]) / dy
        dvdy[-1] = (v_hem[-1] - v_hem[-2]) / dy

        # Find where dv/dy changes sign around the extremum
        # Check interval [ext_idx-1, ext_idx] and [ext_idx, ext_idx+1]
        dvdy_left = dvdy[ext_idx - 1]
        dvdy_center = dvdy[ext_idx]
        dvdy_right = dvdy[ext_idx + 1]

        # Try left interval first [ext_idx-1, ext_idx]
        if dvdy_left * dvdy_center < 0:
            # Sign change in left interval - linear interpolate
            # y_zero = y[i] - dvdy[i] * (y[i+1] - y[i]) / (dvdy[i+1] - dvdy[i])
            y_left = y_hem[ext_idx - 1]
            y_center = y_hem[ext_idx]
            refined_lat = y_left - dvdy_left * (y_center - y_left) / (dvdy_center - dvdy_left)
        elif dvdy_center * dvdy_right < 0:
            # Sign change in right interval [ext_idx, ext_idx+1]
            y_center = y_hem[ext_idx]
            y_right = y_hem[ext_idx + 1]
            refined_lat = y_center - dvdy_center * (y_right - y_center) / (dvdy_right - dvdy_center)
        else:
            # No clear sign change - use grid point
            refined_lat = grid_center_lat

        # Interpolate v at the refined latitude using linear interpolation
        # Find which interval contains refined_lat
        if refined_lat < y_hem[ext_idx]:
            # In left interval
            y_lo, y_hi = y_hem[ext_idx - 1], y_hem[ext_idx]
            v_lo, v_hi = v_hem[ext_idx - 1], v_hem[ext_idx]
        else:
            # In right interval or at grid point
            y_lo, y_hi = y_hem[ext_idx], y_hem[ext_idx + 1]
            v_lo, v_hi = v_hem[ext_idx], v_hem[ext_idx + 1]

        # Linear interpolation for v
        frac = (refined_lat - y_lo) / (y_hi - y_lo) if y_hi != y_lo else 0.0
        refined_strength = v_lo + frac * (v_hi - v_lo)

        return refined_lat, refined_strength

    def record_day(
        self,
        day: int,
        u: np.ndarray,
        v: np.ndarray,
        y: np.ndarray,
        dy: float,
        beta: float,
    ):
        """
        Compute and store diagnostics for a given day.

        Args:
            day: Day number (0-indexed)
            u: Daily-averaged zonal wind (ny,)
            v: Daily-averaged meridional wind (ny,)
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

        # Find cell centers (where v extremizes)
        north_center, north_strength = self.find_cell_center(v, y, dy, "north")
        south_center, south_strength = self.find_cell_center(v, y, dy, "south")

        self.north_cell_center_lat[day] = north_center
        self.north_cell_strength[day] = north_strength
        self.south_cell_center_lat[day] = south_center
        self.south_cell_strength[day] = south_strength

        # Compute cell edges using jet positions
        ascending, north_desc, south_desc = self.compute_cell_edges(
            v, y, north_lat, south_lat
        )
        self.ascending_edge_lat[day] = ascending
        self.north_descending_edge_lat[day] = north_desc
        self.south_descending_edge_lat[day] = south_desc

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
            "north_cell_center_lat": self.north_cell_center_lat[mask],
            "north_cell_strength": self.north_cell_strength[mask],
            "south_cell_center_lat": self.south_cell_center_lat[mask],
            "south_cell_strength": self.south_cell_strength[mask],
            "ascending_edge_lat": self.ascending_edge_lat[mask],
            "north_descending_edge_lat": self.north_descending_edge_lat[mask],
            "south_descending_edge_lat": self.south_descending_edge_lat[mask],
        }
