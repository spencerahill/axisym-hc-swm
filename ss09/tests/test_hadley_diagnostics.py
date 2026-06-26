import pytest
import numpy as np
from ss09.hadley_diagnostics import HadleyDiagnostics


class TestHadleyDiagnostics:

    @pytest.fixture
    def basic_grid(self):
        """Create a basic symmetric grid"""
        ny = 51
        y = np.linspace(-15751e3, 15751e3, ny)
        dy = np.diff(y)[0]
        beta = 2e-11
        return y, dy, beta, ny

    def test_initialization(self, basic_grid):
        """Test that diagnostics initializes correctly"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=100)

        assert diag.rossby_number.shape == (100, ny)
        assert diag.north_jet_lat.shape == (100,)
        assert diag.north_jet_magnitude.shape == (100,)
        assert diag.south_jet_lat.shape == (100,)
        assert diag.south_jet_magnitude.shape == (100,)
        assert diag.north_cell_center_lat.shape == (100,)
        assert diag.north_cell_strength.shape == (100,)
        assert diag.south_cell_center_lat.shape == (100,)
        assert diag.south_cell_strength.shape == (100,)
        assert diag.days_recorded == 0

        # Check pre-filled with NaN
        assert np.all(np.isnan(diag.rossby_number))
        assert np.all(np.isnan(diag.north_jet_lat))
        assert np.all(np.isnan(diag.north_cell_center_lat))

    def test_rossby_number_computation(self, basic_grid):
        """Test Rossby number computation"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create linear u profile: u = cy
        c = 1e-5  # constant slope
        u = c * y

        rossby = diag.compute_rossby_number(u, y, dy, beta)

        # Expected: Ro = (du/dy) / (beta*y) = c / (beta*y)
        # Away from equator, should equal c / (beta*y)
        mask = np.abs(y) > dy
        expected = c / (beta * y[mask])
        assert np.allclose(rossby[mask], expected)

        # At equator, should be NaN
        equator_mask = np.abs(y) < dy
        assert np.all(np.isnan(rossby[equator_mask]))

    def test_rossby_number_equator_threshold(self, basic_grid):
        """Test that equator masking uses dy threshold"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        u = np.ones(ny) * 10.0  # Constant wind
        rossby = diag.compute_rossby_number(u, y, dy, beta)

        # Count NaN points (should be those within dy of equator)
        n_nan = np.sum(np.isnan(rossby))

        # Should be at least 1 point (exactly at or near equator)
        assert n_nan >= 1

        # Far from equator should not be NaN
        far_from_eq = np.abs(y) > 5 * dy
        assert not np.any(np.isnan(rossby[far_from_eq]))

    def test_find_jet_northern_hemisphere(self, basic_grid):
        """Test jet finding in northern hemisphere"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create wind profile with jet in northern subtropics
        jet_lat = 5000e3  # 5000 km north
        jet_mag = 25.0  # m/s
        u = jet_mag * np.exp(-((y - jet_lat) / 2000e3) ** 2)

        lat, mag = diag.find_jet_position(u, y, "north")

        # Should find the jet near the prescribed location
        assert np.abs(lat - jet_lat) < 0.1 * dy  # Within 10% of grid spacing (~63 km)
        assert np.abs(mag - jet_mag) < 0.1  # Close to peak magnitude

    def test_find_jet_southern_hemisphere(self, basic_grid):
        """Test jet finding in southern hemisphere"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create wind profile with jet in southern subtropics
        jet_lat = -6000e3  # 6000 km south
        jet_mag = 20.0  # m/s
        u = jet_mag * np.exp(-((y - jet_lat) / 2000e3) ** 2)

        lat, mag = diag.find_jet_position(u, y, "south")

        # Should find the jet near the prescribed location
        assert np.abs(lat - jet_lat) < 0.1 * dy  # Within 10% of grid spacing
        assert np.abs(mag - jet_mag) < 0.5  # Relaxed tolerance for grid discretization

    def test_find_jet_asymmetric_profile(self, basic_grid):
        """Test that northern and southern jets are found separately"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create asymmetric profile with different jets
        u_north = 30.0 * np.exp(-((y - 5000e3) / 2000e3) ** 2)
        u_south = 15.0 * np.exp(-((y + 7000e3) / 2000e3) ** 2)
        u = u_north + u_south

        north_lat, north_mag = diag.find_jet_position(u, y, "north")
        south_lat, south_mag = diag.find_jet_position(u, y, "south")

        # Northern jet should be stronger and different latitude
        assert north_mag > south_mag
        assert north_lat > 0
        assert south_lat < 0
        assert np.abs(north_lat - south_lat) > 10e3  # Different positions

    def test_find_jet_no_hemisphere_data(self):
        """Test jet finding when hemisphere has no data"""
        # Grid that doesn't include northern hemisphere
        y = np.linspace(-15751e3, -1000e3, 25)
        u = np.ones_like(y) * 10.0

        diag = HadleyDiagnostics(ny=25, total_days=10)
        lat, mag = diag.find_jet_position(u, y, "north")

        # Should return NaN when hemisphere not present
        assert np.isnan(lat)
        assert np.isnan(mag)

    def test_record_day_integration(self, basic_grid):
        """Test that record_day correctly stores all diagnostics"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create realistic wind profile
        u = (
            20.0 * np.exp(-((y - 5000e3) / 3000e3) ** 2)
            + 15.0 * np.exp(-((y + 6000e3) / 3000e3) ** 2)
        )
        v = np.zeros_like(y)  # Simple v for this test

        # Record day 0
        diag.record_day(0, u, v, y, dy, beta)

        assert diag.days_recorded == 1
        assert not np.all(np.isnan(diag.rossby_number[0]))
        assert not np.isnan(diag.north_jet_lat[0])
        assert not np.isnan(diag.north_jet_magnitude[0])
        assert not np.isnan(diag.south_jet_lat[0])
        assert not np.isnan(diag.south_jet_magnitude[0])

        # Day 1 should still be NaN
        assert np.all(np.isnan(diag.rossby_number[1]))

    def test_record_multiple_days(self, basic_grid):
        """Test recording multiple days"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)
        v = np.zeros_like(y)  # Simple v for this test

        for day in range(5):
            # Vary jet position over time
            u = (20.0 + day) * np.exp(-((y - (5000 + day * 100) * 1e3) / 3000e3) ** 2)
            diag.record_day(day, u, v, y, dy, beta)

        assert diag.days_recorded == 5

        # Check that jet positions vary
        assert not np.allclose(diag.north_jet_lat[:5], diag.north_jet_lat[0])
        assert not np.allclose(
            diag.north_jet_magnitude[:5], diag.north_jet_magnitude[0]
        )

    def test_get_diagnostics_dict(self, basic_grid):
        """Test conversion to dictionary for xarray"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=100)
        v = np.zeros_like(y)  # Simple v for this test

        # Record 5 days
        for day in range(5):
            u = 20.0 * np.exp(-((y - 5000e3) / 3000e3) ** 2)
            diag.record_day(day, u, v, y, dy, beta)

        diag_dict = diag.get_diagnostics_dict()

        # Check all keys present (including cell edge, cell center, and width keys)
        assert "rossby_number" in diag_dict
        assert "north_jet_lat" in diag_dict
        assert "north_jet_magnitude" in diag_dict
        assert "south_jet_lat" in diag_dict
        assert "south_jet_magnitude" in diag_dict
        assert "north_cell_center_lat" in diag_dict
        assert "north_cell_strength" in diag_dict
        assert "south_cell_center_lat" in diag_dict
        assert "south_cell_strength" in diag_dict
        assert "ascending_edge_lat" in diag_dict
        assert "north_descending_edge_lat" in diag_dict
        assert "south_descending_edge_lat" in diag_dict
        assert "north_hadley_width" in diag_dict
        assert "south_hadley_width" in diag_dict

        # Check filtered to recorded days
        assert diag_dict["rossby_number"].shape == (5, ny)
        assert diag_dict["north_jet_lat"].shape == (5,)
        assert diag_dict["north_cell_center_lat"].shape == (5,)
        assert diag_dict["ascending_edge_lat"].shape == (5,)

        # Check no trailing NaN days included
        assert not np.all(np.isnan(diag_dict["rossby_number"][-1]))

    def test_get_diagnostics_dict_empty(self, basic_grid):
        """Test that empty diagnostics returns empty dict"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # No days recorded
        diag_dict = diag.get_diagnostics_dict()

        assert diag_dict == {}

    def test_invalid_hemisphere_raises(self, basic_grid):
        """Test that invalid hemisphere argument raises error"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        u = np.ones(ny) * 10.0

        with pytest.raises(ValueError, match="hemisphere must be"):
            diag.find_jet_position(u, y, "west")

    def test_find_jet_with_interpolation_accuracy(self, basic_grid):
        """Test that interpolation achieves sub-grid accuracy"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create Gaussian jet at off-grid location
        true_jet_lat = 5137e3  # Not on grid
        true_jet_mag = 25.0
        u = true_jet_mag * np.exp(-((y - true_jet_lat) / 2000e3) ** 2)

        lat, mag = diag.find_jet_position(u, y, "north")

        # Should find jet within 10% of grid spacing
        assert np.abs(lat - true_jet_lat) < 0.1 * dy
        # Magnitude should match grid-point max
        assert np.abs(mag - true_jet_mag) < 0.5

    def test_find_jet_boundary_maximum(self, basic_grid):
        """Test fallback when maximum is at hemisphere boundary"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create monotonically increasing wind in northern hemisphere
        u = np.where(y > 0, y / 1e6, 0)  # Linear increase

        lat, mag = diag.find_jet_position(u, y, "north")

        # Should return northernmost point (boundary) - no interpolation
        assert lat == y[y > 0][-1]
        assert not np.isnan(lat)

    def test_find_jet_near_boundary(self):
        """Test local interpolation when max is near hemisphere boundary"""
        # Grid where northern hemisphere starts at index 25
        y = np.linspace(-15751e3, 15751e3, 51)
        diag = HadleyDiagnostics(ny=51, total_days=10)

        # Create jet very close to equator (y=0+)
        u = 20.0 * np.exp(-(y - 500e3) ** 2 / (1000e3) ** 2)

        lat, mag = diag.find_jet_position(u, y, "north")

        # Should still find jet near 500 km (even though close to boundary)
        assert np.abs(lat - 500e3) < 200e3  # Within reasonable range
        assert not np.isnan(lat)

    def test_find_jet_local_interpolation_improves_accuracy(self, basic_grid):
        """Test that local interpolation refines grid-point maximum"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create smooth Gaussian jet offset from grid points
        true_jet_lat = 6253e3  # Deliberately between grid points
        u = 20.0 * np.exp(-((y - true_jet_lat) / 2000e3) ** 2)

        lat, mag = diag.find_jet_position(u, y, "north")

        # Interpolated position should be closer to true position than nearest grid point
        grid_points = y[y > 0]
        nearest_grid_dist = np.min(np.abs(grid_points - true_jet_lat))
        interp_dist = np.abs(lat - true_jet_lat)

        assert interp_dist < nearest_grid_dist  # Interpolation improves accuracy

    def test_get_diagnostics_dict_restart_scenario(self, basic_grid):
        """Test that diagnostics dict has correct size when starting from non-zero day (restart)."""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=100)
        v = np.zeros_like(y)  # Simple v for this test

        # Simulate restart: record days 50-59 only (like restarting from day 50)
        for day in range(50, 60):
            u = 20.0 * np.exp(-((y - 5000e3) / 3000e3) ** 2)
            diag.record_day(day, u, v, y, dy, beta)

        diag_dict = diag.get_diagnostics_dict()

        # Should have exactly 10 days, not 60 (days 50-59 inclusive)
        assert diag_dict["rossby_number"].shape == (10, ny)
        assert diag_dict["north_jet_lat"].shape == (10,)
        assert diag_dict["south_jet_lat"].shape == (10,)
        assert diag_dict["north_jet_magnitude"].shape == (10,)
        assert diag_dict["south_jet_magnitude"].shape == (10,)
        assert diag_dict["north_cell_center_lat"].shape == (10,)
        assert diag_dict["north_cell_strength"].shape == (10,)
        assert diag_dict["south_cell_center_lat"].shape == (10,)
        assert diag_dict["south_cell_strength"].shape == (10,)
        assert diag_dict["ascending_edge_lat"].shape == (10,)
        assert diag_dict["north_descending_edge_lat"].shape == (10,)
        assert diag_dict["south_descending_edge_lat"].shape == (10,)
        assert diag_dict["north_hadley_width"].shape == (10,)
        assert diag_dict["south_hadley_width"].shape == (10,)

        # None should be NaN (all days were recorded)
        assert not np.any(np.isnan(diag_dict["north_jet_lat"]))
        assert not np.any(np.isnan(diag_dict["north_cell_center_lat"]))


class TestCellEdgeDiagnostics:
    """Tests for Hadley cell edge detection."""

    @pytest.fixture
    def basic_grid(self):
        """Create a basic symmetric grid"""
        ny = 51
        y = np.linspace(-15751e3, 15751e3, ny)
        dy = np.diff(y)[0]
        beta = 2e-11
        return y, dy, beta, ny

    def test_find_zero_crossings_simple(self, basic_grid):
        """Test finding zero crossings with simple linear v profile"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Linear profile that crosses zero at y=0
        v = y / 1e6  # Simple linear: v = y * 1e-6

        crossings = diag.find_zero_crossings(v, y)

        # Should find exactly one crossing near y=0
        assert len(crossings) == 1
        assert np.abs(crossings[0]) < dy  # Within one grid cell of equator

    def test_find_zero_crossings_multiple(self, basic_grid):
        """Test finding multiple zero crossings"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create v profile with 3 crossings (typical Hadley cell pattern)
        # v > 0 in NH (poleward), v < 0 in SH (poleward)
        # with zero crossings at ~-5000km, ~0, ~5000km
        v = np.sin(np.pi * y / 10000e3)

        crossings = diag.find_zero_crossings(v, y)

        # Should find 3 crossings
        assert len(crossings) == 3

    def test_find_zero_crossings_no_crossing(self, basic_grid):
        """Test when v doesn't change sign"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Constant positive v
        v = np.ones_like(y) * 5.0

        crossings = diag.find_zero_crossings(v, y)

        assert len(crossings) == 0

    def test_compute_cell_edges_realistic(self, basic_grid):
        """Test cell edge detection with realistic Hadley cell pattern"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create realistic v profile with single peak per hemisphere:
        # - Positive v in NH (poleward flow in upper branch)
        # - Negative v in SH (poleward flow in upper branch)
        # - Decays toward zero poleward
        center_lat = 3000e3
        width = 3000e3
        v = np.where(y >= 0,
                     5.0 * np.exp(-((y - center_lat) / width)**2),
                     -5.0 * np.exp(-((y + center_lat) / width)**2))

        ascending, north_desc, south_desc = diag.compute_cell_edges(v, y)

        # Ascending edge should be near equator
        assert np.abs(ascending) < dy

        # Descending edges should be in appropriate hemispheres
        assert north_desc > 0
        assert south_desc < 0

    def test_compute_cell_edges_ascending_between_descending(self, basic_grid):
        """Test that ascending edge is always between descending edges"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create v profile with clear Hadley structure
        center_lat = 3000e3
        width = 3000e3
        v = np.where(y >= 0,
                     3.0 * np.exp(-((y - center_lat) / width)**2),
                     -3.0 * np.exp(-((y + center_lat) / width)**2))

        ascending, north_desc, south_desc = diag.compute_cell_edges(v, y)

        # Ascending must be between descending edges
        assert south_desc < ascending < north_desc

    def test_compute_cell_edges_no_crossings(self, basic_grid):
        """Test cell edge detection when no zero crossings exist"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Constant v (no crossings, no v extremum structure)
        v = np.ones_like(y) * 2.0

        ascending, north_desc, south_desc = diag.compute_cell_edges(v, y)

        # All should be NaN when v has no structure
        assert np.isnan(ascending)
        assert np.isnan(north_desc)
        assert np.isnan(south_desc)

    def test_compute_cell_edges_one_hemisphere_only(self, basic_grid):
        """Test cell edge detection when v only exists in one hemisphere"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # v only in NH (zero in SH)
        v = np.where(y > 0, 5.0 * np.exp(-((y - 3000e3) / 2000e3)**2), 0.0)

        ascending, north_desc, south_desc = diag.compute_cell_edges(v, y)

        # Southern descending should be NaN (no v structure in SH)
        assert np.isnan(south_desc)
        # Ascending may or may not be found depending on cell center detection
        # Northern descending should be found
        assert not np.isnan(north_desc)

    def test_compute_cell_edges_linear_interpolation_accuracy(self, basic_grid):
        """Test that linear interpolation gives accurate crossing positions"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create v profile with known zero crossing at y = 3000 km
        # v = y - 3000e3 crosses zero exactly at y = 3000 km
        target_crossing = 3000e3
        v = (y - target_crossing) / 1e6

        crossings = diag.find_zero_crossings(v, y)

        # Should find exactly one crossing very close to 3000 km
        assert len(crossings) == 1
        # Linear interpolation should be exact for linear function
        assert np.abs(crossings[0] - target_crossing) < 1.0  # Within 1 meter

    def test_record_day_with_realistic_v(self, basic_grid):
        """Test that record_day computes cell edges correctly"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create realistic u profile with jets
        u = (
            20.0 * np.exp(-((y - 5000e3) / 3000e3) ** 2)
            + 15.0 * np.exp(-((y + 5000e3) / 3000e3) ** 2)
        )

        # Create realistic v profile with single peak per hemisphere
        center_lat = 3000e3
        width = 3000e3
        v = np.where(y >= 0,
                     5.0 * np.exp(-((y - center_lat) / width)**2),
                     -5.0 * np.exp(-((y + center_lat) / width)**2))

        diag.record_day(0, u, v, y, dy, beta)

        # Check that cell edges were recorded
        assert not np.isnan(diag.ascending_edge_lat[0])
        assert not np.isnan(diag.north_descending_edge_lat[0])
        assert not np.isnan(diag.south_descending_edge_lat[0])

        # Check physical consistency
        assert diag.south_descending_edge_lat[0] < diag.ascending_edge_lat[0]
        assert diag.ascending_edge_lat[0] < diag.north_descending_edge_lat[0]


class TestCellCenterDiagnostics:
    """Tests for Hadley cell center (v extremum) detection."""

    @pytest.fixture
    def basic_grid(self):
        """Create a basic symmetric grid"""
        ny = 51
        y = np.linspace(-15751e3, 15751e3, ny)
        dy = np.diff(y)[0]
        beta = 2e-11
        return y, dy, beta, ny

    def test_find_cell_center_symmetric_profile(self, basic_grid):
        """Test cell center detection with symmetric v profile"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create symmetric v profile: positive in NH, negative in SH
        # v = v_max * sin(pi*y / (2*L)) gives max at y=L, min at y=-L
        L = 5000e3
        v_max = 8.0
        v = v_max * np.sin(np.pi * y / (2 * L))

        north_lat, north_strength = diag.find_cell_center(v, y, dy, "north")
        south_lat, south_strength = diag.find_cell_center(v, y, dy, "south")

        # Northern cell: max v near y=5000km
        assert np.abs(north_lat - L) < 0.15 * dy
        assert north_strength > 0

        # Southern cell: min v near y=-5000km
        assert np.abs(south_lat + L) < 0.15 * dy
        assert south_strength < 0

    def test_find_cell_center_gaussian_profile(self, basic_grid):
        """Test cell center detection with Gaussian v profile"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create Gaussian profiles: v > 0 peak in NH, v < 0 peak in SH
        v_north = 6.0 * np.exp(-((y - 4000e3) / 2000e3) ** 2)
        v_south = -4.0 * np.exp(-((y + 5500e3) / 2000e3) ** 2)
        v = v_north + v_south

        north_lat, north_strength = diag.find_cell_center(v, y, dy, "north")
        south_lat, south_strength = diag.find_cell_center(v, y, dy, "south")

        # Northern cell should be near 4000 km with positive strength
        assert np.abs(north_lat - 4000e3) < 0.15 * dy
        assert north_strength > 0
        assert np.abs(north_strength - 6.0) < 0.5

        # Southern cell should be near -5500 km with negative strength
        assert np.abs(south_lat + 5500e3) < 0.15 * dy
        assert south_strength < 0
        assert np.abs(south_strength + 4.0) < 0.5

    def test_find_cell_center_linear_interpolation_accuracy(self, basic_grid):
        """Test that linear interpolation achieves sub-grid accuracy"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create Gaussian with peak at off-grid location
        true_center = 5137e3  # Deliberately between grid points
        v_max = 7.0
        v = v_max * np.exp(-((y - true_center) / 2000e3) ** 2)

        lat, strength = diag.find_cell_center(v, y, dy, "north")

        # Should find center within 15% of grid spacing
        assert np.abs(lat - true_center) < 0.15 * dy

    def test_find_cell_center_strength_not_below_grid_extremum(self, basic_grid):
        """Cell strength must not fall below the grid-point extremum.

        For an asymmetric peak the refined latitude lands off-grid; linearly
        interpolating v there returns a value below the grid maximum and thus
        understates the cell strength. Report the grid-point extremum instead.
        """
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Asymmetric NH bump with the grid maximum at an interior point.
        v = np.zeros(ny)
        nh = np.where(y > 0)[0]
        peak = nh[len(nh) // 2]
        v[peak - 1] = 3.0
        v[peak] = 5.0
        v[peak + 1] = 4.0
        v[peak + 2] = 2.0

        lat, strength = diag.find_cell_center(v, y, dy, "north")
        assert strength == np.max(v[y > 0])

    def test_find_cell_center_boundary_maximum(self, basic_grid):
        """Test fallback when extremum is at hemisphere boundary"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create monotonically increasing v in northern hemisphere
        v = np.where(y > 0, y / 1e6, 0)  # Linear increase

        lat, strength = diag.find_cell_center(v, y, dy, "north")

        # Should return northernmost point (boundary) - no interpolation
        assert lat == y[y > 0][-1]
        assert not np.isnan(lat)

    def test_find_cell_center_invalid_hemisphere(self, basic_grid):
        """Test that invalid hemisphere raises error"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)
        v = np.sin(y / 1e6)

        with pytest.raises(ValueError, match="hemisphere must be"):
            diag.find_cell_center(v, y, dy, "west")

    def test_find_cell_center_no_hemisphere_data(self):
        """Test cell center when hemisphere has no data"""
        # Grid that doesn't include northern hemisphere
        y = np.linspace(-15751e3, -1000e3, 25)
        dy = np.diff(y)[0]
        v = np.ones_like(y) * -5.0

        diag = HadleyDiagnostics(ny=25, total_days=10)
        lat, strength = diag.find_cell_center(v, y, dy, "north")

        # Should return NaN when hemisphere not present
        assert np.isnan(lat)
        assert np.isnan(strength)

    def test_cell_center_recorded_in_diagnostics(self, basic_grid):
        """Test that cell center is recorded during record_day"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create realistic profiles
        u = 20.0 * np.exp(-((y - 5000e3) / 3000e3) ** 2)
        v = 5.0 * np.sin(np.pi * y / 10000e3)

        diag.record_day(0, u, v, y, dy, beta)

        # Check that cell centers were recorded
        assert not np.isnan(diag.north_cell_center_lat[0])
        assert not np.isnan(diag.north_cell_strength[0])
        assert not np.isnan(diag.south_cell_center_lat[0])
        assert not np.isnan(diag.south_cell_strength[0])

        # Check physical consistency
        assert diag.north_cell_center_lat[0] > 0  # North hemisphere
        assert diag.south_cell_center_lat[0] < 0  # South hemisphere
        assert diag.north_cell_strength[0] > 0    # Poleward (positive) in NH
        assert diag.south_cell_strength[0] < 0    # Poleward (negative) in SH

    def test_cell_center_in_diagnostics_dict(self, basic_grid):
        """Test that cell center appears in diagnostics dict"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        u = 20.0 * np.exp(-((y - 5000e3) / 3000e3) ** 2)
        v = 5.0 * np.sin(np.pi * y / 10000e3)

        for day in range(3):
            diag.record_day(day, u, v, y, dy, beta)

        diag_dict = diag.get_diagnostics_dict()

        # Check all cell center keys present
        assert "north_cell_center_lat" in diag_dict
        assert "north_cell_strength" in diag_dict
        assert "south_cell_center_lat" in diag_dict
        assert "south_cell_strength" in diag_dict

        # Check shapes
        assert diag_dict["north_cell_center_lat"].shape == (3,)
        assert diag_dict["north_cell_strength"].shape == (3,)
        assert diag_dict["south_cell_center_lat"].shape == (3,)
        assert diag_dict["south_cell_strength"].shape == (3,)

    def test_cell_center_interpolation_improves_accuracy(self, basic_grid):
        """Test that linear interpolation refines grid-point extremum"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create smooth Gaussian v offset from grid points
        true_center = 6253e3  # Deliberately between grid points
        v = 8.0 * np.exp(-((y - true_center) / 2000e3) ** 2)

        lat, strength = diag.find_cell_center(v, y, dy, "north")

        # Interpolated position should be closer to true position than nearest grid point
        grid_points = y[y > 0]
        nearest_grid_dist = np.min(np.abs(grid_points - true_center))
        interp_dist = np.abs(lat - true_center)

        assert interp_dist < nearest_grid_dist  # Interpolation improves accuracy

    def test_cell_center_restart_scenario(self, basic_grid):
        """Test cell center diagnostics in restart scenario"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=100)

        u = 20.0 * np.exp(-((y - 5000e3) / 3000e3) ** 2)
        v = 5.0 * np.sin(np.pi * y / 10000e3)

        # Simulate restart: record days 50-59 only
        for day in range(50, 60):
            diag.record_day(day, u, v, y, dy, beta)

        diag_dict = diag.get_diagnostics_dict()

        # Should have exactly 10 days
        assert diag_dict["north_cell_center_lat"].shape == (10,)
        assert diag_dict["south_cell_center_lat"].shape == (10,)
        assert diag_dict["north_cell_strength"].shape == (10,)
        assert diag_dict["south_cell_strength"].shape == (10,)

        # None should be NaN
        assert not np.any(np.isnan(diag_dict["north_cell_center_lat"]))
        assert not np.any(np.isnan(diag_dict["south_cell_center_lat"]))


class TestHadleyCellWidth:
    """Tests for Hadley cell width diagnostics."""

    @pytest.fixture
    def basic_grid(self):
        """Create a basic symmetric grid"""
        ny = 51
        y = np.linspace(-15751e3, 15751e3, ny)
        dy = np.diff(y)[0]
        beta = 2e-11
        return y, dy, beta, ny

    def test_initialization_includes_width_arrays(self, basic_grid):
        """Test that width arrays are initialized correctly"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=100)

        assert diag.north_hadley_width.shape == (100,)
        assert diag.south_hadley_width.shape == (100,)
        assert np.all(np.isnan(diag.north_hadley_width))
        assert np.all(np.isnan(diag.south_hadley_width))

    def test_width_computed_correctly(self, basic_grid):
        """Test that width is computed as abs(edge difference) in km"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create realistic profiles with single peak per hemisphere
        u = (
            20.0 * np.exp(-((y - 5000e3) / 3000e3) ** 2)
            + 15.0 * np.exp(-((y + 5000e3) / 3000e3) ** 2)
        )
        # v profile with peaks at ~3000 km, decaying poleward
        center_lat = 3000e3
        width = 3000e3
        v = np.where(y >= 0,
                     5.0 * np.exp(-((y - center_lat) / width)**2),
                     -5.0 * np.exp(-((y + center_lat) / width)**2))

        diag.record_day(0, u, v, y, dy, beta)

        # Check widths are positive
        assert diag.north_hadley_width[0] > 0
        assert diag.south_hadley_width[0] > 0

        # Check widths are reasonable (10% threshold at ~6900 km from peak at 3000 km)
        assert 3000 < diag.north_hadley_width[0] < 12000
        assert 3000 < diag.south_hadley_width[0] < 12000

        # Verify width matches edge difference
        expected_north = abs(diag.north_descending_edge_lat[0] - diag.ascending_edge_lat[0]) / 1000
        expected_south = abs(diag.ascending_edge_lat[0] - diag.south_descending_edge_lat[0]) / 1000
        assert np.isclose(diag.north_hadley_width[0], expected_north)
        assert np.isclose(diag.south_hadley_width[0], expected_south)

    def test_width_always_positive(self, basic_grid):
        """Test that width is always positive regardless of edge order"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create realistic v profile with single peak per hemisphere
        u = 20.0 * np.exp(-((y - 5000e3) / 3000e3) ** 2)
        center_lat = 3000e3
        width = 3000e3
        v = np.where(y >= 0,
                     5.0 * np.exp(-((y - center_lat) / width)**2),
                     -5.0 * np.exp(-((y + center_lat) / width)**2))

        diag.record_day(0, u, v, y, dy, beta)

        # Width must always be positive (or NaN)
        if not np.isnan(diag.north_hadley_width[0]):
            assert diag.north_hadley_width[0] > 0
        if not np.isnan(diag.south_hadley_width[0]):
            assert diag.south_hadley_width[0] > 0

    def test_width_nan_when_edges_undefined(self, basic_grid):
        """Test that width is NaN when cell edges cannot be determined"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Constant v (no zero crossings)
        u = 20.0 * np.exp(-((y - 5000e3) / 3000e3) ** 2)
        v = np.ones_like(y) * 2.0

        diag.record_day(0, u, v, y, dy, beta)

        # Widths should be NaN since edges aren't defined
        assert np.isnan(diag.north_hadley_width[0])
        assert np.isnan(diag.south_hadley_width[0])

    def test_width_in_diagnostics_dict(self, basic_grid):
        """Test that widths appear in diagnostics dict"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        u = 20.0 * np.exp(-((y - 5000e3) / 3000e3) ** 2)
        center_lat = 3000e3
        width = 3000e3
        v = np.where(y >= 0,
                     5.0 * np.exp(-((y - center_lat) / width)**2),
                     -5.0 * np.exp(-((y + center_lat) / width)**2))

        for day in range(3):
            diag.record_day(day, u, v, y, dy, beta)

        diag_dict = diag.get_diagnostics_dict()

        assert "north_hadley_width" in diag_dict
        assert "south_hadley_width" in diag_dict
        assert diag_dict["north_hadley_width"].shape == (3,)
        assert diag_dict["south_hadley_width"].shape == (3,)

    def test_width_restart_scenario(self, basic_grid):
        """Test width diagnostics in restart scenario"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=100)

        u = 20.0 * np.exp(-((y - 5000e3) / 3000e3) ** 2)
        center_lat = 3000e3
        width = 3000e3
        v = np.where(y >= 0,
                     5.0 * np.exp(-((y - center_lat) / width)**2),
                     -5.0 * np.exp(-((y + center_lat) / width)**2))

        # Simulate restart: record days 50-59 only
        for day in range(50, 60):
            diag.record_day(day, u, v, y, dy, beta)

        diag_dict = diag.get_diagnostics_dict()

        # Should have exactly 10 days
        assert diag_dict["north_hadley_width"].shape == (10,)
        assert diag_dict["south_hadley_width"].shape == (10,)

    def test_asymmetric_widths(self, basic_grid):
        """Test that cells can have different widths"""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create asymmetric profiles
        u = (
            20.0 * np.exp(-((y - 5000e3) / 3000e3) ** 2)
            + 15.0 * np.exp(-((y + 7000e3) / 3000e3) ** 2)
        )
        # Asymmetric v: wider cell in south (larger width parameter)
        # v peaks at ~3000 km in each hemisphere, but with different widths
        north_width = 2500e3  # Narrower cell
        south_width = 4000e3  # Wider cell
        center = 3000e3
        v = np.where(y >= 0,
                     5.0 * np.exp(-((y - center) / north_width)**2),
                     -5.0 * np.exp(-((y + center) / south_width)**2))

        diag.record_day(0, u, v, y, dy, beta)

        # If both widths are defined, southern should be larger
        if not np.isnan(diag.north_hadley_width[0]) and not np.isnan(diag.south_hadley_width[0]):
            assert diag.south_hadley_width[0] > diag.north_hadley_width[0]


class TestThresholdBasedDescendingEdge:
    """Tests for threshold-based descending edge detection."""

    @pytest.fixture
    def basic_grid(self):
        """Create a basic symmetric grid"""
        ny = 51
        y = np.linspace(-15751e3, 15751e3, ny)
        dy = np.diff(y)[0]
        beta = 2e-11
        return y, dy, beta, ny

    def test_find_descending_edge_threshold_basic(self, basic_grid):
        """Test threshold-based descending edge detection with smooth decay."""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create v profile that decays smoothly without zero crossing
        # v = exp(-y^2/(2*sigma^2)) * sign for each hemisphere
        sigma = 3000e3
        v = np.where(y >= 0,
                     np.exp(-y**2 / (2*sigma**2)),
                     -np.exp(-y**2 / (2*sigma**2)))

        # Find descending edges
        north_desc = diag.find_descending_edge_threshold(v, y, "north")
        south_desc = diag.find_descending_edge_threshold(v, y, "south")

        # Edges should be in correct hemispheres
        assert north_desc > 0
        assert south_desc < 0

        # Check that edges are found (not NaN)
        assert not np.isnan(north_desc)
        assert not np.isnan(south_desc)

    def test_find_descending_edge_threshold_at_10_percent(self, basic_grid):
        """Test that edge is found where |v| drops to ~10% of extremum."""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create Gaussian v profile with known decay
        sigma = 4000e3
        v_max = 10.0
        v = np.where(y >= 0,
                     v_max * np.exp(-y**2 / (2*sigma**2)),
                     -v_max * np.exp(-y**2 / (2*sigma**2)))

        north_desc = diag.find_descending_edge_threshold(v, y, "north")

        # At 10% threshold, |v| = 0.1 * v_max = 1.0
        # exp(-y^2/(2*sigma^2)) = 0.1 => y = sqrt(-2*sigma^2*ln(0.1))
        expected_y = np.sqrt(-2 * sigma**2 * np.log(0.1))

        # Should be within a grid spacing of expected
        assert np.abs(north_desc - expected_y) < dy

    def test_find_descending_edge_threshold_custom_fraction(self, basic_grid):
        """Test threshold detection with custom threshold fraction."""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create Gaussian v profile
        sigma = 4000e3
        v = np.where(y >= 0,
                     np.exp(-y**2 / (2*sigma**2)),
                     -np.exp(-y**2 / (2*sigma**2)))

        # 10% threshold should give farther edge than 50% threshold
        edge_10 = diag.find_descending_edge_threshold(v, y, "north", threshold_fraction=0.1)
        edge_50 = diag.find_descending_edge_threshold(v, y, "north", threshold_fraction=0.5)

        assert edge_10 > edge_50  # 10% edge is farther poleward

    def test_find_descending_edge_threshold_invalid_hemisphere(self, basic_grid):
        """Test that invalid hemisphere raises error."""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)
        v = np.sin(y / 1e6)

        with pytest.raises(ValueError, match="hemisphere must be"):
            diag.find_descending_edge_threshold(v, y, "west")

    def test_find_descending_edge_threshold_zero_extremum(self, basic_grid):
        """Test that zero extremum returns NaN."""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # v = 0 everywhere in NH
        v = np.where(y >= 0, 0.0, -1.0)

        north_desc = diag.find_descending_edge_threshold(v, y, "north")

        assert np.isnan(north_desc)

    def test_find_descending_edge_threshold_never_drops_below(self, basic_grid):
        """Test NaN returned when v never drops below threshold."""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Very wide Gaussian that never drops below 10% in domain
        sigma = 50000e3  # Much larger than domain
        v = np.where(y >= 0,
                     np.exp(-y**2 / (2*sigma**2)),
                     -np.exp(-y**2 / (2*sigma**2)))

        north_desc = diag.find_descending_edge_threshold(v, y, "north")

        # Should return NaN since threshold never reached
        assert np.isnan(north_desc)


class TestComputeCellEdgesThreshold:
    """Tests for updated compute_cell_edges using threshold approach."""

    @pytest.fixture
    def basic_grid(self):
        """Create a basic symmetric grid"""
        ny = 51
        y = np.linspace(-15751e3, 15751e3, ny)
        dy = np.diff(y)[0]
        beta = 2e-11
        return y, dy, beta, ny

    def test_compute_cell_edges_asymptotic_v(self, basic_grid):
        """Test cell edge detection when v asymptotes to zero."""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create v profile that crosses zero at equator but asymptotes poleward
        sigma = 4000e3
        v = y / sigma * np.exp(-y**2 / (2*sigma**2))  # Has zero at equator, asymptotes to 0

        # New signature: no jet position parameters
        ascending, north_desc, south_desc = diag.compute_cell_edges(v, y)

        # All edges should be found
        assert not np.isnan(ascending)
        assert not np.isnan(north_desc)
        assert not np.isnan(south_desc)

        # Ascending should be near equator (where v=0 crossing is)
        assert abs(ascending) < dy

        # Descending edges in correct hemispheres
        assert north_desc > 0
        assert south_desc < 0

        # Ordering: south_desc < ascending < north_desc
        assert south_desc < ascending < north_desc

    def test_compute_cell_edges_no_jet_params(self, basic_grid):
        """Test that compute_cell_edges works without jet position parameters."""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create realistic v profile with single peak per hemisphere
        # v peaks at ~3000 km and decays poleward (like a real Hadley cell)
        center_lat = 3000e3
        width = 3000e3
        v = np.where(y >= 0,
                     5.0 * np.exp(-((y - center_lat) / width)**2),
                     -5.0 * np.exp(-((y + center_lat) / width)**2))

        # New signature: only v and y
        ascending, north_desc, south_desc = diag.compute_cell_edges(v, y)

        # Should find valid edges
        assert not np.isnan(ascending)
        assert not np.isnan(north_desc)
        assert not np.isnan(south_desc)

    def test_compute_cell_edges_uses_cell_centers_for_ascending(self, basic_grid):
        """Test that ascending edge search uses cell centers as bounds."""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create v profile with single peak per hemisphere
        # v crosses zero only at equator
        center_lat = 3000e3
        width = 3000e3
        v = np.where(y >= 0,
                     5.0 * np.exp(-((y - center_lat) / width)**2),
                     -5.0 * np.exp(-((y + center_lat) / width)**2))

        ascending, north_desc, south_desc = diag.compute_cell_edges(v, y)

        # Ascending should be near equator
        assert np.abs(ascending) < 1000e3

    def test_compute_cell_edges_realistic_hadley_pattern(self, basic_grid):
        """Test with realistic Hadley cell v pattern."""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create realistic v profile:
        # - Positive v in NH (poleward flow in upper branch)
        # - Negative v in SH (poleward flow in upper branch)
        # - Zero at equator (ascending) and decaying poleward
        v_max = 5.0
        center_lat = 3000e3
        width = 3000e3

        # Gaussian poleward flow centered at ±center_lat
        v = np.where(y >= 0,
                     v_max * np.exp(-((y - center_lat) / width)**2),
                     -v_max * np.exp(-((y + center_lat) / width)**2))

        ascending, north_desc, south_desc = diag.compute_cell_edges(v, y)

        # Check that all edges are found
        assert not np.isnan(ascending)
        assert not np.isnan(north_desc)
        assert not np.isnan(south_desc)

        # Physical ordering
        assert south_desc < ascending < north_desc

    def test_record_day_uses_new_signature(self, basic_grid):
        """Test that record_day works with updated compute_cell_edges."""
        y, dy, beta, ny = basic_grid
        diag = HadleyDiagnostics(ny=ny, total_days=10)

        # Create realistic profiles with single peak per hemisphere
        u = 20.0 * np.exp(-((y - 5000e3) / 3000e3) ** 2)
        center_lat = 3000e3
        width = 3000e3
        v = np.where(y >= 0,
                     5.0 * np.exp(-((y - center_lat) / width)**2),
                     -5.0 * np.exp(-((y + center_lat) / width)**2))

        # This should work without error after the fix
        diag.record_day(0, u, v, y, dy, beta)

        # Should have valid cell widths
        assert not np.isnan(diag.north_hadley_width[0])
        assert not np.isnan(diag.south_hadley_width[0])
