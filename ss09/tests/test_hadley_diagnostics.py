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
        assert diag.days_recorded == 0

        # Check pre-filled with NaN
        assert np.all(np.isnan(diag.rossby_number))
        assert np.all(np.isnan(diag.north_jet_lat))

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

        # Record day 0
        diag.record_day(0, u, y, dy, beta)

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

        for day in range(5):
            # Vary jet position over time
            u = (20.0 + day) * np.exp(-((y - (5000 + day * 100) * 1e3) / 3000e3) ** 2)
            diag.record_day(day, u, y, dy, beta)

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

        # Record 5 days
        for day in range(5):
            u = 20.0 * np.exp(-((y - 5000e3) / 3000e3) ** 2)
            diag.record_day(day, u, y, dy, beta)

        diag_dict = diag.get_diagnostics_dict()

        # Check all keys present
        assert "rossby_number" in diag_dict
        assert "north_jet_lat" in diag_dict
        assert "north_jet_magnitude" in diag_dict
        assert "south_jet_lat" in diag_dict
        assert "south_jet_magnitude" in diag_dict

        # Check filtered to recorded days
        assert diag_dict["rossby_number"].shape == (5, ny)
        assert diag_dict["north_jet_lat"].shape == (5,)

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
