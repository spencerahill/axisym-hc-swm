"""Tests for the shared center-v output loader.

Model output stores v on the ny-1 faces (staggered) or the ny centers
(collocated). load_centered normalizes both to the centers so analysis code
has one grid to reason about, and the reconstruction is bit-for-bit the same
one the model uses internally for its own diagnostics (v_faces_to_centers).
"""

import numpy as np

from ss09.sw_config import SWConfig
from ss09.sw_model import SWModel, v_faces_to_centers
from ss09.theta_e import ThetaEConfig, Sin2Profile
from ss09.read_output import load_centered


def _write_two_days(grid, path):
    """Populate two days of daily results with distinctive fields and write
    an output file. Returns (config, daily_v) for cross-checking."""
    config = SWConfig(total_integration_days=2, ny=51, dt=3600, grid=grid)
    model = SWModel(config, Sin2Profile(ThetaEConfig()))
    rng = np.random.default_rng(3)
    daily_v = []
    for day in range(2):
        u = rng.normal(0, 5, config.ny)
        v = rng.normal(0, 1, config.nv)
        theta = 300 + rng.normal(0, 2, config.ny)
        model.results.store_day(day, float(day + 1), u, v, theta)
        daily_v.append(v)
    ds = model.results.to_xarray(
        config, model.theta_e_profile, model.steady_state_detector,
        model.hadley_diagnostics,
    )
    ds.to_netcdf(path)
    return config, daily_v


def test_load_centered_staggered_reconstructs_faces(tmp_path):
    """A staggered file has v on y_face; load_centered returns v on the ny
    centers, reconstructed exactly as the model does internally."""
    path = tmp_path / "stag.nc"
    config, daily_v = _write_two_days("staggered", path)

    y, u, v, T = load_centered(str(path))
    assert y.shape == (config.ny,)
    assert v.shape == (2, config.ny)  # centers, not faces
    for day in range(2):
        np.testing.assert_array_equal(v[day], v_faces_to_centers(daily_v[day]))


def test_load_centered_collocated_passthrough(tmp_path):
    """A collocated file already has v on the centers; load_centered returns it
    unchanged."""
    path = tmp_path / "collo.nc"
    config, daily_v = _write_two_days("collocated", path)

    y, u, v, T = load_centered(str(path))
    assert v.shape == (2, config.ny)
    for day in range(2):
        np.testing.assert_array_equal(v[day], daily_v[day])


def test_load_centered_time_mean(tmp_path):
    """ndays time-means the last N days and still returns center-v."""
    path = tmp_path / "stag.nc"
    config, daily_v = _write_two_days("staggered", path)

    y, u, v, T = load_centered(str(path), ndays=1)
    assert v.shape == (config.ny,)
    np.testing.assert_array_equal(v, v_faces_to_centers(daily_v[-1]))


def test_output_v_face_coordinate_attrs(tmp_path):
    """The staggered output file carries v on a y_face coordinate tagged with
    its grid, and u/T stay on y."""
    import xarray as xr
    path = tmp_path / "stag.nc"
    _write_two_days("staggered", path)
    ds = xr.open_dataset(path, decode_timedelta=False)
    assert ds["v"].dims == ("time", "y_face")
    assert ds["v"].attrs.get("grid") == "staggered_face"
    assert ds["u"].dims == ("time", "y")
    assert ds["T"].dims == ("time", "y")
    assert "y_face" in ds.coords
