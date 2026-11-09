"""Tests for the spherical geometry option: the grid and its geometry arrays.

On the sphere the model's coordinate is y = a sin(lat), on a grid uniform in
y from pole to pole, so cells of equal width in y have equal area. The
geometry enters the operators only through arrays computed once by the
configuration: cos(lat) at the centers and faces, its square, its reciprocal
(zeroed at the two pole centers), the Coriolis parameter f = (2 Omega / a) y,
and the metric factor tan(lat)/a at the faces. On the beta plane the cosine
arrays are exactly 1.0 and f is beta*y, so every operator reduces to the
beta-plane one bit for bit.

Expected values are rebuilt from raw numpy trigonometry (arcsin, cos, tan),
independently of the configuration's own algebraic forms.
"""

import numpy as np
import pytest

from ss09.sw_config import SWConfig

A_EARTH = 6.371e6
OMEGA_WGS84 = 7.292115e-5


# --- the spherical grid ------------------------------------------------------

def test_sphere_grid_is_uniform_in_y_pole_to_pole():
    cfg = SWConfig(geometry="sphere", ny=325)
    m = 162
    assert cfg.radius == A_EARTH
    assert cfg.rot_rate == OMEGA_WGS84
    assert cfg.y.shape == (325,)
    assert cfg.y[0] == -A_EARTH and cfg.y[-1] == A_EARTH
    assert cfg.y[m] == 0.0
    assert cfg.domain_size == 2 * A_EARTH
    assert cfg.dy == 2 * A_EARTH / 324
    # uniform to roundoff
    assert np.max(np.abs(np.diff(cfg.y) - cfg.dy)) < 1e-6
    # the faces are the midpoints, all strictly inside the poles
    assert cfg.y_v.shape == (324,)
    np.testing.assert_allclose(cfg.y_v, 0.5 * (cfg.y[:-1] + cfg.y[1:]), rtol=0)
    assert np.all(np.abs(cfg.y_v) < A_EARTH)


@pytest.mark.parametrize("ny", [21, 81, 325, 649])
def test_sphere_grid_exactly_antisymmetric(ny):
    """Exact mirror parity needs y == -y[::-1] bit for bit; np.linspace does
    not give that on these grids, a (k/m) does."""
    cfg = SWConfig(geometry="sphere", ny=ny)
    assert np.array_equal(cfg.y, -cfg.y[::-1])
    assert np.array_equal(cfg.y_v, -cfg.y_v[::-1])


def test_linspace_is_not_antisymmetric_on_the_sphere_grid():
    """Positive control for the test above: the beta-plane construction,
    np.linspace(-a, a, 325), is off its own mirror image at some points, so
    the antisymmetry test can fail."""
    y_lin = np.linspace(-A_EARTH, A_EARTH, 325)
    assert np.sum(y_lin != -y_lin[::-1]) > 0


def test_sphere_requires_odd_ny():
    with pytest.raises(ValueError, match="odd"):
        SWConfig(geometry="sphere", ny=324)


def test_sphere_custom_planet():
    cfg = SWConfig(geometry="sphere", ny=41, radius=3.0e6, rot_rate=1.0e-4)
    assert cfg.y[-1] == 3.0e6
    np.testing.assert_allclose(cfg.geom.f, (2.0e-4 / 3.0e6) * cfg.y, rtol=1e-15)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(domain_size=1.0e7), "domain_size"),
        (dict(beta=2.3e-11), "beta"),
        (dict(grid="collocated"), "collocated"),
        (dict(enable_moisture=True), "moisture"),
        (dict(geometry="cylinder"), "geometry"),
    ],
)
def test_sphere_refusals(kwargs, match):
    base = dict(geometry="sphere", ny=41)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        SWConfig(**base)


@pytest.mark.parametrize(
    "kwargs, match",
    [(dict(radius=3.0e6), "radius"), (dict(rot_rate=1.0e-4), "rot_rate")],
)
def test_beta_plane_refuses_inert_planet_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SWConfig(ny=41, **kwargs)


# --- geometry arrays ---------------------------------------------------------

@pytest.mark.parametrize("grid", ["staggered", "collocated"])
def test_beta_plane_geometry_arrays_are_exactly_flat(grid):
    cfg = SWConfig(ny=51, grid=grid)
    g = cfg.geom
    assert not g.sphere
    for name in ("cos_lat", "cos2_lat", "inv_cos_lat"):
        arr = getattr(g, name)
        assert arr.shape == cfg.y.shape
        assert np.all(arr == 1.0), name
    assert g.cos_lat_face.shape == cfg.y_v.shape
    assert np.all(g.cos_lat_face == 1.0)
    # f is beta*y bit for bit, the exact product the beta-plane operators form
    assert np.array_equal(g.f, cfg.beta * cfg.y)
    assert np.array_equal(g.f_face, cfg.beta * cfg.y_v)
    assert g.lat is None and g.lat_face is None


def test_sphere_geometry_arrays_match_trigonometry():
    cfg = SWConfig(geometry="sphere", ny=325)
    g = cfg.geom
    assert g.sphere
    lat = np.arcsin(cfg.y / A_EARTH)
    lat_face = np.arcsin(cfg.y_v / A_EARTH)
    np.testing.assert_allclose(g.lat, np.rad2deg(lat), rtol=0, atol=1e-12)
    np.testing.assert_allclose(g.lat_face, np.rad2deg(lat_face), rtol=0, atol=1e-12)
    np.testing.assert_allclose(g.cos_lat, np.cos(lat), rtol=0, atol=1e-15)
    np.testing.assert_allclose(g.cos_lat_face, np.cos(lat_face), rtol=0, atol=1e-15)
    np.testing.assert_allclose(g.cos2_lat, np.cos(lat) ** 2, rtol=0, atol=1e-15)
    inner = slice(1, -1)
    np.testing.assert_allclose(
        g.inv_cos_lat[inner], 1.0 / np.cos(lat[inner]), rtol=1e-12
    )
    np.testing.assert_allclose(
        g.f, 2 * OMEGA_WGS84 * np.sin(lat), rtol=1e-14, atol=1e-20
    )
    np.testing.assert_allclose(
        g.f_face, 2 * OMEGA_WGS84 * np.sin(lat_face), rtol=1e-14, atol=1e-20
    )
    np.testing.assert_allclose(
        g.metric_face, np.tan(lat_face) / A_EARTH, rtol=1e-12, atol=1e-25
    )


def test_sphere_geometry_poles_and_symmetry():
    cfg = SWConfig(geometry="sphere", ny=325)
    g = cfg.geom
    # cos(lat) is exactly zero at the pole centers, and so is its reciprocal
    # array (the pole u tendency is discarded by the u = 0 condition)
    assert g.cos_lat[0] == 0.0 and g.cos_lat[-1] == 0.0
    assert g.inv_cos_lat[0] == 0.0 and g.inv_cos_lat[-1] == 0.0
    assert np.all(g.cos_lat[1:-1] > 0.0)
    assert np.all(g.cos_lat_face > 0.0)
    # exact mirror symmetry of the even arrays, antisymmetry of the odd ones
    for name in ("cos_lat", "cos2_lat", "inv_cos_lat"):
        arr = getattr(g, name)
        assert np.array_equal(arr, arr[::-1]), name
    assert np.array_equal(g.cos_lat_face, g.cos_lat_face[::-1])
    assert np.array_equal(g.f, -g.f[::-1])
    assert np.array_equal(g.f_face, -g.f_face[::-1])
    assert np.array_equal(g.metric_face, -g.metric_face[::-1])
    # the outermost interior centers and faces, as the plan states them
    assert round(g.lat[-2], 2) == 83.63
    assert round(g.lat_face[-1], 2) == 85.50
