"""Unit tests for the staggered C-grid geometry and flux-form operators.

These are Stage-0 tests of the numerics rewrite: every spatial operator is
checked in isolation on analytic inputs, plus the discrete conservation and
symmetry-parity properties the rewrite is meant to guarantee.
"""

import numpy as np

from ss09.grid import (
    StaggeredGrid,
    grad_c2f,
    div_f2c,
    avg_f2c,
    avg_c2f,
    lap_face_dirichlet,
    lap_center_neumann,
    ddy_center,
    ddy_upwind,
)


DOMAIN = 15751e3 * 2


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def test_grid_even_n_has_face_at_equator_no_center():
    g = StaggeredGrid(n=50, domain_size=DOMAIN)
    # dy = domain / N
    assert np.isclose(g.dy, DOMAIN / 50)
    # N centers, N+1 faces
    assert g.yc.shape == (50,)
    assert g.yf.shape == (51,)
    # boundary faces sit on the walls
    assert np.isclose(g.yf[0], -DOMAIN / 2)
    assert np.isclose(g.yf[-1], DOMAIN / 2)
    # exactly one face at the equator, no center at the equator
    assert np.any(np.isclose(g.yf, 0.0))
    assert not np.any(np.isclose(g.yc, 0.0))


def test_grid_centers_symmetric_about_equator():
    g = StaggeredGrid(n=50, domain_size=DOMAIN)
    np.testing.assert_allclose(g.yc, -g.yc[::-1], atol=1e-6)
    np.testing.assert_allclose(g.yf, -g.yf[::-1], atol=1e-6)


def test_grid_matches_legacy_resolution():
    # Legacy collocated grid used ny=51 -> dy = domain/(ny-1) = domain/50.
    # New N=50 centers give the same dy.
    g = StaggeredGrid(n=50, domain_size=DOMAIN)
    legacy_dy = DOMAIN / (51 - 1)
    assert np.isclose(g.dy, legacy_dy)


# --------------------------------------------------------------------------
# First-derivative operators (exact on linear fields)
# --------------------------------------------------------------------------
def test_grad_c2f_linear_field_is_exact_on_interior():
    g = StaggeredGrid(n=50, domain_size=DOMAIN)
    a, b = 3.7e-6, 12.0
    c = a * g.yc + b
    f = grad_c2f(c, g.dy)
    assert f.shape == (g.n + 1,)
    np.testing.assert_allclose(f[1:-1], a, rtol=1e-9)


def test_div_f2c_linear_field_is_exact():
    g = StaggeredGrid(n=50, domain_size=DOMAIN)
    a, b = -2.1e-6, 5.0
    f = a * g.yf + b
    c = div_f2c(f, g.dy)
    assert c.shape == (g.n,)
    np.testing.assert_allclose(c, a, rtol=1e-9)


def test_ddy_center_linear_field_is_exact_everywhere():
    g = StaggeredGrid(n=50, domain_size=DOMAIN)
    a, b = 1.3e-6, -4.0
    c = a * g.yc + b
    d = ddy_center(c, g.dy)
    assert d.shape == (g.n,)
    # exact for a linear field at every center, boundaries included
    np.testing.assert_allclose(d, a, rtol=1e-9)


# --------------------------------------------------------------------------
# Averaging operators
# --------------------------------------------------------------------------
def test_avg_f2c_constant():
    g = StaggeredGrid(n=50, domain_size=DOMAIN)
    f = np.full(g.n + 1, 7.0)
    np.testing.assert_allclose(avg_f2c(f), 7.0)
    assert avg_f2c(f).shape == (g.n,)


def test_avg_c2f_interior_is_midpoint():
    g = StaggeredGrid(n=50, domain_size=DOMAIN)
    c = np.arange(g.n, dtype=float)
    f = avg_c2f(c)
    assert f.shape == (g.n + 1,)
    np.testing.assert_allclose(f[1:-1], 0.5 * (c[:-1] + c[1:]))


# --------------------------------------------------------------------------
# Laplacians
# --------------------------------------------------------------------------
def test_lap_face_dirichlet_quadratic_interior():
    g = StaggeredGrid(n=50, domain_size=DOMAIN)
    f = g.yf ** 2
    lap = lap_face_dirichlet(f, g.dy)
    assert lap.shape == (g.n + 1,)
    np.testing.assert_allclose(lap[1:-1], 2.0, rtol=1e-7)
    # boundary faces (v=0 walls) do not evolve
    assert lap[0] == 0.0
    assert lap[-1] == 0.0


def test_lap_center_neumann_constant_is_zero():
    g = StaggeredGrid(n=50, domain_size=DOMAIN)
    c = np.full(g.n, 5.0)
    lap = lap_center_neumann(c, g.dy)
    assert lap.shape == (g.n,)
    np.testing.assert_allclose(lap, 0.0, atol=1e-12)


def test_lap_center_neumann_quadratic_interior():
    g = StaggeredGrid(n=50, domain_size=DOMAIN)
    c = g.yc ** 2
    lap = lap_center_neumann(c, g.dy)
    np.testing.assert_allclose(lap[1:-1], 2.0, rtol=1e-7)


# --------------------------------------------------------------------------
# Symmetry parity (exact, to roundoff)
# --------------------------------------------------------------------------
def _even_center(g):
    return np.cos(np.pi * g.yc / g.domain_size)  # even about y=0


def _odd_face(g):
    return np.sin(2 * np.pi * g.yf / g.domain_size)  # odd about y=0


def test_grad_c2f_of_even_center_is_odd_on_faces():
    g = StaggeredGrid(n=50, domain_size=DOMAIN)
    c = _even_center(g)
    f = grad_c2f(c, g.dy)
    # odd about the equator: f[j] = -f[N-j]
    np.testing.assert_allclose(f[1:-1], -f[1:-1][::-1], atol=1e-12)


def test_div_f2c_of_odd_face_is_even_on_centers():
    g = StaggeredGrid(n=50, domain_size=DOMAIN)
    f = _odd_face(g)
    c = div_f2c(f, g.dy)
    np.testing.assert_allclose(c, c[::-1], atol=1e-12)


def test_ddy_center_of_even_is_odd():
    g = StaggeredGrid(n=50, domain_size=DOMAIN)
    c = _even_center(g)
    d = ddy_center(c, g.dy)
    np.testing.assert_allclose(d, -d[::-1], atol=1e-12)


# --------------------------------------------------------------------------
# Upwind advective gradient (original SS09 v du/dy discretization)
# --------------------------------------------------------------------------
def test_ddy_upwind_picks_upwind_side():
    g = StaggeredGrid(n=6, domain_size=DOMAIN)
    c = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0])  # forward gaps: 1,2,3,4,5
    # positive velocity -> backward difference; the first point has no upwind
    # neighbor, so it stays 0
    gp = ddy_upwind(c, np.ones(6), g.dy)
    np.testing.assert_allclose(gp * g.dy, [0, 1, 2, 3, 4, 5])
    # negative velocity -> forward difference; the last point stays 0
    gn = ddy_upwind(c, -np.ones(6), g.dy)
    np.testing.assert_allclose(gn * g.dy, [1, 2, 3, 4, 5, 0])
    # zero velocity -> no advection
    np.testing.assert_allclose(ddy_upwind(c, np.zeros(6), g.dy), 0.0)


def test_ddy_upwind_advective_term_preserves_parity():
    g = StaggeredGrid(n=50, domain_size=DOMAIN)
    c = _even_center(g)                                   # even
    vel = np.sin(2 * np.pi * g.yc / g.domain_size)        # odd velocity
    # the momentum tendency -vel * du/dy must be even from symmetric data
    term = vel * ddy_upwind(c, vel, g.dy)
    np.testing.assert_allclose(term, term[::-1], atol=1e-12)
