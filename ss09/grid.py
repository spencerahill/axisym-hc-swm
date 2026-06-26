"""Staggered (Arakawa C) grid geometry and flux-form spatial operators.

One-dimensional equatorial beta-plane. The domain ``[-L, L]`` (``L =
domain_size / 2``) is divided into ``N`` cells of width ``dy = domain_size / N``.

  - **Cell centers** (``N`` points) carry ``u`` and ``theta`` (and later ``W``)::

        yc[i] = -L + (i + 0.5) * dy,    i = 0 .. N-1

  - **Cell faces** (``N+1`` points) carry ``v``::

        yf[j] = -L + j * dy,            j = 0 .. N

    The two boundary faces (``j = 0, N``) sit on the walls where ``v = 0``,
    leaving ``N-1`` interior ``v`` unknowns.

With ``N`` even the centers are symmetric about the equator with **no point at
``y = 0``** (avoiding ``sgn(0)`` in the eddy-momentum term) and a **face exactly
at ``y = 0``** (so an odd ``v`` field vanishes there exactly). ``N`` even is
therefore recommended.

The gravity wave lives in the ``v``-``theta`` couple. On the C-grid ``d/dy`` of a
center field lands on faces (a single adjacent difference) and ``d/dy`` of a
face field lands on centers, each with no ``2*dy`` null space. This removes the
computational mode that the old collocated scheme needed ``k_v`` and the Asselin
filter to suppress.

All operators are pure functions of NumPy arrays and the grid spacing ``dy``;
they hold no state and are tested in isolation (``test_grid.py``).
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class StaggeredGrid:
    """1-D Arakawa C-grid: ``n`` cell centers and ``n+1`` cell faces."""

    n: int  # number of cell centers
    domain_size: float
    dy: float = field(init=False)
    yc: np.ndarray = field(init=False)  # cell centers, shape (n,)
    yf: np.ndarray = field(init=False)  # cell faces, shape (n+1,)

    def __post_init__(self):
        if self.n < 2:
            raise ValueError(f"StaggeredGrid needs n >= 2, got {self.n}")
        half = self.domain_size / 2
        self.dy = self.domain_size / self.n
        self.yc = -half + (np.arange(self.n) + 0.5) * self.dy
        self.yf = -half + np.arange(self.n + 1) * self.dy


# --------------------------------------------------------------------------
# First-derivative operators
# --------------------------------------------------------------------------
def grad_c2f(c: np.ndarray, dy: float) -> np.ndarray:
    """``d/dy`` of a center field, returned on faces (length ``n+1``).

    Interior faces use the single adjacent difference ``(c[j] - c[j-1]) / dy``.
    The two boundary faces are set to 0: they feed only the pressure-gradient
    term of the ``v`` equation, which is never evaluated on the (non-evolving)
    wall faces.
    """
    out = np.zeros(len(c) + 1)
    out[1:-1] = (c[1:] - c[:-1]) / dy
    return out


def div_f2c(f: np.ndarray, dy: float) -> np.ndarray:
    """``d/dy`` of a face field, returned on centers (length ``n``).

    Exact single adjacent difference ``(f[i+1] - f[i]) / dy``. This is both the
    divergence ``d_y v`` operator and the building block of the flux divergence.
    """
    return (f[1:] - f[:-1]) / dy


def avg_f2c(f: np.ndarray) -> np.ndarray:
    """Two-point average of a face field onto centers (length ``n``)."""
    return 0.5 * (f[:-1] + f[1:])


def avg_c2f(c: np.ndarray) -> np.ndarray:
    """Two-point average of a center field onto faces (length ``n+1``).

    Boundary faces are filled with the nearest center value. Those entries are
    unused downstream (the momentum flux ``v*u_hat`` vanishes there because
    ``v = 0``, and the wall faces of ``v`` do not evolve), but filling them
    avoids spurious zeros/NaNs.
    """
    out = np.empty(len(c) + 1)
    out[1:-1] = 0.5 * (c[:-1] + c[1:])
    out[0] = c[0]
    out[-1] = c[-1]
    return out


def flux_div_vu(v_face: np.ndarray, u_center: np.ndarray, dy: float) -> np.ndarray:
    """Conservative momentum-flux divergence ``d_y(v u)`` on centers.

    The flux ``F = v * u_hat`` is formed on faces (``u_hat`` interpolated from
    centers), then differenced back to centers. Because ``v = 0`` on the wall
    faces, the boundary fluxes vanish and the domain integral of the result is
    exactly zero (discrete conservation of ``integral u dy`` from advection).
    """
    flux = v_face * avg_c2f(u_center)
    return div_f2c(flux, dy)


def lap_face_dirichlet(f: np.ndarray, dy: float) -> np.ndarray:
    """3-point Laplacian on faces with Dirichlet ``v = 0`` walls (length ``n+1``).

    Boundary faces return 0 (they do not evolve); interior faces use the
    standard ``(f[j+1] - 2 f[j] + f[j-1]) / dy^2`` with the wall values
    ``f[0] = f[N] = 0`` carried in the array.
    """
    out = np.zeros_like(f)
    out[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / dy ** 2
    return out


def lap_center_neumann(c: np.ndarray, dy: float) -> np.ndarray:
    """3-point Laplacian on centers with Neumann ``d_y(.) = 0`` walls (length ``n``).

    The zero-gradient boundary is imposed with a ghost cell equal to the edge
    value, so the end points reduce to ``(c[1] - c[0]) / dy^2`` and
    ``(c[-2] - c[-1]) / dy^2``.
    """
    out = np.empty_like(c)
    out[1:-1] = (c[2:] - 2 * c[1:-1] + c[:-2]) / dy ** 2
    out[0] = (c[1] - c[0]) / dy ** 2
    out[-1] = (c[-2] - c[-1]) / dy ** 2
    return out


def ddy_center(c: np.ndarray, dy: float) -> np.ndarray:
    """``d/dy`` of a center field, returned on centers (length ``n``).

    Used by the eddy-momentum flux divergence, which is a local (non-flux)
    term living on centers. Interior points are the centered second-order
    difference ``(c[i+1] - c[i-1]) / (2 dy)``; the two end points use a
    first-order one-sided difference. Equivalent to averaging adjacent face
    gradients with the boundary-face gradient held constant, which preserves
    the even -> odd parity exactly on an even-``N`` grid.
    """
    g = np.empty(len(c) + 1)
    g[1:-1] = (c[1:] - c[:-1]) / dy
    g[0] = g[1]
    g[-1] = g[-2]
    return avg_f2c(g)
