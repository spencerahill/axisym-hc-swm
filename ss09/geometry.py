"""Geometry arrays for the beta plane and the sphere.

On the sphere the model's meridional coordinate is y = a sin(lat), for
planetary radius a, on a grid uniform in y from pole to pole. y is a
coordinate, not meridional distance (which is a d(lat) = dy / cos(lat)), and
cells of equal width in y have equal area, so the area-weighted budgets keep
their beta-plane discrete form. In this coordinate the spherical operators
are the beta-plane ones with factors of c = cos(lat) = (1 - y^2/a^2)^(1/2):

    relative vorticity     -d(c u)/dy
    divergence              d(c v)/dy
    (1/a) d/d(lat)          c d/dy
    u tan(lat)/a            y u / (a^2 c)

and the Coriolis parameter is exactly f = 2 Omega sin(lat) = (2 Omega / a) y.

Every operator takes its geometry from the arrays built here, once per
configuration. On the beta plane the cosine arrays are exactly 1.0 and f is
beta*y, the same product the beta-plane operators form, so each operator
performs the beta-plane floating-point operations apart from multiplications
by exactly 1.0, which return their operand unchanged: a beta-plane run is
bit-for-bit what it was before the sphere existed. The metric term is skipped
on the beta plane rather than multiplied by zero, since adding +0.0 would turn
a -0.0 into +0.0.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Geometry:
    """Geometry arrays on the ny centers and the ny-1 faces (or, on the
    collocated grid, the ny centers again).

    sphere: False on the beta plane, True on the sphere.
    cos_lat, cos_lat_face: cos(lat) at the centers and faces; exactly 1.0 on
        the beta plane, exactly 0.0 at the two pole centers on the sphere.
    cos2_lat: cos(lat)^2 at the centers.
    inv_cos_lat: 1/cos(lat) at the centers, set to 0.0 at the pole centers,
        whose u tendency the condition u = 0 there discards (a zero keeps the
        filtered leapfrog history at the poles finite).
    f, f_face: the Coriolis parameter, beta*y on the beta plane and
        (2 Omega / a) y on the sphere.
    metric_face: tan(lat)/a = y / (a^2 cos(lat)) at the faces on the sphere;
        zeros on the beta plane, where the metric term is skipped.
    lat, lat_face: latitude in degrees on the sphere; None on the beta plane.
    """

    sphere: bool
    cos_lat: np.ndarray
    cos_lat_face: np.ndarray
    cos2_lat: np.ndarray
    inv_cos_lat: np.ndarray
    f: np.ndarray
    f_face: np.ndarray
    metric_face: np.ndarray
    lat: Optional[np.ndarray] = None
    lat_face: Optional[np.ndarray] = None


def beta_plane_geometry(y: np.ndarray, y_v: np.ndarray, beta: float) -> Geometry:
    """The flat geometry: unit cosines, f = beta*y, no metric term."""
    ones = np.ones_like(y)
    return Geometry(
        sphere=False,
        cos_lat=ones,
        cos_lat_face=np.ones_like(y_v),
        cos2_lat=ones.copy(),
        inv_cos_lat=ones.copy(),
        f=beta * y,
        f_face=beta * y_v,
        metric_face=np.zeros_like(y_v),
    )


def sphere_grid(radius: float, ny: int) -> np.ndarray:
    """Centers y_k = a (k/m) for the integers k = -m..m, m = (ny-1)/2.

    Negating k negates the quotient exactly, so y is exactly antisymmetric
    about the equator (np.linspace(-a, a, ny) is not, at most sizes), which
    the model's exact mirror parity requires; the endpoints are exactly +-a
    and the middle point exactly 0.
    """
    m = (ny - 1) // 2
    return radius * (np.arange(-m, m + 1) / m)


def _cos_from_sin(s: np.ndarray) -> np.ndarray:
    """cos(lat) from sin(lat) as ((1 - s)(1 + s))^(1/2): exactly zero at
    s = +-1 and exactly even in s, since the product commutes."""
    return np.sqrt((1.0 - s) * (1.0 + s))


def sphere_geometry(
    y: np.ndarray, y_v: np.ndarray, radius: float, rot_rate: float
) -> Geometry:
    """The spherical geometry in the coordinate y = a sin(lat)."""
    s = y / radius
    s_v = y_v / radius
    c = _cos_from_sin(s)
    c_v = _cos_from_sin(s_v)
    inv_c = np.zeros_like(c)
    interior = c > 0.0
    inv_c[interior] = 1.0 / c[interior]
    coriolis_coef = 2.0 * rot_rate / radius
    return Geometry(
        sphere=True,
        cos_lat=c,
        cos_lat_face=c_v,
        cos2_lat=c * c,
        inv_cos_lat=inv_c,
        f=coriolis_coef * y,
        f_face=coriolis_coef * y_v,
        metric_face=y_v / (radius**2 * c_v),
        lat=np.rad2deg(np.arcsin(np.clip(s, -1.0, 1.0))),
        lat_face=np.rad2deg(np.arcsin(s_v)),
    )
