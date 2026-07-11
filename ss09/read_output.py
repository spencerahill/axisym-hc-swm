"""Read model output NetCDF files with v normalized to the u/theta centers.

The staggered grid writes the meridional wind v on the ny-1 interior faces
(the y_face coordinate); the collocated grid writes it on the ny centers (y,
shared with u and theta). ``load_centered`` returns v reconstructed onto the
centers in both cases, so analysis and figure code works off a single grid
regardless of which layout produced the file. The reconstruction is the same
one the model uses for its own center diagnostics (``v_faces_to_centers``), so
loaded center-v matches what u felt during the run.
"""

from typing import Optional, Tuple

import numpy as np
import xarray as xr

from .sw_model import v_faces_to_centers


def _faces_to_centers(v: np.ndarray) -> np.ndarray:
    """Reconstruct v at the centers from a face array shaped (y_face,) or
    (time, y_face)."""
    if v.ndim == 1:
        return v_faces_to_centers(v)
    return np.stack([v_faces_to_centers(row) for row in v])


def load_centered(
    path: str, ndays: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load (y, u, v, T) from a model output file with v on the ny centers.

    Args:
        path: model output NetCDF path (collocated or staggered).
        ndays: if given, time-mean the last ``ndays`` records of each field and
            return them as 1-D (y,) arrays; otherwise return the full
            (time, y) arrays.

    Returns:
        (y, u, v, T). v is on the ny centers: passed through for a collocated
        file, reconstructed from the faces for a staggered one.
    """
    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        y = ds["y"].values
        v_on_faces = "y_face" in ds["v"].dims

        def field(name: str) -> np.ndarray:
            da = ds[name]
            if ndays is not None:
                da = da.isel(time=slice(-ndays, None)).mean("time")
            return da.values

        u = field("u")
        temp = field("T")
        v = field("v")
    finally:
        ds.close()

    if v_on_faces:
        v = _faces_to_centers(v)
    return y, u, v, temp
