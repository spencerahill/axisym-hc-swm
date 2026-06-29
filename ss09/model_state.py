from typing import NamedTuple
import numpy as np


class ModelState(NamedTuple):
    """Container for the staggered model state at a single timestep.

    On the Arakawa C-grid u and theta live on the N cell centers while v lives
    on the N+1 cell faces, so u/theta and v have different lengths.
    """

    t: float  # time in seconds
    u: np.ndarray  # instantaneous zonal wind (N cell centers)
    v: np.ndarray  # instantaneous meridional wind (N+1 cell faces, v=0 at walls)
    theta: np.ndarray  # instantaneous potential temperature (N cell centers)
    y: np.ndarray  # cell-center meridional coordinate (N,)
