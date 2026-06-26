from typing import NamedTuple
import numpy as np


class ModelState(NamedTuple):
    """Container for model state at a single timestep."""

    t: float  # time in seconds
    u: np.ndarray  # instantaneous zonal wind
    v: np.ndarray  # instantaneous meridional wind
    theta: np.ndarray  # instantaneous potential temperature
    y: np.ndarray  # meridional distance from equator
