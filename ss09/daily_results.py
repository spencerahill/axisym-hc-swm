import numpy as np
import xarray as xr
from .model_state import ModelState
from .theta_e import ThetaEProfile
from .sw_config import SWConfig


class DailyResults:
    """Handler for daily-averaged model output."""

    def __init__(self, total_days: int, ny: int):
        self.time = np.zeros(total_days)
        self.u = np.zeros([total_days, ny])
        self.v = np.zeros([total_days, ny])
        self.theta = np.zeros([total_days, ny])

    def store_day(
        self, day: int, time: float, u: np.ndarray, v: np.ndarray, theta: np.ndarray
    ):
        """Store results for one day."""
        self.time[day] = time
        self.u[day] = u
        self.v[day] = v
        self.theta[day] = theta

    def to_xarray(self, config: SWConfig, theta_e_profile: ThetaEProfile) -> xr.Dataset:
        """Convert results to xarray Dataset with metadata."""
        mask = self.time != 0
        time_filtered = self.time[mask]

        # Calculate theta_e for the output
        state = ModelState(
            0.0,
            np.zeros_like(config.y),
            np.zeros_like(config.y),
            np.zeros_like(config.y),
            config.y,
        )
        theta_e = theta_e_profile(state)

        data_vars = {
            "u": xr.DataArray(
                data=self.u[mask],
                dims=["time", "y"],
                coords={"time": time_filtered, "y": config.y},
                attrs={"units": "m/s", "long_name": "zonal wind"},
            ),
            "v": xr.DataArray(
                data=self.v[mask],
                dims=["time", "y"],
                coords={"time": time_filtered, "y": config.y},
                attrs={"units": "m/s", "long_name": "meridional wind"},
            ),
            "T": xr.DataArray(
                data=self.theta[mask] / 1.6,
                dims=["time", "y"],
                coords={"time": time_filtered, "y": config.y},
                attrs={"units": "K", "long_name": "temperature"},
            ),
            "theta_e": xr.DataArray(
                data=theta_e,
                dims=["y"],
                coords={"y": config.y},
                attrs={"units": "K", "long_name": "equilibrium potential temperature"},
            ),
        }

        time_coord = xr.DataArray(
            data=time_filtered,
            dims=["time"],
            coords={"time": time_filtered},
            attrs={
                "units": "days since 0000-01-01 00:00:00.0",
                "calendar": "noleap",
                "long_name": "time",
            },
        )

        return xr.Dataset(
            data_vars=data_vars, coords={"time": time_coord, "y": config.y}
        )
