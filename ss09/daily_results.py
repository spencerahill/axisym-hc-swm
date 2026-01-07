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

    def to_xarray(
        self, config: SWConfig, theta_e_profile: ThetaEProfile, steady_state_detector=None
    ) -> xr.Dataset:
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

        # Add steady-state convergence history if available
        if steady_state_detector is not None and steady_state_detector.enabled:
            conv_history = steady_state_detector.get_history_dict()
            if conv_history:  # Only add if we have data
                data_vars['steady_state_kinetic_energy'] = xr.DataArray(
                    data=conv_history['kinetic_energy'],
                    dims=['time'],
                    attrs={
                        'units': '(m/s)^2',
                        'long_name': 'domain-averaged kinetic energy for steady-state detection'
                    }
                )
                data_vars['steady_state_temp_variance'] = xr.DataArray(
                    data=conv_history['temp_variance'],
                    dims=['time'],
                    attrs={
                        'units': 'K',
                        'long_name': 'temperature standard deviation for steady-state detection'
                    }
                )

                # Add v field smoothness history if available
                if 'v_smoothness' in conv_history:
                    data_vars['v_neighbor_correlation'] = xr.DataArray(
                        data=conv_history['v_smoothness'],
                        dims=['time'],
                        attrs={
                            'units': 'dimensionless',
                            'long_name': 'v field neighbor correlation (smoothness indicator)',
                            'description': 'Correlation between adjacent grid points. Values > 0.5 indicate smooth field, < 0.5 indicates grid-scale oscillations.'
                        }
                    )
                    data_vars['v_grid_variance'] = xr.DataArray(
                        data=conv_history['v_grid_variance'],
                        dims=['time'],
                        attrs={
                            'units': 'm^-2 s^-2',
                            'long_name': 'v field grid-scale variance',
                            'description': 'Variance of second spatial derivative (d²v/dy²). High values indicate grid-scale noise.'
                        }
                    )

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
