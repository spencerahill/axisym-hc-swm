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
        self, config: SWConfig, theta_e_profile: ThetaEProfile, steady_state_detector=None,
        hadley_diagnostics=None
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

        # Add Hadley cell diagnostics if available
        if hadley_diagnostics is not None:
            hadley_diags = hadley_diagnostics.get_diagnostics_dict()
            if hadley_diags:  # Only add if we have data
                # Note: hadley_diags are already filtered to recorded days, don't apply mask
                # 2D: Rossby number
                data_vars['rossby_number'] = xr.DataArray(
                    data=hadley_diags['rossby_number'],
                    dims=['time', 'y'],
                    attrs={
                        'units': 'dimensionless',
                        'long_name': 'local Rossby number',
                        'description': 'Ratio of relative vorticity (du/dy) to planetary vorticity (beta*y). NaN near equator to avoid singularity.'
                    }
                )

                # 1D: Northern hemisphere jet
                data_vars['north_jet_lat'] = xr.DataArray(
                    data=hadley_diags['north_jet_lat'],
                    dims=['time'],
                    attrs={
                        'units': 'm',
                        'long_name': 'northern hemisphere subtropical jet latitude',
                        'description': 'Latitude of maximum zonal wind in northern hemisphere (y > 0)'
                    }
                )
                data_vars['north_jet_magnitude'] = xr.DataArray(
                    data=hadley_diags['north_jet_magnitude'],
                    dims=['time'],
                    attrs={
                        'units': 'm/s',
                        'long_name': 'northern hemisphere subtropical jet magnitude',
                        'description': 'Maximum zonal wind speed in northern hemisphere'
                    }
                )

                # 1D: Southern hemisphere jet
                data_vars['south_jet_lat'] = xr.DataArray(
                    data=hadley_diags['south_jet_lat'],
                    dims=['time'],
                    attrs={
                        'units': 'm',
                        'long_name': 'southern hemisphere subtropical jet latitude',
                        'description': 'Latitude of maximum zonal wind in southern hemisphere (y < 0)'
                    }
                )
                data_vars['south_jet_magnitude'] = xr.DataArray(
                    data=hadley_diags['south_jet_magnitude'],
                    dims=['time'],
                    attrs={
                        'units': 'm/s',
                        'long_name': 'southern hemisphere subtropical jet magnitude',
                        'description': 'Maximum zonal wind speed in southern hemisphere'
                    }
                )

                # Hadley cell edge latitudes
                data_vars['ascending_edge_lat'] = xr.DataArray(
                    data=hadley_diags['ascending_edge_lat'],
                    dims=['time'],
                    attrs={
                        'units': 'm',
                        'long_name': 'ascending branch edge latitude',
                        'description': 'Latitude where v=0 between the two descending edges (ITCZ/ascending branch)'
                    }
                )
                data_vars['north_descending_edge_lat'] = xr.DataArray(
                    data=hadley_diags['north_descending_edge_lat'],
                    dims=['time'],
                    attrs={
                        'units': 'm',
                        'long_name': 'northern descending branch edge latitude',
                        'description': 'Latitude where v=0 closest to northern subtropical jet (descending branch)'
                    }
                )
                data_vars['south_descending_edge_lat'] = xr.DataArray(
                    data=hadley_diags['south_descending_edge_lat'],
                    dims=['time'],
                    attrs={
                        'units': 'm',
                        'long_name': 'southern descending branch edge latitude',
                        'description': 'Latitude where v=0 closest to southern subtropical jet (descending branch)'
                    }
                )

                # Hadley cell center (v extremum) latitudes and strengths
                data_vars['north_cell_center_lat'] = xr.DataArray(
                    data=hadley_diags['north_cell_center_lat'],
                    dims=['time'],
                    attrs={
                        'units': 'm',
                        'long_name': 'northern Hadley cell center latitude',
                        'description': 'Latitude where poleward meridional wind (v) is maximum in northern hemisphere'
                    }
                )
                data_vars['north_cell_strength'] = xr.DataArray(
                    data=hadley_diags['north_cell_strength'],
                    dims=['time'],
                    attrs={
                        'units': 'm/s',
                        'long_name': 'northern Hadley cell strength',
                        'description': 'Maximum poleward meridional wind (v) in northern hemisphere'
                    }
                )
                data_vars['south_cell_center_lat'] = xr.DataArray(
                    data=hadley_diags['south_cell_center_lat'],
                    dims=['time'],
                    attrs={
                        'units': 'm',
                        'long_name': 'southern Hadley cell center latitude',
                        'description': 'Latitude where poleward meridional wind (v) is minimum in southern hemisphere'
                    }
                )
                data_vars['south_cell_strength'] = xr.DataArray(
                    data=hadley_diags['south_cell_strength'],
                    dims=['time'],
                    attrs={
                        'units': 'm/s',
                        'long_name': 'southern Hadley cell strength',
                        'description': 'Minimum meridional wind (v) in southern hemisphere (most negative = strongest poleward)'
                    }
                )

        # Time coordinate without CF conventions - stored as plain numeric values (days)
        time_coord = xr.DataArray(
            data=time_filtered,
            dims=["time"],
            coords={"time": time_filtered},
            attrs={
                "units": "days",  # Descriptive only - not CF convention
                "long_name": "time",
            },
        )

        return xr.Dataset(
            data_vars=data_vars, coords={"time": time_coord, "y": config.y}
        )
