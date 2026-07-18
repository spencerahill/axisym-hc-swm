from typing import Optional
import numpy as np
import xarray as xr
from .model_state import ModelState
from .theta_e import ThetaEProfile
from .sw_config import SWConfig


# Specs for the optional diagnostic output variables, used to build the
# (otherwise highly repetitive) xr.DataArray entries in to_xarray.
# Steady-state vars: (output_name, history_dict_key, attrs); all dims ("time",).
_STEADY_STATE_VARS = [
    (
        "steady_state_kinetic_energy",
        "kinetic_energy",
        {
            "units": "(m/s)^2",
            "long_name": "domain-averaged kinetic energy for steady-state detection",
        },
    ),
    (
        "steady_state_temp_variance",
        "temp_variance",
        {
            "units": "K",
            "long_name": "temperature standard deviation for steady-state detection",
        },
    ),
]
_V_SMOOTHNESS_VARS = [
    (
        "v_neighbor_correlation",
        "v_smoothness",
        {
            "units": "dimensionless",
            "long_name": "v field neighbor correlation (smoothness indicator)",
            "description": "Correlation between adjacent grid points. Values > 0.5 indicate smooth field, < 0.5 indicates grid-scale oscillations.",
        },
    ),
    (
        "v_grid_variance",
        "v_grid_variance",
        {
            "units": "m^-2 s^-2",
            "long_name": "v field grid-scale variance",
            "description": "Variance of second spatial derivative (d²v/dy²). High values indicate grid-scale noise.",
        },
    ),
]
# Hadley diagnostics: (name, dims, attrs); name is also the diagnostics-dict key.
_HADLEY_VARS = [
    (
        "rossby_number",
        ["time", "y"],
        {
            "units": "dimensionless",
            "long_name": "local Rossby number",
            "description": "Ratio of relative vorticity (du/dy) to planetary vorticity (beta*y). NaN near equator to avoid singularity.",
        },
    ),
    (
        "north_jet_lat",
        ["time"],
        {
            "units": "m",
            "long_name": "northern hemisphere subtropical jet latitude",
            "description": "Latitude of maximum zonal wind in northern hemisphere (y > 0)",
        },
    ),
    (
        "north_jet_magnitude",
        ["time"],
        {
            "units": "m/s",
            "long_name": "northern hemisphere subtropical jet magnitude",
            "description": "Maximum zonal wind speed in northern hemisphere",
        },
    ),
    (
        "south_jet_lat",
        ["time"],
        {
            "units": "m",
            "long_name": "southern hemisphere subtropical jet latitude",
            "description": "Latitude of maximum zonal wind in southern hemisphere (y < 0)",
        },
    ),
    (
        "south_jet_magnitude",
        ["time"],
        {
            "units": "m/s",
            "long_name": "southern hemisphere subtropical jet magnitude",
            "description": "Maximum zonal wind speed in southern hemisphere",
        },
    ),
    (
        "ascending_edge_lat",
        ["time"],
        {
            "units": "m",
            "long_name": "ascending branch edge latitude",
            "description": "Latitude where v=0 between the two descending edges (ITCZ/ascending branch)",
        },
    ),
    (
        "north_descending_edge_lat",
        ["time"],
        {
            "units": "m",
            "long_name": "northern descending branch edge latitude",
            "description": "Latitude where v=0 closest to northern subtropical jet (descending branch)",
        },
    ),
    (
        "south_descending_edge_lat",
        ["time"],
        {
            "units": "m",
            "long_name": "southern descending branch edge latitude",
            "description": "Latitude where v=0 closest to southern subtropical jet (descending branch)",
        },
    ),
    (
        "north_cell_center_lat",
        ["time"],
        {
            "units": "m",
            "long_name": "northern Hadley cell center latitude",
            "description": "Latitude where poleward meridional wind (v) is maximum in northern hemisphere",
        },
    ),
    (
        "north_cell_strength",
        ["time"],
        {
            "units": "m/s",
            "long_name": "northern Hadley cell strength",
            "description": "Maximum poleward meridional wind (v) in northern hemisphere",
        },
    ),
    (
        "south_cell_center_lat",
        ["time"],
        {
            "units": "m",
            "long_name": "southern Hadley cell center latitude",
            "description": "Latitude where poleward meridional wind (v) is minimum in southern hemisphere",
        },
    ),
    (
        "south_cell_strength",
        ["time"],
        {
            "units": "m/s",
            "long_name": "southern Hadley cell strength",
            "description": "Minimum meridional wind (v) in southern hemisphere (most negative = strongest poleward)",
        },
    ),
    (
        "north_hadley_width",
        ["time"],
        {
            "units": "km",
            "long_name": "northern Hadley cell width",
            "description": "Meridional extent from ascending edge to northern descending edge",
        },
    ),
    (
        "south_hadley_width",
        ["time"],
        {
            "units": "km",
            "long_name": "southern Hadley cell width",
            "description": "Meridional extent from ascending edge to southern descending edge",
        },
    ),
]


class DailyResults:
    """Handler for daily-averaged model output."""

    def __init__(
        self, total_days: int, ny: int, store_theta_e: bool = False,
        nv: Optional[int] = None, store_moisture: bool = False
    ):
        # u and theta live on the ny centers; v may live on a different grid
        # (nv = ny-1 interior faces for the staggered layout). nv defaults to
        # ny, the collocated case.
        if nv is None:
            nv = ny
        self.nv = nv
        self.time = np.zeros(total_days)
        self.u = np.zeros([total_days, ny])
        self.v = np.zeros([total_days, nv])
        self.theta = np.zeros([total_days, ny])
        self.store_theta_e = store_theta_e
        if store_theta_e:
            self.theta_e = np.zeros([total_days, ny])
        # Moist V1: daily-mean W and applied P on the centers, plus the
        # minimum instantaneous W of each day (an undershoot monitor a daily
        # mean would hide).
        self.store_moisture = store_moisture
        if store_moisture:
            self.w = np.zeros([total_days, ny])
            self.precip = np.zeros([total_days, ny])
            self.w_min = np.zeros(total_days)

    def store_day(
        self, day: int, time: float, u: np.ndarray, v: np.ndarray, theta: np.ndarray,
        theta_e: np.ndarray = None, w: Optional[np.ndarray] = None,
        precip: Optional[np.ndarray] = None, w_min: Optional[float] = None
    ):
        """Store results for one day."""
        self.time[day] = time
        self.u[day] = u
        self.v[day] = v
        self.theta[day] = theta
        if self.store_theta_e and theta_e is not None:
            self.theta_e[day] = theta_e
        if self.store_moisture and w is not None:
            self.w[day] = w
            self.precip[day] = precip
            self.w_min[day] = w_min

    def to_xarray(
        self, config: SWConfig, theta_e_profile: ThetaEProfile, steady_state_detector=None,
        hadley_diagnostics=None
    ) -> xr.Dataset:
        """Convert results to xarray Dataset with metadata."""
        mask = self.time != 0
        time_filtered = self.time[mask]

        # v may live on a different grid than u/theta. On the staggered layout
        # it sits on the ny-1 interior faces (y_face); on the collocated layout
        # it shares the ny centers (y), unchanged from earlier output files.
        staggered = getattr(config, "grid", "collocated") == "staggered"
        if staggered:
            v_dims = ["time", "y_face"]
            v_coords = {"time": time_filtered, "y_face": config.y_v}
            v_attrs = {
                "units": "m/s",
                "long_name": "meridional wind",
                "grid": "staggered_face",
            }
        else:
            v_dims = ["time", "y"]
            v_coords = {"time": time_filtered, "y": config.y}
            v_attrs = {"units": "m/s", "long_name": "meridional wind"}

        data_vars = {
            "u": xr.DataArray(
                data=self.u[mask],
                dims=["time", "y"],
                coords={"time": time_filtered, "y": config.y},
                attrs={"units": "m/s", "long_name": "zonal wind"},
            ),
            "v": xr.DataArray(
                data=self.v[mask],
                dims=v_dims,
                coords=v_coords,
                attrs=v_attrs,
            ),
            "T": xr.DataArray(
                data=self.theta[mask] / 1.6,
                dims=["time", "y"],
                coords={"time": time_filtered, "y": config.y},
                attrs={"units": "K", "long_name": "temperature"},
            ),
        }

        # Add theta_e: 2D (time, y) if time-varying, 1D (y) if static
        if self.store_theta_e:
            data_vars["theta_e"] = xr.DataArray(
                data=self.theta_e[mask],
                dims=["time", "y"],
                coords={"time": time_filtered, "y": config.y},
                attrs={"units": "K", "long_name": "equilibrium potential temperature"},
            )
        else:
            # Static theta_e - compute once at t=0
            state = ModelState(
                0.0,
                np.zeros_like(config.y),
                np.zeros_like(config.y),
                np.zeros_like(config.y),
                config.y,
            )
            theta_e = theta_e_profile(state)
            data_vars["theta_e"] = xr.DataArray(
                data=theta_e,
                dims=["y"],
                coords={"y": config.y},
                attrs={"units": "K", "long_name": "equilibrium potential temperature"},
            )

        # Moist V1 output: daily-mean W and applied P, plus the monitoring
        # scalars (FV cell-weighted domain mean, and the day's minimum
        # instantaneous W).
        if self.store_moisture:
            data_vars["W"] = xr.DataArray(
                data=self.w[mask],
                dims=["time", "y"],
                coords={"time": time_filtered, "y": config.y},
                attrs={
                    "units": "kg m-2",
                    "long_name": "column water vapor (daily mean)",
                },
            )
            data_vars["P"] = xr.DataArray(
                data=self.precip[mask],
                dims=["time", "y"],
                coords={"time": time_filtered, "y": config.y},
                attrs={
                    "units": "kg m-2 s-1",
                    "long_name": "precipitation (daily mean of the applied sink)",
                },
            )
            w_masked = self.w[mask]
            ny = w_masked.shape[1]
            w_mean = (
                0.5 * w_masked[:, 0]
                + w_masked[:, 1:-1].sum(axis=1)
                + 0.5 * w_masked[:, -1]
            ) / (ny - 1)
            data_vars["W_mean"] = xr.DataArray(
                data=w_mean,
                dims=["time"],
                attrs={
                    "units": "kg m-2",
                    "long_name": "domain-mean column water vapor",
                    "description": (
                        "FV cell-weighted mean (half cells at the walls), the "
                        "measure under which transport conserves total water"
                    ),
                },
            )
            data_vars["W_min"] = xr.DataArray(
                data=self.w_min[mask],
                dims=["time"],
                attrs={
                    "units": "kg m-2",
                    "long_name": "minimum instantaneous column water vapor",
                    "description": (
                        "Minimum over the day's timesteps and the domain; "
                        "negative values indicate transport undershoot"
                    ),
                },
            )

        # Add steady-state convergence history if available
        if steady_state_detector is not None and steady_state_detector.enabled:
            conv_history = steady_state_detector.get_history_dict()
            if conv_history:  # Only add if we have data
                for name, key, attrs in _STEADY_STATE_VARS:
                    data_vars[name] = xr.DataArray(
                        data=conv_history[key], dims=["time"], attrs=attrs
                    )
                # Add v field smoothness history if available
                if "v_smoothness" in conv_history:
                    for name, key, attrs in _V_SMOOTHNESS_VARS:
                        data_vars[name] = xr.DataArray(
                            data=conv_history[key], dims=["time"], attrs=attrs
                        )

        # Add Hadley cell diagnostics if available
        if hadley_diagnostics is not None:
            hadley_diags = hadley_diagnostics.get_diagnostics_dict()
            if hadley_diags:  # Only add if we have data
                # Note: hadley_diags are already filtered to recorded days, don't apply mask
                for name, dims, attrs in _HADLEY_VARS:
                    data_vars[name] = xr.DataArray(
                        data=hadley_diags[name], dims=dims, attrs=attrs
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

        coords = {"time": time_coord, "y": config.y}
        if staggered:
            coords["y_face"] = xr.DataArray(
                data=config.y_v,
                dims=["y_face"],
                attrs={
                    "units": "m",
                    "long_name": "meridional distance from equator (v faces)",
                    "grid": "staggered_face",
                },
            )

        return xr.Dataset(data_vars=data_vars, coords=coords)
