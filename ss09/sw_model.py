"""
This file shows the python code for S-S model.
"""

import os
import logging
import numpy as np
import xarray as xr
from typing import Tuple
from dataclasses import dataclass, field
from .model_state import ModelState
from .theta_e import ThetaEProfile

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
SECONDS_PER_DAY = 86400  # Number of seconds in a day
THETA_TO_TEMP = 1 / 1.6  # Inverse of (p_s/p_t)^(R/c_p)


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
        self, config: "SWConfig", theta_e_profile: ThetaEProfile
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


@dataclass
class SWConfig:
    """Configuration for the Shallow Water Model."""

    total_integration_days: int = 250
    gravity: float = 9.81
    height: float = 16e3
    beta: float = 2e-11
    t_ref: float = 300.0
    output_path: str = "./model_output/output.nc"
    k_v: float = 7786 * 100
    epsilon_u: float = 1e-8
    delta_z: float = 60
    delta: float = 4e3
    tau: float = 37.0 * SECONDS_PER_DAY
    v_d: float = 2.5
    dt: int = 3600
    ny: int = 51
    domain_size: float = 15751e3 * 2
    dy: float = field(init=False)
    y: np.ndarray = field(init=False)
    asselin_filt_coef: float = 0.04

    def __post_init__(self):
        self.dy = self.domain_size / (self.ny - 1)
        self.y = np.linspace(-self.domain_size / 2, self.domain_size / 2, self.ny)


class SWModel:
    def __init__(self, config: SWConfig, theta_e_profile: ThetaEProfile):
        self.config = config
        self.theta_e_profile = theta_e_profile
        self.results = DailyResults(config.total_integration_days, config.ny)
        self.current_time = 0.0

    def get_state(self, u: np.ndarray, v: np.ndarray, theta: np.ndarray) -> ModelState:
        """Get model state from current instantaneous values."""
        return ModelState(t=self.current_time, u=u, v=v, theta=theta, y=self.config.y)

    def init_current_step_vars(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize variables for the current step."""
        u_now = np.zeros(self.config.ny)
        v_now = np.zeros(self.config.ny)
        theta_now = self.theta_e_profile(
            self.get_state(u_now, v_now, np.zeros(self.config.ny))
        )
        return u_now, v_now, theta_now

    def init_prev_step_vars(
        self, u_now: np.ndarray, v_now: np.ndarray, theta_now: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize variables for the previous step."""
        u_prev = u_now - self.config.dt * self.get_dudt(u_now, v_now, theta_now)
        v_prev = v_now - self.config.dt * self.get_dvdt(u_now, v_now, theta_now)
        theta_prev = theta_now - self.config.dt * self.get_dthetadt(
            u_now, v_now, theta_now
        )
        return u_prev, v_prev, theta_prev

    def init_temp_storage(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize temporary storage for daily averages."""
        steps_per_day = int(SECONDS_PER_DAY / self.config.dt)
        u_temp = np.zeros([steps_per_day, self.config.ny])
        v_temp = np.zeros([steps_per_day, self.config.ny])
        theta_temp = np.zeros([steps_per_day, self.config.ny])
        time_temp = np.zeros(steps_per_day)
        return u_temp, v_temp, theta_temp, time_temp

    def du_dy_upwind(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Calculate the upwind gradient of u with respect to y based on velocity v."""
        grad_u_upwind = np.zeros_like(u)
        # For positive velocity (backward difference)
        mask_pos = v > 0
        grad_u_upwind[1:][mask_pos[1:]] = (
            u[1:][mask_pos[1:]] - u[:-1][mask_pos[1:]]
        ) / self.config.dy
        # For negative velocity (forward difference)
        mask_neg = v < 0
        grad_u_upwind[:-1][mask_neg[:-1]] = (
            u[1:][mask_neg[:-1]] - u[:-1][mask_neg[:-1]]
        ) / self.config.dy
        return grad_u_upwind

    def edd_mom_flux_div(self, grad_u: np.ndarray) -> np.ndarray:
        """Calculate the eddy momentum flux divergence."""
        return self.config.v_d * np.sign(self.config.y) * grad_u

    def rayleigh_drag_u(self, u: np.ndarray) -> np.ndarray:
        """Calculate the Rayleigh drag for u."""
        return u * self.config.epsilon_u

    def vert_mom_advec(
        self, u: np.ndarray, v: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Calculate the vertical momentum advection."""
        state = self.get_state(u, v, theta)
        dv_dy = np.gradient(v, self.config.dy)
        return u * dv_dy * np.heaviside(self.theta_e_profile(state) - theta, 0.5)

    def get_dudt(self, u: np.ndarray, v: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Calculate the time derivative of u."""

        return (
            v * (self.config.beta * self.config.y - self.du_dy_upwind(u, v))
            - self.vert_mom_advec(u, v, theta)
            - self.rayleigh_drag_u(u)
            - self.edd_mom_flux_div(np.gradient(u, self.config.dy))
        )

    def diffusion_v(self, v: np.ndarray) -> np.ndarray:
        """Calculate the diffusion term for v."""
        dv_dy = np.gradient(v, self.config.dy)
        return np.gradient(dv_dy, self.config.dy) * self.config.k_v

    def get_dvdt(self, u: np.ndarray, v: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Calculate the time derivative of v."""
        dtemp_dy = np.gradient(theta * THETA_TO_TEMP, self.config.dy)
        return (
            -self.config.beta * self.config.y * u
            - self.config.gravity * self.config.height * dtemp_dy / self.config.t_ref
            + self.diffusion_v(v)
        ) / 2

    def get_dthetadt(
        self, u: np.ndarray, v: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Calculate the time derivative of theta."""
        state = self.get_state(u, v, theta)
        return (
            self.theta_e_profile(state) - theta
        ) / self.config.tau - self.config.delta * self.config.delta_z * np.gradient(
            v, self.config.dy
        ) / self.config.height

    def leapfrog_step(
        self,
        u_prev: np.ndarray,
        v_prev: np.ndarray,
        theta_prev: np.ndarray,
        u_now: np.ndarray,
        v_now: np.ndarray,
        theta_now: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform a leapfrog step."""
        u_after = u_prev + 2 * self.config.dt * self.get_dudt(u_now, v_now, theta_now)
        v_after = v_prev + 2 * self.config.dt * self.get_dvdt(u_now, v_now, theta_now)
        theta_after = theta_prev + 2 * self.config.dt * self.get_dthetadt(
            u_now, v_now, theta_now
        )
        return u_after, v_after, theta_after

    def apply_asselin_filter(
        self,
        u_prev: np.ndarray,
        v_prev: np.ndarray,
        theta_prev: np.ndarray,
        u_after: np.ndarray,
        v_after: np.ndarray,
        theta_after: np.ndarray,
        u_now: np.ndarray,
        v_now: np.ndarray,
        theta_now: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the Asselin filter."""
        u_prev = u_now + self.config.asselin_filt_coef * (u_after + u_prev - 2 * u_now)
        v_prev = v_now + self.config.asselin_filt_coef * (v_after + v_prev - 2 * v_now)
        theta_prev = theta_now + self.config.asselin_filt_coef * (
            theta_after + theta_prev - 2 * theta_now
        )
        return u_prev, v_prev, theta_prev

    def enforce_boundary_conditions(self, u_now: np.ndarray, v_now: np.ndarray):
        """Enforce boundary conditions."""
        u_now[0] = 0
        u_now[-1] = 0
        v_now[0] = 0
        v_now[-1] = 0

    def store_temp_results(
        self,
        u_temp: np.ndarray,
        v_temp: np.ndarray,
        theta_temp: np.ndarray,
        time_temp: np.ndarray,
        u_now: np.ndarray,
        v_now: np.ndarray,
        theta_now: np.ndarray,
        timestamp: float,
        j: int,
    ):
        """Store temporary results for daily averaging."""
        u_temp[j - 1] = u_now
        v_temp[j - 1] = v_now
        theta_temp[j - 1] = theta_now
        time_temp[j - 1] = timestamp / SECONDS_PER_DAY

    def store_daily_avgs(
        self,
        u_temp: np.ndarray,
        v_temp: np.ndarray,
        theta_temp: np.ndarray,
        time_temp: np.ndarray,
        day: int,
    ):
        """Store daily averages."""
        self.results.store_day(
            day,
            np.mean(time_temp),
            np.mean(u_temp, axis=0),
            np.mean(v_temp, axis=0),
            np.mean(theta_temp, axis=0),
        )

    def reset_temp_storage(
        self,
        u_temp: np.ndarray,
        v_temp: np.ndarray,
        theta_temp: np.ndarray,
        time_temp: np.ndarray,
    ):
        """Reset temporary storage arrays."""
        u_temp.fill(0)
        v_temp.fill(0)
        theta_temp.fill(0)
        time_temp.fill(0)

    def run_sim(self):
        """Run the S-S model simulation."""
        u_now, v_now, theta_now = self.init_current_step_vars()
        u_prev, v_prev, theta_prev = self.init_prev_step_vars(u_now, v_now, theta_now)

        total_time_steps = int(
            SECONDS_PER_DAY * self.config.total_integration_days / self.config.dt
        )
        day = 0

        u_temp, v_temp, theta_temp, time_temp = self.init_temp_storage()

        for i in range(total_time_steps):
            self.current_time = i * self.config.dt

            u_after, v_after, theta_after = self.leapfrog_step(
                u_prev, v_prev, theta_prev, u_now, v_now, theta_now
            )

            u_prev, v_prev, theta_prev = self.apply_asselin_filter(
                u_prev,
                v_prev,
                theta_prev,
                u_after,
                v_after,
                theta_after,
                u_now,
                v_now,
                theta_now,
            )

            u_now, v_now, theta_now = u_after, v_after, theta_after

            self.enforce_boundary_conditions(u_now, v_now)

            ind_within_day = (i + 1) % int(SECONDS_PER_DAY / self.config.dt)

            self.store_temp_results(
                u_temp,
                v_temp,
                theta_temp,
                time_temp,
                u_now,
                v_now,
                theta_now,
                self.current_time,
                ind_within_day,
            )

            if (i + 1) % int(SECONDS_PER_DAY / self.config.dt) == 0:
                self.store_daily_avgs(u_temp, v_temp, theta_temp, time_temp, day)
                self.reset_temp_storage(u_temp, v_temp, theta_temp, time_temp)
                day += 1
                logging.info(f"Day {day} finished.")

            if np.isnan(u_now).any():
                logging.warning("NaN detected in u_now, breaking the loop.")
                break

    def save_results(self):
        """Save the simulation results to a NetCDF file."""
        ds = self.results.to_xarray(self.config, self.theta_e_profile)

        # Add coordinate attributes
        ds.y.attrs.update(
            {"units": "m", "long_name": "meridional distance from equator"}
        )

        # Add global attributes
        global_attrs = {
            "title": "Shallow Water Model Output",
            "creation_date": str(np.datetime64("now")),
            **{
                key: getattr(self.config, key)
                for key in self.config.__dataclass_fields__
            },
        }
        ds.attrs.update(global_attrs)

        # Save to file
        os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
        try:
            ds.to_netcdf(self.config.output_path)
            logging.info(f"Results successfully saved to {self.config.output_path}")
        except Exception as e:
            logging.error(f"Failed to save results: {str(e)}")
            raise
