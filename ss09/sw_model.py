"""
This file shows the python code for S-S model.
"""

import os
import logging
import numpy as np
from typing import Tuple, NamedTuple
from .model_state import ModelState
from .theta_e import ThetaEProfile
from .sw_config import SWConfig
from .daily_results import DailyResults

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
SECONDS_PER_DAY = 86400  # Number of seconds in a day
THETA_TO_TEMP = 1 / 1.6  # Inverse of (p_s/p_t)^(R/c_p)


class AuxiliaryVars(NamedTuple):
    u: np.ndarray
    v: np.ndarray
    theta: np.ndarray


class TempVars(NamedTuple):
    u: np.ndarray
    v: np.ndarray
    theta: np.ndarray
    time: np.ndarray


class SWModel:
    def __init__(self, config: SWConfig, theta_e_profile: ThetaEProfile):
        self.config = config
        self.theta_e_profile = theta_e_profile
        self.state = ModelState(
            t=0.0,
            u=np.zeros(config.ny),
            v=np.zeros(config.ny),
            theta=np.zeros(config.ny),
            y=config.y,
        )
        self.state = self.state._replace(theta=self.theta_e_profile(self.state))
        self.results = DailyResults(config.total_integration_days, config.ny)
        self.temp_vars = TempVars(
            u=np.zeros((0, config.ny)),
            v=np.zeros((0, config.ny)),
            theta=np.zeros((0, config.ny)),
            time=np.zeros(0),
        )
        self.vars_prev_step = AuxiliaryVars(
            u=np.zeros(config.ny), v=np.zeros(config.ny), theta=np.zeros(config.ny)
        )
        self.vars_next_step = AuxiliaryVars(
            u=np.zeros(config.ny), v=np.zeros(config.ny), theta=np.zeros(config.ny)
        )

    def init_prev_step_vars(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize variables for the previous step."""
        u_prev = self.state.u - self.config.dt * self.du_dt()
        v_prev = self.state.v - self.config.dt * self.dv_dt()
        theta_prev = self.state.theta - self.config.dt * self.dtheta_dt()
        return u_prev, v_prev, theta_prev

    def init_temp_storage(self):
        """Initialize temporary storage for daily averages."""
        steps_per_day = int(SECONDS_PER_DAY / self.config.dt)
        self.temp_vars = TempVars(
            u=np.zeros([steps_per_day, self.config.ny]),
            v=np.zeros([steps_per_day, self.config.ny]),
            theta=np.zeros([steps_per_day, self.config.ny]),
            time=np.zeros(steps_per_day),
        )

    def du_dy_upwind(self) -> np.ndarray:
        """Calculate the upwind gradient of u with respect to y based on velocity v."""
        grad_u_upwind = np.zeros_like(self.state.u)
        # For positive velocity (backward difference)
        mask_pos = self.state.v > 0
        grad_u_upwind[1:][mask_pos[1:]] = (
            self.state.u[1:][mask_pos[1:]] - self.state.u[:-1][mask_pos[1:]]
        ) / self.config.dy
        # For negative velocity (forward difference)
        mask_neg = self.state.v < 0
        grad_u_upwind[:-1][mask_neg[:-1]] = (
            self.state.u[1:][mask_neg[:-1]] - self.state.u[:-1][mask_neg[:-1]]
        ) / self.config.dy
        return grad_u_upwind

    def edd_mom_flux_div_u(self) -> np.ndarray:
        """Calculate the eddy momentum flux divergence."""
        return (
            self.config.v_d
            * np.sign(self.config.y)
            * np.gradient(self.state.u, self.config.dy)
        )

    def rayleigh_drag_u(self) -> np.ndarray:
        """Calculate the Rayleigh drag for u."""
        return self.state.u * self.config.epsilon_u

    def vert_advec_u(self) -> np.ndarray:
        """Calculate the vertical momentum advection."""
        dv_dy = np.gradient(self.state.v, self.config.dy)
        return (
            self.state.u
            * dv_dy
            * np.heaviside(self.theta_e_profile(self.state) - self.state.theta, 0.5)
        )

    def coriolis_term(self, u_or_v: np.ndarray) -> np.ndarray:
        """Calculate the Coriolis term for the u equation."""
        return self.config.beta * self.config.y * u_or_v

    def merid_advec_u(self) -> np.ndarray:
        """Calculate the meridional advection term for u."""
        return self.state.v * self.du_dy_upwind()

    def du_dt(self) -> np.ndarray:
        """Calculate the time derivative of u."""
        vert_advec_u_term = (
            self.vert_advec_u() if self.config.include_vert_advec_u else 0
        )
        return (
            self.coriolis_term(self.state.v)
            - self.merid_advec_u()
            - vert_advec_u_term
            - self.rayleigh_drag_u()
            - self.edd_mom_flux_div_u()
        )

    def dv_dy(self) -> np.ndarray:
        """Calculate the gradient of v with respect to y."""
        return np.gradient(self.state.v, self.config.dy)

    def diffusion_v(self) -> np.ndarray:
        """Calculate the diffusion term for v."""
        return np.gradient(self.dv_dy(), self.config.dy) * self.config.k_v

    def dp_dy_term(self) -> np.ndarray:
        """Calculate the pressure gradient term."""
        dtemp_dy = np.gradient(self.state.theta * THETA_TO_TEMP, self.config.dy)
        return self.config.gravity * self.config.height * dtemp_dy / self.config.t_ref

    def dv_dt(self) -> np.ndarray:
        """Calculate the time derivative of v."""
        return (
            -self.coriolis_term(self.state.u) - self.dp_dy_term() + self.diffusion_v()
        ) / 2

    def newt_cool_term(self) -> np.ndarray:
        """Newtonian cooling term."""
        return (self.theta_e_profile(self.state) - self.state.theta) / self.config.tau

    def vert_advec_theta(self) -> np.ndarray:
        """Calculate the vertical advection term for theta."""
        return (
            -self.config.delta
            * self.config.delta_z
            * np.gradient(self.state.v, self.config.dy)
            / self.config.height
        )

    def eddy_heat_flux(self) -> np.ndarray:
        """Calculate the eddy heat flux divergence."""
        if self.config.coeff_eddy_heat_diff == 0.0:
            return np.zeros_like(self.state.theta)
        dtheta_dy = np.gradient(self.state.theta, self.config.dy)
        return self.config.coeff_eddy_heat_diff * np.gradient(dtheta_dy, self.config.dy)

    def dtheta_dt(self) -> np.ndarray:
        """Calculate the time derivative of theta."""
        return self.newt_cool_term() + self.vert_advec_theta() + self.eddy_heat_flux()

    def leapfrog_step(self, prev: np.ndarray, time_deriv_func) -> np.ndarray:
        """Perform a leapfrog step for a single variable."""
        return prev + 2 * self.config.dt * time_deriv_func()

    def asselin_filt(
        self, prev: np.ndarray, after: np.ndarray, now: np.ndarray
    ) -> np.ndarray:
        """Apply the Asselin filter to a single variable."""
        return now + self.config.asselin_filt_coef * (after + prev - 2 * now)

    def enforce_boundary_conditions(self):
        """Enforce boundary conditions."""
        self.state.u[0] = 0
        self.state.u[-1] = 0
        self.state.v[0] = 0
        self.state.v[-1] = 0

    def store_temp_results(self, timestamp: float, j: int):
        """Store temporary results for daily averaging."""
        self.temp_vars.u[j - 1] = self.state.u
        self.temp_vars.v[j - 1] = self.state.v
        self.temp_vars.theta[j - 1] = self.state.theta
        self.temp_vars.time[j - 1] = timestamp / SECONDS_PER_DAY

    def store_daily_avgs(self, day: int):
        """Store daily averages."""
        self.results.store_day(
            day,
            np.mean(self.temp_vars.time),
            np.mean(self.temp_vars.u, axis=0),
            np.mean(self.temp_vars.v, axis=0),
            np.mean(self.temp_vars.theta, axis=0),
        )

    def reset_temp_storage(self):
        """Reset temporary storage arrays."""
        self.temp_vars = TempVars(
            u=np.zeros_like(self.temp_vars.u),
            v=np.zeros_like(self.temp_vars.v),
            theta=np.zeros_like(self.temp_vars.theta),
            time=np.zeros_like(self.temp_vars.time),
        )

    def calc_ind_within_day(self, current_step: int) -> int:
        """Calculate the index within the day."""
        return (current_step + 1) % int(SECONDS_PER_DAY / self.config.dt)

    def run_sim(self):
        """Run the S-S model simulation."""
        self.vars_prev_step = AuxiliaryVars(*self.init_prev_step_vars())

        total_time_steps = int(
            SECONDS_PER_DAY * self.config.total_integration_days / self.config.dt
        )
        day = 0

        self.init_temp_storage()

        for i in range(total_time_steps):
            self.state = self.state._replace(t=i * self.config.dt)

            self.vars_next_step = self.vars_next_step._replace(
                u=self.leapfrog_step(self.vars_prev_step.u, self.du_dt),
                v=self.leapfrog_step(self.vars_prev_step.v, self.dv_dt),
                theta=self.leapfrog_step(self.vars_prev_step.theta, self.dtheta_dt),
            )

            self.vars_prev_step = self.vars_prev_step._replace(
                u=self.asselin_filt(
                    self.vars_prev_step.u, self.vars_next_step.u, self.state.u
                ),
                v=self.asselin_filt(
                    self.vars_prev_step.v, self.vars_next_step.v, self.state.v
                ),
                theta=self.asselin_filt(
                    self.vars_prev_step.theta,
                    self.vars_next_step.theta,
                    self.state.theta,
                ),
            )

            self.state = self.state._replace(
                u=self.vars_next_step.u,
                v=self.vars_next_step.v,
                theta=self.vars_next_step.theta,
            )

            self.enforce_boundary_conditions()

            ind_within_day = self.calc_ind_within_day(i)
            self.store_temp_results(self.state.t, ind_within_day)
            if ind_within_day == 0:
                self.store_daily_avgs(day)
                self.reset_temp_storage()
                day += 1
                logging.info(f"Day {day} finished.")

            if np.isnan(self.state.u).any():
                logging.warning("NaN detected in u, breaking the loop.")
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
                key: (
                    str(getattr(self.config, key))
                    if isinstance(getattr(self.config, key), bool)
                    else getattr(self.config, key)
                )
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
