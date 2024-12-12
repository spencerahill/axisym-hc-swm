"""
This file shows the python code for S-S model.
"""

import os
import argparse
import logging
import numpy as np
import xarray as xr
from typing import Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
SECONDS_PER_DAY = 86400  # Number of seconds in a day


@dataclass
class SWConfig:
    total_integration_days: int = 365 * 15
    gravity: float = 9.81
    height: float = 16e3
    beta: float = 2e-11
    t_ref: float = 300.0
    output_path: str = "./model_output/output.nc"
    k_v: float = 7786 * 100
    epsilon_u: float = 1e-8
    delta_z: float = 60
    delta_y: float = 50
    delta: float = 4e3
    tau: float = 37 * 24 * 3600
    y_one: float = 9439e3
    y_0: float = 0
    v_d: float = 2.5
    dt: int = 30  # Time step size
    ny: int = 801  # Number of grid points in the y-direction
    domain_size: float = 15751e3 * 2  # Total domain size in the y-direction
    dy: float = None  # Grid spacing in the y-direction
    y: np.ndarray = None  # Y coordinates
    asselin_filt_coef: float = 0.04  # Coefficient for the Asselin filter
    seconds_per_day: int = 86400  # Number of seconds in a day

    def __post_init__(self):
        self.dy = self.domain_size / (self.ny - 1)
        self.y = np.linspace(
            start=-self.domain_size / 2, stop=self.domain_size / 2, num=self.ny
        )


class SWModel:
    def __init__(self, config: SWConfig):
        self.config = config
        self.u, self.v, self.theta, self.time = self.init_vars()
        self.THETA_E = self.calc_theta_e()

    def init_vars(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize the model variables."""
        u = np.zeros([self.config.total_integration_days, self.config.ny])
        v = np.zeros([self.config.total_integration_days, self.config.ny])
        theta = np.zeros([self.config.total_integration_days, self.config.ny])
        time = np.zeros(self.config.total_integration_days)
        return u, v, theta, time

    def calc_theta_e(self) -> np.ndarray:
        """Calculate the radiative-convective equilibrium (RCE) temperature θₑ."""
        theta_00 = 330
        return np.where(
            np.abs(self.config.y - self.config.y_0) < self.config.y_one,
            theta_00
            - self.config.delta_y
            * (
                np.sin(
                    0.5 * np.pi * (self.config.y - self.config.y_0) / self.config.y_one
                )
                ** 2
            ),
            theta_00 - self.config.delta_y,
        )

    def run_sim(self):
        """Run the S-S model simulation."""
        u_now, v_now, theta_now = self.init_current_step_vars()
        u_prev, v_prev, theta_prev = self.init_prev_step_vars(u_now, v_now, theta_now)

        total_time_steps = self.calc_total_time_steps()
        timestamp = 0
        day = 0

        u_temp, v_temp, theta_temp, time_temp = self.init_temp_storage()

        for i in range(total_time_steps):
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

            u_now, v_now, theta_now = self.update_current_step(
                u_after, v_after, theta_after
            )

            self.enforce_boundary_conditions(u_now, v_now)

            timestamp += self.config.dt

            j = (i + 1) % int(SECONDS_PER_DAY / self.config.dt)

            self.store_temp_results(
                u_temp,
                v_temp,
                theta_temp,
                time_temp,
                u_now,
                v_now,
                theta_now,
                timestamp,
                j,
            )

            if (i + 1) % int(SECONDS_PER_DAY / self.config.dt) == 0:
                self.store_daily_avgs(u_temp, v_temp, theta_temp, time_temp, day)
                self.reset_temp_storage(u_temp, v_temp, theta_temp, time_temp)
                day += 1
                logging.info(f"Day {day} finished.")

            if np.isnan(u_now).any():
                logging.warning("NaN detected in u_now, breaking the loop.")
                break

    def init_current_step_vars(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize variables for the current step."""
        u_now = np.zeros(self.config.ny)
        v_now = np.zeros(self.config.ny)
        theta_now = self.THETA_E
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

    def calc_total_time_steps(self) -> int:
        """Calculate the total number of time steps."""
        return int(
            SECONDS_PER_DAY * self.config.total_integration_days / self.config.dt
        )

    def init_temp_storage(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize temporary storage for daily averages."""
        u_temp = np.zeros([int(SECONDS_PER_DAY / self.config.dt), self.config.ny])
        v_temp = np.zeros([int(SECONDS_PER_DAY / self.config.dt), self.config.ny])
        theta_temp = np.zeros([int(SECONDS_PER_DAY / self.config.dt), self.config.ny])
        time_temp = np.zeros(int(SECONDS_PER_DAY / self.config.dt))
        return u_temp, v_temp, theta_temp, time_temp

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
        """Apply the Asselin filter to reduce numerical oscillations."""
        u_prev = u_now + self.config.asselin_filt_coef * (u_after + u_prev - 2 * u_now)
        v_prev = v_now + self.config.asselin_filt_coef * (v_after + v_prev - 2 * v_now)
        theta_prev = theta_now + self.config.asselin_filt_coef * (
            theta_after + theta_prev - 2 * theta_now
        )
        return u_prev, v_prev, theta_prev

    def update_current_step(
        self, u_after: np.ndarray, v_after: np.ndarray, theta_after: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update the current step variables."""
        return u_after, v_after, theta_after

    def enforce_boundary_conditions(self, u_now: np.ndarray, v_now: np.ndarray):
        """Enforce boundary conditions on the current step variables."""
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
        time_temp[j - 1] = timestamp / 86400

    def store_daily_avgs(
        self,
        u_temp: np.ndarray,
        v_temp: np.ndarray,
        theta_temp: np.ndarray,
        time_temp: np.ndarray,
        day: int,
    ):
        """Store daily averages of the simulation results."""
        self.u[day] = np.mean(u_temp, axis=0)
        self.v[day] = np.mean(v_temp, axis=0)
        self.theta[day] = np.mean(theta_temp, axis=0)
        self.time[day] = np.mean(time_temp, axis=0)

    def reset_temp_storage(
        self,
        u_temp: np.ndarray,
        v_temp: np.ndarray,
        theta_temp: np.ndarray,
        time_temp: np.ndarray,
    ):
        """Reset temporary storage for the next day's calculations."""
        u_temp.fill(0)
        v_temp.fill(0)
        theta_temp.fill(0)
        time_temp.fill(0)

    def get_dudt(self, u: np.ndarray, v: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Calculate the time derivative of u."""
        grad_u = np.gradient(u, self.config.dy)
        grad_v = np.gradient(v, self.config.dy)
        grad_u_adv_pos_v = np.zeros_like(u)
        grad_u_adv_pos_v[1:] = (u[1:] - u[:-1]) / self.config.dy
        grad_u_adv_neg_v = np.zeros_like(u)
        grad_u_adv_neg_v[:-1] = (u[1:] - u[:-1]) / self.config.dy
        grad_u_adv = np.where(v > 0, grad_u_adv_pos_v, grad_u_adv_neg_v)
        s = self.config.v_d * np.sign(self.config.y - self.config.y_0) * grad_u
        f = u * self.config.epsilon_u
        vt = u * grad_v * np.heaviside(self.THETA_E - theta, 0.5)
        dudt = v * (self.config.beta * self.config.y - grad_u_adv) - vt - f - s
        return dudt

    def get_dvdt(self, u: np.ndarray, v: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Calculate the time derivative of v."""
        grad_v = np.gradient(v, self.config.dy)
        grad_T = np.gradient(theta / 1.6, self.config.dy)
        diffusion_v = np.gradient(grad_v, self.config.dy) * self.config.k_v
        return (
            -self.config.beta * self.config.y * u
            - self.config.gravity * self.config.height * grad_T / self.config.t_ref
            + diffusion_v
        ) / 2

    def get_dthetadt(
        self, u: np.ndarray, v: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Calculate the time derivative of theta."""
        grad_v = np.gradient(v, self.config.dy)
        return (
            (self.THETA_E - theta) / self.config.tau
            - self.config.delta * self.config.delta_z * grad_v / self.config.height
        )

    def save_results(self):
        """Save the simulation results to a NetCDF file."""
        u_xr = xr.DataArray(
            data=self.u[self.time != 0],
            dims=["time", "y"],
            coords={"time": self.time[self.time != 0], "y": self.config.y},
            attrs=dict(units="m/s"),
        )

        v_xr = xr.DataArray(
            data=self.v[self.time != 0],
            dims=["time", "y"],
            coords={"time": self.time[self.time != 0], "y": self.config.y},
            attrs=dict(units="m/s"),
        )

        temp_xr = xr.DataArray(
            data=self.theta[self.time != 0] / 1.6,
            dims=["time", "y"],
            coords={"time": self.time[self.time != 0], "y": self.config.y},
            attrs=dict(units="K"),
        )

        t_xr = xr.DataArray(
            data=self.time[self.time != 0],
            dims=["time"],
            coords={"time": self.time[self.time != 0]},
            attrs=dict(units="days since 0000-01-01 00:00:00.0", calendar="noleap"),
        )

        thetae_xr = xr.DataArray(
            data=self.THETA_E,
            dims=["y"],
            coords={"y": self.config.y},
            attrs=dict(units="K"),
        )

        ds = u_xr.to_dataset(name="u")
        ds["v"] = v_xr
        ds["T"] = temp_xr
        ds["theta_e"] = thetae_xr
        ds["time"] = t_xr

        ds.attrs["DELTA_Z"] = self.config.delta_z
        ds.attrs["DELTA_Y"] = self.config.delta_y
        ds.attrs["BETA"] = self.config.beta
        ds.attrs["K_V"] = self.config.k_v
        ds.attrs["EPSILON_U"] = self.config.epsilon_u
        ds.attrs["DELTA"] = self.config.delta
        ds.attrs["Y_ONE"] = self.config.y_one
        ds.attrs["Y_0"] = self.config.y_0
        ds.attrs["V_D"] = self.config.v_d

        os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
        ds.to_netcdf(self.config.output_path)
        logging.info(f"Results saved to {self.config.output_path}")


def main():
    args = parse_arguments()
    config = SWConfig(
        total_integration_days=args.total_integration_days,
        gravity=args.gravity,
        height=args.height,
        beta=args.beta,
        t_ref=args.t_ref,
        output_path=args.output_path,
    )
    model = SWModel(config)
    model.run_sim()
    model.save_results()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the S-S model simulation.")
    parser.add_argument(
        "--total_integration_days",
        type=int,
        default=365 * 15,
        help="Total number of integration days (default: 365*15)",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=9.81,
        help="Gravitational acceleration (default: 9.81 m/s^2)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=16e3,
        help="Height of the model (default: 16000 m)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2e-11,
        help="Beta parameter (default: 2e-11)",
    )
    parser.add_argument(
        "--t_ref",
        type=float,
        default=300.0,
        help="Reference temperature (default: 300 K)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./model_output/output.nc",
        help="Path to save the output NetCDF file (default: ./model_output/output.nc)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
