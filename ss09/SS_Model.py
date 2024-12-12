"""
This file shows the python code for S-S model.
"""

import os
import argparse
import logging
import numpy as np
import xarray as xr
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
DT = 30  # satisfies CFL requirement if wind speed is not insanely large
SAVEPATH = "./model_output/"  # save path
SECONDS_PER_DAY = 86400  # Number of seconds in a day

# Model constants (default values)
DEFAULT_BETA = 2e-11
K_V = 7786 * 100
EPSILON_U = 1e-8
DELTA_Z = 60
DELTA_Y = 50
DEFAULT_T_REF = 300.0
DELTA = 4e3
DEFAULT_GRAVITY = 9.81
DEFAULT_HEIGHT = 16e3
TAU = 37 * 24 * 3600
Y_ONE = 9439e3
Y_0 = 0
V_D = 2.5

# Dimensions
NY = 801
DY = 15751e3 * 2 / (NY - 1)
Y = np.linspace(start=-15751e3, stop=15751e3, num=NY)


def initialize_variables(
    total_integration_days: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize the model variables."""
    u = np.zeros([total_integration_days, NY])
    v = np.zeros([total_integration_days, NY])
    theta = np.zeros([total_integration_days, NY])
    time = np.zeros(total_integration_days)
    return u, v, theta, time


def calculate_theta_e() -> np.ndarray:
    """Calculate the radiative-convective equilibrium (RCE) temperature θₑ."""
    theta_00 = 330
    return np.where(
        np.abs(Y - Y_0) < Y_ONE,
        theta_00 - DELTA_Y * (np.sin(0.5 * np.pi * (Y - Y_0) / Y_ONE) ** 2),
        theta_00 - DELTA_Y,
    )


def get_dudt(
    u: np.ndarray,
    v: np.ndarray,
    theta: np.ndarray,
    THETA_E: np.ndarray,
    beta: float,
    gravity: float,
) -> np.ndarray:
    """Calculate the time derivative of u."""
    grad_u = np.gradient(u, DY)
    grad_v = np.gradient(v, DY)
    grad_u_advection_positive_v = np.zeros_like(u)
    grad_u_advection_positive_v[1:] = (u[1:] - u[:-1]) / DY
    grad_u_advection_negative_v = np.zeros_like(u)
    grad_u_advection_negative_v[:-1] = (u[1:] - u[:-1]) / DY
    grad_u_adv = np.where(
        v > 0, grad_u_advection_positive_v, grad_u_advection_negative_v
    )
    s = V_D * np.sign(Y - Y_0) * grad_u
    f = u * EPSILON_U
    vt = u * grad_v * np.heaviside(THETA_E - theta, 0.5)
    dudt = v * (beta * Y - grad_u_adv) - vt - f - s
    return dudt


def get_dvdt(
    u: np.ndarray,
    v: np.ndarray,
    theta: np.ndarray,
    beta: float,
    gravity: float,
    height: float,
    t_ref: float,
) -> np.ndarray:
    """Calculate the time derivative of v."""
    grad_v = np.gradient(v, DY)
    grad_T = np.gradient(theta / 1.6, DY)
    diffusion_v = np.gradient(grad_v, DY) * K_V
    return (-beta * Y * u - gravity * height * grad_T / t_ref + diffusion_v) / 2


def get_dthetadt(
    u: np.ndarray, v: np.ndarray, theta: np.ndarray, THETA_E: np.ndarray, height: float
) -> np.ndarray:
    """Calculate the time derivative of theta."""
    grad_v = np.gradient(v, DY)
    return (THETA_E - theta) / TAU - DELTA * DELTA_Z * grad_v / height


def run_simulation(
    total_integration_days: int,
    beta: float,
    gravity: float,
    height: float,
    t_ref: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the S-S model simulation."""
    u, v, theta, time = initialize_variables(total_integration_days)
    THETA_E = calculate_theta_e()

    u_now, v_now, theta_now = initialize_current_step_variables(THETA_E)
    u_before, v_before, theta_before = initialize_previous_step_variables(
        u_now, v_now, theta_now, THETA_E, beta, gravity, height, t_ref
    )

    total_time_steps = calculate_total_time_steps(total_integration_days)
    timestamp = 0
    day = 0

    u_temp, v_temp, theta_temp, time_temp = initialize_temp_storage()

    for i in range(total_time_steps):
        u_after, v_after, theta_after = leapfrog_step(
            u_before,
            v_before,
            theta_before,
            u_now,
            v_now,
            theta_now,
            THETA_E,
            beta,
            gravity,
            height,
            t_ref,
        )

        u_before, v_before, theta_before = apply_asselin_filter(
            u_before,
            v_before,
            theta_before,
            u_after,
            v_after,
            theta_after,
            u_now,
            v_now,
            theta_now,
        )

        u_now, v_now, theta_now = update_current_step(u_after, v_after, theta_after)

        enforce_boundary_conditions(u_now, v_now)

        timestamp += DT

        j = (i + 1) % int(SECONDS_PER_DAY / DT)

        store_temporary_results(
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

        if (i + 1) % int(SECONDS_PER_DAY / DT) == 0:
            store_daily_averages(
                u, v, theta, time, u_temp, v_temp, theta_temp, time_temp, day
            )
            reset_temp_storage(u_temp, v_temp, theta_temp, time_temp)
            day += 1
            logging.info(f"Day {day} finished.")

        if np.isnan(u_now).any():
            logging.warning("NaN detected in u_now, breaking the loop.")
            break

    return u, v, theta, time


def initialize_current_step_variables(
    THETA_E: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize variables for the current step."""
    u_now = np.zeros(NY)
    v_now = np.zeros(NY)
    theta_now = THETA_E
    return u_now, v_now, theta_now


def initialize_previous_step_variables(
    u_now: np.ndarray,
    v_now: np.ndarray,
    theta_now: np.ndarray,
    THETA_E: np.ndarray,
    beta: float,
    gravity: float,
    height: float,
    t_ref: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize variables for the previous step."""
    u_before = u_now - DT * get_dudt(u_now, v_now, theta_now, THETA_E, beta, gravity)
    v_before = v_now - DT * get_dvdt(
        u_now, v_now, theta_now, beta, gravity, height, t_ref
    )
    theta_before = theta_now - DT * get_dthetadt(
        u_now, v_now, theta_now, THETA_E, height
    )
    return u_before, v_before, theta_before


def calculate_total_time_steps(total_integration_days: int) -> int:
    """Calculate the total number of time steps."""
    return int(SECONDS_PER_DAY * total_integration_days / DT)


def initialize_temp_storage() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize temporary storage for daily averages."""
    u_temp = np.zeros([int(SECONDS_PER_DAY / DT), NY])
    v_temp = np.zeros([int(SECONDS_PER_DAY / DT), NY])
    theta_temp = np.zeros([int(SECONDS_PER_DAY / DT), NY])
    time_temp = np.zeros(int(SECONDS_PER_DAY / DT))
    return u_temp, v_temp, theta_temp, time_temp


def leapfrog_step(
    u_before: np.ndarray,
    v_before: np.ndarray,
    theta_before: np.ndarray,
    u_now: np.ndarray,
    v_now: np.ndarray,
    theta_now: np.ndarray,
    THETA_E: np.ndarray,
    beta: float,
    gravity: float,
    height: float,
    t_ref: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform a leapfrog step."""
    u_after = u_before + 2 * DT * get_dudt(
        u_now, v_now, theta_now, THETA_E, beta, gravity
    )
    v_after = v_before + 2 * DT * get_dvdt(
        u_now, v_now, theta_now, beta, gravity, height, t_ref
    )
    theta_after = theta_before + 2 * DT * get_dthetadt(
        u_now, v_now, theta_now, THETA_E, height
    )
    return u_after, v_after, theta_after


def apply_asselin_filter(
    u_before: np.ndarray,
    v_before: np.ndarray,
    theta_before: np.ndarray,
    u_after: np.ndarray,
    v_after: np.ndarray,
    theta_after: np.ndarray,
    u_now: np.ndarray,
    v_now: np.ndarray,
    theta_now: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the Asselin filter to reduce numerical oscillations."""
    u_before = u_now + 0.04 * (u_after + u_before - 2 * u_now)
    v_before = v_now + 0.04 * (v_after + v_before - 2 * v_now)
    theta_before = theta_now + 0.04 * (theta_after + theta_before - 2 * theta_now)
    return u_before, v_before, theta_before


def update_current_step(
    u_after: np.ndarray, v_after: np.ndarray, theta_after: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update the current step variables."""
    return u_after, v_after, theta_after


def enforce_boundary_conditions(u_now: np.ndarray, v_now: np.ndarray):
    """Enforce boundary conditions on the current step variables."""
    u_now[0] = 0
    u_now[-1] = 0
    v_now[0] = 0
    v_now[-1] = 0


def store_temporary_results(
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


def store_daily_averages(
    u: np.ndarray,
    v: np.ndarray,
    theta: np.ndarray,
    time: np.ndarray,
    u_temp: np.ndarray,
    v_temp: np.ndarray,
    theta_temp: np.ndarray,
    time_temp: np.ndarray,
    day: int,
):
    """Store daily averages of the simulation results."""
    u[day] = np.mean(u_temp, axis=0)
    v[day] = np.mean(v_temp, axis=0)
    theta[day] = np.mean(theta_temp, axis=0)
    time[day] = np.mean(time_temp, axis=0)


def reset_temp_storage(
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


def save_results(
    u: np.ndarray,
    v: np.ndarray,
    theta: np.ndarray,
    time: np.ndarray,
    THETA_E: np.ndarray,
):
    """Save the simulation results to a NetCDF file."""
    u_xr = xr.DataArray(
        data=u[time != 0],
        dims=["time", "y"],
        coords={"time": time[time != 0], "y": Y},
        attrs=dict(units="m/s"),
    )

    v_xr = xr.DataArray(
        data=v[time != 0],
        dims=["time", "y"],
        coords={"time": time[time != 0], "y": Y},
        attrs=dict(units="m/s"),
    )

    temp_xr = xr.DataArray(
        data=theta[time != 0] / 1.6,
        dims=["time", "y"],
        coords={"time": time[time != 0], "y": Y},
        attrs=dict(units="K"),
    )

    t_xr = xr.DataArray(
        data=time[time != 0],
        dims=["time"],
        coords={"time": time[time != 0]},
        attrs=dict(units="days since 0000-01-01 00:00:00.0", calendar="noleap"),
    )

    thetae_xr = xr.DataArray(
        data=THETA_E,
        dims=["y"],
        coords={"y": Y},
        attrs=dict(units="K"),
    )

    ds = u_xr.to_dataset(name="u")
    ds["v"] = v_xr
    ds["T"] = temp_xr
    ds["theta_e"] = thetae_xr
    ds["time"] = t_xr

    ds.attrs["DELTA_Z"] = DELTA_Z
    ds.attrs["DELTA_Y"] = DELTA_Y
    ds.attrs["BETA"] = beta
    ds.attrs["K_V"] = K_V
    ds.attrs["EPSILON_U"] = EPSILON_U
    ds.attrs["DELTA"] = DELTA
    ds.attrs["Y_ONE"] = Y_ONE
    ds.attrs["Y_0"] = Y_0
    ds.attrs["V_D"] = V_D

    if not os.path.isdir(SAVEPATH):
        os.mkdir(SAVEPATH)

    ds.to_netcdf(SAVEPATH + "output.nc")
    logging.info(f"Results saved to {SAVEPATH}output.nc")


def main(
    total_integration_days: int,
    gravity: float,
    height: float,
    beta: float,
    t_ref: float,
):
    """Main function to run the simulation and save results."""
    try:
        u, v, theta, time = run_simulation(
            total_integration_days, beta, gravity, height, t_ref
        )
        THETA_E = calculate_theta_e()
        save_results(u, v, theta, time, THETA_E)
    except Exception as e:
        logging.error(f"An error occurred during the simulation: {e}")
        raise

    # Add more detailed logging
    logging.debug(
        "Starting simulation with total_integration_days=%d", total_integration_days
    )


if __name__ == "__main__":
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
        default=DEFAULT_GRAVITY,
        help="Gravitational acceleration (default: 9.81 m/s^2)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=DEFAULT_HEIGHT,
        help="Height of the model (default: 16000 m)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help="Beta parameter (default: 2e-11)",
    )
    parser.add_argument(
        "--t_ref",
        type=float,
        default=DEFAULT_T_REF,
        help="Reference temperature (default: 300 K)",
    )
    args = parser.parse_args()
    main(args.total_integration_days, args.gravity, args.height, args.beta, args.t_ref)
