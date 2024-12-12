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

# Model constants
BETA = 2e-11
K_V = 7786 * 100
EPSILON_U = 1e-8
DELTA_Z = 60
DELTA_Y = 50
T_REF = 300.0
DELTA = 4e3
GRAVITY = 9.81
HEIGHT = 16e3
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
    u: np.ndarray, v: np.ndarray, theta: np.ndarray, THETA_E: np.ndarray
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
    dudt = v * (BETA * Y - grad_u_adv) - vt - f - s
    return dudt


def get_dvdt(u: np.ndarray, v: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Calculate the time derivative of v."""
    grad_v = np.gradient(v, DY)
    grad_T = np.gradient(theta / 1.6, DY)
    diffusion_v = np.gradient(grad_v, DY) * K_V
    return (-BETA * Y * u - GRAVITY * HEIGHT * grad_T / T_REF + diffusion_v) / 2


def get_dthetadt(
    u: np.ndarray, v: np.ndarray, theta: np.ndarray, THETA_E: np.ndarray
) -> np.ndarray:
    """Calculate the time derivative of theta."""
    grad_v = np.gradient(v, DY)
    return (THETA_E - theta) / TAU - DELTA * DELTA_Z * grad_v / HEIGHT


def run_simulation(
    total_integration_days: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the S-S model simulation."""
    u, v, theta, time = initialize_variables(total_integration_days)
    THETA_E = calculate_theta_e()

    u_thisstep = np.zeros(NY)
    v_thisstep = np.zeros(NY)
    theta_thisstep = THETA_E

    u_before = u_thisstep - DT * get_dudt(
        u_thisstep, v_thisstep, theta_thisstep, THETA_E
    )
    v_before = v_thisstep - DT * get_dvdt(u_thisstep, v_thisstep, theta_thisstep)
    theta_before = theta_thisstep - DT * get_dthetadt(
        u_thisstep, v_thisstep, theta_thisstep, THETA_E
    )

    total_time_steps = int(86400 * total_integration_days / DT)
    timestamp = 0
    day = 0

    u_temp = np.zeros([int(86400 / DT), NY])
    v_temp = np.zeros([int(86400 / DT), NY])
    theta_temp = np.zeros([int(86400 / DT), NY])
    time_temp = np.zeros(int(86400 / DT))

    for i in range(total_time_steps):
        # Leap frog
        u_after = u_before + 2 * DT * get_dudt(
            u_thisstep, v_thisstep, theta_thisstep, THETA_E
        )
        v_after = v_before + 2 * DT * get_dvdt(u_thisstep, v_thisstep, theta_thisstep)
        theta_after = theta_before + 2 * DT * get_dthetadt(
            u_thisstep, v_thisstep, theta_thisstep, THETA_E
        )

        u_before = u_thisstep + 0.04 * (u_after + u_before - 2 * u_thisstep)
        v_before = v_thisstep + 0.04 * (v_after + v_before - 2 * v_thisstep)
        theta_before = theta_thisstep + 0.04 * (
            theta_after + theta_before - 2 * theta_thisstep
        )

        u_thisstep = u_after
        v_thisstep = v_after
        theta_thisstep = theta_after

        u_thisstep[0] = 0
        u_thisstep[-1] = 0
        v_thisstep[0] = 0
        v_thisstep[-1] = 0

        timestamp += DT

        j = (i + 1) % int(86400 / DT)

        u_temp[j - 1] = u_thisstep
        v_temp[j - 1] = v_thisstep
        theta_temp[j - 1] = theta_thisstep
        time_temp[j - 1] = timestamp / 86400

        if (i + 1) % int(86400 / DT) == 0:
            u[day] = np.mean(u_temp, axis=0)
            v[day] = np.mean(v_temp, axis=0)
            theta[day] = np.mean(theta_temp, axis=0)
            time[day] = np.mean(time_temp, axis=0)

            u_temp = np.zeros([int(86400 / DT), NY])
            v_temp = np.zeros([int(86400 / DT), NY])
            theta_temp = np.zeros([int(86400 / DT), NY])
            time_temp = np.zeros(int(86400 / DT))

            day += 1

            logging.info(f"Day {day} finished.")

        if np.isnan(u_thisstep).any():
            logging.warning("NaN detected in u_thisstep, breaking the loop.")
            break

    return u, v, theta, time


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
    ds.attrs["BETA"] = BETA
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


def main(total_integration_days: int):
    """Main function to run the simulation and save results."""
    try:
        u, v, theta, time = run_simulation(total_integration_days)
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
    args = parser.parse_args()
    main(args.total_integration_days)
