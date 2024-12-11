"""
This file shows the python code for S-S model.
"""

import os

import numpy as np
import xarray as xr

## ----------------------------------------------
# model configuration
DT = 30  # satisfies CFL requirement if wind speed is not insanely large
TOTAL_INTEGRATION_DAYS = 365 * 15  # integrate how many years
SAVEPATH = "./model_output/"  # save path
# ----------------------------------------------

## ----------------------------------------------
# Define constants
BETA = 2e-11
K_V = 7786 * 100
EPSILON_U = 1e-8
DELTA_Z = 60
DELTA_Y = 50
T_REF = 300
DELTA = 4e3
GRAVITY = 9.81
HEIGHT = 16e3
TAU = 37 * 24 * 3600
Y_ONE = 9439e3
# Y_0=666e3
Y_0 = 0
V_D = 2.5
# ----------------------------------------------


# ----------------------------------------------
# define dimensions
NY = 801
DY = 15751e3 * 2 / (NY - 1)
Y = np.linspace(start=-15751e3, stop=15751e3, num=NY)
# ----------------------------------------------

# ----------------------------------------------
# Define variables u, v and θ. (time, y)
u = np.zeros([TOTAL_INTEGRATION_DAYS, NY])
v = np.zeros([TOTAL_INTEGRATION_DAYS, NY])
theta = np.zeros([TOTAL_INTEGRATION_DAYS, NY])
time = np.zeros(TOTAL_INTEGRATION_DAYS)

# Define the radiative-convective equilibrium (RCE) temperature θₑ
theta_00 = 330
# THETA_E=np.where(np.abs(Y)<Y_ONE,theta_00-DELTA_Y*(Y/Y_ONE)**2,theta_00-DELTA_Y)
THETA_E = np.where(
    np.abs(Y - Y_0) < Y_ONE,
    theta_00 - DELTA_Y * (np.sin(0.5 * np.pi * (Y - Y_0) / Y_ONE) ** 2),
    theta_00 - DELTA_Y,
)
# ----------------------------------------------


# ----------------------------------------------
# Define necessary Functions
# Heaviside function: np.heaviside(x1, x2), x2 could be 0.5, i.e., np.heaviside(x, 0.5)


# First PDE about u:
def get_dudt(u, v, theta):
    grad_u = np.gradient(u, DY)
    grad_v = np.gradient(v, DY)
    grad_u_advection_positive_v = np.zeros_like(u)
    grad_u_advection_positive_v[1:] = (u[1:] - u[:-1]) / DY
    grad_u_advection_negative_v = np.zeros_like(u)
    grad_u_advection_negative_v[:-1] = (u[1:] - u[:-1]) / DY
    grad_u_adv = np.where(
        v > 0, grad_u_advection_positive_v, grad_u_advection_negative_v
    )
    # s=V_D*np.heaviside(u, 0.5)*np.sign(Y)*grad_u
    s = V_D * np.sign(Y - Y_0) * grad_u
    # s=0
    f = u * EPSILON_U
    vt = u * grad_v * np.heaviside(THETA_E - theta, 0.5)
    dudt = v * (BETA * Y - grad_u_adv) - vt - f - s
    return dudt


# Second PDE about v:
def get_dvdt(u, v, theta):
    grad_v = np.gradient(v, DY)
    grad_T = np.gradient(theta / 1.6, DY)
    diffusion_v = np.gradient(grad_v, DY) * K_V
    return (-BETA * Y * u - GRAVITY * HEIGHT * grad_T / T_REF + diffusion_v) / 2


# Third PDE about theta:
def get_dthetadt(u, v, theta):
    grad_v = np.gradient(v, DY)
    return (THETA_E - theta) / TAU - DELTA * DELTA_Z * grad_v / HEIGHT


# First step: forward Eular
u_thisstep = np.zeros(NY)
v_thisstep = np.zeros(NY)
theta_thisstep = THETA_E

u_before = u_thisstep - DT * get_dudt(u_thisstep, v_thisstep, theta_thisstep)
v_before = v_thisstep - DT * get_dvdt(u_thisstep, v_thisstep, theta_thisstep)
theta_before = theta_thisstep - DT * get_dthetadt(
    u_thisstep, v_thisstep, theta_thisstep
)


total_time_steps = int(86400 * TOTAL_INTEGRATION_DAYS / DT)
timestamp = 0
day = 0

u_temp = np.zeros([int(86400 / DT), NY])
v_temp = np.zeros([int(86400 / DT), NY])
theta_temp = np.zeros([int(86400 / DT), NY])
time_temp = np.zeros(int(86400 / DT))

for i in range(total_time_steps):
    # Leap frog

    u_after = u_before + 2 * DT * get_dudt(u_thisstep, v_thisstep, theta_thisstep)
    v_after = v_before + 2 * DT * get_dvdt(u_thisstep, v_thisstep, theta_thisstep)
    theta_after = theta_before + 2 * DT * get_dthetadt(
        u_thisstep, v_thisstep, theta_thisstep
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

        print(f"Day {day} finished.")

    if np.isnan(u_thisstep).any():
        break

u_xr = xr.DataArray(
    data=u[time != 0],
    dims=["time", "y"],
    coords={"time": time[time != 0], "y": Y},
    attrs=dict(
        units="m/s",
    ),
)

v_xr = xr.DataArray(
    data=v[time != 0],
    dims=["time", "y"],
    coords={"time": time[time != 0], "y": Y},
    attrs=dict(
        units="m/s",
    ),
)

temp_xr = xr.DataArray(
    data=theta[time != 0] / 1.6,
    dims=["time", "y"],
    coords={"time": time[time != 0], "y": Y},
    attrs=dict(
        units="K",
    ),
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
    attrs=dict(
        units="K",
    ),
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
