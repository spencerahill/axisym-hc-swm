"""Figure: repo gate-on vs gateless(=published code) at ny=801/dt=30/v_d=2.5.

Panel (a): u profiles, mean over a matched late window.
Panel (b): daily time series of domain max u (spinup + any drift).
Panel (c): zoom on the poleward flank region where the gated model's
spurious jet develops.

Usage: fig_gate_pathology.py FIGDIR GATE_ON_NC GATELESS_NC
GATE_ON_NC: repo run with H(u) gate (1500 d). GATELESS_NC: original-code run
(daily records; a matched window is selected from it).
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

M_PER_DEG = np.pi * 6.371e6 / 180.0

FIGDIR = pathlib.Path(sys.argv[1]).resolve()
FIGDIR.mkdir(parents=True, exist_ok=True)
ds_on = xr.open_dataset(sys.argv[2], decode_times=False)
ds_off = xr.open_dataset(sys.argv[3], decode_times=False)

n_on = ds_on.sizes["time"]
window = slice(n_on - 300, n_on)
u_on = ds_on["u"].isel(time=window).mean("time")
u_off = ds_off["u"].isel(time=window).mean("time")
lat_on = ds_on["y"].values / M_PER_DEG
lat_off = ds_off["y"].values / M_PER_DEG

umax_on = ds_on["u"].max("y")
umax_off = ds_off["u"].isel(time=slice(0, n_on)).max("y")

fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
axs[0].plot(lat_off, u_off, color="#08306b", lw=1.5,
            label="gate OFF (= published code)")
axs[0].plot(lat_on, u_on, color="#d95f02", lw=1.2, label="gate ON")
axs[0].set_xlabel("beta-plane latitude [deg]")
axs[0].set_ylabel("u [m/s]")
axs[0].legend(fontsize=8, frameon=False)

axs[1].plot(np.arange(1, len(umax_off) + 1), umax_off, color="#08306b", lw=1.2)
axs[1].plot(np.arange(1, len(umax_on) + 1), umax_on, color="#d95f02", lw=1.2)
axs[1].set_xlabel("day")
axs[1].set_ylabel("domain max u [m/s]")

m_on = np.abs(lat_on) >= 70
m_off = np.abs(lat_off) >= 70
axs[2].plot(lat_off[m_off], u_off.values[m_off], color="#08306b", lw=1.5)
axs[2].plot(lat_on[m_on], u_on.values[m_on], color="#d95f02", lw=1.2)
axs[2].set_xlabel("beta-plane latitude [deg]")
axs[2].set_ylabel("u [m/s]")
axs[2].set_title("poleward flank zoom (|lat| >= 70)", fontsize=9)

for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)

i = int(np.argmax(np.abs(u_on.values)))
print(f"gate ON:  max|u| = {float(np.abs(u_on).max()):.1f} m/s at "
      f"{lat_on[i]:+.1f} deg (day {n_on-300}-{n_on} mean)")
print(f"gate OFF: max|u| = {float(np.abs(u_off).max()):.1f} m/s")
print(f"gate ON  final-day domain max u: {float(umax_on[-1]):.1f}")
print(f"gate OFF matched-day domain max u: {float(umax_off[-1]):.1f}")

fig.suptitle("EMFD H(u) gate ON vs OFF: ny=801, dt=30 s, v_d=2.5, "
             f"days {n_on-300}-{n_on} mean", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIGDIR / "fig4_gate_pathology.png", dpi=150)
print(f"wrote {FIGDIR}/fig4_gate_pathology.png")
