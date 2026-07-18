"""Analyze the Moist V1 D-ladder equilibria.

Loads the gated moist runs at D = {0, 1, 2} x 1e6 m^2/s, extracts the
steady-state column-water-vapor field W(y) and precipitation P(y), and
evaluates the diagnostic MSE-budget quantities the V1 spec calls for
(guides/moist_axisymmetric_model_spec.pdf):

  gross moist stability   Hhat(y) = Shat - L_v (2a-1) W
  diagnostic eddy MSE flux   F_eddy(y) = -L_v D dW/dy
  mean MSE flux              F_mean(y) = v Hhat

with Shat = C d Dz / H the gross dry stability and the spec's pinned
column constants C = 5.2e6 J/m^2/K, L_v = C * Lambda_conv = 2.5e6 J/kg.

Checks the equilibria against the plan's expectations: quiescent collars at
W_c + tau_c E_0, precipitation localized at the ITCZ with W there pinned near
W_c, subtropical W minima, and Hhat < 0 wherever W exceeds Shat/(L_v(2a-1)).

Usage:
    python scripts/moist_v1_analysis.py [run_dir] [out_png]

run_dir defaults to model_output/moist_v1_validation (subdirs D0/D1/D2);
out_png defaults to <run_dir>/moist_v1_ladder.png.
"""

import sys
import os

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Spec-pinned column constants (guides/moist_axisymmetric_model_spec.pdf).
C_COLUMN = 5.2e6  # J m^-2 K^-1, column heat capacity c_p Pi Dp / g
L_V = 2.5e6  # J kg^-1, latent heat; C * Lambda_conv with Lambda_conv = 0.48

D_VALUES = (0.0, 1.0e6, 2.0e6)
D_LABELS = ("D=0", "D=1e6", "D=2e6")
D_DIRS = ("D0", "D1", "D2")


def load_equilibrium(path, last_n=10):
    """Return a dict of the equilibrium (last-n-day mean) fields and the
    config attributes for one run."""
    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        y = ds["y"].values
        n = ds.sizes["time"]
        sl = slice(max(0, n - last_n), n)
        w = ds["W"].isel(time=sl).mean("time").values
        p = ds["P"].isel(time=sl).mean("time").values
        a = float(ds.attrs["cwv_frac"])
        d_w = float(ds.attrs["d_w"])
        w_crit = float(ds.attrs["w_crit"])
        tau_c = float(ds.attrs["tau_c"])
        evap = float(ds.attrs["evap"])
        delta = float(ds.attrs["delta"])
        delta_z = float(ds.attrs["delta_z"])
        height = float(ds.attrs["height"])
        days = int(ds["time"].values[-1])
    finally:
        ds.close()
    shat = C_COLUMN * delta * delta_z / height
    dy = y[1] - y[0]
    hhat = shat - L_V * (2.0 * a - 1.0) * w
    eddy_flux = -L_V * d_w * np.gradient(w, dy)
    return dict(
        y=y, w=w, p=p, a=a, d_w=d_w, w_crit=w_crit, tau_c=tau_c, evap=evap,
        shat=shat, hhat=hhat, eddy_flux=eddy_flux, days=days,
        w_collar=w_crit + tau_c * evap,
        w_hhat_zero=shat / (L_V * (2.0 * a - 1.0)),
    )


def scorecard(runs):
    """Print the plan's expectation checks for each run."""
    print(f"{'':6} {'days':>5} {'W(eq)':>8} {'W_min':>8} {'y_min':>8} "
          f"{'W_collar':>9} {'collar':>8} {'Pmax':>10} {'y_Pmax':>8} "
          f"{'Hhat(0)':>10} {'W>zero%':>8}")
    for lab, r in zip(D_LABELS, runs):
        y_mm = r["y"] / 1e6
        i_eq = int(np.argmin(np.abs(r["y"])))
        i_wmin = int(np.argmin(r["w"]))
        i_pmax = int(np.argmax(r["p"]))
        collar = 0.5 * (r["w"][0] + r["w"][-1])
        frac_super = float(np.mean(r["w"] > r["w_hhat_zero"])) * 100
        print(f"{lab:6} {r['days']:>5} {r['w'][i_eq]:>8.2f} "
              f"{r['w'][i_wmin]:>8.2f} {y_mm[i_wmin]:>8.2f} "
              f"{r['w_collar']:>9.3f} {collar:>8.3f} "
              f"{r['p'][i_pmax]*86400:>10.4f} {y_mm[i_pmax]:>8.2f} "
              f"{r['hhat'][i_eq]:>10.3e} {frac_super:>8.1f}")
    r0 = runs[0]
    print(f"\nConstants: Shat={r0['shat']:.3e} J/m^2, "
          f"L_v={L_V:.2e}, (2a-1)={2*r0['a']-1:.2f}, "
          f"W_c={r0['w_crit']:.1f}, W(Hhat=0)={r0['w_hhat_zero']:.2f} kg/m^2")
    print("P shown as mm/day (P[kg/m^2/s] * 86400).")


def make_figure(runs, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    colors = ("#4477AA", "#EE6677", "#228833")
    for lab, r, c in zip(D_LABELS, runs, colors):
        y = r["y"] / 1e6
        axes[0, 0].plot(y, r["w"], color=c, label=lab)
        axes[0, 1].plot(y, r["p"] * 86400, color=c, label=lab)
        axes[1, 0].plot(y, r["hhat"] / 1e6, color=c, label=lab)
        axes[1, 1].plot(y, r["eddy_flux"] / 1e6, color=c, label=lab)
    r0 = runs[0]
    axes[0, 0].axhline(r0["w_crit"], ls=":", c="gray", lw=1, label="W_c")
    axes[0, 0].axhline(r0["w_collar"], ls="--", c="gray", lw=1,
                       label="W_c+tau_c E_0")
    axes[0, 0].set_ylabel("W (kg m$^{-2}$)")
    axes[0, 0].set_title("Column water vapor")
    axes[0, 1].set_ylabel("P (mm day$^{-1}$)")
    axes[0, 1].set_title("Precipitation")
    axes[1, 0].axhline(0, ls=":", c="gray", lw=1)
    axes[1, 0].set_ylabel(r"$\hat H$ (MJ m$^{-2}$)")
    axes[1, 0].set_title("Gross moist stability")
    axes[1, 1].axhline(0, ls=":", c="gray", lw=1)
    axes[1, 1].set_ylabel(r"$-L_v D\,\partial_y W$ (MW m$^{-1}$)")
    axes[1, 1].set_title("Diagnostic eddy MSE flux")
    for ax in axes[1, :]:
        ax.set_xlabel("y (Mm)")
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"\nWrote {out_png}")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "model_output/moist_v1_validation"
    out_png = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run_dir, "moist_v1_ladder.png")
    runs = [load_equilibrium(os.path.join(run_dir, d, "out.nc")) for d in D_DIRS]
    scorecard(runs)
    make_figure(runs, out_png)


if __name__ == "__main__":
    main()
