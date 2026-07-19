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

Checks the equilibria against the plan's expectations: quiescent plateaus at
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

from ss09.read_output import load_centered
from ss09.sw_model import muscl_mc_du_dy

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
        u = ds["u"].isel(time=sl).mean("time").values
        a = float(ds.attrs["cwv_frac"])
        d_w = float(ds.attrs["d_w"])
        w_crit = float(ds.attrs["w_crit"])
        tau_c = float(ds.attrs["tau_c"])
        evap = float(ds.attrs["evap"])
        v_d = float(ds.attrs["v_d"])
        beta = float(ds.attrs["beta"])
        eps_u = float(ds.attrs["epsilon_u"])
        delta = float(ds.attrs["delta"])
        delta_z = float(ds.attrs["delta_z"])
        height = float(ds.attrs["height"])
        days = int(ds["time"].values[-1])
    finally:
        ds.close()
    # v reconstructed at the centers (last-n-day mean), for the mean MSE flux.
    _, _, v_all, _ = load_centered(path)
    v = v_all[max(0, n - last_n):n].mean(axis=0)
    shat = C_COLUMN * delta * delta_z / height
    dy = y[1] - y[0]
    hhat = shat - L_V * (2.0 * a - 1.0) * w
    # Mean MSE flux and its two large opposing components (energy units).
    dse_flux = shat * v                         # mean DSE flux  Shat v
    lvq_flux = -L_V * (2.0 * a - 1.0) * w * v    # mean moisture flux -L_v(2a-1)Wv
    mean_flux = v * hhat                         # = dse_flux + lvq_flux
    eddy_flux = -L_V * d_w * np.gradient(w, dy)
    # Zonal-momentum balance terms (the eddy-driven overturning: f v ~ EMFD).
    fv = beta * y * v
    emfd = v_d * np.heaviside(u, 0.5) * np.sign(y) * muscl_mc_du_dy(u, dy, y)
    drag = eps_u * u
    return dict(
        y=y, w=w, p=p, u=u, v=v, a=a, d_w=d_w, w_crit=w_crit, tau_c=tau_c, evap=evap,
        delta=delta, delta_z=delta_z, height=height,
        shat=shat, hhat=hhat, dse_flux=dse_flux, lvq_flux=lvq_flux,
        eddy_flux=eddy_flux, mean_flux=mean_flux, fv=fv, emfd=emfd, drag=drag,
        days=days, w_plateau=w_crit + tau_c * evap,
        w_hhat_zero=shat / (L_V * (2.0 * a - 1.0)),
    )


def load_timeseries(path):
    """Return the daily-scalar time series for a convergence figure."""
    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        out = dict(
            t=ds["time"].values,
            ke=ds["steady_state_kinetic_energy"].values,
            tvar=ds["steady_state_temp_variance"].values,
            jet_lat=ds["north_jet_lat"].values / 1e6,
            w_mean=ds["W_mean"].values,
            w_min=ds["W_min"].values,
            gate_day=int(ds["time"].values[-1]),
        )
    finally:
        ds.close()
    return out


def scorecard(runs):
    """Print the plan's expectation checks for each run."""
    print(f"{'':6} {'days':>5} {'W(eq)':>8} {'W_min':>8} {'y_min':>8} "
          f"{'W_q_pred':>9} {'plateau':>8} {'Pmax':>10} {'y_Pmax':>8} "
          f"{'Hhat(0)':>10} {'W>zero%':>8}")
    for lab, r in zip(D_LABELS, runs):
        y_mm = r["y"] / 1e6
        i_eq = int(np.argmin(np.abs(r["y"])))
        i_wmin = int(np.argmin(r["w"]))
        i_pmax = int(np.argmax(r["p"]))
        plateau = 0.5 * (r["w"][0] + r["w"][-1])
        frac_super = float(np.mean(r["w"] > r["w_hhat_zero"])) * 100
        print(f"{lab:6} {r['days']:>5} {r['w'][i_eq]:>8.2f} "
              f"{r['w'][i_wmin]:>8.2f} {y_mm[i_wmin]:>8.2f} "
              f"{r['w_plateau']:>9.3f} {plateau:>8.3f} "
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
    axes[0, 0].axhline(r0["w_plateau"], ls="--", c="gray", lw=1,
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


def make_convergence_figure(ts, gate_day, out_png):
    """Time series showing the run reaching steady state and the slow-drift
    gate firing. The dry metrics are identical across the ladder (W passive),
    so one run (D=1e6) represents them all."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    t = ts["t"]
    axes[0, 0].plot(t, ts["ke"], color="#4477AA")
    axes[0, 0].set_ylabel("KE (m$^2$ s$^{-2}$)")
    axes[0, 0].set_title("Domain kinetic energy")
    axes[0, 1].plot(t, ts["tvar"], color="#4477AA")
    axes[0, 1].set_ylabel(r"std($\theta$) (K)")
    axes[0, 1].set_title("Temperature variance")
    axes[1, 0].plot(t, ts["jet_lat"], color="#EE6677")
    axes[1, 0].set_ylabel("north jet latitude (Mm)")
    axes[1, 0].set_title("Slow jet-position mode (drift gate)")
    axes[1, 1].plot(t, ts["w_mean"], color="#228833", label=r"$\langle W\rangle$")
    axes[1, 1].plot(t, ts["w_min"], color="#CCBB44", label="min $W$")
    axes[1, 1].set_ylabel("W (kg m$^{-2}$)")
    axes[1, 1].set_title("Column moisture spin-up")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.axvline(gate_day, ls="--", c="gray", lw=1)
        ax.grid(alpha=0.3)
    for ax in axes[1, :]:
        ax.set_xlabel("day")
    axes[0, 0].annotate(f"gate fires\nday {gate_day}", xy=(gate_day, axes[0, 0].get_ylim()[1]),
                        xytext=(-90, -14), textcoords="offset points", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def make_dsensitivity_figure(runs, out_png):
    """The D leverage on W (hidden by the overlay), the eddy-flux D-scaling,
    the equilibrium overturning that sets the mean flux, and the MSE-flux
    decomposition into mean (v Hhat) and eddy (-L_v D dW/dy) parts."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = ("#4477AA", "#EE6677", "#228833")
    y = runs[0]["y"] / 1e6
    for lab, r, c in zip(D_LABELS[1:], runs[1:], colors[1:]):
        axes[0, 0].plot(y, r["w"] - runs[0]["w"], color=c, label=f"{lab} - D=0")
    axes[0, 0].set_title("D leverage on W: W(D) - W(D=0)")
    axes[0, 0].set_ylabel(r"$\Delta W$ (kg m$^{-2}$)")
    for lab, r, c in zip(D_LABELS, runs, colors):
        axes[0, 1].plot(y, r["eddy_flux"] / 1e6, color=c, label=lab)
    axes[0, 1].set_title(r"Eddy MSE flux $-L_v D\,\partial_y W$")
    axes[0, 1].set_ylabel("MW m$^{-1}$")
    r1 = runs[1]  # dry circulation is identical across the ladder
    axes[1, 0].plot(y, r1["v"], color="#4477AA")
    axes[1, 0].set_title(r"Equilibrium overturning v (|v|$_{max}$ %.2f m s$^{-1}$, set by $v_d$)"
                         % np.max(np.abs(r1["v"])))
    axes[1, 0].set_ylabel("v (m s$^{-1}$)")
    axes[1, 1].plot(y, r1["mean_flux"] / 1e6, color="#4477AA", label=r"mean MSE $v\hat H$")
    axes[1, 1].plot(y, r1["eddy_flux"] / 1e6, color="#EE6677",
                    label=r"eddy $-L_vD\partial_yW$ (D=1e6)")
    axes[1, 1].set_title("Mean vs eddy MSE flux (both small; see budget fig)")
    axes[1, 1].set_ylabel("MW m$^{-1}$")
    for ax in axes.flat:
        ax.axhline(0, ls=":", c="gray", lw=1)
        ax.set_xlabel("y (Mm)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def make_budget_figure(r, out_png):
    """The eddy-driven momentum balance that sets the overturning, and the
    mean MSE flux resolved into its large opposing DSE and moisture parts.
    One run (D=1e6) represents the ladder; the dry circulation is identical."""
    y = r["y"] / 1e6
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # (a) zonal-momentum balance: f v ~ EMFD, drag negligible
    axes[0].plot(y, r["fv"] * 1e5, color="#4477AA", label=r"$\beta y\,v$ (Coriolis)")
    axes[0].plot(y, r["emfd"] * 1e5, color="#EE6677", ls="--",
                 label=r"EMFD $v_d\mathcal{H}(u)\,\mathrm{sgn}(y)\,\partial_y u$")
    axes[0].plot(y, r["drag"] * 1e5, color="#228833", label=r"$\varepsilon_u u$ (drag)")
    axes[0].set_title(r"Zonal-momentum balance: $\beta y\,v \approx$ EMFD"
                      f"  (|v|$_{{max}}$={np.max(np.abs(r['v'])):.2f} m/s)")
    axes[0].set_ylabel(r"tendency ($\times10^{-5}$ m s$^{-2}$)")
    # (b) mean MSE flux decomposition: DSE poleward, L_v q equatorward, small net
    axes[1].plot(y, r["dse_flux"] / 1e6, color="#CC6677", label=r"DSE $\hat S v$")
    axes[1].plot(y, r["lvq_flux"] / 1e6, color="#4477AA",
                 label=r"$L_v q$ $-L_v(2a{-}1)Wv$")
    axes[1].plot(y, r["mean_flux"] / 1e6, color="k", lw=2, label=r"MSE $\hat H v$ (net)")
    axes[1].plot(y, r["eddy_flux"] / 1e6, color="#999933", ls=":",
                 label=r"eddy $-L_vD\partial_yW$")
    axes[1].set_title("Mean MSE flux = large poleward DSE + larger equatorward $L_vq$")
    axes[1].set_ylabel("northward flux (MW m$^{-1}$)")
    for ax in axes:
        ax.axhline(0, ls=":", c="gray", lw=1)
        ax.set_xlabel("y (Mm)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "model_output/moist_v1_validation"
    out_png = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run_dir, "moist_v1_ladder.png")
    runs = [load_equilibrium(os.path.join(run_dir, d, "out.nc")) for d in D_DIRS]
    scorecard(runs)
    make_figure(runs, out_png)
    base, ext = os.path.splitext(out_png)
    ts = load_timeseries(os.path.join(run_dir, "D1", "out.nc"))
    make_convergence_figure(ts, ts["gate_day"], base + "_convergence" + ext)
    make_dsensitivity_figure(runs, base + "_dsensitivity" + ext)
    make_budget_figure(runs[1], base + "_budget" + ext)


if __name__ == "__main__":
    main()
