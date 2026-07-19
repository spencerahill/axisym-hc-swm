"""Analyze the Moist V1 tau_c and Delta_z sensitivity scans.

Companion to moist_v1_analysis.py (D-ladder) and moist_v1_wc_scan_analysis.py
(W_c scan). Two ladders, both reusing the D-ladder D1 run as their center
point (tau_c = 14400 s, Delta_z = 60 K, W_c = 50, D = 1e6):

tau_c ladder (tc1800/tc7200/tc57600/tc172800). Moist-only: W is passive and
tau_c enters nothing dry, so the dry fields must be bit-identical to D1 (the
script checks every daily u field). tau_c sets the quiescent collar
W = W_c + tau_c E_0, the linear deepening of Hhat = Shat - L_v(2a-1)W with
tau_c, and the amplitude of the transport-driven W departures from the
collar (relaxation time tau_c times the local moisture-flux convergence).

Delta_z ladder (dz45/dz68p2/dz75/dz90). Delta_z sits in the dry theta
equation's vertical-advection term (vert_advec_theta), so these runs change
the dry circulation itself as well as the diagnosed gross stabilities:
Shat = C delta Delta_z / H, hence the crossover W* = Shat/(L_v(2a-1)) moves
proportionally to Delta_z, and the sign of Hhat at the (nearly tau_c-set) W
level flips at Delta_z* = H L_v(2a-1)(W_c + tau_c E_0)/(C delta) ~ 68.2 K.
The dz68p2 rung sits at that predicted crossover.

Usage:
    python scripts/moist_v1_tauc_dz_analysis.py [scan_dir] [d1_path]

scan_dir defaults to model_output/moist_v1_tauc_dz (subdirs tc*/dz*);
d1_path defaults to model_output/moist_v1_validation/D1/out.nc.
Figures are written into scan_dir.
"""

import sys
import os

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moist_v1_analysis import load_equilibrium, C_COLUMN, L_V

TC_VALUES = (1800.0, 7200.0, 14400.0, 57600.0, 172800.0)
TC_DIRS = ("tc1800", "tc7200", None, "tc57600", "tc172800")  # None -> D1
DZ_VALUES = (45.0, 60.0, 68.2, 75.0, 90.0)
DZ_DIRS = ("dz45", None, "dz68p2", "dz75", "dz90")

# Flux-decomposition colors (validated categorical triple + black net).
C_DSE, C_LVQ, C_EDDY = "#CC6677", "#0077BB", "#999933"


def run_paths(scan_dir, d1_path, dirs):
    return [d1_path if d is None else os.path.join(scan_dir, d, "out.nc")
            for d in dirs]


def load_extras(path):
    """Daily dry u (for the bit-invariance check) and the trailing W_mean
    drift (the slow-drift gate watches only dry metrics, so W convergence
    needs its own check)."""
    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        u_daily = ds["u"].values
        w_mean = ds["W_mean"].values
    finally:
        ds.close()
    tail = w_mean[-100:]
    return dict(u_daily=u_daily, w_drift=float(tail.max() - tail.min()))


def i_star_of(r):
    """Index of the northern max-|v| latitude of this run's circulation."""
    north = r["y"] > 0
    return int(np.flatnonzero(north)[np.argmax(np.abs(r["v"][north]))])


def dz_star_of_w(r, w):
    """The Delta_z at which Hhat = 0 for a given W (inverts Shat(Delta_z))."""
    ds_dz = C_COLUMN * r["delta"] / r["height"]  # dShat/dDelta_z
    return L_V * (2.0 * r["a"] - 1.0) * np.asarray(w) / ds_dz


def scorecard_tc(runs, extras, u_ref):
    print("=== tau_c ladder (Delta_z=60; dry fields must be bit-identical "
          "to D1) ===")
    print(f"{'tau_c':>7} {'days':>5} {'collar':>8} {'pred':>8} {'|c-p|':>8} "
          f"{'dW+':>7} {'dW-':>7} {'Pmax':>7} {'Pmin':>7} {'Hhat(0)':>9} "
          f"{'Hh<0%':>6} {'Wdrift':>8} {'max|du|':>8} {'max|dP|':>8}")
    p_ref = runs[2]["p"]
    for r, ex in zip(runs, extras):
        collar = 0.5 * (r["w"][0] + r["w"][-1])
        i_eq = int(np.argmin(np.abs(r["y"])))
        du = (np.abs(ex["u_daily"] - u_ref).max()
              if ex["u_daily"].shape == u_ref.shape else np.nan)
        dp = np.abs(r["p"] - p_ref).max() * 86400
        print(f"{r['tau_c']:>7.0f} {r['days']:>5} {collar:>8.3f} "
              f"{r['w_collar']:>8.3f} {abs(collar - r['w_collar']):>8.1e} "
              f"{r['w'].max() - collar:>7.3f} {collar - r['w'].min():>7.3f} "
              f"{r['p'].max() * 86400:>7.3f} {r['p'].min() * 86400:>7.3f} "
              f"{r['hhat'][i_eq] / 1e6:>9.2f} "
              f"{np.mean(r['hhat'] < 0) * 100:>6.1f} {ex['w_drift']:>8.1e} "
              f"{du:>8.1e} {dp:>8.4f}")
    print("collar/pred/dW in kg/m^2, P in mm/day, Hhat in MJ/m^2; Wdrift = "
          "range of W_mean over the trailing 100 d; max|du| vs D1 over all "
          "daily fields (0.0 = bit-identical dry circulation); max|dP| vs "
          "D1 equilibrium P in mm/day.")


def scorecard_dz(runs, extras):
    print("\n=== Delta_z ladder (tau_c=14400; dry circulation changes with "
          "Delta_z) ===")
    print(f"{'dz':>6} {'days':>5} {'Shat':>9} {'W*':>7} {'u_jet':>7} "
          f"{'max|v|':>7} {'y*':>6} {'W(0)':>7} {'Hhat(0)':>9} {'Hh<0%':>6} "
          f"{'DSE*':>8} {'Lvq*':>8} {'net*':>8} {'eddy*':>8} {'Wdrift':>8}")
    for r, ex in zip(runs, extras):
        i_eq = int(np.argmin(np.abs(r["y"])))
        i_s = i_star_of(r)
        print(f"{r['delta_z']:>6.1f} {r['days']:>5} {r['shat']:>9.3e} "
              f"{r['w_hhat_zero']:>7.2f} {r['u'].max():>7.2f} "
              f"{np.abs(r['v']).max():>7.2f} {r['y'][i_s] / 1e6:>6.2f} "
              f"{r['w'][i_eq]:>7.2f} {r['hhat'][i_eq] / 1e6:>9.2f} "
              f"{np.mean(r['hhat'] < 0) * 100:>6.1f} "
              f"{r['dse_flux'][i_s] / 1e6:>8.2f} "
              f"{r['lvq_flux'][i_s] / 1e6:>8.2f} "
              f"{r['mean_flux'][i_s] / 1e6:>8.2f} "
              f"{r['eddy_flux'][i_s] / 1e6:>8.2f} {ex['w_drift']:>8.1e}")
    r0 = runs[1]
    print(f"Predicted crossover Delta_z* = "
          f"{dz_star_of_w(r0, r0['w_collar']):.2f} K (collar W = "
          f"{r0['w_collar']:.3f}); Shat in J/m^2, W in kg/m^2, u/v in m/s, "
          f"Hhat in MJ/m^2, fluxes (at each run's own y*) in MW/m.")


def make_tc_ladder_figure(runs, colors, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for r, c in zip(runs, colors):
        y = r["y"] / 1e6
        lab = f"$\\tau_c$={r['tau_c']:.0f} s"
        collar = 0.5 * (r["w"][0] + r["w"][-1])
        axes[0, 0].plot(y, r["w"], color=c, label=lab)
        axes[0, 1].plot(y, r["p"] * 86400, color=c, label=lab)
        axes[1, 0].plot(y, r["hhat"] / 1e6, color=c, label=lab)
        axes[1, 1].plot(y, r["w"] - collar, color=c, label=lab)
    axes[0, 0].set_ylabel("W (kg m$^{-2}$)")
    axes[0, 0].set_title(r"W: collar rises as $W_c + \tau_c E_0$")
    axes[0, 1].set_ylabel("P (mm day$^{-1}$)")
    axes[0, 1].set_title(r"P: equilibrium profile nearly $\tau_c$-independent")
    axes[1, 0].axhline(0, ls=":", c="gray", lw=1)
    axes[1, 0].set_ylabel(r"$\hat H$ (MJ m$^{-2}$)")
    axes[1, 0].set_title(r"$\hat H$ deepens linearly with $\tau_c$ "
                         "(all moisture-mode at $W_c$=50)")
    axes[1, 1].axhline(0, ls=":", c="gray", lw=1)
    axes[1, 1].set_ylabel(r"$W -$ collar (kg m$^{-2}$)")
    axes[1, 1].set_title(r"Transport departures grow with $\tau_c$")
    for ax in axes[1, :]:
        ax.set_xlabel("y (Mm)")
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def make_tc_summary_figure(runs, out_png):
    tc = np.array([r["tau_c"] for r in runs])
    r0 = runs[2]
    i_s = i_star_of(r0)
    collar = np.array([0.5 * (r["w"][0] + r["w"][-1]) for r in runs])
    pred = np.array([r["w_collar"] for r in runs])
    dwp = np.array([r["w"].max() - c for r, c in zip(runs, collar)])
    dwm = np.array([c - r["w"].min() for r, c in zip(runs, collar)])
    h0 = np.array([r["hhat"][int(np.argmin(np.abs(r["y"])))]
                   for r in runs]) / 1e6
    hmin = np.array([r["hhat"].min() for r in runs]) / 1e6
    hmax = np.array([r["hhat"].max() for r in runs]) / 1e6
    tc_fine = np.geomspace(tc[0], tc[-1], 200)
    h_pred = (r0["shat"] - L_V * (2 * r0["a"] - 1)
              * (r0["w_crit"] + tc_fine * r0["evap"])) / 1e6

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.semilogx(tc_fine, r0["w_crit"] + tc_fine * r0["evap"], "-", c="gray",
                lw=1, label=r"predicted $W_c + \tau_c E_0$")
    ax.semilogx(tc, collar, "o", c="#0077BB", label="measured collar")
    ax.set_ylabel("collar W (kg m$^{-2}$)")
    ax.set_title("Quiescent collar tracks the local balance "
                 f"(max dev {np.abs(collar - pred).max():.1e})")

    ax = axes[0, 1]
    ax.loglog(tc, dwp, "o-", c="#CC6677", label=r"ITCZ bump max$(W)-$collar")
    ax.loglog(tc, dwm, "o-", c="#0077BB", label=r"subtrop deficit collar$-$min$(W)$")
    ax.loglog(tc, dwp[2] * tc / tc[2], "--", c="gray", lw=1,
              label=r"slope 1 ($\propto\tau_c$)")
    ax.set_ylabel(r"$|W -$ collar$|$ (kg m$^{-2}$)")
    ax.set_title(r"Departure amplitude scales $\sim\tau_c$")

    ax = axes[1, 0]
    ax.semilogx(tc_fine, h_pred, "-", c="gray", lw=1,
                label=r"predicted $\hat S - L_v(2a{-}1)(W_c+\tau_c E_0)$")
    ax.fill_between(tc, hmin, hmax, color="#BBBBBB", alpha=0.5,
                    label="domain range")
    ax.semilogx(tc, h0, "o", c="#0077BB", label=r"$\hat H(0)$ (ITCZ)")
    ax.axhline(0, ls=":", c="gray", lw=1)
    ax.set_ylabel(r"$\hat H$ (MJ m$^{-2}$)")
    ax.set_title(r"$\hat H$ deepens with $\tau_c$; no sign change at $W_c=50$")

    ax = axes[1, 1]
    dse = np.array([r["dse_flux"][i_s] for r in runs]) / 1e6
    lvq = np.array([r["lvq_flux"][i_s] for r in runs]) / 1e6
    net = np.array([r["mean_flux"][i_s] for r in runs]) / 1e6
    eddy = np.array([r["eddy_flux"][i_s] for r in runs]) / 1e6
    ax.semilogx(tc, dse, "o-", c=C_DSE, label=r"DSE $\hat S v$")
    ax.semilogx(tc, lvq, "o-", c=C_LVQ, label=r"$L_vq$ $-L_v(2a{-}1)Wv$")
    ax.semilogx(tc, net, "o-", c="k", lw=2, label=r"net mean $v\hat H$")
    ax.semilogx(tc, eddy, "o--", c=C_EDDY, label=r"eddy $-L_vD\partial_yW$")
    ax.axhline(0, ls=":", c="gray", lw=1)
    ax.set_ylabel(f"northward flux at y*={r0['y'][i_s] / 1e6:.2f} Mm "
                  "(MW m$^{-1}$)")
    ax.set_title("Mean-flux components (dry v fixed across the ladder)")

    for ax in axes.flat:
        ax.set_xlabel(r"$\tau_c$ (s)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def make_dz_ladder_figure(runs, colors, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for r, c in zip(runs, colors):
        y = r["y"] / 1e6
        lab = f"$\\Delta_z$={r['delta_z']:g} K"
        axes[0, 0].plot(y, r["w"], color=c, label=lab)
        axes[0, 1].plot(y, r["p"] * 86400, color=c, label=lab)
        axes[1, 0].plot(y, r["hhat"] / 1e6, color=c, label=lab)
        axes[1, 1].plot(y, r["mean_flux"] / 1e6, color=c, label=lab)
    axes[0, 0].set_ylabel("W (kg m$^{-2}$)")
    axes[0, 0].set_title(r"W: collar pinned by $W_c+\tau_c E_0$, structure "
                         "by the changing dry v")
    axes[0, 1].set_ylabel("P (mm day$^{-1}$)")
    axes[0, 1].set_title("Precipitation")
    axes[1, 0].axhline(0, ls=":", c="gray", lw=1)
    axes[1, 0].set_ylabel(r"$\hat H$ (MJ m$^{-2}$)")
    axes[1, 0].set_title(r"$\hat H$ sign flips across the ladder "
                         r"($\hat S \propto \Delta_z$)")
    axes[1, 1].axhline(0, ls=":", c="gray", lw=1)
    axes[1, 1].set_ylabel(r"$v\hat H$ (MW m$^{-1}$)")
    axes[1, 1].set_title("Mean MSE flux reverses at the crossover")
    for ax in axes[1, :]:
        ax.set_xlabel("y (Mm)")
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def make_dz_summary_figure(runs, tc_runs, out_png):
    dz = np.array([r["delta_z"] for r in runs])
    r0 = runs[1]
    dz_fine = np.linspace(dz[0], dz[-1], 200)
    h_pred = (C_COLUMN * r0["delta"] * dz_fine / r0["height"]
              - L_V * (2 * r0["a"] - 1) * r0["w_collar"]) / 1e6
    dz_star = dz_star_of_w(r0, r0["w_collar"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    h0 = np.array([r["hhat"][int(np.argmin(np.abs(r["y"])))]
                   for r in runs]) / 1e6
    hmin = np.array([r["hhat"].min() for r in runs]) / 1e6
    hmax = np.array([r["hhat"].max() for r in runs]) / 1e6
    ax.plot(dz_fine, h_pred, "-", c="gray", lw=1,
            label=r"predicted $\hat S(\Delta_z) - L_v(2a{-}1)\,$collar")
    ax.fill_between(dz, hmin, hmax, color="#BBBBBB", alpha=0.5,
                    label="domain range")
    ax.plot(dz, h0, "o", c="#0077BB", label=r"$\hat H(0)$ (ITCZ)")
    ax.axhline(0, ls=":", c="gray", lw=1)
    ax.axvline(dz_star, ls="--", c="gray", lw=1,
               label=f"predicted $\\Delta_z^*$={dz_star:.1f} K")
    ax.set_xlabel(r"$\Delta_z$ (K)")
    ax.set_ylabel(r"$\hat H$ (MJ m$^{-2}$)")
    ax.set_title(r"$\hat H$ crosses zero at the predicted $\Delta_z^*$")

    ax = axes[0, 1]
    dse = np.array([r["dse_flux"][i_star_of(r)] for r in runs]) / 1e6
    lvq = np.array([r["lvq_flux"][i_star_of(r)] for r in runs]) / 1e6
    net = np.array([r["mean_flux"][i_star_of(r)] for r in runs]) / 1e6
    eddy = np.array([r["eddy_flux"][i_star_of(r)] for r in runs]) / 1e6
    ax.plot(dz, dse, "o-", c=C_DSE, label=r"DSE $\hat S v$")
    ax.plot(dz, lvq, "o-", c=C_LVQ, label=r"$L_vq$ $-L_v(2a{-}1)Wv$")
    ax.plot(dz, net, "o-", c="k", lw=2, label=r"net mean $v\hat H$")
    ax.plot(dz, eddy, "o--", c=C_EDDY, label=r"eddy $-L_vD\partial_yW$")
    ax.axhline(0, ls=":", c="gray", lw=1)
    ax.axvline(dz_star, ls="--", c="gray", lw=1)
    ax.set_xlabel(r"$\Delta_z$ (K)")
    ax.set_ylabel("northward flux at each run's y* (MW m$^{-1}$)")
    ax.set_title(r"Both $\hat S$ and the dry v respond to $\Delta_z$")

    ax = axes[1, 0]
    ujet = np.array([r["u"].max() for r in runs])
    vmax = np.array([np.abs(r["v"]).max() for r in runs])
    ax.plot(dz, 100 * ujet / ujet[1], "o-", c="#CC6677",
            label=f"jet max u (100% = {ujet[1]:.1f} m/s)")
    ax.plot(dz, 100 * vmax / vmax[1], "o-", c="#0077BB",
            label=f"max |v| (100% = {vmax[1]:.2f} m/s)")
    ax.axhline(100, ls=":", c="gray", lw=1)
    ax.set_xlabel(r"$\Delta_z$ (K)")
    ax.set_ylabel(r"% of $\Delta_z=60$ value")
    ax.set_title("Dry-circulation response to the stratification")

    ax = axes[1, 1]
    tc = np.array([r["tau_c"] for r in tc_runs])
    tc_fine = np.geomspace(tc[0] / 2, tc[-1] * 2, 200)
    ax.semilogx(tc_fine,
                dz_star_of_w(r0, r0["w_crit"] + tc_fine * r0["evap"]),
                "-", c="k", lw=1.5,
                label=r"$\hat H_{collar}=0$: $\Delta_z^*(\tau_c)$")
    band_lo = dz_star_of_w(r0, np.array([r["w"].min() for r in tc_runs]))
    band_hi = dz_star_of_w(r0, np.array([r["w"].max() for r in tc_runs]))
    ax.fill_between(tc, band_lo, band_hi, color="#CCBB44", alpha=0.5,
                    label="mixed-sign band (measured W range)")
    for rr, tcv, dzv in (
            [(r, r["tau_c"], 60.0) for r in tc_runs]
            + [(r, 14400.0, r["delta_z"]) for r in runs]):
        frac = float(np.mean(rr["hhat"] < 0))
        fc = ("#CC6677" if frac == 1.0
              else "#0077BB" if frac == 0.0 else "none")
        ax.plot(tcv, dzv, "o", mfc=fc, mec="k", ms=8)
    ax.plot([], [], "o", mfc="#CC6677", mec="k",
            label=r"run: $\hat H<0$ everywhere")
    ax.plot([], [], "o", mfc="#0077BB", mec="k",
            label=r"run: $\hat H>0$ everywhere")
    ax.plot([], [], "o", mfc="none", mec="k", label="run: mixed sign")
    ax.set_xlabel(r"$\tau_c$ (s)")
    ax.set_ylabel(r"$\Delta_z$ (K)")
    ax.set_title(r"Regime diagram: the scans cross one boundary")

    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def main():
    scan_dir = (sys.argv[1] if len(sys.argv) > 1
                else "model_output/moist_v1_tauc_dz")
    d1_path = (sys.argv[2] if len(sys.argv) > 2
               else "model_output/moist_v1_validation/D1/out.nc")

    tc_paths = run_paths(scan_dir, d1_path, TC_DIRS)
    dz_paths = run_paths(scan_dir, d1_path, DZ_DIRS)
    tc_runs = [load_equilibrium(p) for p in tc_paths]
    dz_runs = [load_equilibrium(p) for p in dz_paths]
    for r, tc in zip(tc_runs, TC_VALUES):
        assert r["tau_c"] == tc, f"run has tau_c={r['tau_c']}, expected {tc}"
        assert r["delta_z"] == 60.0 and r["w_crit"] == 50.0 and r["d_w"] == 1e6
    for r, dzv in zip(dz_runs, DZ_VALUES):
        assert r["delta_z"] == dzv, \
            f"run has delta_z={r['delta_z']}, expected {dzv}"
        assert r["tau_c"] == 14400.0 and r["w_crit"] == 50.0 and r["d_w"] == 1e6

    tc_extras = [load_extras(p) for p in tc_paths]
    dz_extras = [load_extras(p) for p in dz_paths]
    u_ref = tc_extras[2]["u_daily"]

    scorecard_tc(tc_runs, tc_extras, u_ref)
    scorecard_dz(dz_runs, dz_extras)

    # Ordered parameter -> sequential ramp (CVD-safe), dark = large value.
    tc_colors = plt.cm.viridis(np.linspace(0.85, 0.05, len(tc_runs)))
    dz_colors = plt.cm.viridis(np.linspace(0.85, 0.05, len(dz_runs)))
    make_tc_ladder_figure(
        tc_runs, tc_colors, os.path.join(scan_dir, "moist_v1_tauc_scan.png"))
    make_tc_summary_figure(
        tc_runs, os.path.join(scan_dir, "moist_v1_tauc_scan_summary.png"))
    make_dz_ladder_figure(
        dz_runs, dz_colors, os.path.join(scan_dir, "moist_v1_dz_scan.png"))
    make_dz_summary_figure(
        dz_runs, tc_runs, os.path.join(scan_dir, "moist_v1_dz_scan_summary.png"))


if __name__ == "__main__":
    main()
