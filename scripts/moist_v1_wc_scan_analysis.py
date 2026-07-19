"""Analyze the Moist V1 W_c scan across the gross-moist-stability crossover.

Companion to moist_v1_analysis.py (the D-ladder). Loads the gated moist runs
at W_c = {35, 40, 44, 55} kg/m^2 (fixed D = 1e6 m^2/s) plus the reused
D-ladder D1 run as the W_c = 50 point, and shows how the equilibrium sits
relative to the gross-moist-stability crossover

  Hhat(y) = Shat - L_v (2a-1) W = 0   at   W* = Shat / (L_v (2a-1)) ~ 44.6,

i.e. a critical W_c* = W* - tau_c E_0 ~ 43.9 given the quiescent equilibrium
W = W_c + tau_c E_0. Below W_c* the mean circulation exports MSE from the
ITCZ (Hhat > 0, dry-side); above it the mean MSE flux is up-gradient
(Hhat < 0, moisture-mode side), and near W_c* the mean flux nearly vanishes
so the diagnostic eddy flux -L_v D dW/dy is the only MSE transport left.

The dry circulation is identical across the scan (W is passive), so every
run fires the slow-drift gate at the same day and differences are purely in
the moisture fields.

Usage:
    python scripts/moist_v1_wc_scan_analysis.py [scan_dir] [wc50_path] [out_png]

scan_dir defaults to model_output/moist_v1_wc_scan (subdirs wc35/wc40/wc44/wc55);
wc50_path defaults to model_output/moist_v1_validation/D1/out.nc;
out_png defaults to <scan_dir>/moist_v1_wc_scan.png.
"""

import sys
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moist_v1_analysis import load_equilibrium

WC_VALUES = (35.0, 40.0, 44.0, 50.0, 55.0)
WC_DIRS = ("wc35", "wc40", "wc44", None, "wc55")  # None -> the reused D1 run


def run_paths(scan_dir, wc50_path):
    return [wc50_path if d is None else os.path.join(scan_dir, d, "out.nc")
            for d in WC_DIRS]


def scorecard(runs):
    """Per-run equilibrium checks; y* is the northern max-|v| latitude of the
    (shared) dry circulation, where the signed fluxes are compared."""
    r0 = runs[0]
    north = r0["y"] > 0
    i_star = int(np.flatnonzero(north)[np.argmax(np.abs(r0["v"][north]))])
    y_star = r0["y"][i_star] / 1e6
    print(f"Signed northward fluxes evaluated at y* = {y_star:.2f} Mm "
          f"(northern max |v| of the shared dry circulation)\n")
    print(f"{'W_c':>5} {'days':>5} {'W(0)':>7} {'collar':>7} {'pred':>7} "
          f"{'Pmax':>7} {'Pmin':>7} {'Hhat(0)':>9} {'Hh<0%':>6} "
          f"{'DSE*':>8} {'Lvq*':>8} {'net*':>8} {'eddy*':>8}")
    for r in runs:
        i_eq = int(np.argmin(np.abs(r["y"])))
        collar = 0.5 * (r["w"][0] + r["w"][-1])
        frac_neg = float(np.mean(r["hhat"] < 0.0)) * 100
        print(f"{r['w_crit']:>5.0f} {r['days']:>5} {r['w'][i_eq]:>7.2f} "
              f"{collar:>7.3f} {r['w_collar']:>7.3f} "
              f"{r['p'].max()*86400:>7.3f} {r['p'].min()*86400:>7.3f} "
              f"{r['hhat'][i_eq]/1e6:>9.2f} {frac_neg:>6.1f} "
              f"{r['dse_flux'][i_star]/1e6:>8.2f} {r['lvq_flux'][i_star]/1e6:>8.2f} "
              f"{r['mean_flux'][i_star]/1e6:>8.2f} {r['eddy_flux'][i_star]/1e6:>8.2f}")
    print(f"\nW(Hhat=0) = {r0['w_hhat_zero']:.2f} kg/m^2; predicted crossover "
          f"W_c* = {r0['w_hhat_zero'] - r0['tau_c'] * r0['evap']:.2f} kg/m^2.")
    print("W and collar in kg/m^2, P in mm/day, Hhat in MJ/m^2, fluxes in MW/m.")


def make_ladder_figure(runs, colors, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for r, c in zip(runs, colors):
        y = r["y"] / 1e6
        lab = f"$W_c$={r['w_crit']:.0f}"
        axes[0, 0].plot(y, r["w"], color=c, label=lab)
        axes[0, 1].plot(y, r["p"] * 86400, color=c, label=lab)
        axes[1, 0].plot(y, r["hhat"] / 1e6, color=c, label=lab)
        axes[1, 1].plot(y, r["mean_flux"] / 1e6, color=c, label=lab)
    axes[0, 0].set_ylabel("W (kg m$^{-2}$)")
    axes[0, 0].set_title("Column water vapor")
    axes[0, 1].set_ylabel("P (mm day$^{-1}$)")
    axes[0, 1].set_title("Precipitation")
    axes[1, 0].axhline(0, ls=":", c="gray", lw=1)
    axes[1, 0].set_ylabel(r"$\hat H$ (MJ m$^{-2}$)")
    axes[1, 0].set_title("Gross moist stability: sign flips across the scan")
    axes[1, 1].axhline(0, ls=":", c="gray", lw=1)
    axes[1, 1].set_ylabel(r"$v\hat H$ (MW m$^{-1}$)")
    axes[1, 1].set_title("Mean MSE flux: reverses at the crossover")
    for ax in axes[1, :]:
        ax.set_xlabel("y (Mm)")
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"\nWrote {out_png}")


def make_crossover_figure(runs, colors, out_png):
    """The scan summary: Hhat vs W_c through zero, the mean-flux components
    behind the net (decomposed, not just the residual), and the structure
    that survives when the mean flux vanishes."""
    r0 = runs[0]
    wc = np.array([r["w_crit"] for r in runs])
    wc_star = r0["w_hhat_zero"] - r0["tau_c"] * r0["evap"]
    north = r0["y"] > 0
    i_star = int(np.flatnonzero(north)[np.argmax(np.abs(r0["v"][north]))])
    y_star = r0["y"][i_star] / 1e6

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    h0 = np.array([r["hhat"][int(np.argmin(np.abs(r["y"])))] for r in runs]) / 1e6
    hmin = np.array([r["hhat"].min() for r in runs]) / 1e6
    hmax = np.array([r["hhat"].max() for r in runs]) / 1e6
    ax.fill_between(wc, hmin, hmax, color="#BBBBBB", alpha=0.5,
                    label="domain range")
    ax.plot(wc, h0, "o-", color="#4477AA", label=r"$\hat H(y=0)$ (ITCZ)")
    ax.axhline(0, ls=":", c="gray", lw=1)
    ax.axvline(wc_star, ls="--", c="gray", lw=1,
               label=f"predicted $W_c^*$={wc_star:.1f}")
    ax.set_xlabel(r"$W_c$ (kg m$^{-2}$)")
    ax.set_ylabel(r"$\hat H$ (MJ m$^{-2}$)")
    ax.set_title("Gross moist stability crosses zero at the predicted $W_c^*$")

    ax = axes[0, 1]
    dse = np.array([r["dse_flux"][i_star] for r in runs]) / 1e6
    lvq = np.array([r["lvq_flux"][i_star] for r in runs]) / 1e6
    net = np.array([r["mean_flux"][i_star] for r in runs]) / 1e6
    eddy = np.array([r["eddy_flux"][i_star] for r in runs]) / 1e6
    ax.plot(wc, dse, "o-", color="#CC6677", label=r"DSE $\hat S v$")
    ax.plot(wc, lvq, "o-", color="#4477AA", label=r"$L_vq$ $-L_v(2a{-}1)Wv$")
    ax.plot(wc, net, "o-", color="k", lw=2, label=r"net mean $v\hat H$")
    ax.plot(wc, eddy, "o--", color="#999933", label=r"eddy $-L_vD\partial_yW$")
    ax.axhline(0, ls=":", c="gray", lw=1)
    ax.axvline(wc_star, ls="--", c="gray", lw=1)
    ax.set_xlabel(r"$W_c$ (kg m$^{-2}$)")
    ax.set_ylabel(f"northward flux at y*={y_star:.2f} Mm (MW m$^{{-1}}$)")
    ax.set_title("Mean-flux components: net reverses, eddy flux persists")

    ax = axes[1, 0]
    for r, c in zip(runs, colors):
        ax.plot(r["y"] / 1e6, r["w"] - r["w_collar"], color=c,
                label=f"$W_c$={r['w_crit']:.0f}")
    ax.axhline(0, ls=":", c="gray", lw=1)
    ax.set_xlabel("y (Mm)")
    ax.set_ylabel(r"$W - (W_c + \tau_c E_0)$ (kg m$^{-2}$)")
    ax.set_title("Transport-driven W structure (collar removed)")

    ax = axes[1, 1]
    for r, c in zip(runs, colors):
        ax.plot(r["y"] / 1e6, r["eddy_flux"] / 1e6, color=c,
                label=f"$W_c$={r['w_crit']:.0f}")
    ax.axhline(0, ls=":", c="gray", lw=1)
    ax.set_xlabel("y (Mm)")
    ax.set_ylabel(r"$-L_v D\,\partial_y W$ (MW m$^{-1}$)")
    ax.set_title("Diagnostic eddy MSE flux across the scan")

    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def main():
    scan_dir = sys.argv[1] if len(sys.argv) > 1 else "model_output/moist_v1_wc_scan"
    wc50_path = (sys.argv[2] if len(sys.argv) > 2
                 else "model_output/moist_v1_validation/D1/out.nc")
    out_png = (sys.argv[3] if len(sys.argv) > 3
               else os.path.join(scan_dir, "moist_v1_wc_scan.png"))
    runs = [load_equilibrium(p) for p in run_paths(scan_dir, wc50_path)]
    for r, wc in zip(runs, WC_VALUES):
        assert r["w_crit"] == wc, f"run has w_crit={r['w_crit']}, expected {wc}"
        assert r["d_w"] == 1.0e6, f"run has d_w={r['d_w']}, expected 1e6"
    # Ordered parameter -> sequential ramp (CVD-safe), dark = large W_c.
    colors = plt.cm.viridis(np.linspace(0.85, 0.05, len(runs)))
    scorecard(runs)
    make_ladder_figure(runs, colors, out_png)
    base, ext = os.path.splitext(out_png)
    make_crossover_figure(runs, colors, base + "_crossover" + ext)


if __name__ == "__main__":
    main()
