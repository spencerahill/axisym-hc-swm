"""Publication-comparison figures: original SS_Model.py runs vs Zhang et al. 2025.

Builds three PNGs from the climatology.npz files produced by
analyze_original_runs.py:
  fig1_full_domain.png   -- Zhang25 Fig. 3 analog (u, v, T vs latitude, 0-80N)
                            with AMC references and figure-read anchor targets
  fig2_near_equator.png  -- Zhang25 Fig. 2 analog (u, v, 0-5N) with beta*y^2/2,
                            beta*y^2/3, cubic references
  fig3_artifact_scan.png -- whole-domain (+/-141 deg) artifact scan: u, v, and
                            the 2dy sawtooth diagnostic on log scale

Usage: figures_original_runs.py FIGDIR RUNDIR [RUNDIR ...]
Each RUNDIR must contain climatology.npz + summary.json; the run label is the
directory basename.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

M_PER_DEG = np.pi * 6.371e6 / 180.0
BETA = 2e-11

FIGDIR = pathlib.Path(sys.argv[1]).resolve()
FIGDIR.mkdir(parents=True, exist_ok=True)
RUNDIRS = [pathlib.Path(p) for p in sys.argv[2:]]

# fixed styling per run key (sequential blues by v_d; dashed = no vert advec)
STYLE = {
    "amc_vd0_novert": dict(color="#9ecae1", ls="--", label="v_d=0, no vert adv"),
    "vd0_vert": dict(color="#6baed6", ls="-", label="v_d=0"),
    "vd0125_novert": dict(color="#4292c6", ls="--", label="v_d=0.0125, no vert adv"),
    "vd0125_vert": dict(color="#4292c6", ls="-", label="v_d=0.0125"),
    "vd05_vert": dict(color="#2171b5", ls="-", label="v_d=0.5"),
    "vd25_vert": dict(color="#08306b", ls="-", label="v_d=2.5"),
}

# Anchor targets read off Zhang et al. 2025 figures (accuracy ~ +/-5-10%).
ANCHORS = {
    "u": [(26, 50, "vd=0 jet (Fig 3a)"), (50, 28, "vd=2.5 jet (Fig 3a)")],
    "v": [(25, 0.36, "vd=2.5 (Fig 3b)")],
    "T": [(0, 205, "vd=0 (Fig 3c)"), (0, 199.5, "vd=2.5 (Fig 3c)"),
          (80, 173.5, "all (Fig 3c)")],
}

CONFIG_LINE = ("original SS_Model.py: ny=801, dt=30 s, 15 yr, last-5-yr mean; "
               "eps_u=1e-8, K_V=7786x100 (eff. /2), sin2 theta_E, Dy=50 K, y0=0")


def load(rundir):
    d = np.load(rundir / "climatology.npz")
    s = json.loads((rundir / "summary.json").read_text())
    return d, s


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)


def fig_full_domain(runs):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    latref = np.linspace(0, 80, 400)
    yref = latref * M_PER_DEG
    axs[0].plot(latref, 0.5 * BETA * yref**2, "k:", lw=1.4, label="AMC $\\beta y^2/2$")
    axs[0].plot(latref, BETA * yref**2 / 3, ":", color="0.45", lw=1.4,
                label="$\\beta y^2/3$")
    for d, s, key in runs:
        m = (d["lat"] >= 0) & (d["lat"] <= 80)
        axs[0].plot(d["lat"][m], d["u"][m], **STYLE[key])
        axs[1].plot(d["lat"][m], d["v"][m], **STYLE[key])
        axs[2].plot(d["lat"][m], d["T"][m], **STYLE[key])
    for ax, var, ylab in [(axs[0], "u", "u [m/s]"), (axs[1], "v", "v [m/s]"),
                          (axs[2], "T", "T [K]")]:
        for (la, val, note) in ANCHORS[var]:
            ax.plot(la, val, "x", color="#d95f02", ms=9, mew=2.2, zorder=5)
        ax.set_xlabel("latitude [deg]")
        ax.set_ylabel(ylab)
        style_ax(ax)
    axs[0].set_ylim(-5, 85)
    axs[1].set_ylim(-0.15, 0.4)
    axs[0].plot([], [], "x", color="#d95f02", ms=9, mew=2.2,
                label="Zhang25 fig-read anchors")
    axs[0].legend(fontsize=8, frameon=False)
    axs[1].legend([plt.Line2D([], [], **STYLE[k]) for _, _, k in runs],
                  [STYLE[k]["label"] for _, _, k in runs],
                  fontsize=8, frameon=False)
    fig.suptitle("Original code vs Zhang et al. 2025 Fig. 3 (NH; runs are "
                 "exactly hemispherically symmetric)\n" + CONFIG_LINE, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGDIR / "fig1_full_domain.png", dpi=150)
    plt.close(fig)


def fig_near_equator(runs):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    latref = np.linspace(0, 5, 200)
    yref = latref * M_PER_DEG
    axs[0].plot(latref, 0.5 * BETA * yref**2, "k:", lw=1.6, label="$\\beta y^2/2$")
    axs[0].plot(latref, BETA * yref**2 / 3, ":", color="0.45", lw=1.6,
                label="$\\beta y^2/3$")
    ref_v = None
    for d, s, key in runs:
        m = (d["lat"] >= 0) & (d["lat"] <= 5)
        axs[0].plot(d["lat"][m], d["u"][m], **STYLE[key])
        axs[1].plot(d["lat"][m], d["v"][m], **STYLE[key])
        if key == "amc_vd0_novert":
            ref_v = (d["lat"][m], d["v"][m])
    if ref_v is not None:
        axs[1].plot(*ref_v, color="0.6", lw=3, alpha=0.5, zorder=0,
                    label="canonical-AMC v (gray ref, cf. Fig 2)")
    axs[0].set_ylim(0, 2.2)
    axs[0].set_ylabel("u [m/s]")
    axs[1].set_ylabel("v [m/s]")
    for ax in axs:
        ax.set_xlabel("latitude [deg]")
        style_ax(ax)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Near-equator scalings vs Zhang et al. 2025 Fig. 2\n"
                 + CONFIG_LINE, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGDIR / "fig2_near_equator.png", dpi=150)
    plt.close(fig)


def fig_artifact_scan(runs):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for d, s, key in runs:
        axs[0].plot(d["lat"], d["u"], **STYLE[key])
        axs[1].plot(d["lat"], d["v"], **STYLE[key])
        saw = np.full_like(d["u"], np.nan)
        saw[1:-1] = np.abs(d["u"][1:-1] - 0.5 * (d["u"][:-2] + d["u"][2:]))
        axs[2].semilogy(d["lat"], np.maximum(saw, 1e-12), **STYLE[key])
    axs[0].set_ylabel("u [m/s]")
    axs[1].set_ylabel("v [m/s]")
    axs[2].set_ylabel("u sawtooth $|u_i - \\frac{1}{2}(u_{i-1}+u_{i+1})|$ [m/s]")
    for ax in axs:
        ax.set_xlabel("beta-plane latitude [deg]")
        style_ax(ax)
    axs[2].axhline(1.0, color="#d95f02", lw=1, ls="--")
    axs[2].annotate("1 m/s", (0.02, 1.25), xycoords=("axes fraction", "data"),
                    fontsize=8, color="#d95f02")
    axs[1].legend([plt.Line2D([], [], **STYLE[k]) for _, _, k in runs],
                  [STYLE[k]["label"] for _, _, k in runs],
                  fontsize=8, frameon=False)
    fig.suptitle("Whole-domain artifact scan (full +/-15751 km domain)\n"
                 + CONFIG_LINE, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGDIR / "fig3_artifact_scan.png", dpi=150)
    plt.close(fig)


def ro_table(runs):
    print("\nLocal Rossby number Ro = (du/dy)/(beta*y), NH interior (cf. Zhang25 Fig. 4):")
    for d, s, key in runs:
        y = d["y"]
        dy = y[1] - y[0]
        dudy = np.gradient(d["u"], dy)
        with np.errstate(divide="ignore", invalid="ignore"):
            ro = np.where(np.abs(y) > 2 * dy, dudy / (BETA * y), np.nan)
        m = (d["lat"] > 1) & (d["lat"] < 80)
        i = int(np.nanargmax(np.where(m, ro, -np.inf)))
        print(f"  {key:>16}: max Ro = {ro[i]:.3f} at {d['lat'][i]:.1f} deg"
              f"   (v_max lat = {s['lat_v_absmax_deg']:.1f} deg)")
    print("\nu(3 deg) ratios [computed]:")
    for d, s, key in runs:
        i3 = int(np.argmin(np.abs(d["lat"] - 3.0)))
        y3 = d["y"][i3]
        u3 = d["u"][i3]
        print(f"  {key:>16}: u(3deg)={u3:7.4f}  /(by^2/2)={u3/(0.5*BETA*y3**2):.3f}"
              f"  /(by^2/3)={u3/(BETA*y3**2/3):.3f}")
    print("\nv(5 deg) [computed, cf. Zhang25 Fig. 2 v panels]:")
    for d, s, key in runs:
        i5 = int(np.argmin(np.abs(d["lat"] - 5.0)))
        print(f"  {key:>16}: v(5deg)={d['v'][i5]:.4f} m/s")


def main():
    runs = []
    for rd in RUNDIRS:
        if (rd / "climatology.npz").exists():
            d, s = load(rd)
            runs.append((d, s, rd.name))
        else:
            print(f"skipping {rd.name}: no climatology.npz")
    fig_full_domain(runs)
    fig_near_equator(runs)
    fig_artifact_scan(runs)
    ro_table(runs)
    print(f"\nfigures written to {FIGDIR}")


if __name__ == "__main__":
    main()
