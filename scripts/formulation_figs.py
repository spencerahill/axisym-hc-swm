"""Overlay figures for the unified-formulation suite (test vs reference).

Produces the standard figure set comparing a run's climatology against a
stored reference climatology: full-domain u/v/T overlays with difference
panels, a near-equator zoom with the power-law fit, westerly-terminus zooms
with grid-point markers, and a drift/transient time-series figure. Reference
is drawn as a solid gray line, test as a dashed blue line (redundant
color/dash/width encoding so exact overlap stays visible).

Usage:
    python scripts/formulation_figs.py OUTPUT_NC REF_NPZ OUTDIR
        [--days N] [--test-label L] [--ref-label L]
"""
import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anchor_compare import M_PER_DEG, near_eq_powerlaw  # noqa: E402

Mm = 1e6
REF_C, TEST_C, DIFF_C = "#52514e", "#2a78d6", "#4a3aa7"
GRID_KW = {"color": "#e1e0d9", "linewidth": 0.7}


def style(ax):
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def overlay(ax, y, f, yr, fr, labels):
    ax.plot(yr / Mm, fr, color=REF_C, lw=2.4, label=labels[1])
    ax.plot(y / Mm, f, color=TEST_C, lw=1.6, ls="--", label=labels[0])
    style(ax)


def diffpanel(ax, y, f, yr, fr, unit):
    d = f - np.interp(y, yr, fr)
    ax.plot(y / Mm, d, color=DIFF_C, lw=1.4)
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    i = int(np.argmax(np.abs(d)))
    ax.annotate(f"max|Δ| = {abs(d[i]):.3g} {unit}\nat {y[i] / Mm:+.2f} Mm",
                xy=(0.02, 0.95), xycoords="axes fraction", va="top",
                fontsize=8, color="#52514e")
    style(ax)
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_nc")
    p.add_argument("ref_npz")
    p.add_argument("outdir")
    p.add_argument("--days", type=int, default=1825)
    p.add_argument("--test-label", default="gate-on + upwind")
    p.add_argument("--ref-label", default="gateless reference")
    args = p.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.output_nc, decode_timedelta=False)
    y = ds["y"].values
    nt = ds.sizes["time"]
    navg = min(nt, args.days)
    avg = slice(nt - navg, nt)
    u = ds["u"].values[avg].mean(axis=0)
    v = ds["v"].values[avg].mean(axis=0)
    T = ds["T"].values[avg].mean(axis=0)
    ref = np.load(args.ref_npz)
    yr, ur, vr, Tr = ref["y"], ref["u"], ref["v"], ref["T"]
    labels = (args.test_label, args.ref_label)

    # --- fig 1: full-domain overlays + differences -----------------------
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    for row, (nm, f, fr, unit) in enumerate(
            [("u", u, ur, "m/s"), ("v", v, vr, "m/s"), ("T", T, Tr, "K")]):
        overlay(axes[row, 0], y, f, yr, fr, labels)
        axes[row, 0].set_ylabel(f"{nm} [{unit}]")
        diffpanel(axes[row, 1], y, f, yr, fr, unit)
        axes[row, 1].set_ylabel(f"Δ{nm} [{unit}]")
    axes[0, 0].legend(frameon=False, fontsize=9)
    for ax in axes[2]:
        ax.set_xlabel("y [Mm]")
    fig.suptitle(f"{labels[0]} vs {labels[1]}: last-{navg}-d climatology")
    fig.tight_layout()
    f1 = outdir / "fig1_overlays_full.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    # --- fig 2: near-equator zoom + power law ----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    m = np.abs(y) <= 3 * Mm
    mr = np.abs(yr) <= 3 * Mm
    overlay(axes[0], y[m], u[m], yr[mr], ur[mr], labels)
    axes[0].set_xlabel("y [Mm]")
    axes[0].set_ylabel("u [m/s]")
    axes[0].set_title("near-equator u (|y| ≤ 3 Mm)", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    diffpanel(axes[1], y[m], u[m], yr, ur, "m/s")
    axes[1].set_xlabel("y [Mm]")
    axes[1].set_ylabel("Δu [m/s]")
    axes[1].set_title("difference (test − ref)", fontsize=10)

    ax = axes[2]
    for f_, y_, c, lw, ls, lb in [(ur, yr, REF_C, 2.4, "-", labels[1]),
                                  (u, y, TEST_C, 1.6, "--", labels[0])]:
        nh = (y_ > 0) & (f_ > 0)
        ax.loglog(y_[nh], f_[nh], color=c, lw=lw, ls=ls, label=lb)
        pfit, afit = near_eq_powerlaw(y_ / M_PER_DEG, f_)
        yy = np.geomspace(1.0 * M_PER_DEG, 3.5 * M_PER_DEG, 20)
        ax.loglog(yy, afit * yy ** pfit, color=c, lw=3.5, alpha=0.35)
        ax.annotate(f"p = {pfit:.3f}", xy=(0.03, 0.9 if c == REF_C else 0.8),
                    xycoords="axes fraction", color=c, fontsize=9)
    ax.axvspan(1.0 * M_PER_DEG, 3.5 * M_PER_DEG, color="#e1e0d9", alpha=0.4)
    ax.set_xlim(5e4, 3e6)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("u [m/s]")
    ax.set_title("NH power law u = A·y^p (fit band shaded)", fontsize=10)
    style(ax)
    fig.suptitle(f"near-equator structure: {labels[0]} vs {labels[1]}")
    fig.tight_layout()
    f2 = outdir / "fig2_zoom_neareq.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)

    # --- fig 3: terminus zooms with grid-point markers --------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (lo, hi), ttl in [(axes[0], (-11 * Mm, -7 * Mm), "SH terminus"),
                              (axes[1], (7 * Mm, 11 * Mm), "NH terminus")]:
        mw = (y >= lo) & (y <= hi)
        mwr = (yr >= lo) & (yr <= hi)
        ax.plot(yr[mwr] / Mm, ur[mwr], color=REF_C, lw=2.4, marker=".",
                ms=4, label=labels[1])
        ax.plot(y[mw] / Mm, u[mw], color=TEST_C, lw=1.2, ls="--", marker=".",
                ms=4, label=labels[0])
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set_xlabel("y [Mm]")
        ax.set_title(ttl, fontsize=10)
        style(ax)
    axes[0].set_ylabel("u [m/s]")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"westerly terminus (7–11 Mm): {labels[0]} vs {labels[1]}")
    fig.tight_layout()
    f3 = outdir / "fig3_zoom_terminus.png"
    fig.savefig(f3, dpi=150)
    plt.close(fig)

    # --- fig 4: drift + near-equator transient ---------------------------
    t = np.asarray(ds["time"].values, dtype=float)
    umax_t = np.abs(ds["u"].values).max(axis=1)
    band = (np.abs(y) <= 2 * Mm) & (np.abs(y) > 0)
    umin_band_t = ds["u"].values[:, band].min(axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, umax_t, color=TEST_C, lw=1.4)
    nd = min(500, nt - 1)
    axes[0].annotate(f"drift over last {nd} d: "
                     f"{umax_t[-1] - umax_t[-1 - nd]:+.4f} m/s",
                     xy=(0.02, 0.92), xycoords="axes fraction", fontsize=9,
                     color="#52514e")
    axes[0].set_ylabel("daily max|u| [m/s]")
    axes[1].plot(t, umin_band_t, color=TEST_C, lw=1.4)
    axes[1].axhline(0, color="#c3c2b7", lw=0.8)
    axes[1].annotate(f"final-day band min: {umin_band_t[-1]:+.4f} m/s "
                     f"(last-{navg}-d mean of band min: "
                     f"{umin_band_t[avg].mean():+.4f})",
                     xy=(0.02, 0.08), xycoords="axes fraction", fontsize=9,
                     color="#52514e")
    axes[1].set_ylabel("min u, 0<|y|≤2 Mm [m/s]")
    axes[1].set_xlabel("day")
    for ax in axes:
        style(ax)
    fig.suptitle(f"steadiness and near-equator transient: {labels[0]}")
    fig.tight_layout()
    f4 = outdir / "fig4_drift.png"
    fig.savefig(f4, dpi=150)
    plt.close(fig)

    for f in (f1, f2, f3, f4):
        print(f"saved: {f}")


if __name__ == "__main__":
    main()
