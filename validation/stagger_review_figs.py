"""Figures backing the staggered-v review claims, with self-checks.

Four figures, one claim each, into
model_output/formulation_suite/mc_stencil/review_figs/:

  fig1_ripple: the standing v ripple is visible gridpoint noise in the
      collocated run and absent in the staggered run (zoom overlay), and
      the roughness profile drops to the gateless floor domain-wide
      (log panel with the banded reduction factors).
  fig2_climate: u is unchanged between collocated and staggered except
      the terminus notch (overlay + difference panel).
  fig3_timing: the staggered extension's warm-start transient settles
      well before the averaging window (left), and B1's cold-start
      ripple bands and notch reach equilibrium by day ~100-200 (middle,
      right), which sizes a short ny=1601 probe.
  fig4_traps: the staggered output stores v on faces under center
      labels (near-equator zoom: as-labeled points shifted dy/2 and
      nonzero at y=0), and the run's mirror parity drifts at the
      1e-10 level from the Laplacian association order (day series).

Each panel's underlying numbers are printed as checks so the figures
can be verified against the review's claims before use.
"""
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cmp_utils import sawtooth  # noqa: E402

Mm = 1e6
BANDS = [(0, 2), (2, 5), (5, 8)]
MC = pathlib.Path("model_output/formulation_suite/mc_stencil")
OUT = MC / "review_figs"
GRAY, BLUE, ORANGE = "#52514e", "#2a78d6", "#bf7300"
GRID_KW = {"color": "#e1e0d9", "linewidth": 0.7}


def style(ax):
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def banded(y, v):
    sv = sawtooth(v)
    return [float(np.nanmax(np.where(
        (np.abs(y) >= a * Mm) & (np.abs(y) < b * Mm), sv, np.nan)))
        for a, b in BANDS]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    b1 = xr.open_dataset(MC / "b1_y0p0000_gateon_mc/output.nc",
                         decode_timedelta=False)
    y = b1["y"].values
    dy = y[1] - y[0]
    nt = b1.sizes["time"]
    u_c = b1["u"].values[nt - 1825:].mean(axis=0)
    v_c = b1["v"].values[nt - 1825:].mean(axis=0)
    v_c_daily = b1["v"].values
    u_c_daily = b1["u"].values
    b1.close()

    st = xr.open_dataset(MC / "staggered_v/output.nc", decode_timedelta=False)
    ntf = st.sizes["time"]
    u_s = st["u"].values[ntf - 100:].mean(axis=0)
    f_s = st["v"].values[ntf - 100:].mean(axis=0)
    f_daily = st["v"].values
    u_s_daily = st["u"].values
    st.close()
    yf = y[:-1] + 0.5 * dy
    fv = f_s[:-1]

    ref = np.load(
        "model_output/validation_20260709/runs/vd25_vert/climatology.npz")
    v_g = ref["v"]

    # ---------- fig 1: the ripple ----------
    lo, hi = 6.2, 8.0
    win_c = (y >= lo * Mm) & (y <= hi * Mm)
    wiggle = np.abs(np.diff(v_c[win_c], 2)).max() / 2
    vrange = v_c[win_c].max() - v_c[win_c].min()
    print(f"check fig1a: collocated wiggle {wiggle:.4f} vs window range "
          f"{vrange:.4f} ({100 * wiggle / vrange:.0f}% -> visible)")
    rip_c, rip_s, rip_g = banded(y, v_c), banded(yf, fv), banded(y, v_g)
    print("check fig1b bands: collocated", [f"{x:.5f}" for x in rip_c],
          "staggered", [f"{x:.6f}" for x in rip_s],
          "gateless", [f"{x:.6f}" for x in rip_g])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    win_f = (yf >= lo * Mm) & (yf <= hi * Mm)
    win_g = win_c
    ax.plot(y[win_g] / Mm, v_g[win_g], color=GRAY, lw=2.4,
            label="gateless reference (centers)")
    ax.plot(y[win_c] / Mm, v_c[win_c], color=BLUE, lw=1.2, ls="--",
            marker="o", ms=3.5, label="collocated gate-on+mc (centers)")
    ax.plot(yf[win_f] / Mm, fv[win_f], color=ORANGE, lw=1.6,
            marker="s", ms=3.5, label="staggered-v (faces)")
    ax.set_xlabel("y [Mm]")
    ax.set_ylabel("v [m/s]")
    ax.set_title("time-mean v, one gridpoint per marker (6.2-8 Mm)")
    ax.legend(fontsize=8)
    style(ax)

    ax = axes[1]
    nh_c = y > 0
    nh_f = yf > 0
    ax.semilogy(y[nh_c] / Mm, np.maximum(sawtooth(v_c)[nh_c], 1e-9),
                color=BLUE, lw=1.2, ls="--", label="collocated gate-on+mc")
    ax.semilogy(yf[nh_f] / Mm, np.maximum(sawtooth(fv)[nh_f], 1e-9),
                color=ORANGE, lw=1.6, label="staggered-v")
    ax.semilogy(y[nh_c] / Mm, np.maximum(sawtooth(v_g)[nh_c], 1e-9),
                color=GRAY, lw=2.4, label="gateless reference")
    for (a, b), rc, rs in zip(BANDS, rip_c, rip_s):
        ax.annotate(f"{rc / rs:.0f}x", xy=((a + b) / 2, 1.6e-3),
                    ha="center", fontsize=9, color="#3d3d3a")
    ax.set_xlim(0, 9)
    ax.set_ylim(1e-7, 3e-2)
    ax.set_xlabel("y [Mm]")
    ax.set_ylabel("gridpoint roughness of v [m/s]")
    ax.set_title("sawtooth(v) by latitude; band reductions annotated")
    ax.legend(fontsize=8, loc="upper left")
    style(ax)
    fig.suptitle("Claim 1: staggering v removes the standing interior "
                 "ripple down to the gateless noise floor", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_ripple.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- fig 2: climate ----------
    du = u_s - u_c
    i = int(np.argmax(np.abs(du)))
    print(f"check fig2: max|du| {np.abs(du).max():.3f} at {y[i] / Mm:+.2f} "
          f"Mm; interior (|y|<8Mm) max "
          f"{np.abs(np.where(np.abs(y) < 8 * Mm, du, 0)).max():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(y / Mm, u_c, color=BLUE, lw=2.2, ls="--",
            label="collocated gate-on+mc")
    ax.plot(y / Mm, u_s, color=ORANGE, lw=1.4, label="staggered-v")
    ax.set_xlabel("y [Mm]")
    ax.set_ylabel("u [m/s]")
    ax.set_title("time-mean u: the two curves overlap")
    ax.legend(fontsize=8)
    style(ax)

    ax = axes[1]
    ax.plot(y / Mm, du, color="#4a3aa7", lw=1.4)
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    itip = int(np.argmin(u_c))  # collocated notch tip
    dtip = u_s[int(np.argmin(u_s))] - u_c[itip]
    ax.annotate(
        f"all differences sit in the notch zone (8.5-9.7 Mm):\n"
        f"tip deepens {dtip:+.2f} m/s ({100 * abs(dtip) / abs(u_c[itip]):.0f}%"
        f" of its depth);\nrecovery-flank gridpoints differ by up to "
        f"{np.abs(du).max():.1f} m/s;\ninterior max only "
        f"{np.abs(np.where(np.abs(y) < 8 * Mm, du, 0)).max():.2f} m/s",
        xy=(0.5, 0.97), xycoords="axes fraction", va="top", ha="center",
        fontsize=8, color="#3d3d3a",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
    ax.set_xlabel("y [Mm]")
    ax.set_ylabel("u difference [m/s]")
    ax.set_title("staggered minus collocated u")
    style(ax)
    fig.suptitle("Claim 2: the climate is unchanged; the only visible "
                 "difference is a slightly deeper terminus notch", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_climate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- fig 3: timing ----------
    def daily_bands(vdaily, ygrid, ndays, faces=False):
        out = np.empty((ndays, len(BANDS)))
        for t in range(ndays):
            vt = vdaily[t][:-1] if faces else vdaily[t]
            out[t] = banded(ygrid, vt)
        return out

    rip_ext = daily_bands(f_daily, yf, ntf, faces=True)
    rip_cold = daily_bands(v_c_daily, y, 400)
    notch_band = (y >= -10.5 * Mm) & (y <= -7 * Mm)
    notch_cold = np.array([u_c_daily[t][notch_band].min() for t in range(400)])
    print(f"check fig3a: extension band[0,2) day1 {rip_ext[0, 0]:.6f} -> "
          f"day90+ mean {rip_ext[90:, 0].mean():.6f}")
    print(f"check fig3b: cold band[0,2) day5 {rip_cold[5, 0]:.4f}, day200 "
          f"{rip_cold[199, 0]:.5f}, equil {rip_c[0]:.5f}")
    print(f"check fig3c: cold notch day5 {notch_cold[5]:.2f}, day150 "
          f"{notch_cold[150]:.2f}, equil last-1825d "
          f"{np.array([u_c_daily[t][notch_band].min() for t in range(nt - 1825, nt, 100)]).mean():.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    labels = [f"|y| in [{a},{b}) Mm" for a, b in BANDS]
    shades = ["#2a78d6", "#7aa7e0", "#b7cdee"]
    ax = axes[0]
    for k in range(3):
        ax.semilogy(np.arange(ntf), rip_ext[:, k], color=shades[k], lw=1.4,
                    label=labels[k])
    ax.axvspan(100, 200, color="#efeee8", zorder=0)
    ax.annotate("averaging\nwindow", xy=(150, 2.5e-4), ha="center",
                fontsize=8, color="#3d3d3a")
    ax.set_xlabel("day of staggered extension")
    ax.set_ylabel("banded max roughness of v [m/s]")
    ax.set_title("warm-start transient settles by day ~90")
    ax.legend(fontsize=8)
    style(ax)

    ax = axes[1]
    for k in range(3):
        ax.semilogy(np.arange(400), rip_cold[:, k], color=shades[k], lw=1.4,
                    label=labels[k])
        ax.axhline(rip_c[k], color=shades[k], lw=0.8, ls=":")
    ax.set_xlabel("day of B1 cold start")
    ax.set_title("cold start: ripple reaches equilibrium (dotted)\n"
                 "by day ~100-200")
    ax.legend(fontsize=8)
    style(ax)

    ax = axes[2]
    ax.plot(np.arange(400), notch_cold, color="#4a3aa7", lw=1.4)
    ax.axhline(-28.05, color="#9a9990", lw=0.8, ls=":")
    ax.set_xlabel("day of B1 cold start")
    ax.set_ylabel("notch depth min(u) [m/s]")
    ax.set_title("cold start: notch depth at equilibrium\n(dotted) by day ~110")
    style(ax)
    fig.suptitle("Claim 3: ripple and notch equilibrate in the first "
                 "~200 days, so a ~300-day ny=1601 probe suffices", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_timing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- fig 4: artifact traps ----------
    zoom = 0.45
    wc = np.abs(y) <= zoom * Mm
    wf = np.abs(yf) <= zoom * Mm
    ieq = int(np.argmin(np.abs(y)))
    print(f"check fig4a: staggered file value at the y=0 slot "
          f"{f_s[ieq]:+.5f} (true face position {yf[ieq] / 1e3:.1f} km); "
          f"collocated v(0) {v_c[ieq]:+.5f}")
    par = np.array([np.abs(u_s_daily[t] - u_s_daily[t][::-1]).max()
                    for t in range(ntf)])
    print(f"check fig4b: parity day1 {par[0]:.2e} -> day200 {par[-1]:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(y[wc] / Mm, v_c[wc], color=BLUE, lw=1.2, ls="--", marker="o",
            ms=4, label="collocated v at centers (v(0) = 0)")
    ax.plot(yf[wf] / Mm, fv[wf], color=ORANGE, lw=1.6, marker="s", ms=4,
            label="staggered v at TRUE face positions")
    ax.plot(y[wc] / Mm, f_s[wc], color=ORANGE, lw=1.0, ls=":", marker="s",
            ms=4, markerfacecolor="none",
            label="same numbers read as the file labels them")
    ax.axvline(0, color="#c3c2b7", lw=0.8)
    ax.set_xlabel("y [Mm]")
    ax.set_ylabel("v [m/s]")
    ax.set_title("near-equator v: the file's labels shift every value\n"
                 "half a cell poleward (open squares)")
    ax.legend(fontsize=8)
    style(ax)

    ax = axes[1]
    ax.semilogy(np.arange(ntf), np.maximum(par, 1e-16), color="#4a3aa7",
                lw=1.4)
    ax.set_xlabel("day of staggered extension")
    ax.set_ylabel("max |u(y) - u(-y)| [m/s]")
    ax.set_title("mirror-parity drift of the staggered patch\n"
                 "(collocated model: exactly 0 for 15 years)")
    style(ax)
    fig.suptitle("Claim 4: two patch-level debts; neither touches the "
                 "ripple result", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_traps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"figures saved under {OUT}")


if __name__ == "__main__":
    main()
