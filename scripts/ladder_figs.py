"""Two-panel figure for the R2/R3 resolution ladder.

(a) log-log convergence of the formulation differences that Tier 1
flagged: exponent deficit at matched ny and gate-on/gateless u excess at
fixed y, vs grid spacing, with a slope-1 (first-order) reference line.
(b) SH terminus zoom: the gate-on notch at ny=201/401/801 over the
gateless ny=801 reference, showing the fixed-physical-width, convergent
structure.

Usage: python scripts/ladder_figs.py [OUTPATH]
"""
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
DAYS = 1825
SUITE = pathlib.Path("model_output/formulation_suite")
RUNS = {
    (201, "gateon"): SUITE / "ladder_ny201_gateon/output.nc",
    (201, "gateless"): SUITE / "ladder_ny201_gateless/output.nc",
    (401, "gateon"): SUITE / "ladder_ny401_gateon/output.nc",
    (401, "gateless"): SUITE / "ladder_ny401_gateless/output.nc",
    (801, "gateon"): SUITE / "tier1_y0p0000_gateon_upwind/output.nc",
}
REF801 = pathlib.Path(
    "model_output/validation_20260709/runs/vd25_vert/climatology.npz")
RAMP = {201: "#86b6ef", 401: "#3987e5", 801: "#104281"}


def load(path):
    if path.suffix == ".npz":
        d = np.load(path)
        return d["y"], d["u"]
    ds = xr.open_dataset(path, decode_timedelta=False)
    nt = ds.sizes["time"]
    avg = slice(nt - min(nt, DAYS), nt)
    return ds["y"].values, ds["u"].values[avg].mean(axis=0)


def main():
    outpath = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
        SUITE / "fig_ladder_convergence.png"
    fields = {k: load(p) for k, p in RUNS.items()}
    fields[(801, "gateless")] = load(REF801)

    dys, dps, ex315, ex630 = [], [], [], []
    for ny in (201, 401, 801):
        yon, uon = fields[(ny, "gateon")]
        yoff, uoff = fields[(ny, "gateless")]
        dys.append((yon[1] - yon[0]) / 1e3)
        p_on, _ = near_eq_powerlaw(yon / M_PER_DEG, uon, 1.5, 7.0)
        p_off, _ = near_eq_powerlaw(yoff / M_PER_DEG, uoff, 1.5, 7.0)
        dps.append(p_off - p_on)
        for yp, acc in [(315e3, ex315), (630e3, ex630)]:
            i = int(np.argmin(np.abs(yon - yp)))
            j = int(np.argmin(np.abs(yoff - yp)))
            acc.append(uon[i] / uoff[j] - 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for series, c, lb in [(dps, "#2a78d6", "exponent deficit Δp"),
                          (ex315, "#1baf7a", "u excess at y=315 km"),
                          (ex630, "#4a3aa7", "u excess at y=630 km")]:
        ax.loglog(dys, series, "o-", color=c, lw=1.6, ms=6, label=lb)
    xr_ = np.array([dys[-1], dys[0]])
    ax.loglog(xr_, dps[-1] * xr_ / xr_[0], "--", color="#898781", lw=1.2,
              label="slope 1 (first order)")
    ax.set_xlabel("dy [km]")
    ax.set_ylabel("gate-on vs gateless difference")
    ax.set_title("near-equator formulation differences vs dy", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color="#e1e0d9", lw=0.7, which="both")
    ax.set_axisbelow(True)

    ax = axes[1]
    yoff, uoff = fields[(801, "gateless")]
    m = (yoff >= -10.5 * Mm) & (yoff <= -7.5 * Mm)
    ax.plot(yoff[m] / Mm, uoff[m], color="#52514e", lw=2.4,
            label="gateless ny=801")
    for ny in (201, 401, 801):
        y, u = fields[(ny, "gateon")]
        m = (y >= -10.5 * Mm) & (y <= -7.5 * Mm)
        ax.plot(y[m] / Mm, u[m], color=RAMP[ny], lw=1.4, marker=".", ms=5,
                label=f"gate-on ny={ny}")
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set_xlabel("y [Mm]")
    ax.set_ylabel("u [m/s]")
    ax.set_title("SH terminus notch across resolution", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color="#e1e0d9", lw=0.7)
    ax.set_axisbelow(True)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    fig.suptitle("R2/R3 ladder: exponent deficit is first-order numerics; "
                 "the notch is converged physics", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"saved: {outpath}")


if __name__ == "__main__":
    main()
