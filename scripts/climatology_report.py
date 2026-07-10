"""Climatology report for a model output NetCDF.

Prints the standard steady-state anchors (hemispheric jets, tropical
easterlies, meridional-wind extrema, u = 0 crossings), a whole-domain
grid-scale artifact scan, and a steadiness check; optionally saves the
time-mean fields to npz for figure scripts. Generalizes the per-arm analysis
from the 2026-07-10 gate experiment so parameter sweeps get it for free.

Usage:
    python scripts/climatology_report.py OUTPUT_NC [--days N] [--save-npz PATH]
"""
import argparse
import pathlib
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cmp_utils import sawtooth  # noqa: E402

Mm = 1e6


def report_extremum(name, y, field, mask, sign):
    vals = np.where(mask, field, np.nan)
    i = int(np.nanargmax(sign * vals))
    print(f"{name}: {field[i]:+8.3f} at y = {y[i] / Mm:+6.2f} Mm")
    return i


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_nc", help="model output NetCDF file")
    p.add_argument("--days", type=int, default=1825,
                   help="averaging window from the end of the record "
                        "(default 1825; capped at the record length)")
    p.add_argument("--save-npz", default=None,
                   help="path to save the climatology (y, u, v, T, theta_e)")
    args = p.parse_args()

    ds = xr.open_dataset(args.output_nc, decode_timedelta=False)
    y = ds["y"].values
    nt = ds.sizes["time"]
    nan_days = int(np.isnan(ds["u"].values).any(axis=1).sum())
    navg = min(nt, args.days)
    print(f"=== {args.output_nc} ===")
    print(f"{nt} days saved, NaN days: {nan_days}; averaging last {navg}")

    avg = slice(nt - navg, nt)
    u = ds["u"].values[avg].mean(axis=0)
    v = ds["v"].values[avg].mean(axis=0)
    T = ds["T"].values[avg].mean(axis=0)
    if args.save_npz:
        np.savez(args.save_npz, y=y, u=u, v=v, T=T,
                 theta_e=ds["theta_e"].values)
        print(f"climatology saved: {args.save_npz}")

    report_extremum("SH jet u_max", y, u, y < 0, +1)
    report_extremum("NH jet u_max", y, u, y > 0, +1)

    core = np.abs(y) < 5 * Mm
    imin = report_extremum("tropical u_min", y, u, core, -1)
    neg = np.where(u < 0)[0]
    if len(neg) and u[imin] < 0:
        blocks = np.split(neg, np.where(np.diff(neg) != 1)[0] + 1)
        blk = [b for b in blocks if imin in b]
        if blk:
            b = blk[0]
            print(f"easterly band containing u_min: y = "
                  f"[{y[b[0]] / Mm:+.2f}, {y[b[-1]] / Mm:+.2f}] Mm")

    report_extremum("v min", y, v, np.abs(y) < 8 * Mm, -1)
    report_extremum("v max", y, v, np.abs(y) < 8 * Mm, +1)
    ieq = int(np.argmin(np.abs(y)))
    print(f"equator: u = {u[ieq]:+.4f}, v = {v[ieq]:+.4f}, T = {T[ieq]:.2f}")

    sgn = np.sign(u)
    cross = np.where(np.diff(sgn) != 0)[0]
    ycs = [y[i] - u[i] * (y[i + 1] - y[i]) / (u[i + 1] - u[i]) for i in cross]
    txt = ", ".join(f"{c / Mm:+.2f}" for c in ycs)
    print(f"u = 0 crossings ({len(ycs)}) [Mm]: {txt}")

    for nm, f in [("u", u), ("v", v)]:
        s = sawtooth(f)
        k = int(np.nanargmax(s))
        inner = np.abs(y) < 8 * Mm
        ki = int(np.nanargmax(np.where(inner, s, np.nan)))
        print(f"{nm} sawtooth max: {s[k]:.3f} at y = {y[k] / Mm:+.2f} Mm "
              f"(interior: {s[ki]:.3f} at {y[ki] / Mm:+.2f} Mm)")

    i = int(np.argmax(np.abs(u)))
    j = int(np.argmax(np.abs(v)))
    print(f"domain max|u| = {abs(u[i]):.2f} at {y[i] / Mm:+.2f} Mm; "
          f"max|v| = {abs(v[j]):.3f} at {y[j] / Mm:+.2f} Mm")

    umax_t = np.abs(ds["u"].values).max(axis=1)
    nd = min(500, nt - 1)
    print(f"max|u| drift over last {nd} d: {umax_t[-1] - umax_t[-1 - nd]:+.3f} m/s")


if __name__ == "__main__":
    main()
