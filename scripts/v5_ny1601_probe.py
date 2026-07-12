"""V5: ny=1601 resolution probe of the terminus notch and standing v ripple.

The gate-on+mc+staggered production formulation has two grid-scale features
whose resolution behavior was left open at ny=801:

  - the terminus notch (the steady easterly spike at each westerly cell edge),
    whose DEPTH was still deepening across the ny=201/401/801 ladder
    (-23.3/-26.1/-28.2 upwind; -28.80 staggered-mc at ny=801), so it was not
    yet confirmed converged; and
  - the standing interior v ripple, which the staggered grid drops onto the
    gateless noise floor (~8.8e-5 m/s) at ny=801.

This doubles the resolution to ny=1601 (dt=15, the k_v Asselin-budget limit)
and reports both features against the ny=801 staggered equilibrium, so we can
say whether the notch geometry has converged and the ripple stays on the floor
at finer resolution. It is a SHORT probe (cold, ~300 d): the ripple and notch
equilibrate by day ~200 (ripple_equilibration.py), but the near-equator
exponent is a multi-year observable and is NOT assessed here.

Usage:
    python scripts/v5_ny1601_probe.py [--test DIR] [--ref DIR] [--days N]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ss09.read_output import load_centered  # noqa: E402
from cmp_utils import sawtooth  # noqa: E402

Mm = 1e6
BANDS = [(0, 2), (2, 5), (5, 8)]
GATELESS_FLOOR = 8.8e-5
SUITE = pathlib.Path("model_output/formulation_suite/staggered_v_prod")


def notch_metrics(y, u, hemi):
    """Terminus-notch geometry in one hemisphere's [7, 10.5] Mm band."""
    if hemi == "sh":
        band = (y >= -10.5 * Mm) & (y <= -7 * Mm)
    else:
        band = (y >= 7 * Mm) & (y <= 10.5 * Mm)
    ub, yb = u[band], y[band]
    dy = y[1] - y[0]
    i = int(np.argmin(ub))
    return {
        "depth": float(ub[i]),
        "pos_Mm": float(yb[i] / Mm),
        "w_deep_km": float(np.sum(ub < -5) * dy / 1e3),
        "w_easterly_km": float(np.sum(ub < 0) * dy / 1e3),
    }


def banded_ripple_faces(path, days):
    """Max sawtooth(v) by |y| band, on the native staggered faces."""
    ds = xr.open_dataset(path, decode_timedelta=False)
    y = ds["y"].values
    yf = 0.5 * (y[:-1] + y[1:])
    v_faces = ds["v"].values[-days:].mean(axis=0)
    ds.close()
    sv = sawtooth(v_faces)
    return [float(np.nanmax(np.where(
        (np.abs(yf) >= a * Mm) & (np.abs(yf) < b * Mm), sv, np.nan)))
        for a, b in BANDS]


def load_run(run, days):
    """Center-grid u/T/v, banded face ripple, and health metrics for a run."""
    nc = pathlib.Path(run) / "output.nc"
    y, u, _, T = load_centered(str(nc), ndays=days)
    ds = xr.open_dataset(nc, decode_timedelta=False)
    u_t = ds["u"].values
    nan_days = int(np.isnan(u_t).any(axis=1).sum())
    umax_t = np.abs(u_t).max(axis=1)
    nd = min(200, ds.sizes["time"] - 1)
    drift = float(umax_t[-1] - umax_t[-1 - nd])
    ny = ds.sizes["y"]
    ds.close()
    interior = np.abs(y) <= 8 * Mm
    return {
        "ny": ny, "y": y, "u": u, "T": T,
        "nan_days": nan_days, "drift": drift, "drift_days": nd,
        "u_sawtooth_int": float(np.nanmax(sawtooth(u)[interior])),
        "parity_u": float(np.max(np.abs(u - u[::-1]))),
        "jet": float(u[y > 0].max()),
        "T_eq": float(T[int(np.argmin(np.abs(y)))]),
        "ripple": banded_ripple_faces(nc, days),
        "notch": {h: notch_metrics(y, u, h) for h in ("sh", "nh")},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", default=str(SUITE / "v5_ny1601"))
    p.add_argument("--ref", default=str(SUITE),
                   help="ny=801 staggered equilibrium (output.nc)")
    p.add_argument("--days", type=int, default=100,
                   help="last-N-day averaging window (both runs)")
    p.add_argument("--json", default=str(SUITE / "v5_ny1601/analysis.json"))
    args = p.parse_args()

    test = load_run(args.test, args.days)
    ref = load_run(args.ref, args.days)

    print(f"\n=== V5 ny=1601 probe vs ny={ref['ny']} staggered "
          f"(last {args.days} d) ===\n")
    print(f"{'metric':>26} {'ny=' + str(ref['ny']):>12} "
          f"{'ny=' + str(test['ny']):>12}   note")
    rows = [
        ("NaN days", ref["nan_days"], test["nan_days"], "want 0"),
        ("drift max|u| (m/s)", f"{ref['drift']:+.4g}",
         f"{test['drift']:+.4g}", f"over last {test['drift_days']} d"),
        ("parity max|u(y)-u(-y)|", f"{ref['parity_u']:.2g}",
         f"{test['parity_u']:.2g}", "y0=0 -> want 0"),
        ("NH jet (m/s)", f"{ref['jet']:.3f}", f"{test['jet']:.3f}",
         "climate anchor"),
        ("T_eq (K)", f"{ref['T_eq']:.3f}", f"{test['T_eq']:.3f}", ""),
        ("interior sawtooth(u)", f"{ref['u_sawtooth_int']:.5f}",
         f"{test['u_sawtooth_int']:.5f}", "|y|<=8 Mm"),
    ]
    for name, a, b, note in rows:
        print(f"{name:>26} {str(a):>12} {str(b):>12}   {note}")

    print("\n  terminus notch (depth m/s, pos Mm, deep-core & easterly width):")
    print(f"{'arm/ny':>16} {'depth':>9} {'pos Mm':>8} {'w(u<-5) km':>11} "
          f"{'w(u<0) km':>10}")
    for hemi in ("sh", "nh"):
        for label, d in ((f"ny{ref['ny']} {hemi}", ref),
                         (f"ny{test['ny']} {hemi}", test)):
            n = d["notch"][hemi]
            print(f"{label:>16} {n['depth']:>9.2f} {n['pos_Mm']:>8.2f} "
                  f"{n['w_deep_km']:>11.0f} {n['w_easterly_km']:>10.0f}")
    d_depth = abs(test["notch"]["nh"]["depth"] - ref["notch"]["nh"]["depth"])
    print(f"\n  notch depth change ny{ref['ny']}->ny{test['ny']} (NH): "
          f"{d_depth:.2f} m/s "
          f"(ladder step was ~1.9 m/s at 401->801; converging if <)")

    print("\n  banded max sawtooth(v) on faces (want ~gateless floor "
          f"{GATELESS_FLOOR:.1e}):")
    print(f"{'band Mm':>10} {'ny=' + str(ref['ny']):>12} "
          f"{'ny=' + str(test['ny']):>12} {'xfloor':>8}")
    for (a, b), rr, rt in zip(BANDS, ref["ripple"], test["ripple"]):
        print(f"[{a},{b}) {'':>3} {rr:>12.6f} {rt:>12.6f} "
              f"{rt / GATELESS_FLOOR:>7.1f}x")

    out = {
        "days_window": args.days,
        "ref_ny": ref["ny"], "test_ny": test["ny"],
        "ref": {k: ref[k] for k in
                ("nan_days", "drift", "parity_u", "jet", "T_eq",
                 "u_sawtooth_int", "ripple", "notch")},
        "test": {k: test[k] for k in
                 ("nan_days", "drift", "parity_u", "jet", "T_eq",
                  "u_sawtooth_int", "ripple", "notch")},
        "notch_depth_change_nh": d_depth,
    }
    pathlib.Path(args.json).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
