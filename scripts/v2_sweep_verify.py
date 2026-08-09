"""Check the sweep against numbers earlier sessions recorded for the same runs.

Two configurations in this sweep were run before, in July 2026, and their
equilibria were written into CLAUDE.md and into
docs/era5_calibration_report.pdf. Those numbers were produced by a different
session with different analysis code, so reproducing them is an independent
check on this sweep's whole chain: the manifest's parameters, the runs, and the
diagnostics that read them.

Differences are expected, and their size is the point. The earlier runs used the
slow-drift-gated stopping criterion and finished near day 3700-4000; these are
fixed 5800-day integrations. Anything that still moves between those two lengths
shows up here as a discrepancy, which is information rather than a failure.

Every expectation below is quoted with the file it came from. Nothing is from
memory.

Usage:
    python scripts/v2_sweep_verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "/home/shill/py/axisym-hc-swm")

from scripts.v2_sweep_diagnostics import load_run, scalars  # noqa: E402
from scripts.v2_sweep_manifest import SWEEP_DIR, w_star  # noqa: E402

# (label, expected, tolerance as a fraction, source)
# Sources:
#   CLAUDE.md, "Regime warning" paragraph of the Moist V2 section
#   docs/era5_calibration_report.tex, Table tab:score and Table tab:fluxcompare
EXPECT_WC50 = [
    ("peak precipitation (mm/day)", 78.0, 0.10, "CLAUDE.md V2 regime warning"),
    ("second band latitude (Mm)", 5.6, 0.12, "CLAUDE.md V2 regime warning"),
    ("second band rate (mm/day)", 18.7, 0.20, "CLAUDE.md V2 regime warning"),
    ("fraction with P identically zero", 0.36, 0.25,
     "CLAUDE.md V2 regime warning"),
    ("terminus notch latitude (Mm)", 7.48, 0.12, "CLAUDE.md V2 regime warning"),
    ("terminus notch depth (m/s)", -45.0, 0.20, "CLAUDE.md V2 regime warning"),
    ("subtropical jet (m/s)", 73.8, 0.10, "era5 report Table tab:score"),
    ("peak |v| (m/s)", 5.86, 0.15, "era5 report Table tab:score"),
    ("dtheta at 16 deg (K)", 1.69, 0.15, "era5 report Table tab:score"),
    ("dtheta at 24 deg (K)", 7.69, 0.15, "era5 report Table tab:score"),
    ("dtheta at 33 deg (K)", 21.33, 0.15, "era5 report Table tab:score"),
    ("mean MSE flux at 24 deg (MW/m)", 2.3, 0.60,
     "era5 report Table tab:fluxcompare"),
]
EXPECT_WC40 = [
    ("peak precipitation (mm/day)", 9.0, 0.15, "CLAUDE.md V2 regime warning"),
    ("mean MSE flux at 24 deg (MW/m)", 11.2, 0.25,
     "era5 report Table tab:fluxcompare, a=0.85 W_c=40 row"),
    ("eddy MSE flux at 24 deg (MW/m)", 0.5, 0.60,
     "era5 report Table tab:fluxcompare, a=0.85 W_c=40 row"),
    ("subtropical jet (m/s)", 60.8, 0.10, "era5 report Table tab:score"),
    ("peak |v| (m/s)", 2.27, 0.15, "era5 report Table tab:score"),
    ("dtheta at 24 deg (K)", 5.13, 0.15, "era5 report Table tab:score"),
]


def secondary_band(r):
    """Latitude and rate of the strongest precipitation peak away from the
    ITCZ, which CLAUDE.md records at 5.6 Mm and 18.7 mm/day for W_c = 50."""
    from scipy.signal import find_peaks
    p = r["p"] * 86400.0
    y = r["y"]
    pk, _ = find_peaks(p, height=1.2 * r["evap"] * 86400.0,
                       prominence=0.25 * r["evap"] * 86400.0)
    off = [i for i in pk if abs(y[i]) > 2.0e6]
    if not off:
        return np.nan, np.nan
    i = max(off, key=lambda j: p[j])
    return abs(float(y[i])) / 1e6, float(p[i])


def terminus_notch(r):
    """The strongest easterly in the extratropics, which CLAUDE.md records for
    W_c = 50 at 7.48 Mm and -45 m/s."""
    y, u = r["y"], r["u"]
    m = np.abs(y) > 4.0e6
    if not m.any():
        return np.nan, np.nan
    i = int(np.flatnonzero(m)[np.argmin(u[m])])
    return abs(float(y[i])) / 1e6, float(u[i])


def measured(r, s):
    band_lat, band_rate = secondary_band(r)
    notch_lat, notch_depth = terminus_notch(r)
    return {
        "peak precipitation (mm/day)": s["p_max"],
        "second band latitude (Mm)": band_lat,
        "second band rate (mm/day)": band_rate,
        "fraction with P identically zero": s["dry_frac"],
        "terminus notch latitude (Mm)": notch_lat,
        "terminus notch depth (m/s)": notch_depth,
        "subtropical jet (m/s)": s["jet"],
        "peak |v| (m/s)": s["v_max"],
        "dtheta at 16 deg (K)": s["dtheta16"],
        "dtheta at 24 deg (K)": s["dtheta24"],
        "dtheta at 33 deg (K)": s["dtheta33"],
        "mean MSE flux at 24 deg (MW/m)": s["f_mean_ref"],
        "eddy MSE flux at 24 deg (MW/m)": s["f_eddy_ref"],
    }


def check(name, expects, tree=SWEEP_DIR):
    path = os.path.join(tree, name, "out.nc")
    r = load_run(path, last_n=200)
    if r is None:
        print(f"{name}: no usable output at {path}")
        return None
    s = scalars(r, name)
    got = measured(r, s)
    print(f"\n=== {name} ({r['days']} days, r = {s['r']:.3f}) ===")
    print(f"{'quantity':36s} {'expected':>10} {'measured':>10} "
          f"{'rel diff':>9}  verdict   source")
    n_pass = n_fail = 0
    rows = []
    for label, exp, tol, src in expects:
        val = got.get(label, np.nan)
        rel = abs(val - exp) / max(abs(exp), 1e-30) if np.isfinite(val) else np.nan
        ok = np.isfinite(rel) and rel <= tol
        n_pass += ok
        n_fail += not ok
        print(f"{label:36s} {exp:>10.3g} {val:>10.4g} {rel:>8.1%}   "
              f"{'agrees' if ok else 'DIFFERS':7s}  {src}")
        rows.append(dict(label=label, expected=exp, measured=float(val),
                         rel=float(rel), ok=bool(ok), source=src, tol=tol))
    print(f"{n_pass} of {n_pass + n_fail} within tolerance")
    return dict(name=name, days=r["days"], r=s["r"], rows=rows,
                n_pass=int(n_pass), n_fail=int(n_fail))


def main():
    out = []
    for name, exp in (("A_a085_wc50", EXPECT_WC50),
                      ("A_a085_wc40", EXPECT_WC40)):
        res = check(name, exp)
        if res:
            out.append(res)

    # The Hhat = 0 crossover, which CLAUDE.md and the ERA5 report both put at
    # 44.25 kg/m^2 for a = 0.85. Derived, so this is a check on the constants
    # rather than on a run.
    ws = w_star(0.85)
    print(f"\nHhat = 0 crossover at a = 0.85: {ws:.4f} kg/m^2 against the "
          f"44.25 recorded in CLAUDE.md, relative difference "
          f"{abs(ws - 44.25) / 44.25:.2%}")

    # The mean-state warming, predicted as tau * Lambda * E_0 = 71 K in
    # SCIENCE.md 3.6, measured at the wall where the column is quiescent.
    for name in ("A_a085_wc50", "A_a079_wc40"):
        r = load_run(os.path.join(SWEEP_DIR, name, "out.nc"))
        if r is None:
            continue
        s = scalars(r, name)
        pred = r["tau"] * r["lam"] * r["evap"]
        print(f"{name}: quiescent column sits {s['theta_offset']:.2f} K above "
              f"its radiative target, against the predicted "
              f"tau*Lambda*E_0 = {pred:.2f} K "
              f"({abs(s['theta_offset'] - pred) / pred:.2%})")

    total_fail = sum(o["n_fail"] for o in out)
    print(f"\n{total_fail} quantities outside tolerance across "
          f"{sum(len(o['rows']) for o in out)} comparisons")


if __name__ == "__main__":
    main()
