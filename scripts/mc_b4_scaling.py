"""B4 scaling check: does the MC stencil's exponent deficit shrink ~dy^2?

Computes the near-equator exponent deficit dp (arm minus matched-ny
gateless reference) for the mc and upwind gate-on arms at ny = 401 and
801, on both pre-registered fit windows, plus the fixed-y u excess at the
315/630 km probes. Under second-order behavior, dp(401)/dp(801) ~ 4 (the
upwind arms measured ~2, first order); the pre-registered B4 target is
dp(401, mc) <= ~0.1 on [1.5, 7] deg.

Usage:
    python scripts/mc_b4_scaling.py [--json PATH]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anchor_compare import M_PER_DEG, near_eq_powerlaw  # noqa: E402

DAYS = 1825
SUITE = pathlib.Path("model_output/formulation_suite")
RUNS = {
    (401, "mc"): SUITE / "mc_stencil/b4_ny401_gateon_mc/output.nc",
    (401, "upwind"): SUITE / "ladder_ny401_gateon/output.nc",
    (401, "gateless"): SUITE / "ladder_ny401_gateless/output.nc",
    (801, "mc"): SUITE / "mc_stencil/b1_y0p0000_gateon_mc/output.nc",
    (801, "upwind"): SUITE / "tier1_y0p0000_gateon_upwind/output.nc",
}
REF801 = pathlib.Path(
    "model_output/validation_20260709/runs/vd25_vert/climatology.npz")
FIT_WINDOWS = [(1.0, 3.5), (1.5, 7.0)]
Y_PROBES = [315.0e3, 630.0e3]


def load(path):
    if path.suffix == ".npz":
        d = np.load(path)
        return d["y"], d["u"]
    ds = xr.open_dataset(path, decode_timedelta=False)
    nt = ds.sizes["time"]
    avg = slice(nt - min(nt, DAYS), nt)
    y, u = ds["y"].values, ds["u"].values[avg].mean(axis=0)
    ds.close()
    return y, u


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    fields = {k: load(v) for k, v in RUNS.items() if v.exists()}
    fields[(801, "gateless")] = load(REF801)
    missing = [k for k in RUNS if k not in fields]
    for k in missing:
        print(f"[not on disk: {k} -> {RUNS[k]}]")

    out = {"dp": {}, "probe_excess": {}}
    print("=== exponent deficit dp = p(arm) - p(gateless at same ny) ===")
    print(f"{'window':>12} {'ny':>4} {'mc':>8} {'upwind':>8}")
    for lo, hi in FIT_WINDOWS:
        for ny in (401, 801):
            if (ny, "gateless") not in fields:
                continue
            y_g, u_g = fields[(ny, "gateless")]
            p_g, _ = near_eq_powerlaw(y_g / M_PER_DEG, u_g, lo, hi)
            row = []
            for arm in ("mc", "upwind"):
                if (ny, arm) not in fields:
                    row.append(np.nan)
                    continue
                y_a, u_a = fields[(ny, arm)]
                p_a, _ = near_eq_powerlaw(y_a / M_PER_DEG, u_a, lo, hi)
                row.append(p_a - p_g)
            out["dp"][f"[{lo},{hi}]_ny{ny}"] = row
            print(f"[{lo},{hi}] {'':>2} {ny:>4} {row[0]:>8.3f} {row[1]:>8.3f}")
        key_hi = out["dp"].get(f"[{lo},{hi}]_ny401")
        key_lo = out["dp"].get(f"[{lo},{hi}]_ny801")
        if key_hi and key_lo:
            for i, arm in enumerate(("mc", "upwind")):
                r = key_hi[i] / key_lo[i] if key_lo[i] else np.nan
                print(f"    {arm}: dp(401)/dp(801) = {r:.2f} "
                      "(4 = 2nd order, 2 = 1st order)")

    print("\n=== fixed-y u excess vs matched gateless (m/s) ===")
    print(f"{'y km':>6} {'ny':>4} {'mc':>10} {'upwind':>10}")
    for yp in Y_PROBES:
        for ny in (401, 801):
            if (ny, "gateless") not in fields:
                continue
            y_g, u_g = fields[(ny, "gateless")]
            ug = u_g[int(np.argmin(np.abs(y_g - yp)))]
            row = []
            for arm in ("mc", "upwind"):
                if (ny, arm) not in fields:
                    row.append(np.nan)
                    continue
                y_a, u_a = fields[(ny, arm)]
                row.append(u_a[int(np.argmin(np.abs(y_a - yp)))] - ug)
            out["probe_excess"][f"{yp / 1e3:.0f}km_ny{ny}"] = row
            print(f"{yp / 1e3:>6.0f} {ny:>4} {row[0]:>10.4f} {row[1]:>10.4f}")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nJSON saved: {args.json}")


if __name__ == "__main__":
    main()
