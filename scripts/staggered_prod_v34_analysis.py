"""V3/V4: production staggered model in the eddying (v_d=0.125) and
off-equatorial (armA, y_0=1000 km) regimes.

V1 already established that the staggered operators reproduce the collocated
physics to <1e-4 m/s, independent of v_d and y_0 (the stencils are spatial and
regime-blind). V3/V4 therefore confirm the model stays stable and sensible in
regimes V1/V2 did not exercise: no NaN, the grid-scale v ripple sits on the
gateless noise floor, the climate anchors match the documented reference values,
and (for the y_0=0 V3 case, whose state is mirror-symmetric) parity is exactly 0.

Reference anchors (documented, provenance noted where uncertain):
  V3 (v_d=0.125): NH jet ~48.3 m/s, near-eq exponent ~2.51 (B2 targets).
  V4 (armA y_0=1e6): SH winter jet ~41.7, u(0) ~-14.16, u_min ~-16.15
      (from the A2 armA warm-start smoke, collocated gate-on+mc).

Usage:
    python scripts/staggered_prod_v34_analysis.py --run DIR [--symmetric]
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
from anchor_compare import M_PER_DEG, near_eq_powerlaw  # noqa: E402

Mm = 1e6
BANDS = [(0, 2), (2, 5), (5, 8)]
GATELESS_FLOOR = 8.8e-5


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--symmetric", action="store_true",
                   help="y_0=0 case: check exact mirror parity")
    p.add_argument("--days", type=int, default=300,
                   help="last-N-day averaging window")
    args = p.parse_args()
    run = pathlib.Path(args.run)

    y, u, v, T = load_centered(str(run / "output.nc"), ndays=args.days)
    dy = y[1] - y[0]
    yf = 0.5 * (y[:-1] + y[1:])
    ds = xr.open_dataset(run / "output.nc", decode_timedelta=False)
    v_faces = ds["v"].values[-args.days:].mean(axis=0)
    nan_days = int(np.isnan(ds["u"].values).any(axis=1).sum())
    umax_t = np.abs(ds["u"].values).max(axis=1)
    nd = min(500, ds.sizes["time"] - 1)
    drift = float(umax_t[-1] - umax_t[-1 - nd])
    ds.close()

    ieq = int(np.argmin(np.abs(y)))
    nh, sh = y > 0, y < 0
    lat_deg = y / M_PER_DEG
    p_nh, _ = near_eq_powerlaw(lat_deg, u)

    out = {
        "nan_days": nan_days, "drift_500d": drift,
        "nh_jet_mag": float(u[nh].max()),
        "nh_jet_lat_Mm": float(y[nh][np.argmax(u[nh])] / Mm),
        "sh_jet_mag": float(u[sh].max()),
        "sh_jet_lat_Mm": float(y[sh][np.argmax(u[sh])] / Mm),
        "u_eq": float(u[ieq]), "u_min": float(u.min()),
        "exponent": p_nh, "T_eq": float(T[ieq]),
    }

    print(f"\n=== V3/V4 analysis: {run.name} (last {args.days} d) ===")
    print(f"  NaN days              {nan_days}          (want 0)")
    print(f"  drift over last 500 d {drift:+.4g} m/s")
    print(f"  NH jet                {out['nh_jet_mag']:.3f} at {out['nh_jet_lat_Mm']:+.2f} Mm")
    print(f"  SH jet                {out['sh_jet_mag']:.3f} at {out['sh_jet_lat_Mm']:+.2f} Mm")
    print(f"  u(0)                  {out['u_eq']:+.4f} m/s")
    print(f"  u_min                 {out['u_min']:+.4f} m/s")
    print(f"  near-eq exponent(NH)  {p_nh:.3f}")
    print(f"  T_eq                  {out['T_eq']:.4f} K")

    print("\n  grid-scale v sawtooth by band (want ~gateless floor 8.8e-5):")
    out["bands"] = []
    for a, b in BANDS:
        m_f = (np.abs(yf) >= a * Mm) & (np.abs(yf) < b * Mm)
        s_f = float(np.nanmax(np.where(m_f, sawtooth(v_faces), np.nan)))
        ratio = s_f / GATELESS_FLOOR
        print(f"    [{a},{b}) Mm   {s_f:.6f}   ({ratio:.1f}x floor)")
        out["bands"].append({"band": [a, b], "sawtooth": s_f})

    if args.symmetric:
        par_u = float(np.max(np.abs(u - u[::-1])))
        par_v = float(np.max(np.abs(v_faces + v_faces[::-1])))
        out["parity_u"], out["parity_v"] = par_u, par_v
        print(f"\n  parity max|u(y)-u(-y)|    {par_u:.3g}   (want exactly 0)")
        print(f"  parity max|v_f+v_f(mirr)| {par_v:.3g}   (want exactly 0)")

    (run / "analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {run / 'analysis.json'}")


if __name__ == "__main__":
    main()
