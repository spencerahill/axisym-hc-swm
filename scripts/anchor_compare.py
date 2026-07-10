"""Anchor comparison of a model climatology against a stored reference.

Computes the standard steady-state anchor set (hemispheric jet magnitude and
position, v extrema, equator values, near-equator power law, sawtooth,
hemispheric parity, u = 0 crossings) for a test run and a reference
climatology, prints both plus deltas, and quantifies interior (|y| <= 8 Mm)
and full-domain max|delta| for u, v, T on a common grid. The recomputed
reference anchors are cross-checked against the stored summary.json, so
feeding the reference npz against itself doubles as the script's self-test
(all deltas must be exactly zero and the cross-check must match).

Crossings here require a strict sign change (u_i * u_{i+1} < 0); a point
where u touches zero exactly (e.g. u(0) = 0 by symmetry) is not counted,
unlike climatology_report.py, so that crossing clusters are meaningful.

Usage:
    python scripts/anchor_compare.py TEST REF_NPZ REF_SUMMARY [--days N]
                                     [--json PATH]

TEST is a model output NetCDF (time-averaged over the last --days records)
or a climatology .npz with keys y, u, v, T as written by
analyze_original_runs.py or climatology_report.py --save-npz.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cmp_utils import report_diff, sawtooth  # noqa: E402

M_PER_DEG = np.pi * 6.371e6 / 180.0  # 111.19 km per degree latitude
Mm = 1e6
INTERIOR_M = 8e6


def near_eq_powerlaw(lat_deg, u, lo=1.0, hi=3.5):
    """Fit u = A * y^p over lat in [lo, hi] deg (NH). Returns p, A.

    Copied verbatim from analyze_original_runs.py so the exponent and
    amplitude are directly comparable to the stored summary.json values.
    """
    m = (lat_deg >= lo) & (lat_deg <= hi) & (u > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    y_m = lat_deg[m] * M_PER_DEG
    coef = np.polyfit(np.log(y_m), np.log(u[m]), 1)
    return float(coef[0]), float(np.exp(coef[1]))


def load_test(path, days):
    """Load a test climatology from output.nc (last-`days` mean) or npz."""
    path = pathlib.Path(path)
    if path.suffix == ".npz":
        d = np.load(path)
        return d["y"], d["u"], d["v"], d["T"], {"source": str(path)}
    import xarray as xr

    ds = xr.open_dataset(path, decode_timedelta=False)
    nt = ds.sizes["time"]
    navg = min(nt, days)
    nan_days = int(np.isnan(ds["u"].values).any(axis=1).sum())
    avg = slice(nt - navg, nt)
    u = ds["u"].values[avg].mean(axis=0)
    v = ds["v"].values[avg].mean(axis=0)
    T = ds["T"].values[avg].mean(axis=0)
    umax_t = np.abs(ds["u"].values).max(axis=1)
    nd = min(500, nt - 1)
    meta = {"source": str(path), "nt": nt, "navg": navg,
            "nan_days": nan_days, "drift_days": nd,
            "drift_umax": float(umax_t[-1] - umax_t[-1 - nd])}
    return ds["y"].values, u, v, T, meta


def compute_anchors(y, u, v, T):
    """Anchor dict for one climatology (all floats / lists, JSON-safe)."""
    lat = y / M_PER_DEG
    ieq = int(np.argmin(np.abs(y)))
    a = {}
    for hemi, mask in [("sh", y < 0), ("nh", y > 0)]:
        i = int(np.argmax(np.where(mask, u, -np.inf)))
        a[f"u_max_{hemi}"] = float(u[i])
        a[f"u_max_{hemi}_y_Mm"] = float(y[i] / Mm)
        j = int(np.argmax(np.where(mask, np.abs(v), -np.inf)))
        a[f"v_absmax_{hemi}"] = float(abs(v[j]))
        a[f"v_absmax_{hemi}_y_Mm"] = float(y[j] / Mm)
        a[f"v_absmax_{hemi}_sign"] = float(np.sign(v[j]))
    i = int(np.argmax(u))
    a["u_max"] = float(u[i])
    a["u_max_lat_deg"] = float(lat[i])
    i = int(np.argmin(u))
    a["u_min"] = float(u[i])
    a["u_min_y_Mm"] = float(y[i] / Mm)
    a["u_min_lat_deg"] = float(lat[i])
    a["u_eq"] = float(u[ieq])
    a["v_eq"] = float(v[ieq])
    a["T_eq"] = float(T[ieq])
    band = (np.abs(y) <= 2 * Mm) & (np.abs(y) > 0)
    k = int(np.argmin(np.where(band, u, np.inf)))
    a["neareq_umin_2Mm"] = float(u[k])
    a["neareq_umin_2Mm_y_Mm"] = float(y[k] / Mm)
    p, amp = near_eq_powerlaw(lat, u)
    a["near_eq_exponent"] = p
    a["near_eq_amp"] = amp
    interior = np.abs(y) < INTERIOR_M
    for nm, f in [("u", u), ("v", v)]:
        s = sawtooth(f)
        k = int(np.nanargmax(s))
        a[f"sawtooth_{nm}_max"] = float(s[k])
        a[f"sawtooth_{nm}_at_Mm"] = float(y[k] / Mm)
        ki = int(np.nanargmax(np.where(interior, s, np.nan)))
        a[f"sawtooth_{nm}_interior"] = float(s[ki])
        a[f"sawtooth_{nm}_interior_at_Mm"] = float(y[ki] / Mm)
    a["asym_u_max"] = float(np.max(np.abs(u - u[::-1])))
    a["asym_v_max"] = float(np.max(np.abs(v + v[::-1])))
    a["asym_T_max"] = float(np.max(np.abs(T - T[::-1])))
    strict = u[:-1] * u[1:] < 0
    idx = np.where(strict)[0]
    a["u_zero_crossings_Mm"] = [
        float((y[i] - u[i] * (y[i + 1] - y[i]) / (u[i + 1] - u[i])) / Mm)
        for i in idx
    ]
    return a


# summary.json field -> recomputed anchor field (for the ref cross-check)
SUMMARY_MAP = {
    "u_max": "u_max",
    "lat_u_max_deg": "u_max_lat_deg",
    "u_eq": "u_eq",
    "u_min": "u_min",
    "lat_u_min_deg": "u_min_lat_deg",
    "v_absmax_nh": "v_absmax_nh",
    "T_eq": "T_eq",
    "near_eq_exponent": "near_eq_exponent",
    "near_eq_amp": "near_eq_amp",
    "sawtooth_u_max": "sawtooth_u_max",
    "sawtooth_v_max": "sawtooth_v_max",
    "asym_u_max": "asym_u_max",
    "asym_v_max": "asym_v_max",
}

# rows of the anchor-delta table: (label, key, unit, show_pct)
TABLE_ROWS = [
    ("SH jet u_max", "u_max_sh", "m/s", True),
    ("SH jet y", "u_max_sh_y_Mm", "Mm", False),
    ("NH jet u_max", "u_max_nh", "m/s", True),
    ("NH jet y", "u_max_nh_y_Mm", "Mm", False),
    ("u_min (global)", "u_min", "m/s", False),
    ("u_min y", "u_min_y_Mm", "Mm", False),
    ("v_absmax SH", "v_absmax_sh", "m/s", True),
    ("v_absmax SH y", "v_absmax_sh_y_Mm", "Mm", False),
    ("v_absmax NH", "v_absmax_nh", "m/s", True),
    ("v_absmax NH y", "v_absmax_nh_y_Mm", "Mm", False),
    ("u_eq", "u_eq", "m/s", False),
    ("T_eq", "T_eq", "K", False),
    ("near-eq u_min (0<|y|<=2Mm)", "neareq_umin_2Mm", "m/s", False),
    ("near-eq exponent", "near_eq_exponent", "", False),
    ("near-eq amp", "near_eq_amp", "", True),
    ("sawtooth u global", "sawtooth_u_max", "m/s", False),
    ("sawtooth u interior", "sawtooth_u_interior", "m/s", False),
    ("sawtooth v interior", "sawtooth_v_interior", "m/s", False),
    ("parity max|u(y)-u(-y)|", "asym_u_max", "m/s", False),
    ("parity max|v(y)+v(-y)|", "asym_v_max", "m/s", False),
    ("parity max|T(y)-T(-y)|", "asym_T_max", "K", False),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("test", help="test output.nc or climatology.npz")
    p.add_argument("ref_npz", help="reference climatology.npz")
    p.add_argument("ref_summary", help="reference summary.json")
    p.add_argument("--days", type=int, default=1825,
                   help="averaging window (from record end) for a NetCDF "
                        "test input (default 1825)")
    p.add_argument("--json", default=None,
                   help="path to save the full comparison dict as JSON")
    args = p.parse_args()

    yt, ut, vt, Tt, meta = load_test(args.test, args.days)
    ref = np.load(args.ref_npz)
    yr, ur, vr, Tr = ref["y"], ref["u"], ref["v"], ref["T"]
    stored = json.loads(pathlib.Path(args.ref_summary).read_text())

    print(f"=== test: {meta['source']}")
    if "nt" in meta:
        print(f"    {meta['nt']} days saved, NaN days: {meta['nan_days']}; "
              f"averaging last {meta['navg']}")
        print(f"    max|u| drift over last {meta['drift_days']} d: "
              f"{meta['drift_umax']:+.4f} m/s")
    print(f"=== ref:  {args.ref_npz}")
    print(f"    (window per stored summary: {stored.get('avg_days_used')} "
          f"of {stored.get('days_completed')} days)")

    at = compute_anchors(yt, ut, vt, Tt)
    ar = compute_anchors(yr, ur, vr, Tr)

    # cross-check recomputed ref anchors against the stored summary
    bad = []
    for sk, ak in SUMMARY_MAP.items():
        if sk not in stored:
            continue
        sv, av = stored[sk], ar[ak]
        tol = max(1e-9, 1e-9 * abs(sv))
        if not np.isclose(av, sv, rtol=0, atol=tol):
            bad.append(f"{sk}: stored {sv!r} vs recomputed {av!r}")
    if bad:
        print("\n!! REF CROSS-CHECK FAILED (recomputed vs stored summary):")
        for b in bad:
            print(f"   {b}")
    else:
        print(f"\nref cross-check OK: {sum(k in stored for k in SUMMARY_MAP)} "
              "stored summary fields reproduced from the npz")

    print(f"\n{'anchor':>28} {'test':>12} {'ref':>12} {'delta':>12} {'pct':>8}")
    for label, key, unit, show_pct in TABLE_ROWS:
        tv, rv = at[key], ar[key]
        d = tv - rv
        pct = f"{100 * d / rv:+7.2f}%" if show_pct and rv != 0 else ""
        print(f"{label:>28} {tv:>12.5g} {rv:>12.5g} {d:>+12.4g} {pct:>8} {unit}")

    nct, ncr = at["u_zero_crossings_Mm"], ar["u_zero_crossings_Mm"]
    print(f"\nu=0 crossings (strict sign change), test ({len(nct)}): "
          + ", ".join(f"{c:+.2f}" for c in nct))
    print(f"u=0 crossings (strict sign change), ref  ({len(ncr)}): "
          + ", ".join(f"{c:+.2f}" for c in ncr))

    print("\nfield differences (test - ref), common grid:")
    diffs = {}
    inter_t, inter_r = np.abs(yt) <= INTERIOR_M, np.abs(yr) <= INTERIOR_M
    for nm, ft, fr in [("u", ut, ur), ("v", vt, vr), ("T", Tt, Tr)]:
        diffs[f"{nm}_full"] = report_diff(yr, fr, yt, ft, name=f"{nm} full domain")
        diffs[f"{nm}_interior"] = report_diff(
            yr[inter_r], fr[inter_r], yt[inter_t], ft[inter_t],
            name=f"{nm} interior |y|<=8Mm")

    if args.json:
        out = {"meta": meta, "test_anchors": at, "ref_anchors": ar,
               "ref_summary_stored": stored, "diffs": diffs,
               "ref_npz": str(args.ref_npz)}
        pathlib.Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\ncomparison JSON saved: {args.json}")


if __name__ == "__main__":
    main()
