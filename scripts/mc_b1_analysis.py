"""B1 verdict for the MC-limited EMFD stencil (pre-registered criteria).

Evaluates the decisive 15-yr gate-on+mc run at y0=0 against BOTH
references: the validated gateless vd25_vert climatology (physics target)
and the Tier-1 gate-on+upwind run (stencil comparison), applying the
pre-registered criteria of ~/.claude/plans/mc-limited-emfd-stencil.md:

  exponent  dp <= 0.05 vs gateless on [1,3.5] and [1.5,7] deg; amp within 30%
  anchors   jets within 5%, |dlat| <= 0.5 Mm, v_absmax within 5%,
            |dT_eq| <= 0.5 K (vs gateless, Tier-1's passing envelope)
  health    parity max|u(y)-u(-y)| <= 1e-6; |drift| <= 0.05 m/s per 500 d;
            no NaN
  artifacts interior (|y| <= 8 Mm) sawtooth(u) <= 0.1 m/s; whole-domain
            extrema sweep vs both references (printed for judgment)
  notch     REPORT depth/width/position vs upwind; STOP flag if depth
            changes > 5 m/s or width > 30%
  ripple    REPORT banded sawtooth(v) vs upwind (no target, expect <=)

Usage:
    python scripts/mc_b1_analysis.py [--test PATH] [--json PATH]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anchor_compare import M_PER_DEG, compute_anchors, near_eq_powerlaw  # noqa: E402
from cmp_utils import report_diff, sawtooth  # noqa: E402

Mm = 1e6
DAYS = 1825
SUITE = pathlib.Path("model_output/formulation_suite")
TEST_NC = SUITE / "mc_stencil/b1_y0p0000_gateon_mc/output.nc"
UPWIND_NC = SUITE / "tier1_y0p0000_gateon_upwind/output.nc"
REF_NPZ = pathlib.Path(
    "model_output/validation_20260709/runs/vd25_vert/climatology.npz")
FIT_WINDOWS = [(1.0, 3.5), (1.5, 7.0)]
RIPPLE_BANDS = [(0, 2), (2, 5), (5, 8)]


def load_nc(path):
    """Last-DAYS-mean climatology + daily max|u| series + NaN-day count."""
    ds = xr.open_dataset(path, decode_timedelta=False)
    nt = ds.sizes["time"]
    avg = slice(nt - min(nt, DAYS), nt)
    u_t = ds["u"].values
    out = {
        "y": ds["y"].values,
        "u": u_t[avg].mean(axis=0),
        "v": ds["v"].values[avg].mean(axis=0),
        "T": ds["T"].values[avg].mean(axis=0),
        "umax_t": np.abs(u_t).max(axis=1),
        "nan_days": int(np.isnan(u_t).any(axis=1).sum()),
        "nt": nt,
    }
    ds.close()
    return out


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
        "w_deep_pts": int(np.sum(ub < -5)),
        "w_deep_km": float(np.sum(ub < -5) * dy / 1e3),
        "w_easterly_km": float(np.sum(ub < 0) * dy / 1e3),
    }


def banded_ripple(y, v):
    sv = sawtooth(v)
    return [float(np.nanmax(np.where(
        (np.abs(y) >= a * Mm) & (np.abs(y) < b * Mm), sv, np.nan)))
        for a, b in RIPPLE_BANDS]


def extrema_row(y, f):
    imin, imax = int(np.argmin(f)), int(np.argmax(f))
    return (f[imin], y[imin] / Mm, f[imax], y[imax] / Mm)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", default=str(TEST_NC))
    p.add_argument("--json", default=None)
    args = p.parse_args()

    test = load_nc(args.test)
    up = load_nc(UPWIND_NC)
    ref = np.load(REF_NPZ)
    ref = {"y": ref["y"], "u": ref["u"], "v": ref["v"], "T": ref["T"]}

    at = compute_anchors(test["y"], test["u"], test["v"], test["T"])
    ar = compute_anchors(ref["y"], ref["u"], ref["v"], ref["T"])
    au = compute_anchors(up["y"], up["u"], up["v"], up["T"])

    checks = []

    def check(name, value, ok, detail=""):
        checks.append((name, value, bool(ok), detail))

    # --- exponent vs gateless ref, both windows; amp within 30%
    lat_t = test["y"] / M_PER_DEG
    lat_r = ref["y"] / M_PER_DEG
    for lo, hi in FIT_WINDOWS:
        pt, _ = near_eq_powerlaw(lat_t, test["u"], lo, hi)
        pr, _ = near_eq_powerlaw(lat_r, ref["u"], lo, hi)
        check(f"exponent dp [{lo},{hi}] deg", f"{pt:.3f} vs {pr:.3f}",
              abs(pt - pr) <= 0.05, f"dp={pt - pr:+.3f} (limit 0.05)")
    amp_ratio = at["near_eq_amp"] / ar["near_eq_amp"]
    check("near_eq_amp ratio", f"{at['near_eq_amp']:.3g} vs "
          f"{ar['near_eq_amp']:.3g}", 0.7 <= amp_ratio <= 1.3,
          f"ratio={amp_ratio:.3f} (limit [0.7,1.3])")

    # --- anchors vs gateless ref (Tier-1's passing envelope)
    jet_pct = max(abs(at[f"u_max_{h}"] / ar[f"u_max_{h}"] - 1)
                  for h in ("sh", "nh"))
    check("jets within 5%", f"{at['u_max_nh']:.3f} vs {ar['u_max_nh']:.3f}",
          jet_pct <= 0.05, f"max dev {100 * jet_pct:.2f}%")
    dlat = max(abs(at[f"u_max_{h}_y_Mm"] - ar[f"u_max_{h}_y_Mm"])
               for h in ("sh", "nh"))
    check("jet |dlat| <= 0.5 Mm", f"{at['u_max_nh_y_Mm']:.2f} vs "
          f"{ar['u_max_nh_y_Mm']:.2f} Mm", dlat <= 0.5, f"max {dlat:.3f} Mm")
    v_pct = max(abs(at[f"v_absmax_{h}"] / ar[f"v_absmax_{h}"] - 1)
                for h in ("sh", "nh"))
    check("v_absmax within 5%", f"{at['v_absmax_nh']:.4f} vs "
          f"{ar['v_absmax_nh']:.4f}", v_pct <= 0.05,
          f"max dev {100 * v_pct:.2f}%")
    dT = abs(at["T_eq"] - ar["T_eq"])
    check("|dT_eq| <= 0.5 K", f"{at['T_eq']:.3f} vs {ar['T_eq']:.3f}",
          dT <= 0.5, f"dT={dT:.4f} K")

    # --- health
    check("parity max|u(y)-u(-y)| <= 1e-6", f"{at['asym_u_max']:.3g}",
          at["asym_u_max"] <= 1e-6)
    nd = min(500, test["nt"] - 1)
    drift = float(test["umax_t"][-1] - test["umax_t"][-1 - nd])
    check("|drift| <= 0.05 m/s per 500 d", f"{drift:+.4f}",
          abs(drift) <= 0.05, f"over last {nd} d")
    check("no NaN days", str(test["nan_days"]), test["nan_days"] == 0)

    # --- artifacts
    check("interior sawtooth(u) <= 0.1 m/s",
          f"{at['sawtooth_u_interior']:.4f}",
          at["sawtooth_u_interior"] <= 0.1,
          f"at {at['sawtooth_u_interior_at_Mm']:+.2f} Mm")

    # --- notch vs upwind (REPORT + STOP flags)
    notch = {}
    for hemi in ("sh", "nh"):
        nt_ = notch_metrics(test["y"], test["u"], hemi)
        nu_ = notch_metrics(up["y"], up["u"], hemi)
        notch[hemi] = {"mc": nt_, "upwind": nu_}
        d_depth = abs(nt_["depth"] - nu_["depth"])
        w_ratio = (nt_["w_deep_km"] / nu_["w_deep_km"]
                   if nu_["w_deep_km"] else np.nan)
        check(f"notch {hemi} depth shift <= 5 m/s (STOP flag)",
              f"{nt_['depth']:.2f} vs upwind {nu_['depth']:.2f}",
              d_depth <= 5, f"|d|={d_depth:.2f}")
        check(f"notch {hemi} width within 30% (STOP flag)",
              f"{nt_['w_deep_km']:.0f} vs upwind {nu_['w_deep_km']:.0f} km",
              0.7 <= w_ratio <= 1.3, f"ratio={w_ratio:.3f}")

    print("=== B1 pre-registered criteria ===")
    npass = 0
    for name, value, ok, detail in checks:
        tag = "PASS" if ok else "FAIL"
        npass += ok
        print(f"[{tag}] {name:>42}: {value}  {detail}")
    print(f"--- {npass}/{len(checks)} criteria pass")

    # --- notch geometry report
    print("\n=== notch geometry (REPORT; expect ~ -28+-3 m/s, ~600 km, "
          "~8.8-8.9 Mm) ===")
    print(f"{'arm':>10} {'depth':>8} {'pos Mm':>8} {'w(u<-5) pts':>12} "
          f"{'w(u<-5) km':>11} {'w(u<0) km':>10}")
    for hemi in ("sh", "nh"):
        for arm in ("mc", "upwind"):
            n = notch[hemi][arm]
            print(f"{hemi + ' ' + arm:>10} {n['depth']:>8.2f} "
                  f"{n['pos_Mm']:>8.2f} {n['w_deep_pts']:>12d} "
                  f"{n['w_deep_km']:>11.0f} {n['w_easterly_km']:>10.0f}")

    # --- ripple report
    rip_t = banded_ripple(test["y"], test["v"])
    rip_u = banded_ripple(up["y"], up["v"])
    print("\n=== banded max sawtooth(v), REPORT (expect mc <= upwind) ===")
    print(f"{'band Mm':>10} {'mc':>10} {'upwind':>10}")
    for (a, b), rt, ru in zip(RIPPLE_BANDS, rip_t, rip_u):
        print(f"[{a},{b}) {'':>3} {rt:>10.5f} {ru:>10.5f}")

    # --- whole-domain extrema sweep (JUDGE)
    print("\n=== whole-domain extrema sweep (JUDGE: no new extrema) ===")
    print(f"{'field/arm':>16} {'min':>9} {'at Mm':>7} {'max':>9} {'at Mm':>7}")
    for nm in ("u", "v", "T"):
        for label, d in [("mc", test), ("gateless", ref), ("upwind", up)]:
            mn, ymn, mx, ymx = extrema_row(d["y"], d[nm])
            print(f"{nm + ' ' + label:>16} {mn:>9.3f} {ymn:>7.2f} "
                  f"{mx:>9.3f} {ymx:>7.2f}")

    print("\n=== field differences, common grid ===")
    diffs = {}
    for nm in ("u", "v", "T"):
        diffs[f"{nm}_vs_gateless"] = report_diff(
            ref["y"], ref[nm], test["y"], test[nm],
            name=f"{nm} mc vs gateless (full)")
        diffs[f"{nm}_vs_upwind"] = report_diff(
            up["y"], up[nm], test["y"], test[nm],
            name=f"{nm} mc vs upwind (full)")
    inter_t = np.abs(test["y"]) <= 8 * Mm
    inter_r = np.abs(ref["y"]) <= 8 * Mm
    inter_u = np.abs(up["y"]) <= 8 * Mm
    for nm in ("u", "v", "T"):
        diffs[f"{nm}_vs_gateless_int"] = report_diff(
            ref["y"][inter_r], ref[nm][inter_r],
            test["y"][inter_t], test[nm][inter_t],
            name=f"{nm} mc vs gateless (|y|<=8Mm)")
        diffs[f"{nm}_vs_upwind_int"] = report_diff(
            up["y"][inter_u], up[nm][inter_u],
            test["y"][inter_t], test[nm][inter_t],
            name=f"{nm} mc vs upwind (|y|<=8Mm)")

    crossings = {"mc": at["u_zero_crossings_Mm"],
                 "gateless": ar["u_zero_crossings_Mm"],
                 "upwind": au["u_zero_crossings_Mm"]}
    print("\nu=0 strict crossings:")
    for label, c in crossings.items():
        print(f"  {label} ({len(c)}): " + ", ".join(f"{x:+.2f}" for x in c))

    if args.json:
        out = {
            "checks": [{"name": n, "value": v, "pass": ok, "detail": d}
                       for n, v, ok, d in checks],
            "test_anchors": at, "gateless_anchors": ar, "upwind_anchors": au,
            "notch": notch,
            "ripple_bands_Mm": RIPPLE_BANDS,
            "ripple_mc": rip_t, "ripple_upwind": rip_u,
            "diffs": diffs, "crossings": crossings,
            "drift_umax_500d": drift, "nan_days": test["nan_days"],
        }
        pathlib.Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nJSON saved: {args.json}")


if __name__ == "__main__":
    main()
