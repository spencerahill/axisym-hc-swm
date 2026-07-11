"""V1: does the production staggered model reproduce the staggered-v patch?

The patch (scripts/run_staggered_v.py, an in-process StaggeredVModel with a
padding slot and the naive Laplacian association) removed the standing interior
v ripple 24-91x. The production StaggeredSWModel replaces that with no padding,
the symmetric-association Laplacian, and the mirror wall ghost. V1 continues the
same collocated B1 restart (day 5475) 200 days under the production model (via
--migrate-restart) and checks:

  1. the banded v-sawtooth reduction vs the collocated B1 baseline still lands
     in the 24-91x class on the gateless noise floor;
  2. climate anchors (jet, T_eq, notch) move <=0.3% from B1;
  3. mirror parity is now EXACTLY 0 (the patch drifted 3.5e-10 from its
     asymmetric Laplacian association);
  4. the production climatology agrees with the patch run to ~1e-3 m/s (not
     bitwise: the Laplacian association and wall ghost changed).

Prints a table; writes v1_analysis.json.
"""
import json
import pathlib
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ss09.sw_model import v_faces_to_centers  # noqa: E402
from ss09.read_output import load_centered  # noqa: E402
from cmp_utils import sawtooth, report_diff  # noqa: E402

Mm = 1e6
BANDS = [(0, 2), (2, 5), (5, 8)]
SUITE = pathlib.Path("model_output/formulation_suite")
PROD = SUITE / "staggered_v_prod" / "output.nc"
PATCH = SUITE / "mc_stencil" / "staggered_v" / "output.nc"
B1 = SUITE / "mc_stencil" / "b1_y0p0000_gateon_mc" / "output.nc"
GATELESS_FLOOR = 8.8e-5


def _last_mean(path, var, ndays):
    ds = xr.open_dataset(path, decode_timedelta=False)
    nt = ds.sizes["time"]
    f = ds[var].values[nt - min(ndays, nt):].mean(axis=0)
    ds.close()
    return f


def main():
    # production: center-v (load_centered reconstructs from faces) + native faces
    y, u, v_centers, T = load_centered(str(PROD), ndays=100)
    v_faces = _last_mean(PROD, "v", 100)          # native 800 faces
    dy = y[1] - y[0]
    yf = 0.5 * (y[:-1] + y[1:])

    # collocated B1 baseline (v on the 801 centers), last 1825 d as in the patch
    ur = _last_mean(B1, "u", 1825)
    vr = _last_mean(B1, "v", 1825)
    Tr = _last_mean(B1, "T", 1825)

    out = {}
    print("\n=== V1: production staggered vs patch + collocated B1 ===")

    # (1) banded v sawtooth: production faces vs B1 centers
    print("\nbanded max sawtooth(v): production vs B1 collocated")
    print(f"{'band Mm':>10} {'prod':>11} {'B1':>10} {'reduction':>10}")
    out["bands"] = []
    for a, b in BANDS:
        m_f = (np.abs(yf) >= a * Mm) & (np.abs(yf) < b * Mm)
        m_c = (np.abs(y) >= a * Mm) & (np.abs(y) < b * Mm)
        s_f = float(np.nanmax(np.where(m_f, sawtooth(v_faces), np.nan)))
        s_c = float(np.nanmax(np.where(m_c, sawtooth(vr), np.nan)))
        red = s_c / s_f if s_f > 0 else float("inf")
        print(f"[{a},{b}) {'':>3} {s_f:>11.6f} {s_c:>10.5f} {red:>9.1f}x")
        out["bands"].append({"band": [a, b], "prod": s_f, "b1": s_c, "reduction": red})

    inner = np.abs(y) < 8 * Mm
    su = float(np.nanmax(np.where(inner, sawtooth(u), np.nan)))
    sur = float(np.nanmax(np.where(inner, sawtooth(ur), np.nan)))
    out["u_sawtooth_prod"], out["u_sawtooth_b1"] = su, sur
    print(f"interior sawtooth(u): {su:.4f} vs B1 {sur:.4f}")

    # (2) climate anchors vs B1
    ieq = int(np.argmin(np.abs(y)))
    notch = (y >= -10.5 * Mm) & (y <= -7 * Mm)
    out["anchors"] = {
        "jet_umax_prod": float(u.max()), "jet_umax_b1": float(ur.max()),
        "T_eq_prod": float(T[ieq]), "T_eq_b1": float(Tr[ieq]),
        "notch_prod": float(u[notch].min()), "notch_b1": float(ur[notch].min()),
        "v_absmax_prod": float(np.abs(v_faces).max()),
    }
    jet_pct = 100 * (u.max() - ur.max()) / ur.max()
    print("\nclimate anchors (production vs B1):")
    print(f"  jet u_max   {u.max():.4f} vs {ur.max():.4f}  ({jet_pct:+.3f}%)")
    print(f"  T_eq        {T[ieq]:.4f} vs {Tr[ieq]:.4f}  ({T[ieq]-Tr[ieq]:+.4f} K)")
    print(f"  notch       {u[notch].min():.3f} vs {ur[notch].min():.3f}")
    out["jet_pct"] = jet_pct

    # (3) parity: production must be EXACTLY 0
    par_u = float(np.max(np.abs(u - u[::-1])))
    par_v = float(np.max(np.abs(v_faces + v_faces[::-1])))
    out["parity_u"], out["parity_v"] = par_u, par_v
    print(f"\nparity max|u(y)-u(-y)|      {par_u:.3g}   (target exactly 0)")
    print(f"parity max|v_f + v_f(mirr)| {par_v:.3g}   (target exactly 0)")

    # (4) agreement with the patch run (~1e-3 m/s expected)
    if PATCH.exists():
        yp, up, _, Tp = load_centered(str(PATCH), ndays=100)
        vp_faces = _last_mean(PATCH, "v", 100)[:-1]          # drop padding slot
        vp_centers = v_faces_to_centers(vp_faces)
        print("\nagreement with the patch staggered run (last 100 d):")
        du = report_diff(yp, up, y, u, "u prod vs patch")
        dv = report_diff(yp, vp_centers, y, v_centers, "v_centers prod vs patch")
        dT = report_diff(yp, Tp, y, T, "T prod vs patch")
        out["vs_patch"] = {"u": du, "v": dv, "T": dT}

    outpath = PROD.parent / "v1_analysis.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
