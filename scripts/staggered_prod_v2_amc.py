"""V2: AMC (v_d=0) verdict for the production staggered model.

At v_d=0 the EMFD is identically zero, so the model reduces to the axisymmetric
angular-momentum-conserving (AMC) balance. This is the end where the earlier k_u
eddy-viscosity and full-staggering attempts spuriously superrotated the equator,
so it is the check Spencer gated the staggered baseline on. The patch
StaggeredVModel already passed the 10/10 pre-registered AMC gate here; V2 confirms
the production StaggeredSWModel (symmetric Laplacian + mirror wall ghost) does too.

References: the validated patch AMC run (amc_staggered_vd0, which passed 10/10)
and the analytical parabola u_amc = beta*y^2/2 the equatorward cell must not
exceed. Checks: no equatorial superrotation (u_eq ~ 0), no super-AMC (u <= parabola
in the cell), jet / exponent / T_eq match the patch, mirror parity EXACTLY 0, and
agreement with the patch to ~1e-3.

Usage: python scripts/staggered_prod_v2_amc.py [--run DIR]
"""
import argparse
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
from anchor_compare import M_PER_DEG, near_eq_powerlaw  # noqa: E402

Mm = 1e6
BETA = 2e-11
DAYS = 1825
SUITE = pathlib.Path("model_output/formulation_suite")
PATCH_AMC = SUITE / "mc_stencil" / "amc_staggered_vd0"


def _patch_centers(path, ndays):
    """Center climatology from a patch (padding-slot) staggered file."""
    ds = xr.open_dataset(path / "output.nc", decode_timedelta=False)
    nt = ds.sizes["time"]
    avg = slice(nt - min(nt, ndays), nt)
    y = ds["y"].values
    u = ds["u"].values[avg].mean(axis=0)
    T = ds["T"].values[avg].mean(axis=0)
    f = ds["v"].values[avg].mean(axis=0)[:-1]  # drop padding slot
    ds.close()
    return y, u, v_faces_to_centers(f), T


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=str(SUITE / "staggered_v_prod" / "v2_amc"))
    args = p.parse_args()
    run = pathlib.Path(args.run)

    y, u, v, T = load_centered(str(run / "output.nc"), ndays=DAYS)
    ds = xr.open_dataset(run / "output.nc", decode_timedelta=False)
    nan_days = int(np.isnan(ds["u"].values).any(axis=1).sum())
    umax_t = np.abs(ds["u"].values).max(axis=1)
    nd = min(500, ds.sizes["time"] - 1)
    drift = float(umax_t[-1] - umax_t[-1 - nd])
    ds.close()

    ieq = int(np.argmin(np.abs(y)))
    u_amc = BETA * y**2 / 2.0
    cell = np.abs(y) <= 8 * Mm
    super_amc = float(np.max((u - u_amc)[cell]))  # <= 0 means sub-AMC
    lat_deg = y / M_PER_DEG
    p_nh, A_nh = near_eq_powerlaw(lat_deg, u)

    # NH jet
    nh = y > 0
    jet_mag = float(u[nh].max())
    jet_lat = float(y[nh][np.argmax(u[nh])] / Mm)
    par_u = float(np.max(np.abs(u - u[::-1])))
    v_faces = xr.open_dataset(run / "output.nc", decode_timedelta=False)["v"].values[-DAYS:].mean(axis=0)
    par_v = float(np.max(np.abs(v_faces + v_faces[::-1])))

    yp, up, vp, Tp = _patch_centers(PATCH_AMC, DAYS)

    out = {
        "nan_days": nan_days, "u_eq": float(u[ieq]), "super_amc_max": super_amc,
        "jet_mag": jet_mag, "jet_lat_Mm": jet_lat, "exponent": p_nh,
        "T_eq": float(T[ieq]), "parity_u": par_u, "parity_v": par_v,
        "drift_500d": drift,
    }
    print("\n=== V2: production staggered v_d=0 (AMC) ===")
    print(f"  NaN days                 {nan_days}                (want 0)")
    print(f"  u_eq                     {u[ieq]:+.6f} m/s      (want ~0, no superrotation)")
    print(f"  super-AMC max(u-u_amc)   {super_amc:+.4f} m/s      (want <= 0 in the cell)")
    print(f"  NH jet                   {jet_mag:.3f} m/s at {jet_lat:+.3f} Mm")
    print(f"  near-eq exponent         {p_nh:.3f}            (patch AMC 1.926)")
    print(f"  T_eq                     {T[ieq]:.4f} K")
    print(f"  parity max|u(y)-u(-y)|   {par_u:.3g}            (want exactly 0)")
    print(f"  parity max|v_f+v_f(mirr)|{par_v:.3g}            (want exactly 0)")
    print(f"  drift over last 500 d    {drift:+.4g} m/s")

    print("\nagreement with the validated patch AMC run (last 1825 d):")
    out["vs_patch"] = {
        "u": report_diff(yp, up, y, u, "u prod vs patch AMC"),
        "v": report_diff(yp, vp, y, v, "v_centers prod vs patch AMC"),
        "T": report_diff(yp, Tp, y, T, "T prod vs patch AMC"),
    }

    (run / "v2_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {run / 'v2_analysis.json'}")


if __name__ == "__main__":
    main()
