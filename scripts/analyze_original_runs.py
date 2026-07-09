"""Reduce runs of the original SS_Model.py and extract Zhang et al. 2025 anchors.

For each run directory (containing output.nc from run_original_ss_model.py),
compute the last-N-day climatology and the quantities needed to compare against
Zhang et al. (2025) Figs. 2-4: jet magnitude/position, v extrema, T anchors,
the near-equator power-law scaling of u, whole-domain 2dy-mode diagnostics,
steadiness drift, and hemispheric symmetry. Writes <rundir>/summary.json and
<rundir>/climatology.npz; prints the summary.
"""
import argparse
import json
import pathlib

import numpy as np
import xarray as xr

from cmp_utils import sawtooth, flank_mode

M_PER_DEG = np.pi * 6.371e6 / 180.0  # 111.19 km per degree latitude
BETA = 2e-11


def near_eq_powerlaw(lat_deg, u, lo=1.0, hi=3.5):
    """Fit u = A * y^p over lat in [lo, hi] deg (NH). Returns p, A."""
    m = (lat_deg >= lo) & (lat_deg <= hi) & (u > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    y_m = lat_deg[m] * M_PER_DEG
    coef = np.polyfit(np.log(y_m), np.log(u[m]), 1)
    return float(coef[0]), float(np.exp(coef[1]))


def analyze(rundir, avg_days):
    rundir = pathlib.Path(rundir)
    ds = xr.open_dataset(rundir / "output.nc", decode_times=False)
    nt = ds.sizes["time"]
    clim = ds.isel(time=slice(-min(avg_days, nt), None)).mean("time")
    y = ds["y"].values
    lat = y / M_PER_DEG
    u = clim["u"].values
    v = clim["v"].values
    temp = clim["T"].values

    nh = lat > 0
    iu = int(np.argmax(u))
    iv_nh = int(np.argmax(np.where(nh, np.abs(v), -np.inf)))
    i_eq = int(np.argmin(np.abs(y)))
    i80n = int(np.argmin(np.abs(lat - 80.0)))

    # steadiness: per-day max u, first vs second half of the averaging window
    umax_daily = ds["u"].max("y")
    w = min(avg_days, nt) // 2
    drift = float(umax_daily.isel(time=slice(-w, None)).mean()
                  - umax_daily.isel(time=slice(-2 * w, -w)).mean())

    p_exp, p_amp = near_eq_powerlaw(lat, u)

    saw_u = sawtooth(u)
    saw_v = sawtooth(v)
    isu = int(np.nanargmax(saw_u))
    isv = int(np.nanargmax(saw_v))

    summary = {
        "days_completed": int(nt),
        "nan_truncated": bool(nt < int(ds.time.values[-1] + 0.6) if nt else True),
        "avg_days_used": int(min(avg_days, nt)),
        "u_max": float(u[iu]),
        "lat_u_max_deg": float(lat[iu]),
        "u_eq": float(u[i_eq]),
        "u_min": float(u.min()),
        "lat_u_min_deg": float(lat[int(np.argmin(u))]),
        "v_absmax_nh": float(np.abs(v[iv_nh])),
        "v_nh_sign": float(np.sign(v[iv_nh])),
        "lat_v_absmax_deg": float(lat[iv_nh]),
        "T_eq": float(temp[i_eq]),
        "T_80N": float(temp[i80n]),
        "near_eq_exponent": p_exp,
        "near_eq_amp": p_amp,
        "near_eq_amp_over_beta_half": p_amp / (BETA / 2) if np.isfinite(p_amp) else None,
        "near_eq_amp_over_beta_third": p_amp / (BETA / 3) if np.isfinite(p_amp) else None,
        "sawtooth_u_max": float(saw_u[isu]),
        "sawtooth_u_at_deg": float(lat[isu]),
        "sawtooth_v_max": float(saw_v[isv]),
        "sawtooth_v_at_deg": float(lat[isv]),
        "flank_mode_u": flank_mode(y, u),
        "asym_u_max": float(np.max(np.abs(u - u[::-1]))),
        "asym_v_max": float(np.max(np.abs(v + v[::-1]))),
        "drift_umax_half_window": drift,
    }

    np.savez(rundir / "climatology.npz", lat=lat, y=y, u=u, v=v, T=temp,
             theta_e=ds["theta_e"].values)
    (rundir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rundirs", nargs="+")
    p.add_argument("--avg-days", type=int, default=1825)
    args = p.parse_args()
    for rd in args.rundirs:
        print(f"\n=== {pathlib.Path(rd).name} ===")
        s = analyze(rd, args.avg_days)
        print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
