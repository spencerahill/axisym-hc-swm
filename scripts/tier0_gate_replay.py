"""Replay the slow-drift gate on the full tier-0 1a/1b daily histories
(cold-start file days 0-959 + extension days 960-6000) to compute the
exact day a gated run stops (memo run record, 2026-07-17). The gated
trajectory is bitwise-identical to the recorded one, so the replayed
firing day is the gated run's actual stop day; confirmed by the
_gatecheck.nc cold-start run (stopped day 3683). Run from the repo
root."""

import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view

from ss09.read_output import load_centered

BASE = "model_output/fixed_ro_suite/tier0_amc_edge/"
RUNS = {
    "1a": ("amc_ss09_ny801_dt30_vd0.nc", "amc_ss09_ny801_dt30_vd0_day6000.nc"),
    "1b": ("amc_ss09_ny801_dt30_vd0_novert.nc",
           "amc_ss09_ny801_dt30_vd0_novert_day6000.nc"),
}
CLASSIC_WINDOW, CLASSIC_THRESH = 30, 1e-4     # run 1a/1b detector settings
SLOW_WINDOW, SLOW_THRESH = 1158, 0.002        # auto window at eps_u = 1e-8
THETA_00 = 330.0


def series_from(path):
    y, u, v, temp = load_centered(BASE + path)
    ds = xr.open_dataset(BASE + path, decode_timedelta=False)
    jet = ds["north_jet_lat"].values
    ds.close()
    i0 = int(np.argmin(np.abs(y)))
    theta = 1.6 * temp
    ke = np.mean(u**2 + v**2, axis=1)
    tvar = np.std(theta, axis=1)
    v_absmax = np.max(np.abs(v), axis=1)
    depression = THETA_00 - theta[:, i0]
    return jet, v_absmax, depression, ke, tvar


def rel_range(series, window):
    """Trailing-window (max-min)/|mean|, aligned to the window's last day."""
    sw = sliding_window_view(series, window)
    return (sw.max(axis=1) - sw.min(axis=1)) / np.abs(sw.mean(axis=1))


for tag, (orig, ext) in RUNS.items():
    parts = [series_from(orig), series_from(ext)]
    jet, v_absmax, depression, ke, tvar = (
        np.concatenate([p[i] for p in parts]) for i in range(5)
    )
    ndays = jet.size
    days = np.arange(ndays)

    classic_ok = np.zeros(ndays, dtype=bool)
    classic_ok[CLASSIC_WINDOW - 1:] = (
        (rel_range(ke, CLASSIC_WINDOW) < CLASSIC_THRESH)
        & (rel_range(tvar, CLASSIC_WINDOW) < CLASSIC_THRESH)
    )
    slow_ok = np.zeros(ndays, dtype=bool)
    finite = sliding_window_view(
        np.isfinite(jet) & np.isfinite(v_absmax) & np.isfinite(depression),
        SLOW_WINDOW,
    ).all(axis=1)
    slow_ok[SLOW_WINDOW - 1:] = (
        finite
        & (rel_range(jet, SLOW_WINDOW) < SLOW_THRESH)
        & (rel_range(v_absmax, SLOW_WINDOW) < SLOW_THRESH)
        & (rel_range(depression, SLOW_WINDOW) < SLOW_THRESH)
    )
    both = classic_ok & slow_ok
    if not both.any():
        print(f"{tag}: gate never fires within the {ndays}-d record")
        continue
    fire = int(days[both][0])
    print(f"{tag}: gated run stops day {fire} "
          f"(classic alone: day {int(days[classic_ok][0])}); "
          f"vs fixed 6000 d, saves {100 * (1 - fire / 6000):.0f}%")
    for name, s in (("jet", jet), ("v_absmax", v_absmax),
                    ("depression", depression)):
        rr = rel_range(s, SLOW_WINDOW)
        first = int(days[SLOW_WINDOW - 1:][rr < SLOW_THRESH][0])
        print(f"    {name:10s} range<0.2% from day {first}")
