"""When does the standing v ripple reach its equilibrium amplitude?

Computes, from a run's daily output, the day-by-day banded max
sawtooth(v) and terminus-notch depth, then reports the first day after
which each stays within +-20% (ripple) / +-10% (notch) of its
last-1825-d equilibrium value. Sets the minimum useful length of a
cold-start run whose target is the ripple amplitude (e.g. a short
ny=1601 probe).

Usage:
    python scripts/ripple_equilibration.py [OUTPUT_NC]
"""
import pathlib
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cmp_utils import sawtooth  # noqa: E402

Mm = 1e6
BANDS = [(0, 2), (2, 5), (5, 8)]
DEFAULT = ("model_output/formulation_suite/mc_stencil/"
           "b1_y0p0000_gateon_mc/output.nc")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    ds = xr.open_dataset(path, decode_timedelta=False)
    y = ds["y"].values
    v = ds["v"].values
    u = ds["u"].values
    nt = v.shape[0]

    band_masks = [(np.abs(y) >= a * Mm) & (np.abs(y) < b * Mm)
                  for a, b in BANDS]
    notch_band = (y >= -10.5 * Mm) & (y <= -7 * Mm)

    rip = np.empty((nt, len(BANDS)))
    notch = np.empty(nt)
    for t in range(nt):
        sv = sawtooth(v[t])
        for k, m in enumerate(band_masks):
            rip[t, k] = np.nanmax(np.where(m, sv, np.nan))
        notch[t] = u[t][notch_band].min()

    eq = slice(nt - min(nt, 1825), nt)
    print(f"{path}: {nt} days")
    print(f"{'quantity':>16} {'equil value':>12} {'stable from day':>16}")
    for k, (a, b) in enumerate(BANDS):
        target = rip[eq, k].mean()
        ok = np.abs(rip[:, k] - target) <= 0.2 * abs(target)
        # first day from which the criterion holds for 30 consecutive days
        day = next((t for t in range(nt - 30)
                    if ok[t:t + 30].all()), None)
        print(f"  v rip [{a},{b}) Mm {target:>12.5f} {str(day):>16}")
    target = notch[eq].mean()
    ok = np.abs(notch - target) <= 0.1 * abs(target)
    day = next((t for t in range(nt - 30) if ok[t:t + 30].all()), None)
    print(f"{'notch depth':>16} {target:>12.2f} {str(day):>16}")
    ds.close()


if __name__ == "__main__":
    main()
