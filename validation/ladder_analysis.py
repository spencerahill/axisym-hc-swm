"""Resolution-ladder analysis for the unified-formulation suite (R2/R3).

For each available (ny, formulation) 15-yr run, plus the ny=801 pair
(Tier-1 gate-on extension vs the validated gateless vd25_vert
climatology), computes the dy-scaling diagnostics that arbitrate the
Tier-1 failures:

1. near-equator exponent per arm and the deficit between formulations at
   matched ny, on fit windows valid at every resolution, plus the u
   ratio gate-on/gateless at fixed physical y (315 and 630 km, gridpoints
   at all three resolutions);
2. terminus-notch geometry in the gate-on arms: depth, width in
   gridpoints and in km (width in ~constant gridpoints means a
   grid-collapsed numerical feature; ~constant km means physics);
3. interior grid-scale ripple: banded sawtooth(v) and interior
   sawtooth(u);
4. interior (|y| <= 8 Mm) max|du| between formulations at matched ny
   (the stencil-footprint story predicts ~linear-in-dy, 4:2:1 for
   201:401:801).

Runs not yet on disk are skipped with a note, so this can be run before
the full ladder lands.
"""
import pathlib
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anchor_compare import M_PER_DEG, near_eq_powerlaw  # noqa: E402
from cmp_utils import report_diff, sawtooth  # noqa: E402

Mm = 1e6
DAYS = 1825
SUITE = pathlib.Path("model_output/formulation_suite")
RUNS = {
    (201, "gateon"): SUITE / "ladder_ny201_gateon/output.nc",
    (201, "gateless"): SUITE / "ladder_ny201_gateless/output.nc",
    (401, "gateon"): SUITE / "ladder_ny401_gateon/output.nc",
    (401, "gateless"): SUITE / "ladder_ny401_gateless/output.nc",
    (801, "gateon"): SUITE / "tier1_y0p0000_gateon_upwind/output.nc",
}
REF801 = pathlib.Path("model_output/validation_20260709/runs/vd25_vert/climatology.npz")
FIT_WINDOWS = [(1.5, 7.0), (2.0, 7.0), (1.0, 3.5)]
Y_PROBES = [315.0e3, 630.0e3]


def load(path):
    if path.suffix == ".npz":
        d = np.load(path)
        return d["y"], d["u"], d["v"]
    ds = xr.open_dataset(path, decode_timedelta=False)
    nt = ds.sizes["time"]
    avg = slice(nt - min(nt, DAYS), nt)
    return (ds["y"].values, ds["u"].values[avg].mean(axis=0),
            ds["v"].values[avg].mean(axis=0))


def main():
    fields = {}
    for key, path in RUNS.items():
        if path.exists():
            fields[key] = load(path)
        else:
            print(f"[not yet on disk: {key} -> {path}]")
    fields[(801, "gateless")] = load(REF801)

    print("\n=== notch geometry (gate-on arms), SH band [-10.5, -7] Mm ===")
    print(f"{'ny':>4} {'dy km':>6} {'depth m/s':>10} {'at Mm':>7} "
          f"{'w(u<-5) pts':>12} {'w(u<-5) km':>11} {'w(u<0) km':>10}")
    for ny in (201, 401, 801):
        if (ny, "gateon") not in fields:
            continue
        y, u, _ = fields[(ny, "gateon")]
        dy = y[1] - y[0]
        band = (y >= -10.5 * Mm) & (y <= -7 * Mm)
        ub, yb = u[band], y[band]
        i = int(np.argmin(ub))
        n5 = int(np.sum(ub < -5))
        n0 = int(np.sum(ub < 0))
        print(f"{ny:>4} {dy/1e3:>6.1f} {ub[i]:>10.2f} {yb[i]/Mm:>7.2f} "
              f"{n5:>12d} {n5*dy/1e3:>11.0f} {n0*dy/1e3:>10.0f}")

    print("\n=== near-equator exponent p per arm ===")
    hdr = " ".join(f"{f'[{lo},{hi}]':>12}" for lo, hi in FIT_WINDOWS)
    print(f"{'ny':>4} {'arm':>9} {hdr}")
    for ny in (201, 401, 801):
        for arm in ("gateless", "gateon"):
            if (ny, arm) not in fields:
                continue
            y, u, _ = fields[(ny, arm)]
            ps = [near_eq_powerlaw(y / M_PER_DEG, u, lo, hi)[0]
                  for lo, hi in FIT_WINDOWS]
            print(f"{ny:>4} {arm:>9} " + " ".join(f"{p:>12.3f}" for p in ps))

    print("\n=== u(gate-on)/u(gateless) at fixed y (probe of the stencil bias) ===")
    print(f"{'ny':>4} " + " ".join(f"{f'y={yp/1e3:.0f}km':>12}" for yp in Y_PROBES))
    for ny in (201, 401, 801):
        if (ny, "gateon") not in fields or (ny, "gateless") not in fields:
            continue
        yon, uon, _ = fields[(ny, "gateon")]
        yoff, uoff, _ = fields[(ny, "gateless")]
        vals = []
        for yp in Y_PROBES:
            i, j = int(np.argmin(np.abs(yon - yp))), int(np.argmin(np.abs(yoff - yp)))
            vals.append(uon[i] / uoff[j])
        print(f"{ny:>4} " + " ".join(f"{v:>12.4f}" for v in vals))

    print("\n=== interior ripple (gate-on arms): banded max sawtooth(v); interior max sawtooth(u) ===")
    bands = [(0, 2), (2, 5), (5, 8)]
    print(f"{'ny':>4} " + " ".join(f"{f'v [{a},{b})Mm':>12}" for a, b in bands)
          + f" {'u interior':>12}")
    for ny in (201, 401, 801):
        if (ny, "gateon") not in fields:
            continue
        y, u, v = fields[(ny, "gateon")]
        sv, su = sawtooth(v), sawtooth(u)
        vals = [np.nanmax(np.where((np.abs(y) >= a * Mm) & (np.abs(y) < b * Mm), sv, np.nan))
                for a, b in bands]
        inner = np.abs(y) < 8 * Mm
        print(f"{ny:>4} " + " ".join(f"{x:>12.5f}" for x in vals)
              + f" {np.nanmax(np.where(inner, su, np.nan)):>12.5f}")

    print("\n=== interior (|y|<=8 Mm) max|du| between formulations at matched ny ===")
    for ny in (201, 401, 801):
        if (ny, "gateon") not in fields or (ny, "gateless") not in fields:
            continue
        yon, uon, _ = fields[(ny, "gateon")]
        yoff, uoff, _ = fields[(ny, "gateless")]
        mon, moff = np.abs(yon) <= 8 * Mm, np.abs(yoff) <= 8 * Mm
        report_diff(yoff[moff], uoff[moff], yon[mon], uon[mon],
                    name=f"ny={ny} interior u")


if __name__ == "__main__":
    main()
