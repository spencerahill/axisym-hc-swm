"""Item (iii) of the spherical-geometry plan, first half: the timestep ceiling.

Cold-starts the model at the production parameters for a fixed number of
days at each candidate timestep (every candidate divides 86400 s) and judges
each run by its output file: the number of days stored and whether every
stored field is finite. The ceiling is the largest candidate that completes.
Runs the sphere and its beta-plane twin, whose definition is the plan's item
(iv): beta = 2 Omega / a, walls at +-pi a / 2, and the SB08 profile with
y_1 = pi a / 2, which is the spherical forcing written in phi = y / a.

Each run writes to its own directory under --out-dir, and the per-day cost of
each completed run is printed from its wall time.

Usage:
    python validation/sphere_dt_bisection.py --out-dir DIR --ny-sphere 325 \
        --ny-twin 509 --dts 30 40 45 48 54 60 --ndays 400
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import xarray as xr

A = 6.371e6
OMEGA = 7.292115e-5
# The repository this script belongs to; the model runs as `python -m
# ss09.cli` started there, so its own ss09 is the one imported.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sphere_args(ny):
    return ["--geometry", "sphere", "--ny", str(ny)]


def twin_args(ny):
    return [
        "--ny", str(ny), "--beta", repr(2 * OMEGA / A),
        "--domain-size", repr(np.pi * A), "--theta-e-type", "SB08",
        "--y1", repr(np.pi * A / 2),
    ]


def smoke(label, geometry_args, dt, ndays, out_dir):
    run_dir = os.path.join(out_dir, f"{label}_dt{dt}")
    os.makedirs(run_dir, exist_ok=True)
    out = os.path.join(run_dir, "output.nc")
    if os.path.exists(out):
        os.remove(out)
    t0 = time.time()
    subprocess.run(
        [sys.executable, "-m", "ss09.cli", "--backend", "numba", "--dt", str(dt),
         "--ndays", str(ndays), "--output-path", out] + geometry_args,
        check=True, capture_output=True, text=True, cwd=REPO,
    )
    wall = time.time() - t0
    ds = xr.open_dataset(out, decode_times=False)
    days = ds.sizes["time"]
    finite = all(bool(np.all(np.isfinite(ds[v].values))) for v in ("u", "v", "T"))
    umax = float(np.max(np.abs(ds.u.values[-1]))) if days else np.nan
    ds.close()
    ok = days == ndays and finite
    print(f"  {label} dt={dt:3d}: {days:4d} of {ndays} days stored, all finite: "
          f"{finite}, max|u| on the last day {umax:7.2f} m/s, "
          f"{wall / max(days, 1):.4f} s per model day -> {'completes' if ok else 'FAILS'}",
          flush=True)
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ny-sphere", type=int, default=325)
    p.add_argument("--ny-twin", type=int, default=509)
    p.add_argument("--dts", type=int, nargs="+", required=True)
    p.add_argument("--ndays", type=int, default=400)
    args = p.parse_args()
    for dt in args.dts:
        if 86400 % dt:
            sys.exit(f"dt={dt} does not divide 86400")
    for label, geo in (
        (f"sphere_ny{args.ny_sphere}", sphere_args(args.ny_sphere)),
        (f"twin_ny{args.ny_twin}", twin_args(args.ny_twin)),
    ):
        print(f"{label}:", flush=True)
        passed = [dt for dt in sorted(args.dts) if smoke(label, geo, dt, args.ndays, args.out_dir)]
        failed = sorted(set(args.dts) - set(passed))
        print(f"  ceiling among the candidates: {max(passed) if passed else None} s"
              f" (failed: {failed})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
