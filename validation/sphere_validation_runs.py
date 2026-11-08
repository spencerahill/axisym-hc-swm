"""The validation runs of the spherical-geometry plan, items (i) to (iv).

All runs use the numba backend (bitwise identical to the numpy reference),
cold-start at the production parameters, and integrate for a fixed number of
days with no early stop; the spherical forcing is Delta_y = 50 K with its
heating maximum on the equator unless stated. Each run writes into its own
directory under --out-root (restart files are named by day only, so runs
sharing a directory would overwrite each other's), and a run whose output
file already exists is skipped, so the driver can be restarted.

  (i)   drag-limited angular-momentum identity: sphere, v_d = 0, no
        vertical advection, eps_u = 1e-8 (5,800 days) and 1e-9 (58,000 days),
        five drag timescales each
  (iv)  the sphere and its beta-plane twin at Delta_y = 12.5, 25 and 50 K:
        the twin has beta = 2 Omega / a, walls at +-pi a / 2, 508 intervals
        (ny = 509) and the SB08 profile with y_1 = pi a / 2, which is the
        spherical forcing written in phi = y / a; 5,800 days. The 50 K pair is
        also the production pair of item (ii).
  (iii) the resolution twins of the 50 K pair, with the spacing halved:
        sphere ny = 649, twin ny = 1017, 5,800 days

Usage:
    python validation/sphere_validation_runs.py --out-root DIR --dt 30 \
        --dt-fine 8 --jobs 3 [--only LABEL ...] [--list]
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

A = 6.371e6
OMEGA = 7.292115e-5
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAYS = 5800  # five drag timescales at eps_u = 1e-8 (1157 days each)


def sphere(ny):
    return ["--geometry", "sphere", "--ny", str(ny)]


def twin(ny):
    return ["--ny", str(ny), "--beta", repr(2 * OMEGA / A),
            "--domain-size", repr(np.pi * A), "--theta-e-type", "SB08",
            "--y1", repr(np.pi * A / 2)]


def run_list(dt, dt_fine):
    runs = []
    am = ["--vd", "0", "--no-vert-advec-u"]
    runs.append(("i_amc_eps1e-8", sphere(325) + am + ["--eps-u", "1e-8"], DAYS, dt))
    runs.append(("i_amc_eps1e-9", sphere(325) + am + ["--eps-u", "1e-9"], 58000, dt))
    for delta_y in ("12.5", "25", "50"):
        runs.append((f"iv_sphere_dy{delta_y}", sphere(325) + ["--delta-y", delta_y], DAYS, dt))
        runs.append((f"iv_twin_dy{delta_y}", twin(509) + ["--delta-y", delta_y], DAYS, dt))
    runs.append(("iii_sphere_ny649_dy50", sphere(649) + ["--delta-y", "50"], DAYS, dt_fine))
    runs.append(("iii_twin_ny1017_dy50", twin(1017) + ["--delta-y", "50"], DAYS, dt_fine))
    # Diagnostic beyond the plan (added 2026-11-08 after item (iii) failed):
    # the resolution quartet with the eddy term off (v_d = 0), which removes
    # the polar westerly jet and its easterly notch, to separate the
    # spherical discretization's own resolution dependence in the tropics
    # from the remote effect of the polar feature's unconverged position.
    vd0 = ["--delta-y", "50", "--vd", "0"]
    runs.append(("diag_vd0_sphere_ny325", sphere(325) + vd0, DAYS, dt))
    runs.append(("diag_vd0_twin_ny509", twin(509) + vd0, DAYS, dt))
    runs.append(("diag_vd0_sphere_ny649", sphere(649) + vd0, DAYS, dt_fine))
    runs.append(("diag_vd0_twin_ny1017", twin(1017) + vd0, DAYS, dt_fine))
    return runs


def git_sha():
    return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def execute(item):
    label, args, ndays, dt, out_root = item
    run_dir = os.path.join(out_root, label)
    os.makedirs(run_dir, exist_ok=True)
    out = os.path.join(run_dir, "output.nc")
    if os.path.exists(out):
        return label, "skipped (output exists)", 0.0
    cmd = ([sys.executable, "-m", "ss09.cli", "--backend", "numba", "--dt", str(dt),
            "--ndays", str(ndays), "--output-path", out, "--restart-every", "1000"]
           + args)
    t0 = time.time()
    with open(os.path.join(run_dir, "run.log"), "w") as log:
        proc = subprocess.run(["nice", "-n", "10"] + cmd, cwd=REPO, stdout=log,
                              stderr=subprocess.STDOUT)
    wall = time.time() - t0
    with open(os.path.join(run_dir, "command.json"), "w") as f:
        json.dump(dict(cmd=cmd, returncode=proc.returncode, wall_s=wall,
                       s_per_day=wall / ndays, git_sha=git_sha(),
                       host=platform.node()), f, indent=1)
    return label, f"exit {proc.returncode}", wall


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-root", required=True)
    p.add_argument("--dt", type=int, required=True)
    p.add_argument("--dt-fine", type=int, required=True)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--only", nargs="*")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    runs = run_list(args.dt, args.dt_fine)
    if args.only:
        runs = [r for r in runs if r[0] in args.only]
    for label, a, ndays, dt in runs:
        print(f"{label}: {ndays} days at dt={dt}: {' '.join(a)}", flush=True)
    if args.list:
        return 0
    # longest first, so the 58,000-day run starts at once
    items = sorted(((*r, args.out_root) for r in runs), key=lambda r: -r[2] * 86400 / r[3])
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(execute, item) for item in items]
        for fut in as_completed(futures):
            label, status, wall = fut.result()
            print(f"{label}: {status}, {wall / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
