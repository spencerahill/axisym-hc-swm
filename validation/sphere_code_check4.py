"""Code check 4 of the spherical-geometry plan, at the production spacing.

A 30-day spherical run (ny=325, the production grid spacing of 39.3 km at the
equator, production parameters, hemispherically symmetric forcing) must keep
u and theta exactly symmetric and v exactly antisymmetric on every stored
day, and a run continued from a restart file written on day 15 must
reproduce the uninterrupted run bit for bit: every daily field, the final
state and the filtered previous state. Checked on both backends, and the two
backends' daily fields against each other.

Usage:
    python validation/sphere_code_check4.py --out-dir SCRATCH_DIR [--dt 30]
"""

import argparse
import os
import sys

import numpy as np

from ss09.sw_config import SWConfig
from ss09.sw_model import SWModel
from ss09.theta_e import SphericalProfile, ThetaEConfig


def build(out_dir, backend, ndays, dt):
    os.makedirs(out_dir, exist_ok=True)
    cfg = SWConfig(geometry="sphere", ny=325, dt=dt, total_integration_days=ndays,
                   backend=backend, output_path=os.path.join(out_dir, "out.nc"),
                   restart_output_dir=out_dir)
    return SWModel(cfg, SphericalProfile(ThetaEConfig(theta_e_type="sphere")))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dt", type=int, default=30)
    args = p.parse_args()
    checks = []
    runs = {}
    for backend in ("numpy", "numba"):
        root = os.path.join(args.out_dir, backend)
        straight = build(os.path.join(root, "straight"), backend, 30, args.dt)
        straight.run_sim()
        first = build(os.path.join(root, "split"), backend, 15, args.dt)
        first.run_sim()
        cont = build(os.path.join(root, "cont"), backend, 30, args.dt)
        cont.restart_day = cont.load_from_restart(
            os.path.join(root, "split", "restart_day0015.nc"))
        cont.run_sim()
        runs[backend] = straight
        r = straight.results
        days = int(np.count_nonzero(r.time))
        checks.append((f"{backend}: 30 days stored, all finite",
                       days == 30 and bool(np.all(np.isfinite(r.u[:days])))))
        checks.append((f"{backend}: daily u symmetric on all 30 days",
                       float(np.max(np.abs(r.u - r.u[:, ::-1]))) == 0.0))
        checks.append((f"{backend}: daily theta symmetric on all 30 days",
                       float(np.max(np.abs(r.theta - r.theta[:, ::-1]))) == 0.0))
        checks.append((f"{backend}: daily v antisymmetric on all 30 days",
                       float(np.max(np.abs(r.v + r.v[:, ::-1]))) == 0.0))
        print(f"{backend}: max|v| on day 30 = {np.max(np.abs(r.v[29])):.3f} m/s, "
              f"max|u| = {np.max(np.abs(r.u[29])):.2f} m/s")
        same = all(
            np.array_equal(getattr(r, name)[15:30], getattr(cont.results, name)[15:30])
            for name in ("time", "u", "v", "theta")
        )
        same &= all(
            np.array_equal(getattr(straight.state, n), getattr(cont.state, n))
            and np.array_equal(getattr(straight.vars_prev_step, n),
                               getattr(cont.vars_prev_step, n))
            for n in ("u", "v", "theta")
        )
        checks.append((f"{backend}: restart on day 15 reproduces days 15-29 and "
                       "the final state bit for bit", bool(same)))
    checks.append(("numba daily fields equal numpy's bit for bit", all(
        np.array_equal(getattr(runs["numpy"].results, n), getattr(runs["numba"].results, n))
        for n in ("time", "u", "v", "theta"))))
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    n_pass = sum(ok for _, ok in checks)
    print(f"{n_pass} of {len(checks)} checks pass")
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
