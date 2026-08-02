"""Measure run cost and parallel scaling on a given machine.

Re-baselines the wall-time facts that were measured on the old 4-core Intel
MacBook. Reports per-run seconds/day and aggregate throughput at a chosen
concurrency, so the worker cap can be set from a contended timing rather than
a solo one (per the timing-under-contention rule).

Usage:
    python scripts/berimbau_timing.py --ny 801 --dt 30 --backend numba \
        --ndays 10 --workers 1

    # concurrency sweep
    python scripts/berimbau_timing.py --workers 1 2 4 8 12 16
"""

import argparse
import os
import subprocess
import tempfile
import time


def run_one(out_dir, idx, ny, dt, ndays, backend, moist):
    """Launch one model run as a subprocess; return the Popen handle."""
    # Each worker needs its OWN directory: the restart filename is derived
    # from --output-path (not --restart-dir, which is ignored when
    # --output-path is explicit), and encodes only the day, so workers
    # sharing a directory collide on restart_day{NNNN}.nc.
    worker_dir = os.path.join(out_dir, f"w{idx}")
    os.makedirs(worker_dir, exist_ok=True)
    out_path = os.path.join(worker_dir, "run.nc")
    args = [
        "run-sw-model",
        "--ndays", str(ndays),
        "--ny", str(ny),
        "--dt", str(dt),
        "--backend", backend,
        "--output-path", out_path,
    ]
    if moist:
        args.append("--enable-moisture")
    return subprocess.Popen(args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True)


def time_at_concurrency(n_workers, ny, dt, ndays, backend, moist):
    """Run n_workers identical runs concurrently; return (wall, s_per_day)."""
    with tempfile.TemporaryDirectory() as out_dir:
        start = time.perf_counter()
        procs = [
            run_one(out_dir, i, ny, dt, ndays, backend, moist)
            for i in range(n_workers)
        ]
        errs = [p.communicate()[1] for p in procs]
        codes = [p.returncode for p in procs]
        wall = time.perf_counter() - start

    if any(c != 0 for c in codes):
        for i, (c, e) in enumerate(zip(codes, errs)):
            if c != 0:
                print(f"--- worker {i} exit {c} ---\n{e[-3000:]}")
        raise RuntimeError(f"run failed at workers={n_workers}: {codes}")

    # Per-run rate: each worker individually took ~wall seconds for ndays.
    s_per_day = wall / ndays
    throughput = n_workers * ndays / wall
    return wall, s_per_day, throughput


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ny", type=int, default=801)
    p.add_argument("--dt", type=int, default=30)
    p.add_argument("--ndays", type=int, default=10)
    p.add_argument("--backend", default="numba")
    p.add_argument("--moist", action="store_true")
    p.add_argument("--workers", type=int, nargs="+", default=[1])
    args = p.parse_args()

    print(
        f"config: ny={args.ny} dt={args.dt} ndays={args.ndays} "
        f"backend={args.backend} moist={args.moist}"
    )
    print(f"{'workers':>8} {'wall_s':>9} {'s/day/run':>11} "
          f"{'days/s total':>13} {'15yr_run_min':>13}")

    for n in args.workers:
        wall, s_per_day, throughput = time_at_concurrency(
            n, args.ny, args.dt, args.ndays, args.backend, args.moist
        )
        yr15_min = s_per_day * 365 * 15 / 60
        print(f"{n:>8} {wall:>9.2f} {s_per_day:>11.4f} "
              f"{throughput:>13.2f} {yr15_min:>13.1f}")


if __name__ == "__main__":
    main()
