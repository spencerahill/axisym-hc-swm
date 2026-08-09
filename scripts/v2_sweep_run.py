"""Run the moist V2 sweep with a fixed pool of concurrent model processes.

berimbau has 16 physical cores (2x Xeon E5-2620 v4, 32 threads). Threading
*within* a run is counterproductive at these grid sizes, so parallelism means
separate processes, and the measured optimum is 16 concurrent: 138 model-days/s
aggregate for a 34% per-run latency penalty (CLAUDE.md, Integration Backends).

Two modes:
    --smoke N     run every manifest entry for N days into the smoke tree, to
                  catch a configuration that goes unstable before spending
                  hours on it. Judged on the OUTPUT FILE (days present, all
                  fields finite), never on the log.
    (default)     run the real sweep.

Each run gets its own directory, because restart filenames encode only the day
and would otherwise collide (a hard PermissionError on this filesystem).

Status is appended to <tree>/status.jsonl as each run finishes, so progress
survives a driver restart and the analysis can tell a completed run from a
truncated one without re-opening every file.

Usage:
    python scripts/v2_sweep_run.py --smoke 60
    python scripts/v2_sweep_run.py
    python scripts/v2_sweep_run.py --only A_a075_wc30 B_a085_wc44
    python scripts/v2_sweep_run.py --blocks A B C
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/shill/py/axisym-hc-swm")

from scripts.v2_sweep_manifest import (  # noqa: E402
    SMOKE_DIR, SWEEP_DIR, command, cost_days, manifest,
)

NPROC = 16


def check_output(path: str) -> dict:
    """What the output file itself says: days stored and whether the fields
    are finite. A model that goes NaN breaks its loop and still writes a file,
    so the file length is the evidence a run completed, not the exit code."""
    import numpy as np
    import xarray as xr

    if not os.path.exists(path):
        return dict(ok=False, reason="no output file", days=0)
    try:
        ds = xr.open_dataset(path, decode_timedelta=False)
    except Exception as err:  # pragma: no cover - corrupt file
        return dict(ok=False, reason=f"unreadable: {err}", days=0)
    try:
        days = int(ds.sizes["time"])
        last = int(ds["time"].values[-1]) if days else 0
        finite = {}
        for name in ("u", "v", "T", "W", "P"):
            if name in ds:
                arr = ds[name].values
                finite[name] = bool(np.isfinite(arr).all())
        # A run that goes NaN on its first day writes a file with zero stored
        # days, so every reduction below has to tolerate an empty array.
        wmin = (float(ds["W_min"].values.min())
                if ("W_min" in ds and ds["W_min"].size) else float("nan"))
        size_mb = os.path.getsize(path) / 1e6
    finally:
        ds.close()
    ok = bool(days) and all(finite.values())
    if not days:
        return dict(ok=False, reason="zero days stored (diverged immediately)",
                    days=0, last_day=0, finite=finite, w_min=wmin,
                    size_mb=round(size_mb, 1))
    return dict(ok=ok, days=days, last_day=last, finite=finite,
                w_min=wmin, size_mb=round(size_mb, 1),
                reason="" if ok else "non-finite fields")


def run_one(entry: dict, tree: str, ndays: int | None, log_dir: str,
            dt_override: int | None = None) -> dict:
    name = entry["name"]
    if dt_override is not None:
        entry = dict(entry, args=dict(entry["args"], dt=dt_override))
    out_dir = os.path.join(tree, name)
    os.makedirs(out_dir, exist_ok=True)
    cmd = command(entry, out_dir, ndays=ndays)
    log_path = os.path.join(log_dir, f"{name}.log")
    out_nc = os.path.join(out_dir, "out.nc")
    # Delete any previous output first: a run that fails to start would
    # otherwise leave the old file on disk, and reading it back looks exactly
    # like success (the stale-artifact trap in the global Bash rules).
    for stale in (out_nc,):
        if os.path.exists(stale):
            os.remove(stale)
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              cwd="/home/shill/py/axisym-hc-swm")
    wall = time.time() - t0
    status = dict(
        name=name, block=entry["block"], note=entry["note"],
        returncode=proc.returncode, wall_s=round(wall, 1),
        requested_days=ndays if ndays is not None else entry["args"]["ndays"],
        cmd=" ".join(cmd),
    )
    status.update(check_output(out_nc))
    status["s_per_day"] = round(wall / max(status["days"], 1), 4)
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0,
                    help="run every entry for this many days into the smoke tree")
    ap.add_argument("--nproc", type=int, default=NPROC)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--blocks", nargs="*", default=None)
    ap.add_argument("--skip-done", action="store_true",
                    help="skip runs whose output already has the requested days")
    ap.add_argument("--dt-override", type=int, default=None,
                    help="force this dt (stability probing); dt must divide 86400")
    ap.add_argument("--tag", default="",
                    help="suffix on the output tree, so probes do not clobber runs")
    args = ap.parse_args()

    runs = manifest()
    if args.blocks:
        runs = [r for r in runs if r["block"] in args.blocks]
    if args.only:
        runs = [r for r in runs if r["name"] in args.only]

    tree = (SMOKE_DIR if args.smoke else SWEEP_DIR) + args.tag
    ndays = args.smoke if args.smoke else None
    log_dir = os.path.join(tree, "logs")
    os.makedirs(log_dir, exist_ok=True)
    status_path = os.path.join(tree, "status.jsonl")

    if args.skip_done:
        kept = []
        for r in runs:
            want = ndays if ndays is not None else r["args"]["ndays"]
            info = check_output(os.path.join(tree, r["name"], "out.nc"))
            if info["ok"] and info["days"] >= want:
                continue
            kept.append(r)
        print(f"--skip-done: {len(runs) - len(kept)} already complete, "
              f"{len(kept)} to run", flush=True)
        runs = kept

    # Longest first, so the tail of the sweep is short runs and the pool
    # drains evenly instead of waiting on one straggler.
    runs.sort(key=cost_days, reverse=True)
    total_cost = sum(cost_days(r) for r in runs)
    if ndays is not None:
        total_cost = sum(cost_days(dict(r, args=dict(r["args"], ndays=ndays)))
                         for r in runs)
    print(f"{len(runs)} runs, {total_cost:,.0f} equivalent model-days, "
          f"{args.nproc}-way into {tree}"
          + (f", dt forced to {args.dt_override} s" if args.dt_override else ""),
          flush=True)

    t0 = time.time()
    done = 0
    failures = []
    with open(status_path, "a") as sf, ThreadPoolExecutor(args.nproc) as pool:
        futures = {pool.submit(run_one, r, tree, ndays, log_dir,
                              args.dt_override): r for r in runs}
        for fut in as_completed(futures):
            try:
                st = fut.result()
            except Exception as err:  # never let one bad file stop the sweep
                entry = futures[fut]
                st = dict(name=entry["name"], block=entry["block"],
                          note=entry["note"], returncode=-1, wall_s=0.0,
                          requested_days=0, ok=False, days=0,
                          reason=f"driver error: {err}", s_per_day=0.0)
            done += 1
            sf.write(json.dumps(st) + "\n")
            sf.flush()
            flag = "ok " if (st["ok"] and st["days"] >= st["requested_days"]) else "FAIL"
            if flag == "FAIL":
                failures.append(st["name"])
            el = time.time() - t0
            print(f"[{done:3d}/{len(runs)}] {flag} {st['name']:28s} "
                  f"{st['days']:5d} d  {st['wall_s']:7.1f} s  "
                  f"{st['s_per_day']:.3f} s/day  elapsed {el/60:.1f} min",
                  flush=True)

    el = time.time() - t0
    print(f"\nfinished {done} runs in {el/60:.1f} min "
          f"({total_cost/max(el,1):.1f} equivalent model-days/s aggregate)")
    if failures:
        print(f"{len(failures)} FAILED or truncated: {' '.join(failures)}")
    else:
        print("all runs completed with finite fields")


if __name__ == "__main__":
    main()
