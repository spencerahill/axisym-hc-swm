"""Code check 1 of the spherical-geometry plan: on the beta plane, the model
with the spherical option reproduces the unmodified model bit for bit.

Runs the four regression configurations of ss09/tests/test_regression.py
(5 days at ny=801, dt=30: the collocated reproduction path, the staggered
production default, moist V1 and moist V2) once with a reference source tree
(a checkout of the main commit the spherical branch started from) and once
with the tree under test, on the same machine, and compares every data
variable and coordinate of the two output files bit for bit. The stored
regression baselines agree only to cross-machine roundoff (a relative 1e-11),
which is why the reference is a fresh run of the unmodified code and not the
stored files.

Each run is a subprocess started in its tree's root directory, so that
directory's ss09 is imported ahead of any installed copy; the script prints
the imported module path of each tree to confirm.

Usage:
    python validation/sphere_flat_reduction.py --reference REF_TREE \
        --test TEST_TREE --out-dir SCRATCH_DIR [--backend numba]
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import xarray as xr

CONFIGS = {
    "collocated reproduction path": [
        "--grid", "collocated", "--no-emfd-heaviside-gate",
        "--emfd-stencil", "centered",
    ],
    "staggered production default": [],
    "moist V1": ["--enable-moisture"],
    "moist V2": ["--enable-moisture", "--enable-latent-heating"],
}
COMMON = ["--ndays", "5", "--ny", "801", "--dt", "30"]


def imported_from(tree: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", "import ss09; print(ss09.__file__)"],
        cwd=tree, capture_output=True, text=True, check=True,
    )
    return proc.stdout.strip()


def run(tree: str, args: list, out: str) -> None:
    if os.path.exists(out):
        os.remove(out)  # a run that did not happen must yield "no such file"
    subprocess.run(
        [sys.executable, "-m", "ss09.cli"] + COMMON + args + ["--output-path", out],
        cwd=tree, capture_output=True, text=True, check=True,
    )


def compare(ref_path: str, test_path: str) -> list:
    rows = []
    ref = xr.open_dataset(ref_path, decode_times=False)
    test = xr.open_dataset(test_path, decode_times=False)
    names = sorted(set(ref.data_vars) | set(ref.coords))
    missing = [n for n in names if n not in test.variables]
    for name in names:
        if name in missing:
            rows.append((name, None, "MISSING"))
            continue
        a, b = ref[name].values, test[name].values
        if a.shape != b.shape:
            rows.append((name, None, f"SHAPE {a.shape} vs {b.shape}"))
            continue
        same = np.array_equal(a, b, equal_nan=True)
        with np.errstate(invalid="ignore"):
            diff = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
        rows.append((name, diff, "bitwise equal" if same else "DIFFERS"))
    ref.close()
    test.close()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backend", default="numpy", choices=["numpy", "numba"])
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    trees = {"reference": os.path.abspath(args.reference),
             "test": os.path.abspath(args.test)}
    for label, tree in trees.items():
        print(f"{label} tree imports {imported_from(tree)}")
    extra = ["--backend", args.backend] if args.backend == "numba" else []

    n_vars = n_equal = 0
    failures = []
    for name, cfg_args in CONFIGS.items():
        if args.backend == "numba" and "collocated" in name:
            print(f"\n{name}: skipped (numba supports the staggered grid only)")
            continue
        paths = {}
        for label, tree in trees.items():
            slug = name.replace(" ", "_")
            paths[label] = os.path.join(
                args.out_dir, f"{slug}_{args.backend}_{label}.nc")
            run(tree, cfg_args + extra, paths[label])
        rows = compare(paths["reference"], paths["test"])
        print(f"\n{name} ({args.backend}):")
        for var, diff, verdict in rows:
            n_vars += 1
            n_equal += verdict == "bitwise equal"
            if verdict != "bitwise equal":
                failures.append((name, var, verdict))
            shown = "" if diff is None else f"max|delta| = {diff:.3e}"
            print(f"  {var:32s} {verdict:14s} {shown}")
    print(f"\n{n_equal} of {n_vars} variables bitwise equal across "
          f"{len(CONFIGS) - (args.backend == 'numba')} configurations")
    for failure in failures:
        print("  FAIL", *failure)
    return 0 if not failures and n_vars > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
