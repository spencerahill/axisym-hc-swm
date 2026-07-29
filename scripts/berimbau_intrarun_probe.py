"""Can a single SS09 run be sped up by threading within a timestep?

The leapfrog integration is strictly sequential in time, so the only
parallelism available inside one run is spatial: across the ny gridpoints of
a single timestep. This probe mimics that structure with a synthetic kernel
of the same shape as the model's per-step work (a handful of stencil +
elementwise passes over an ny-length array), run with the time loop serial
and the spatial loop either serial or numba-prange parallel.

Reports per-step cost both ways at several ny, against the measured real
per-step cost of the model, so the comparison is anchored to reality.
"""

import time

import numba
import numpy as np
from numba import njit, prange

NSTEPS = 20000


@njit(cache=True, fastmath=False)
def step_serial(u, v, t, un, dy):
    n = u.shape[0]
    for i in range(1, n - 1):
        dudy = (u[i + 1] - u[i - 1]) / (2.0 * dy)
        d2u = (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dy * dy)
        un[i] = u[i] + 0.1 * (-v[i] * dudy + 1e-3 * d2u - 1e-5 * u[i] + t[i])
    return un


@njit(cache=True, parallel=True, fastmath=False)
def step_parallel(u, v, t, un, dy):
    n = u.shape[0]
    for i in prange(1, n - 1):
        dudy = (u[i + 1] - u[i - 1]) / (2.0 * dy)
        d2u = (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dy * dy)
        un[i] = u[i] + 0.1 * (-v[i] * dudy + 1e-3 * d2u - 1e-5 * u[i] + t[i])
    return un


def bench(fn, ny, nsteps=NSTEPS):
    rng = np.random.default_rng(0)
    u = rng.standard_normal(ny)
    v = rng.standard_normal(ny)
    t = rng.standard_normal(ny)
    un = np.zeros(ny)
    dy = 1.0e4

    fn(u, v, t, un, dy)  # JIT warm-up

    start = time.perf_counter()
    for _ in range(nsteps):
        un = fn(u, v, t, un, dy)
        u, un = un, u  # keep the time loop genuinely sequential
    return (time.perf_counter() - start) / nsteps * 1e6  # microseconds/step


def main():
    print(f"numba threads available: {numba.config.NUMBA_NUM_THREADS}")
    print(f"{'ny':>6} {'serial us/step':>15} {'parallel us/step':>17} "
          f"{'speedup':>9}")
    for ny in (801, 1601, 6401, 25601, 102401):
        s = bench(step_serial, ny)
        p = bench(step_parallel, ny)
        print(f"{ny:>6} {s:>15.2f} {p:>17.2f} {s / p:>9.2f}x")


if __name__ == "__main__":
    main()
