"""Does batching an ENSEMBLE into one array beat running separate processes?

This is the CPU half of the JAX question. jax.vmap over an ensemble of runs
compiles to exactly this: the same timestep kernel applied to an
(n_members, ny) array, with the time loop still sequential and shared across
members. Ensemble members are fully independent (no stencil coupling across
the batch axis), so the batch axis parallelizes cleanly, unlike the spatial
axis probed in berimbau_intrarun_probe.py.

The comparison that matters on berimbau is against what we already have for
free: 16 independent processes, measured at 138 model-days/s aggregate. If
batched-and-threaded does not beat that, vmap buys nothing here for
throughput, and JAX's case rests on grad instead.

Reports per-member per-step cost, so lower is better and the numbers are
directly comparable to the 4.43 us/step single-member serial baseline from
berimbau_intrarun_probe.py.
"""

import time

import numpy as np
from numba import njit, prange

NSTEPS = 2000
NY = 801


@njit(cache=True, fastmath=False)
def step_batch_serial(u, v, t, un, dy):
    """Same kernel, batched over members, single-threaded."""
    nm, n = u.shape
    for m in range(nm):
        for i in range(1, n - 1):
            dudy = (u[m, i + 1] - u[m, i - 1]) / (2.0 * dy)
            d2u = (u[m, i + 1] - 2.0 * u[m, i] + u[m, i - 1]) / (dy * dy)
            un[m, i] = u[m, i] + 0.1 * (
                -v[m, i] * dudy + 1e-3 * d2u - 1e-5 * u[m, i] + t[m, i]
            )
    return un


@njit(cache=True, parallel=True, fastmath=False)
def step_batch_parallel(u, v, t, un, dy):
    """Threaded over the ENSEMBLE axis, which is where the work now is."""
    nm, n = u.shape
    for m in prange(nm):
        for i in range(1, n - 1):
            dudy = (u[m, i + 1] - u[m, i - 1]) / (2.0 * dy)
            d2u = (u[m, i + 1] - 2.0 * u[m, i] + u[m, i - 1]) / (dy * dy)
            un[m, i] = u[m, i] + 0.1 * (
                -v[m, i] * dudy + 1e-3 * d2u - 1e-5 * u[m, i] + t[m, i]
            )
    return un


def bench(fn, nm, ny=NY, nsteps=NSTEPS):
    rng = np.random.default_rng(0)
    u = rng.standard_normal((nm, ny))
    v = rng.standard_normal((nm, ny))
    t = rng.standard_normal((nm, ny))
    un = np.zeros((nm, ny))
    dy = 1.0e4

    fn(u, v, t, un, dy)  # JIT warm-up

    start = time.perf_counter()
    for _ in range(nsteps):
        un = fn(u, v, t, un, dy)
        u, un = un, u  # time loop stays sequential, shared by all members
    wall = time.perf_counter() - start
    return wall / nsteps / nm * 1e6  # microseconds per step PER MEMBER


def main():
    print(f"ny={NY}, {NSTEPS} steps, cost in us/step PER MEMBER (lower better)")
    print(f"{'members':>8} {'batched serial':>15} {'batched threaded':>17} "
          f"{'vs 1-proc serial':>17}")
    base = bench(step_batch_serial, 1)
    for nm in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        s = bench(step_batch_serial, nm)
        p = bench(step_batch_parallel, nm)
        best = min(s, p)
        print(f"{nm:>8} {s:>15.3f} {p:>17.3f} {base / best:>16.2f}x")


if __name__ == "__main__":
    main()
