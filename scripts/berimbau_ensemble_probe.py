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

RESULTS (berimbau, 2026-08-02, best of 3):

1. Threading over the ensemble axis WORKS, where threading over the spatial
   axis did not: 256 members reach 0.565 us/step/member, since 256 x 801 =
   205k points finally cover the ~50 us barrier cost.

2. It still loses to plain process parallelism, which is free. In throughput:
   16 processes give ~3.1 member-steps/us (0.264 x the measured 11.9x fleet
   scaling) against ~1.8 for batched-and-threaded. So vmap buys no throughput
   on this CPU, and JAX's remaining case here is grad, not speed.

3. The single-threaded batching gain is NOT real. It looked like 1.6-2.6x, but
   the weight sweep shows it collapsing monotonically as the kernel gets
   heavier (1.64x, 1.28x, 1.14x, 1.05x, 1.05x, 0.99x for nwork 1..32), which is
   the signature of per-call overhead being amortized rather than genuine
   vectorization or cache reuse. At the real SS09 step weight (~30 us, between
   nwork 8 and 16) the gain is 1.05x. CONCLUSION: do not build a batched
   real-model implementation; it is a large port for ~5%.
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


@njit(cache=True, fastmath=False)
def step_batch_serial_heavy(u, v, t, un, dy, nwork):
    """Weight-swept variant, for testing whether the batching gain is real.

    nwork chained tendency evaluations scale the work per step while leaving
    the memory footprint identical, so the kernel can be made as heavy as a
    real model timestep (~30 us at ny=801, roughly 7x this kernel at nwork=1).
    Each pass feeds the next through `val` so nothing can be optimized away,
    and no accumulator crosses the member loop, which would otherwise be
    misread as a parallel reduction.

    This is deliberately a SEPARATE function from step_batch_serial: adding the
    inner loop changes the kernel even at nwork=1, so the two families are not
    comparable and only trends WITHIN this family are meaningful.
    """
    nm, n = u.shape
    for m in range(nm):
        for i in range(1, n - 1):
            val = u[m, i]
            for _ in range(nwork):
                dudy = (u[m, i + 1] - u[m, i - 1]) / (2.0 * dy)
                d2u = (u[m, i + 1] - 2.0 * u[m, i] + u[m, i - 1]) / (dy * dy)
                val = val + 0.1 * (
                    -v[m, i] * dudy + 1e-3 * d2u - 1e-5 * val + t[m, i]
                )
            un[m, i] = val
    return un


def bench(fn, nm, args=(), ny=NY, nsteps=NSTEPS, repeats=3):
    """Best-of-`repeats` us/step per member. Best-of, not mean: the effects
    under test are ~1.3x and background jitter only ever inflates a timing."""
    rng = np.random.default_rng(0)
    u = rng.standard_normal((nm, ny))
    v = rng.standard_normal((nm, ny))
    t = rng.standard_normal((nm, ny))
    un = np.zeros((nm, ny))
    dy = 1.0e4

    fn(u, v, t, un, dy, *args)  # JIT warm-up

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(nsteps):
            un = fn(u, v, t, un, dy, *args)
            u, un = un, u  # time loop stays sequential, shared by all members
        best = min(best, time.perf_counter() - start)
    return best / nsteps / nm * 1e6  # microseconds per step PER MEMBER


def main():
    print(f"ny={NY}, {NSTEPS} steps, best of 3, us/step PER MEMBER (lower better)")
    print(f"{'members':>8} {'batched serial':>15} {'batched threaded':>17} "
          f"{'vs 1-proc serial':>17}")
    base = bench(step_batch_serial, 1)
    for nm in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        s = bench(step_batch_serial, nm)
        p = bench(step_batch_parallel, nm)
        best = min(s, p)
        print(f"{nm:>8} {s:>15.3f} {p:>17.3f} {base / best:>16.2f}x")

    # Does the single-threaded batching gain survive a realistic kernel weight?
    # If the gain is per-call overhead being amortized, it collapses toward 1.0
    # as the kernel gets heavier. If it is genuine vectorization or cache reuse,
    # it persists. The real SS09 step is ~30 us at ny=801; the nwork whose
    # 1-member cost lands near that is the one that decides whether a batched
    # real-model implementation is worth building.
    print("\nDoes the SERIAL batching gain survive a heavier kernel?")
    print("(separate kernel family; compare trends down the column, not to the")
    print(" table above. real SS09 step ~30 us/step at ny=801.)")
    print(f"{'nwork':>6} {'1-member us/step':>17} {'64-member us/step':>18} "
          f"{'batching gain':>14}")
    for nwork in (1, 2, 4, 8, 16, 32):
        one = bench(step_batch_serial_heavy, 1, args=(nwork,))
        many = bench(step_batch_serial_heavy, 64, args=(nwork,))
        print(f"{nwork:>6} {one:>17.3f} {many:>18.3f} {one / many:>13.2f}x")


if __name__ == "__main__":
    main()
