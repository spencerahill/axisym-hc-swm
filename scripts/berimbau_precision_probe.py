"""Is there a speedup available from float32, or from numba fastmath?

Both are speed levers that cost correctness guarantees, so measure the size of
the prize BEFORE weighing the risk. float32 can only pay through SIMD width:
berimbau has AVX2 (256-bit), so 8 float32 lanes against 4 float64, a ceiling of
2x and only if the kernel actually vectorizes. fastmath lets the compiler
reassociate floating-point operations, which breaks the bitwise identity the
numba backend is validated on.

Constants are passed in at the array's own dtype rather than written as Python
literals, which are float64 and would silently promote a float32 kernel back to
float64. The dtype of the result is asserted, so a promoted kernel cannot
masquerade as a float32 one.
"""

import time

import numpy as np
from numba import njit

NSTEPS = 3000


def _make_kernel(fastmath):
    @njit(cache=True, fastmath=fastmath)
    def step(u, v, t, un, dy, two, c1, c2, c3):
        n = u.shape[0]
        for i in range(1, n - 1):
            dudy = (u[i + 1] - u[i - 1]) / (two * dy)
            d2u = (u[i + 1] - two * u[i] + u[i - 1]) / (dy * dy)
            un[i] = u[i] + c1 * (-v[i] * dudy + c2 * d2u - c3 * u[i] + t[i])
        return un
    return step


STEP_PLAIN = _make_kernel(False)
STEP_FAST = _make_kernel(True)


def bench(fn, ny, dtype, repeats=3, nsteps=NSTEPS):
    rng = np.random.default_rng(0)
    u = rng.standard_normal(ny).astype(dtype)
    v = rng.standard_normal(ny).astype(dtype)
    t = rng.standard_normal(ny).astype(dtype)
    un = np.zeros(ny, dtype=dtype)
    consts = [dtype(x) for x in (1.0e4, 2.0, 0.1, 1.0e-3, 1.0e-5)]

    out = fn(u, v, t, un, *consts)
    assert out.dtype == dtype, (
        f"kernel promoted {dtype} to {out.dtype}; constants leaked float64"
    )

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(nsteps):
            un = fn(u, v, t, un, *consts)
            u, un = un, u
        best = min(best, time.perf_counter() - start)
    return best / nsteps * 1e6  # us/step


def main():
    print("us/step, best of 3, single member, single thread")
    print(f"{'ny':>7} {'f64':>9} {'f64+fast':>10} {'f32':>9} {'f32+fast':>10} "
          f"{'f32 gain':>9} {'best gain':>10}")
    for ny in (801, 1601, 6401, 25601):
        f64 = bench(STEP_PLAIN, ny, np.float64)
        f64f = bench(STEP_FAST, ny, np.float64)
        f32 = bench(STEP_PLAIN, ny, np.float32)
        f32f = bench(STEP_FAST, ny, np.float32)
        best = min(f64, f64f, f32, f32f)
        print(f"{ny:>7} {f64:>9.3f} {f64f:>10.3f} {f32:>9.3f} {f32f:>10.3f} "
              f"{f64 / f32:>8.2f}x {f64 / best:>9.2f}x")


if __name__ == "__main__":
    main()
