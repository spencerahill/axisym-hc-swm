"""Reusable comparison/diagnostic helpers for old-vs-new and resolution studies.

Mechanical guards against the "they're on top of each other" eyeball error and
the narrow-inspection error: quantify where two fields disagree, and detect the
2Δy computational mode by sign-alternation rather than by eye.

NumPy + xarray only (no matplotlib), so it is safe to import from any driver.
"""
import numpy as np
import xarray as xr


def steady(path, var, ndays=200):
    """Time-mean of the last ``ndays`` records of ``var`` in a model NetCDF.

    Returns (y [m], field) on the file's native ``y`` grid.
    """
    ds = xr.open_dataset(path)
    f = ds.isel(time=slice(-ndays, None)).mean("time")[var].values
    return ds["y"].values, f


def report_diff(yA, fA, yB, fB, name="", n=4000):
    """Interpolate A and B to a common dense grid; report max|Δ|, its location,
    and RMS. Curves may be on different native grids (e.g. ny=51 vs ny=50); the
    dense common grid preserves grid-scale structure in either. Prints a line
    and returns a dict.
    """
    lo, hi = max(yA.min(), yB.min()), min(yA.max(), yB.max())
    yc = np.linspace(lo, hi, n)
    d = np.interp(yc, yB, fB) - np.interp(yc, yA, fA)
    i = int(np.argmax(np.abs(d)))
    out = {"max_abs": float(abs(d[i])), "at_m": float(yc[i]),
           "rms": float(np.sqrt(np.mean(d ** 2)))}
    print(f"{name:>30}: max|Δ|={out['max_abs']:8.3f} at y={out['at_m']/1e6:+6.2f} Mm"
          f"   RMS={out['rms']:7.3f}")
    return out


def sawtooth(u):
    """|u_i - 0.5(u_{i-1}+u_{i+1})| on interior points (NaN at ends).

    Equals 2x the amplitude of a pure 2Δy mode; falls as O(Δy^2) for smooth u,
    so it isolates grid-scale structure from resolved curvature as resolution
    increases.
    """
    s = np.full_like(u, np.nan, dtype=float)
    s[1:-1] = np.abs(u[1:-1] - 0.5 * (u[:-2] + u[2:]))
    return s


def flank_mode(y, u, flank_min_m=7e6, d2_floor=1e-3):
    """Diagnose the poleward-flank 2Δy mode.

    Returns dict with: ``sawtooth_max`` and its ``at_m`` over the flank region
    (|y| >= flank_min_m), and ``alternating`` (bool) = whether the discrete
    second difference changes sign between adjacent interior flank points with
    magnitude above ``d2_floor`` (the signature of a 2Δy mode, vs a monotonic
    sharp-but-resolved transition). ``alternating=False`` at high resolution
    means the residual sawtooth is resolved physical curvature, not a mode.
    """
    s = sawtooth(u)
    flank = np.abs(y) >= flank_min_m
    sf = np.where(flank, s, np.nan)
    i = int(np.nanargmax(sf))
    d2 = u[:-2] - 2 * u[1:-1] + u[2:]            # interior, len N-2
    fl_int = flank[1:-1]
    alternating = False
    idx = np.where(fl_int)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            continue
        if abs(d2[a]) > d2_floor and abs(d2[b]) > d2_floor and np.sign(d2[a]) != np.sign(d2[b]):
            alternating = True
            break
    return {"sawtooth_max": float(sf[i]), "at_m": float(y[i]),
            "alternating": alternating}
