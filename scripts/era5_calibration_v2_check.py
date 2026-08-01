"""Does the ERA5-consistent ``a`` remove the aggregated V2 regime?

The 2026-07-31 V2 runs at ``W_c = 50`` aggregated into a 78 mm/day equatorial
spike, a second precipitating band near 5.6 Mm, and a bone-dry gap, because the
gross moist stability ``Hhat = Shat - L_v(2a-1)W`` was negative over most of the
domain.  ``scripts/era5_normalization.py`` shows why: the sign change sits at
``W* = Shat/(L_v(2a-1))``, which is 44.3 kg/m^2 at the spec ``a = 0.85`` and
51.5-56.8 kg/m^2 in ERA5, while the quiescent column sits at
``W_c + tau_c E_0 = 50.66``.  So the spec's ``a`` is the one choice that puts a
``W_c = 50`` run below its own crossover.

This script tests the prediction directly: hold everything else fixed, lower
``a`` to the ERA5-consistent end, and check that the aggregated state does not
appear.  The control is the ``W_c = 40`` run, which crosses into positive
stability by lowering the column instead of raising the crossover, and which
already produced a recognizable single-ITCZ Hadley circulation.

Run ``python scripts/era5_calibration_v2_check.py``.
"""

from __future__ import annotations

import os

import numpy as np

from moist_v2_analysis import load_equilibrium, profile_figure, scorecard

ROOT = "model_output/moist_v2"
RUNS = [
    (r"$a$=0.85, $W_c$=50", "m4_v2_dyr75", "#CC3311"),
    (r"$a$=0.77, $W_c$=50", "wc50_a077", "#228833"),
    (r"$a$=0.85, $W_c$=40", "wc40", "#DDAA33"),
]


def bands(r, thresh_mm_day: float = 0.5) -> list[tuple[float, float, float]]:
    """Contiguous precipitating regions: ``(centre Mm, width Mm, peak mm/day)``.

    The aggregated regime's signature is more than one of these, plus a gap
    where P is identically zero; a recognizable Hadley circulation has one.
    """
    p = r["p"] * 86400.0
    on = p > thresh_mm_day
    out = []
    i = 0
    while i < len(on):
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(on) and on[j + 1]:
            j += 1
        seg = slice(i, j + 1)
        wgt = p[seg]
        out.append((float((wgt * r["y"][seg]).sum() / wgt.sum() / 1e6),
                    float((r["y"][j] - r["y"][i]) / 1e6),
                    float(p[seg].max())))
        i = j + 1
    return out


OMEGA, BETA = 7.292e-5, 2.0e-11
# The model's y, and the latitude each maps to through f = beta y = 2 Omega
# sin(phi), which is the mapping era5_normalization.py recommends.
SCORE_Y = [2.0e6, 3.0e6, 4.0e6]


def temp_contrasts(r) -> list[float]:
    """``theta(0) - theta(y)`` at the scorecard's three y.

    Reported in potential temperature, which is what ERA5's free-tropospheric
    theta is, so the two sides of the comparison need no conversion between
    them.  The model's own diagnostic T is theta/1.6.
    """
    y, t = r["y"], r["theta"]
    t0 = float(np.interp(0.0, y, t))
    return [t0 - 0.5 * (float(np.interp(yy, y, t)) + float(np.interp(-yy, y, t)))
            for yy in SCORE_Y]


def report(labels, runs) -> None:
    degs = [np.rad2deg(np.arcsin(BETA * yy / (2 * OMEGA))) for yy in SCORE_Y]
    print("\nPotential-temperature contrast theta(0) - theta(y), directly "
          "comparable\nwith the ERA5 free-tropospheric theta contrast")
    print("contrast (K) at y = " + ", ".join(f"{yy/1e6:.0f} Mm "
                                             f"({d:.0f} deg)"
                                             for yy, d in zip(SCORE_Y, degs)))
    print(f"\n{'run':22}" + "".join(f"{f'{d:.0f} deg':>10}" for d in degs)
          + f"{'jet (m/s)':>11}{'max|v|':>9}")
    print("-" * 74)
    for lab, r in zip(labels, runs):
        print(f"{lab:22}" + "".join(f"{c:>10.2f}" for c in temp_contrasts(r))
              + f"{float(r['u'].max()):>11.1f}"
              f"{float(np.max(np.abs(r['v']))):>9.2f}")
    print("ERA5 values for these columns are printed by "
          "scripts/era5_moisture_budget.py\n(theta contrasts; multiply by "
          "1/1.6 for the model's T) and era5_normalization.py (peak |v|).")

    print(f"\n{'run':22}{'W*':>7}{'W quiescent':>13}{'Hhat/Shat there':>17}"
          f"{'min Hhat':>10}{'dry frac':>10}{'bands':>7}")
    print("-" * 86)
    for lab, r in zip(labels, runs):
        wq = r["w_plateau"]
        wstar = r["w_hhat_zero"]
        dry = float(np.mean(r["p"] <= 0.0))
        print(f"{lab:22}{wstar:>7.1f}{wq:>13.2f}{1 - wq / wstar:>17.3f}"
              f"{np.min(r['hhat']) / 1e6:>10.2f}{dry:>10.3f}"
              f"{len(bands(r)):>7d}")
    print("\nW* = Shat/(L_v(2a-1)), the column where Hhat changes sign; "
          "W quiescent = W_c + tau_c E_0;\nmin Hhat in MJ/m^2; dry frac = "
          "fraction of the domain with P identically zero.")

    for lab, r in zip(labels, runs):
        print(f"\n{lab}: precipitating bands (centre Mm, width Mm, peak mm/day)")
        for c, w, pk in bands(r):
            print(f"    {c:+8.2f} {w:8.2f} {pk:9.2f}")
        u = r["u"]
        y = r["y"]
        print(f"    jet {u.max():.1f} m/s at {y[int(u.argmax())]/1e6:+.2f} Mm; "
              f"min u {u.min():.1f} m/s at {y[int(u.argmin())]/1e6:+.2f} Mm; "
              f"mean MSE flux {r['mean_flux'].max()/1e6:+.1f} MW/m, "
              f"eddy {np.abs(r['eddy_flux']).max()/1e6:.1f} MW/m")


def main() -> None:
    labels = [lab for lab, _, _ in RUNS]
    colors = [c for _, _, c in RUNS]
    runs = [load_equilibrium(os.path.join(ROOT, d, "out.nc"))
            for _, d, _ in RUNS]
    scorecard(labels, runs, "ERA5-consistent a at W_c = 50, against both controls")
    report(labels, runs)
    profile_figure(labels, runs, colors,
                   os.path.join(ROOT, "era5_a_check.png"),
                   "Lowering $a$ to the ERA5-consistent value at $W_c=50$")


if __name__ == "__main__":
    main()
