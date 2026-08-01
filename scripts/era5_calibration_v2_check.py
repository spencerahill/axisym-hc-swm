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


def flux_figure(labels, runs, out_png: str) -> None:
    """The model's own moisture and MSE fluxes, mean against eddy.

    Laid out to be read directly against the ERA5 figure that
    ``scripts/era5_flux_decomposition.py`` produces: same quantities, same
    units (MW/m), same sign convention (positive northward), and a second axis
    giving the latitude each y maps to through f = beta y = 2 Omega sin(phi).

    The model has exactly one eddy term, the moisture diffusion
    -L_v D dW/dy, so its eddy MSE flux and its eddy latent flux are the same
    curve.  Nothing in the model transports dry static energy except the mean
    circulation, which is the structural difference from the observations.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Shared y within each row: the aggregated run's fluxes are three times the
    # others', and per-panel autoscaling would hide exactly that.
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), sharex=True, sharey="row")
    bound = 12.0
    rows = [
        (r"latent flux  $L_v F_q$  (MW m$^{-1}$)",
         [("lvq_flux", "#2c7fb8", "mean circulation", -6.0, 1),
          ("eddy_flux", "#d95f02", "eddies ($-L_vD\\,\\partial_yW$)", 6.5, -1)]),
        (r"MSE flux  $F_h$  (MW m$^{-1}$)",
         [("dse_flux", "#7b3294", "dry static energy (mean)", -6.0, 1),
          ("mean_flux", "#1b7837", "mean MSE", 3.0, -1),
          ("eddy_flux", "#d95f02", "eddies", 8.0, -1),
          ("total_flux", "#000000", "total", 9.5, 1)]),
    ]
    for i, (ylabel, curves) in enumerate(rows):
        for j, (lab, r) in enumerate(zip(labels, runs)):
            ax = axes[i, j]
            yf = r["y_face"] / 1e6
            span = max(np.max(np.abs(r["total_flux"])),
                       np.max(np.abs(r["dse_flux"]))) / 1e6
            for key, colr, text, x0, sgn in curves:
                prof = r[key] / 1e6
                ax.plot(yf, prof, color=colr, lw=1.8)
                if j == 0:
                    k = int(np.abs(yf - x0).argmin())
                    ax.text(yf[k], prof[k] + sgn * 0.09 * span, text,
                            color=colr, fontsize=8,
                            ha="center", va="bottom" if sgn > 0 else "top")
            ax.axhline(0, color="0.6", lw=0.7)
            ax.set_xlim(-bound, bound)
            ax.grid(alpha=0.25)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(lab, fontsize=11)
                sec = ax.secondary_xaxis("top", functions=(
                    lambda y: np.rad2deg(np.arcsin(
                        np.clip(BETA * y * 1e6 / (2 * OMEGA), -1, 1))),
                    lambda d: 2 * OMEGA * np.sin(np.deg2rad(d)) / BETA / 1e6))
                sec.set_xticks([-30, -20, -10, 0, 10, 20, 30])
                sec.set_xlabel("latitude by $f=\\beta y$", fontsize=9)
            if i == 1:
                ax.set_xlabel("y (Mm)")

    fig.suptitle("The model's column fluxes, mean circulation against eddies, "
                 "in the units and sign convention of the ERA5 figure.\n"
                 "The model's only eddy term is the moisture diffusion, so its "
                 "eddy latent and eddy MSE fluxes are one curve.", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


KEYS = ("dse_flux", "lvq_flux", "mean_flux", "eddy_flux", "total_flux")


def flux_table(labels, runs) -> None:
    """Fluxes at y = 3 Mm, whose Coriolis image is 24 deg.

    Signed, and at a single location rather than as peak magnitudes, so it can
    be set beside the 24 deg row that scripts/era5_flux_decomposition.py prints
    without either side having to be re-derived.
    """
    print("\nColumn fluxes at y = 3 Mm (the Coriolis image of 24 deg), MW/m,")
    print("positive northward. Compare with the 24 deg row printed by")
    print("scripts/era5_flux_decomposition.py.")
    print(f"\n{'run':22}{'DSE mean':>10}{'Lv q mean':>11}{'MSE mean':>10}"
          f"{'eddy':>8}{'total':>8}")
    print("-" * 69)
    for lab, r in zip(labels, runs):
        j = int(np.abs(r["y_face"] - 3.0e6).argmin())
        vals = [float(r[k][j]) / 1e6 for k in KEYS]
        print(f"{lab:22}{vals[0]:>10.1f}{vals[1]:>11.1f}{vals[2]:>10.1f}"
              f"{vals[3]:>8.1f}{vals[4]:>8.1f}")
    print("The model's eddy term is the moisture diffusion alone, so its eddy")
    print("column is entirely latent: it has no eddy dry-static-energy flux.")


def main() -> None:
    labels = [lab for lab, _, _ in RUNS]
    colors = [c for _, _, c in RUNS]
    runs = [load_equilibrium(os.path.join(ROOT, d, "out.nc"))
            for _, d, _ in RUNS]
    scorecard(labels, runs, "ERA5-consistent a at W_c = 50, against both controls")
    report(labels, runs)
    flux_table(labels, runs)
    profile_figure(labels, runs, colors,
                   os.path.join(ROOT, "era5_a_check.png"),
                   "Lowering $a$ to the ERA5-consistent value at $W_c=50$")
    flux_figure(labels, runs, os.path.join(ROOT, "era5_a_check_fluxes.png"))


if __name__ == "__main__":
    main()
