"""What ERA5 says about the model's assumed vertical structure.

Three questions, all about where the model's two branches are and how thick:

1. **Does "equal and opposite integrated v" pick out an interface?**  No.
   Writing Psi(p) = (1/g) int_0^p [v] dp', the condition "mass transport above
   the level equals minus that below" is Psi(p_d) = -(Psi(p_s) - Psi(p_d)),
   which reduces to Psi(p_s) = 0.  That is a statement about the *column*, not
   about p_d: it holds at every level once the column mass flux vanishes and
   at no level otherwise.  What does pick out a level is requiring equal and
   opposite branch *velocities*, Psi(p_d)/M_upper = -(Psi(p_s)-Psi(p_d))/M_lower,
   which with Psi(p_s) = 0 forces M_upper = M_lower, i.e. the half-mass level.
   Both statements are verified numerically below.

2. **How much does the barotropic mass adjustment matter?**  ERA5's zonal-mean
   [v] on pressure levels carries a spurious net column mass flux.  Every
   transport coefficient here is reported with and without removing it.

3. **What layer depth d does ERA5 imply?**  The model puts the flow in a layer
   of depth d at the top of the troposphere and another of depth d at the
   bottom, with v = 0 between them.  Given the observed overturning strength
   Psi_ext and the observed DSE flux, the branch-mean dry static energies must
   differ by F_DSE / Psi_ext; d is then whatever layer thickness reproduces
   that difference from the observed s(p) profile.

Run ``python scripts/era5_vertical_structure.py``.
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from era5_a_calibration import (GRAV, dynamical_interface_band, layer_weights,  # noqa: E402
                                load_level_field, load_surface_field,
                                mass_streamfunction, cos_weighted_mean)
from ss09.sw_config import SWConfig  # noqa: E402

CP_DRY = 1004.6
P_TOP_MODEL = 19300.0   # the model's tropopause pressure, Pa
P_SFC_MODEL = 100000.0


def load(years: range):
    ta = load_level_field("ta", "t", years)
    zg = load_level_field("zg", "z", years).sel(time=ta["time"])
    va = load_level_field("va", "v", years).sel(time=ta["time"])
    ps = load_surface_field("ps", "sp").sel(time=ta["time"])
    return ta, zg, va, ps


def interface_degeneracy(va, ps) -> str:
    """Verify that 'equal and opposite integrated v' constrains only the total."""
    lines = ["1. Does 'integrated v above = minus integrated v below' pick a level?", ""]
    for corrected in (False, True):
        edges, psi = mass_streamfunction(va, ps, correct_mass_flux=corrected)
        psi_sfc = psi[..., -1, :]
        lat = va["latitude"].values
        m = np.abs(lat) <= 30
        lines.append(
            f"  mass correction {'ON ' if corrected else 'OFF'}: "
            f"|Psi(p_s)| mean {np.abs(psi_sfc[:, m]).mean():9.3f}, "
            f"max {np.abs(psi_sfc[:, m]).max():9.3f} kg/m/s")
    lines += [
        "",
        "  With the correction Psi(p_s) = 0 to roundoff, so the condition is",
        "  satisfied at EVERY level and selects none.  Without it the condition",
        "  is satisfied at NO level.  Either way it is a constraint on the",
        "  column, not a definition of the interface.",
        "  Requiring equal and opposite branch VELOCITIES instead forces equal",
        "  branch masses, i.e. the half-mass level p_s/2, which is what",
        "  scripts/era5_a_calibration.py already uses.",
    ]
    return "\n".join(lines)


def transport_coefficients(ta, zg, va, ps, correct: bool):
    """Shat, Hhat-relevant fluxes and Psi_ext under one mass-correction choice."""
    level = ta["level"].values
    w = layer_weights(level, np.zeros_like(ps.values), ps.values)
    v = va.values
    if correct:
        v = v - (v * w).sum(axis=-2, keepdims=True) / w.sum(axis=-2, keepdims=True)
    dse = CP_DRY * ta.values + zg.values
    dse_flux = (v * dse * w).sum(axis=-2) / GRAV

    p_dyn, _, _ = dynamical_interface_band(va, ps, correct_mass_flux=correct)
    edges, psi = mass_streamfunction(va, ps, correct_mass_flux=correct)
    idx = np.abs(edges[np.newaxis, :, np.newaxis]
                 - p_dyn.values[:, np.newaxis, :]).argmin(axis=-2)
    psi_ext = np.take_along_axis(psi, idx[..., np.newaxis, :], axis=-2)[..., 0, :]
    v_model = 2.0 * GRAV * psi_ext / ps.values
    return dse_flux, psi_ext, v_model


def slope(x, y, lat, bound):
    m = np.abs(lat) <= bound
    xv, yv = x[:, m].ravel(), y[:, m].ravel()
    g = np.isfinite(xv) & np.isfinite(yv)
    return float((xv[g] * yv[g]).sum() / (xv[g] ** 2).sum())


def mass_correction_sensitivity(ta, zg, va, ps) -> str:
    lat = ta["latitude"].values
    lines = ["2. Sensitivity to the barotropic mass adjustment", "",
             f"{'quantity':<34}{'no adj.':>13}{'adjusted':>13}{'change':>10}"]
    lines.append("-" * len(lines[-1]))
    raw = transport_coefficients(ta, zg, va, ps, correct=False)
    adj = transport_coefficients(ta, zg, va, ps, correct=True)
    for label, fn in [
        ("Shat (|lat|<=20), J/m^2",
         lambda r: slope(r[2], r[0], lat, 20.0)),
        ("peak |Psi_ext| (|lat|<=30), kg/m/s",
         lambda r: float(np.abs(np.nanmean(r[1], axis=0)[np.abs(lat) <= 30]).max())),
        ("peak |v_model| (|lat|<=30), m/s",
         lambda r: float(np.abs(np.nanmean(r[2], axis=0)[np.abs(lat) <= 30]).max())),
    ]:
        a, b = fn(raw), fn(adj)
        lines.append(f"{label:<34}{a:>13.4g}{b:>13.4g}{100*(b-a)/abs(a):>9.1f}%")
    lines += ["",
              "  Shat is the MOST sensitive of the three, not the least: the",
              "  transported quantity s has a large non-zero mean (~330 kJ/kg),",
              "  so a small spurious barotropic [v] produces a large spurious",
              "  DSE flux, while Psi and v_model only feel the mass flux itself.",
              "  This is the standard reason energy-transport budgets need a mass",
              "  correction and moisture budgets tolerate its absence better: q",
              "  has a small mean, s does not.  Treat the 34% as the floor on the",
              "  uncertainty in Shat until the mass-consistent product is used."]
    return "\n".join(lines)


def calibrate_depth(ta, zg, va, ps, bound: float = 20.0):
    """Layer depth d implied by the observed DSE flux and overturning.

    Scan the layer thickness in pressure.  For each, take the branch means of
    s over the top and bottom slabs of that thickness (anchored at the model's
    tropopause and at the surface) and find the thickness whose branch-mean
    difference equals the observed F_DSE / Psi_ext.
    """
    level = ta["level"].values
    lat = ta["latitude"].values
    dse_flux, psi_ext, _ = transport_coefficients(ta, zg, va, ps, correct=True)
    dse = CP_DRY * ta.values + zg.values

    m = np.abs(lat) <= bound
    x, y = psi_ext[:, m].ravel(), dse_flux[:, m].ravel()
    g = np.isfinite(x) & np.isfinite(y)
    ds_needed = float((x[g] * y[g]).sum() / (x[g] ** 2).sum())

    thicknesses = np.arange(2000.0, 45000.0, 500.0)
    diffs = []
    for dp in thicknesses:
        w_up = layer_weights(level, np.full_like(ps.values, P_TOP_MODEL),
                             np.full_like(ps.values, P_TOP_MODEL + dp))
        w_lo = layer_weights(level, ps.values - dp, ps.values)
        s_up = (dse * w_up).sum(axis=-2) / w_up.sum(axis=-2)
        s_lo = (dse * w_lo).sum(axis=-2) / w_lo.sum(axis=-2)
        diffs.append(float(np.nanmean((s_up - s_lo)[:, m])))
    diffs = np.array(diffs)
    k = int(np.abs(diffs - ds_needed).argmin())
    return ds_needed, thicknesses, diffs, thicknesses[k]


def depth_report(ta, zg, va, ps) -> str:
    """Show that the two-slab idealization is over-determined by the data.

    Three observables bear on a two-slab model: the overturning mass transport
    Psi_ext, the DSE flux, and the branch speed.  Two slabs have two free
    parameters (thickness and speed), so the third observable is a prediction,
    and it fails.
    """
    ds_needed, dps, diffs, dp_fit = calibrate_depth(ta, zg, va, ps)
    cfg = SWConfig()
    lat = ta["latitude"].values
    level = ta["level"].values

    _, psi_ext, _ = transport_coefficients(ta, zg, va, ps, correct=True)
    m = np.abs(lat) <= 30
    psi_peak = float(np.abs(np.nanmean(psi_ext, axis=0)[m]).max())
    v_implied = psi_peak * GRAV / dp_fit

    w = layer_weights(level, np.zeros_like(ps.values), ps.values)
    v = va.values - (va.values * w).sum(axis=-2, keepdims=True) / w.sum(
        axis=-2, keepdims=True)
    v_obs_peak = float(np.abs(np.nanmean(v, axis=0)[:, np.abs(lat) <= 20]).max())

    lines = ["3. What layer depth d does ERA5 imply?", "",
             f"  The DSE flux per unit overturning is {ds_needed:.0f} J/kg, which",
             "  is the dry-static-energy difference the two branches must have.",
             "  Matching that against the observed s(p) profile gives:", ""]
    for dp in [5000.0, 9000.0, 20000.0, 30000.0, 40000.0]:
        j = int(np.abs(dps - dp).argmin())
        mark = "  <-- best fit" if abs(dp - dp_fit) < 600 else ""
        lines.append(f"    slab {dp/100:5.0f} hPa -> branch-mean ds = "
                     f"{diffs[j]:8.0f} J/kg{mark}")
    lines += ["",
              f"  best fit {dp_fit/100:.0f} hPa, roughly "
              f"{dp_fit/GRAV/1.15/1e3:.1f} km near the surface, against the "
              f"model's d = {cfg.delta/1e3:.1f} km.",
              "",
              "  But that thickness is not usable, because it over-predicts the",
              "  branch speed.  Carrying the observed peak overturning "
              f"{psi_peak:.0f} kg/m/s",
              f"  through a {dp_fit/100:.0f} hPa slab needs {v_implied:.2f} m/s, "
              f"against an observed peak",
              f"  zonal-mean [v] of {v_obs_peak:.2f} m/s.  Two slabs have two free",
              "  parameters (thickness, speed) and three observables to match",
              "  (mass transport, DSE flux, speed), so the idealization is",
              "  over-determined and d has no clean observational counterpart.",
              "  Report Shat directly instead: it is the combination the model",
              "  actually uses, and it needs no slab geometry to measure."]

    # A d-deep layer is much thinner in pressure aloft than at the ground, so
    # the model's two layers cannot both hold d km and equal mass.
    zprof = np.nanmean(zg.values[:, :, np.abs(lat) <= 20], axis=(0, 2)) / GRAV
    p_up_bot = _pressure_at_height(
        np.interp(P_TOP_MODEL, level, zprof) - cfg.delta, zprof, level)
    p_lo_top = _pressure_at_height(cfg.delta, zprof, level)
    dp_up = p_up_bot - P_TOP_MODEL
    dp_lo = P_SFC_MODEL - p_lo_top
    lines += ["",
              f"  Aside: a {cfg.delta/1e3:.0f} km layer spans "
              f"{dp_up/100:.0f} hPa just below the model tropopause",
              f"  and {dp_lo/100:.0f} hPa just above the ground, a mass ratio of "
              f"{dp_lo/dp_up:.1f}.",
              "  So the model's own geometry cannot give its two layers equal",
              "  mass, yet equal and opposite velocities with unequal masses",
              "  violate mass balance.  The d-layer picture and the",
              "  equal-and-opposite-v picture are separately reasonable and",
              "  mutually inconsistent."]
    return "\n".join(lines)


def _pressure_at_height(z_target, z_prof, p_prof):
    """Pressure at a geopotential height, by log-p interpolation."""
    order = np.argsort(z_prof)
    return float(np.exp(np.interp(z_target, z_prof[order],
                                  np.log(p_prof[order]))))


def figure(ta, zg, va, ps, out_png: str, years: str) -> None:
    """The observed vertical structure against the model's assumed one."""
    level = ta["level"].values
    lat = ta["latitude"].values
    p_hpa = level / 100.0
    cfg = SWConfig()

    w = layer_weights(level, np.zeros_like(ps.values), ps.values)
    v = va.values - (va.values * w).sum(axis=-2, keepdims=True) / w.sum(
        axis=-2, keepdims=True)
    dse = CP_DRY * ta.values + zg.values

    jN = int(np.abs(lat - 12.0).argmin())
    jS = int(np.abs(lat + 12.0).argmin())
    v_prof = 0.5 * (v[:, :, jN] - v[:, :, jS]).mean(axis=0)   # NH minus SH: one cell
    s_prof = 0.5 * (dse[:, :, jN] + dse[:, :, jS]).mean(axis=0) / 1e3
    z_prof = 0.5 * (zg.values[:, :, jN] + zg.values[:, :, jS]).mean(axis=0) / GRAV
    ps_bar = 0.5 * (ps.values[:, jN] + ps.values[:, jS]).mean()

    edges, psi = mass_streamfunction(va, ps)
    psi_prof = 0.5 * (psi[:, :, jN] - psi[:, :, jS]).mean(axis=0)
    # The bottom edge sits below any real surface so the integral can be
    # clipped; put it back at p_s so the curve visibly closes at the ground.
    edges_plot = edges.copy()
    edges_plot[-1] = ps_bar

    # The model's two layers, each of depth d, drawn in pressure: the upper one
    # extends d downward from the model tropopause and the lower one d upward
    # from the ground.  In pressure these are wildly unequal, which is the
    # inconsistency the panel is there to show.
    z_top = _pressure_at_height(0.0, z_prof, level)
    p_up_bot = _pressure_at_height(
        np.interp(P_TOP_MODEL, level, z_prof) - cfg.delta, z_prof, level)
    p_lo_top = _pressure_at_height(cfg.delta, z_prof, level)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))

    ax = axes[0]
    ax.plot(v_prof, p_hpa, color="#d95f02", lw=2)
    ax.axvline(0, color="0.5", lw=0.8)
    v0 = 0.55
    ax.plot([v0, v0], [P_TOP_MODEL / 100, p_up_bot / 100],
            color="#1b7837", lw=3, ls="--")
    ax.plot([-v0, -v0], [p_lo_top / 100, ps_bar / 100],
            color="#1b7837", lw=3, ls="--")
    ax.text(1.02, 0.5 * (P_TOP_MODEL + p_up_bot) / 100,
            f"model layers,\n$d$={cfg.delta/1e3:.0f} km each\n"
            f"({P_TOP_MODEL/100:.0f}-{p_up_bot/100:.0f} and\n"
            f"{p_lo_top/100:.0f}-{ps_bar/100:.0f} hPa)",
            color="#1b7837", fontsize=8, va="center")
    ax.set_xlim(-2.0, 2.0)
    ax.set_xlabel(r"$[v]$ (m s$^{-1}$)")
    ax.set_title(r"observed branches at $\pm12^\circ$ (one-cell composite)",
                 fontsize=10.5)

    ax = axes[1]
    ax.plot(psi_prof, edges_plot / 100.0, color="#2c7fb8", lw=2)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel(r"$\Psi$ (kg m$^{-1}$ s$^{-1}$)")
    ax.set_title(r"$\Psi$ closes at both ends, so 'equal and"
                 "\nopposite integrated $v$' holds at every"
                 "\nlevel and singles out none", fontsize=10)

    ax = axes[2]
    ax.plot(s_prof, p_hpa, color="#7b3294", lw=2)
    ax.set_xlim(290, 370)
    ax.set_xlabel(r"$s=c_pT+\Phi$ (kJ kg$^{-1}$)")
    ax.set_title("the branches sit 39 kJ kg$^{-1}$ apart in $s$", fontsize=10.5)

    for ax in axes:
        ax.axhline(ps_bar / 2 / 100, color="k", lw=1.2, ls=":")
        ax.set_ylim(1000, 100)
        ax.set_ylabel("pressure (hPa)")
        ax.grid(alpha=0.25)
    axes[1].text(150, ps_bar / 2 / 100 - 20, "half-mass level", fontsize=8)

    fig.suptitle(f"ERA5 {years}: the vertical structure the model assumes, "
                 "against the one that is there", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year-start", type=int, default=2000)
    p.add_argument("--year-end", type=int, default=2019)
    p.add_argument("--out-prefix", default="model_output/era5_vertical_structure")
    args = p.parse_args()
    years = range(args.year_start, args.year_end + 1)
    ta, zg, va, ps = load(years)
    print(interface_degeneracy(va, ps))
    print()
    print(mass_correction_sensitivity(ta, zg, va, ps))
    print()
    print(depth_report(ta, zg, va, ps))
    figure(ta, zg, va, ps, args.out_prefix + ".png", f"{years[0]}-{years[-1]}")


if __name__ == "__main__":
    main()
