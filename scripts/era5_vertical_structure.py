"""What ERA5 says about the model's assumed vertical structure.

Three questions, all about where the model's two branches are and how thick:

1. **What level separates the two branches?**  The level of nondivergence,
   where the zonal-mean [v] reverses sign: everything above it is the poleward
   branch and everything below it the equatorward one, so integrating [v] over
   each side gives the full poleward and equatorward transports, and time-mean
   mass conservation makes those two equal and opposite.  The check below is
   that mass conservation, Psi(p_s) = 0, which the barotropic adjustment of
   question 2 is what enforces.  (At a level other than the sign change each
   integral mixes air moving both ways, so an equality there would be a
   cancellation rather than a statement about branches.)  Separately, the model
   gives its branches equal and opposite *velocities* as well as equal and
   opposite mass transports, which forces their masses to be equal, i.e. equal
   depths in PRESSURE.

2. **How much does the barotropic mass adjustment matter?**  ERA5's zonal-mean
   [v] on pressure levels carries a spurious net column mass flux.  Every
   transport coefficient here is reported with and without removing it.

3. **What layer depth d does ERA5 imply?**  The model puts the flow in a layer
   of depth d at the top of the troposphere and another of depth d at the
   bottom, with v = 0 between them.  Given the observed overturning strength
   Psi_ext and the observed DSE flux, the branch-mean dry static energies must
   differ by F_DSE / Psi_ext; d is then whatever layer thickness reproduces
   that difference from the observed s(p) profile.  That is one of three
   routes to a depth, and they do not agree; see
   ``scripts/era5_normalization.py``, which puts all three on one axis.

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
from ss09.moist_constants import gross_dry_stability  # noqa: E402
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
    """Verify the column mass balance the branch interface rests on."""
    lines = ["1. The branch interface and the column mass balance under it", ""]
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
        "  With the correction Psi(p_s) = 0 to roundoff, which is time-mean",
        "  column mass conservation: the poleward and equatorward branch",
        "  transports are equal and opposite.  Without it they are not, by up to",
        "  780 kg/m/s, and every transport coefficient inherits that error.",
        "  The level separating the two branches is where [v] reverses sign,",
        "  i.e. where Psi is extremal, which is what dynamical_interface_band",
        "  returns.  In the deep tropics Psi is nearly flat across the layer",
        "  between the branches, so that level is a band hundreds of hPa wide",
        "  rather than a sharp level, and the figures report it as a band.",
        "  Equal and opposite branch VELOCITIES additionally force equal branch",
        "  masses, i.e. equal depths in PRESSURE, not equal geometric depths.",
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
    shat_model = gross_dry_stability(GRAV, cfg.delta, cfg.delta_z, cfg.height)
    dp_shat = GRAV * shat_model / ds_needed
    j = int(np.abs(dps - dp_shat).argmin())
    lines += ["",
              f"  best fit {dp_fit/100:.0f} hPa, roughly "
              f"{dp_fit/GRAV/1.15/1e3:.1f} km near the surface, against the "
              f"model's d = {cfg.delta/1e3:.1f} km.",
              "",
              "  This is one of three routes to a slab depth, and they disagree:",
              "",
              f"    from the s(p) profile (above)                  "
              f"{dp_fit/100:5.0f} hPa",
              f"    from the model's own Shat = {shat_model:.3e} J/m^2   "
              f"{dp_shat/100:5.0f} hPa",
              "    from the shape of the observed [v]              "
              "  238 hPa   (era5_normalization.py)",
              "",
              f"  The middle one is the definition Shat = dp * ds / g inverted, so"
              f"\n  the model's Shat is equivalent to a {dp_shat/100:.0f} hPa "
              "slab pair.  Two slabs of THAT",
              f"  thickness taken from the observed s(p) profile differ by only "
              f"{diffs[j]:.0f} J/kg,",
              f"  {100*(diffs[j] - ds_needed)/ds_needed:+.0f}% short of the "
              f"{ds_needed:.0f} the transport needs, because the real branches",
              "  correlate [v] with s inside themselves and a uniform slab cannot.",
              "  That residual is the honest uncertainty on d: real, but a ~15%",
              "  effect, not the factor of 2.5 that mixing normalisations produced.",
              "",
              f"  Carrying the observed peak overturning {psi_peak:.0f} kg/m/s "
              f"through a {dp_fit/100:.0f} hPa slab",
              f"  needs {v_implied:.2f} m/s against an observed peak zonal-mean "
              f"[v] of {v_obs_peak:.2f} m/s;",
              f"  through a {dp_shat/100:.0f} hPa slab it needs "
              f"{psi_peak*GRAV/dp_shat:.2f} m/s, which the observed profile "
              "accommodates.",
              "  Report Shat directly where possible: it is the combination the",
              "  model actually uses, and it needs no slab geometry to measure."]

    # A d-deep layer is much thinner in pressure aloft than at the ground, so
    # the model's two layers cannot both hold d km and equal mass.
    zprof = np.nanmean(zg.values[:, :, np.abs(lat) <= 20], axis=(0, 2)) / GRAV
    p_up_bot = _pressure_at_height(
        np.interp(P_TOP_MODEL, level, zprof) - cfg.delta, zprof, level)
    p_lo_top = _pressure_at_height(cfg.delta, zprof, level)
    dp_up = p_up_bot - P_TOP_MODEL
    dp_lo = P_SFC_MODEL - p_lo_top
    lines += ["",
              f"  Why d must be read in pressure: a {cfg.delta/1e3:.0f} km layer "
              f"spans {dp_up/100:.0f} hPa just below the",
              f"  model tropopause and {dp_lo/100:.0f} hPa just above the ground, "
              f"a mass ratio of {dp_lo/dp_up:.1f}.",
              "  Equal GEOMETRIC depths therefore give the two layers unequal",
              "  masses, which equal and opposite velocities cannot have; equal",
              "  PRESSURE depths give equal masses and are consistent.  Both",
              f"  bracket the {dp_shat/100:.0f} hPa the model's Shat implies, which "
              "is why the model's",
              "  dry stability turns out defensible once d is read that way."]
    return "\n".join(lines)


def _pressure_at_height(z_target, z_prof, p_prof):
    """Pressure at a geopotential height, by log-p interpolation."""
    order = np.argsort(z_prof)
    return float(np.exp(np.interp(z_target, z_prof[order],
                                  np.log(p_prof[order]))))


def figure(ta, zg, va, ps, out_png: str, years: str, dp_shat: float) -> None:
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
    # Two ways to give the model's two layers a thickness.  Equal GEOMETRIC
    # depth (d km each) is what the model says literally, and it makes the two
    # layers hold different masses.  Equal PRESSURE depth is what equal and
    # opposite branch velocities require, and dp_shat is the depth the model's
    # own Shat corresponds to.
    v0, v1 = 0.55, 1.15
    for p_a, p_b in [(P_TOP_MODEL, p_up_bot), (p_lo_top, ps_bar)]:
        sign = 1.0 if p_a < 5e4 else -1.0
        ax.plot([sign * v0, sign * v0], [p_a / 100, p_b / 100],
                color="#1b7837", lw=3.5, ls="--", solid_capstyle="butt")
    for p_a, p_b in [(P_TOP_MODEL, P_TOP_MODEL + dp_shat),
                     (ps_bar - dp_shat, ps_bar)]:
        sign = 1.0 if p_a < 5e4 else -1.0
        ax.plot([sign * v1, sign * v1], [p_a / 100, p_b / 100],
                color="#7b3294", lw=3.5, solid_capstyle="butt")
    ax.text(-v0, p_lo_top / 100 - 30,
            f"equal geometric depth, $d$={cfg.delta/1e3:.0f} km each\n"
            f"({(p_up_bot - P_TOP_MODEL)/100:.0f} and "
            f"{(ps_bar - p_lo_top)/100:.0f} hPa: unequal mass)",
            color="#1b7837", fontsize=8, ha="center", va="bottom")
    ax.text(v1, (P_TOP_MODEL + dp_shat) / 100 + 12,
            f"equal pressure depth, $dp$={dp_shat/100:.0f} hPa each\n"
            r"(the model's $\hat S$)",
            color="#7b3294", fontsize=8, ha="center", va="top")
    ax.set_xlim(-2.0, 2.0)
    ax.set_xlabel(r"$[v]$ (m s$^{-1}$)")
    ax.set_title(r"observed branches at $\pm12^\circ$ (one-cell composite)"
                 "\nagainst two readings of the model's layers", fontsize=10)

    ax = axes[1]
    ax.plot(psi_prof, edges_plot / 100.0, color="#2c7fb8", lw=2)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel(r"$\Psi$ (kg m$^{-1}$ s$^{-1}$)")
    ax.set_title(r"$\Psi$ closes at both ends: the two branch"
                 "\ntransports are equal and opposite, and the"
                 "\nsign change of $[v]$ divides them", fontsize=10)

    ax = axes[2]
    ax.plot(s_prof, p_hpa, color="#7b3294", lw=2)
    ax.set_xlim(290, 370)
    ax.set_xlabel(r"$s=c_pT+\Phi$ (kJ kg$^{-1}$)")
    ax.set_title("the observed transport needs the branches"
                 "\n39 kJ kg$^{-1}$ apart in $s$", fontsize=10)

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
    cfg = SWConfig()
    ds_needed = calibrate_depth(ta, zg, va, ps)[0]
    dp_shat = GRAV * gross_dry_stability(
        GRAV, cfg.delta, cfg.delta_z, cfg.height) / ds_needed
    figure(ta, zg, va, ps, args.out_prefix + ".png", f"{years[0]}-{years[-1]}",
           dp_shat)


if __name__ == "__main__":
    main()
