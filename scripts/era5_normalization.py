"""Which velocity is the model's ``v``?  Every transport coefficient depends on it.

The moist model writes three transport terms that all share one symbol ``v``:

    moisture:       d/dy [ -(2a-1) v W ]
    thermodynamic:  (d Delta_z / H) dv/dy
    momentum:       the factor of 2 in the v equation

The first and third read ``v`` as the velocity of two branches that between them
exhaust the column, so each carries half the column mass; the second reads it as
the velocity of a slab of depth ``d`` at the top and another at the bottom, with
``v = 0`` in between.  Those are different velocities, related by

    v_slab / v_half = (p_s / 2) / dp,

with ``dp`` the slab's depth in pressure.  Any coefficient defined as
"transport per unit v" therefore scales linearly with ``dp``, and a coefficient
measured under one reading cannot be used with a coefficient measured under the
other.

This script measures all of them as a function of ``dp``, so the model's own
values can be placed on the same axis:

* ``Shat(dp)``   from the observed dry-static-energy flux,
* ``2a-1 (dp)``  from the observed column moisture flux,
* ``Hhat(dp)``   from the observed column MSE flux,
* the slab depth the observed ``[v]`` profile itself implies, measured from the
  wind shape alone with no energy data, which is what makes the comparison with
  the model's ``Shat = C d Delta_z / H`` a test rather than a fit.

It also carries the two small conversions the report needs: the Coriolis
mapping that turns the model's ``beta`` and ``y`` into latitudes, and the
observed branch speed under both readings.

Run ``python scripts/era5_normalization.py``.
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from era5_a_calibration import (GRAV, column_integral, cos_weighted_mean,  # noqa: E402
                                dynamical_interface_band, layer_weights,
                                load_level_field, load_surface_field,
                                mass_streamfunction, regress)
from ss09.moist_constants import L_V, column_heat_capacity, gross_dry_stability  # noqa: E402
from ss09.sw_config import SWConfig  # noqa: E402

CP_DRY = 1004.6            # J/kg/K, ERA5's dry-air heat capacity
P_TOP_MODEL = 19300.0      # the model's tropopause pressure, Pa
OMEGA = 7.292e-5           # s^-1
EARTH_R = 6.371e6          # m
BAND = 20.0                # latitude band for every regression, degrees

# Slab depths scanned, Pa.  The top of the range is p_s/2, where the slab
# reading degenerates into the half-column one.
DP_SCAN = np.arange(5000.0, 50001.0, 1000.0)


# --------------------------------------------------------------------------
# the observed transports
# --------------------------------------------------------------------------
def build(years: range) -> xr.Dataset:
    """Observed column fluxes and overturning, on one mass-adjusted ``[v]``."""
    ta = load_level_field("ta", "t", years)
    zg = load_level_field("zg", "z", years).sel(time=ta["time"])
    hus = load_level_field("hus", "q", years).sel(time=ta["time"])
    va = load_level_field("va", "v", years).sel(time=ta["time"])
    ps = load_surface_field("ps", "sp").sel(time=ta["time"])
    level = ta["level"].values

    w_full = layer_weights(level, np.zeros_like(ps.values), ps.values)
    v_corr = va.values - (va.values * w_full).sum(axis=-2, keepdims=True) / \
        w_full.sum(axis=-2, keepdims=True)
    dse = CP_DRY * ta.values + zg.values
    mse = dse + L_V * hus.values
    dims, coords = ps.dims, ps.coords

    def flux(field):
        return xr.DataArray((v_corr * field * w_full).sum(axis=-2) / GRAV,
                            dims=dims, coords=coords)

    p_dyn, _, _ = dynamical_interface_band(va, ps)
    edges, psi = mass_streamfunction(va, ps)
    idx = np.abs(edges[np.newaxis, :, np.newaxis]
                 - p_dyn.values[:, np.newaxis, :]).argmin(axis=-2)
    psi_ext = np.take_along_axis(psi, idx[..., np.newaxis, :], axis=-2)[..., 0, :]

    ds = xr.Dataset({
        "psi_ext": xr.DataArray(psi_ext, dims=dims, coords=coords),
        "p_dyn": p_dyn,
        "dse_flux": flux(dse),
        "mse_flux": flux(mse),
        "q_flux": flux(hus.values),
        "w_total": column_integral(hus, np.zeros_like(ps.values), ps.values),
        "ps": ps,
    })
    ds["v_half"] = 2.0 * GRAV * ds["psi_ext"] / ds["ps"]
    ds.attrs["years"] = f"{years[0]}-{years[-1]}"
    # Kept for the slab mass partition and the branch-shape measurement.
    ds["q_clim"] = hus.groupby("time.month").mean("time")
    ds["v_clim"] = va.groupby("time.month").mean("time")
    ds["ps_clim"] = ps.groupby("time.month").mean("time")
    return ds


FIELDS = ["psi_ext", "dse_flux", "mse_flux", "q_flux", "w_total"]


def coefficients(ds: xr.Dataset, dp) -> dict[str, float]:
    """``Shat``, ``2a-1`` and ``Hhat`` for one slab depth (scalar or per-column)."""
    v = GRAV * ds["psi_ext"] / dp
    w20 = float(cos_weighted_mean(ds["w_total"], BAND).mean())
    shat = regress(v, ds["dse_flux"], BAND)
    hhat = regress(v, ds["mse_flux"], BAND)
    coeff = regress(v * ds["w_total"], -ds["q_flux"], BAND)
    return {"shat": shat, "coeff": coeff, "hhat": hhat,
            "hhat_recon": shat - L_V * coeff * w20,
            "w_star": shat / (L_V * coeff), "w": w20}


def season_mean(ds: xr.Dataset, months=None) -> xr.Dataset:
    """Climatological mean over ``months`` (default: all twelve).

    ``coefficients`` applied to the raw monthly data regresses across every
    month and latitude at once, so it weights by transport and is dominated by
    the strong monsoon months.  The model is steady and has no seasonal cycle,
    so a time-mean circulation and its time-mean transport are the closer
    counterpart.  Which time mean matters: the annual mean of the tropical
    overturning is the small residual of two large opposing solstitial cells,
    while the model's single symmetric cell resembles an equinox season.  The
    report runs both.
    """
    sub = ds if months is None else ds.sel(
        time=ds["time"].dt.month.isin(list(months)))
    out = sub[FIELDS].groupby("time.month").mean("time").mean("month")
    out["ps"] = sub["ps"].groupby("time.month").mean("time").mean("month")
    out.attrs = dict(ds.attrs)
    return out


# --------------------------------------------------------------------------
# how thick are the observed branches, from the wind alone?
# --------------------------------------------------------------------------
def branch_depths(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Effective pressure depth of each branch of the annual-mean ``[v]``.

    For a profile that is a top hat of depth ``D``, the participation ratio
    ``(int |v| dp)^2 / int v^2 dp`` returns ``D`` exactly; for a smoothly peaked
    profile it returns the width of the equivalent top hat carrying the same
    mass transport at the same peak-equivalent speed.  It uses only the SHAPE of
    the observed wind, so the depth it returns is independent of every energy
    and moisture measurement, and comparing it with the depth the model's
    ``Shat`` implies is a test rather than a fit.
    """
    level = ds["level"].values
    lat = ds["latitude"].values
    v = ds["v_clim"].mean("month").values                    # (level, lat)
    ps = ds["ps_clim"].mean("month").values                  # (lat,)
    p_d = ds["p_dyn"].groupby("time.month").mean("time").mean("month").values

    w_all = layer_weights(level, np.zeros_like(ps), ps)
    v = v - (v * w_all).sum(axis=0, keepdims=True) / w_all.sum(axis=0, keepdims=True)

    out = []
    for p_top, p_bot in [(np.zeros_like(ps), p_d), (p_d, ps)]:
        w = layer_weights(level, p_top, p_bot)
        num = (np.abs(v) * w).sum(axis=0) ** 2
        den = (v ** 2 * w).sum(axis=0)
        out.append(np.where(den > 0, num / np.maximum(den, 1e-30), np.nan))
    return lat, out[0], out[1]


def slab_partition(ds: xr.Dataset, dp: float) -> xr.DataArray:
    """``(W_lower slab - W_upper slab) / W``: the literal two-slab coefficient.

    This is what ``2a-1`` would be if the atmosphere really moved as two
    uniform slabs of depth ``dp``, the lower one anchored at the ground and the
    upper one at the model's tropopause.  It is the slab-reading counterpart of
    the mass partition ``2a-1`` at the half-mass level, and it differs from the
    flux-matched coefficient by whatever correlation between ``[v]`` and ``[q]``
    the slab idealization discards.
    """
    q = ds["q_clim"]
    level = q["level"].values
    ps = ds["ps_clim"].values
    dims = ("month", "latitude")
    coords = {"month": ds["q_clim"]["month"], "latitude": ds["latitude"]}

    def col(p_top, p_bot):
        w = layer_weights(level, p_top, p_bot)
        return xr.DataArray((q.values * w).sum(axis=-2) / GRAV, dims=dims,
                            coords=coords)

    zeros = np.zeros_like(ps)
    w_up = col(zeros + P_TOP_MODEL, zeros + P_TOP_MODEL + dp)
    w_lo = col(ps - dp, ps)
    return (w_lo - w_up) / col(zeros, ps)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def dp_matching_model(ds: xr.Dataset, shat_model: float) -> float:
    """Slab depth at which the observed ``Shat`` equals the model's."""
    shats = np.array([coefficients(ds, dp)["shat"] for dp in DP_SCAN])
    return float(np.interp(shat_model, shats, DP_SCAN))


def report(ds: xr.Dataset) -> str:
    cfg = SWConfig()
    c_col = column_heat_capacity(GRAV)
    shat_model = gross_dry_stability(GRAV, cfg.delta, cfg.delta_z, cfg.height)
    ps20 = float(cos_weighted_mean(ds["ps"], BAND).mean())

    lat, dp_up, dp_lo = branch_depths(ds)
    core = (np.abs(lat) >= 5.0) & (np.abs(lat) <= 25.0)
    dp_shape = float(np.nanmean(0.5 * (dp_up + dp_lo)[core]))
    dp_fit = dp_matching_model(ds, shat_model)

    c_inv = coefficients(ds, 0.5 * ds["ps"])
    w_obs, w_star_obs = c_inv["w"], c_inv["w_star"]
    a_spec, a_fix = 0.85, 0.5 * (shat_model / (L_V * w_star_obs) + 1.0)
    w_quiescent = 50.0 + 14400.0 * 4.6e-5

    lines = [f"ERA5 {ds.attrs['years']}, |lat| <= {BAND:.0f}, mass-adjusted [v]",
             "",
             "0. The part of the answer that does not depend on the reading",
             "",
             "  Hhat = Shat - L_v (2a-1) W.  Both Shat and (2a-1) are transports",
             "  per unit v, so both scale with whatever normalisation v carries,",
             "  and two combinations of them do not:",
             "",
             f"      W* = Shat / (L_v (2a-1))  = {w_star_obs:.1f} kg/m^2  observed",
             f"      Hhat/Shat = 1 - W/W*      = {1 - w_obs / w_star_obs:.3f}      "
             f"observed at W = {w_obs:.1f}",
             "",
             "  W* is the column at which the gross moist stability changes sign,",
             "  so it alone decides which regime a run is in.  Comparing it with",
             "  the model settles the calibration without choosing a reading:",
             "",
             f"{'':>4}{'a':>8}{'2a-1':>8}{'W* (kg/m^2)':>14}"
             f"{'Hhat/Shat at W=' + f'{w_quiescent:.1f}':>22}"]
    lines.append("  " + "-" * (len(lines[-1]) - 2))
    for label, aval in [("spec", a_spec), ("ERA5", a_fix)]:
        wstar = shat_model / (L_V * (2 * aval - 1))
        lines.append(f"{label:>4}{aval:>8.3f}{2 * aval - 1:>8.3f}{wstar:>14.1f}"
                     f"{1 - w_quiescent / wstar:>22.3f}")
    lines += ["",
              f"  At the model's own Shat = {shat_model:.3e} J/m^2, the default "
              f"a = {a_spec:.2f} puts the",
              f"  sign change at W* = "
              f"{shat_model / (L_V * (2 * a_spec - 1)):.1f} kg/m^2, BELOW the "
              f"quiescent column {w_quiescent:.1f} that the",
              "  V1 defaults produce, so every V2 run at W_c = 50 started off "
              "already in",
              f"  the negative-stability corner.  Matching the observed W* "
              f"instead needs",
              f"  a = {a_fix:.3f}, which puts the same run at "
              f"Hhat/Shat = {1 - w_quiescent / w_star_obs:+.3f}.",
              "",
              "  Nothing above required knowing what v means.  The rest of this",
              "  report is about why the individual symbols looked so wrong.",
              "", "1. The two readings of v", "",
             f"{'reading':<26}{'Shat (J/m^2)':>15}{'2a-1':>9}{'a':>8}"
             f"{'Hhat (J/m^2)':>15}{'W* (kg/m^2)':>13}"]
    lines.append("-" * len(lines[-1]))

    rows = [("half-column, dp=p_s/2", 0.5 * ds["ps"])]
    rows += [(f"slab, dp={int(d/100)} hPa", d) for d in
             (15000.0, 19600.0, 20000.0, 25000.0, 30000.0)]
    for label, dp in rows:
        c = coefficients(ds, dp)
        lines.append(f"{label:<26}{c['shat']:>15.3e}{c['coeff']:>9.3f}"
                     f"{0.5 * (c['coeff'] + 1):>8.3f}{c['hhat']:>15.3e}"
                     f"{c['w_star']:>13.1f}")

    lines += ["",
              "  Every column scales linearly with dp, since v_slab/v_half =",
              f"  (p_s/2)/dp and the observed transports are fixed.  At the "
              f"observed p_s = {ps20/100:.0f} hPa",
              f"  the half-column reading is the dp = {ps20/200:.0f} hPa row of "
              "the same family."]

    lines += ["", "2. Which dp does the observed wind itself pick out?", "",
              f"  Effective branch depth (participation ratio of annual-mean [v],",
              f"  5-25 deg): upper {np.nanmean(dp_up[core])/100:.0f} hPa, "
              f"lower {np.nanmean(dp_lo[core])/100:.0f} hPa, "
              f"mean {dp_shape/100:.0f} hPa.",
              "  This uses the shape of the wind only: no energy, no moisture.",
              "",
              f"  The model's own Shat = C d Delta_z / H = {shat_model:.3e} J/m^2",
              f"  (d = {cfg.delta/1e3:.0f} km, Delta_z = {cfg.delta_z:.0f} K, "
              f"H = {cfg.height/1e3:.0f} km, C = {c_col:.3e} J/m^2/K)",
              f"  reproduces the observed DSE transport at dp = "
              f"{dp_fit/100:.0f} hPa."]
    c_shape = coefficients(ds, dp_shape)
    lines += [f"  At the wind-derived dp = {dp_shape/100:.0f} hPa the observed "
              f"Shat is {c_shape['shat']:.3e},",
              f"  so the model's Shat is "
              f"{100*(shat_model - c_shape['shat'])/c_shape['shat']:+.1f}% "
              "against an independent",
              "  estimate of the same quantity."]

    lines += ["", "3. What the mixed readings did to the gross moist stability",
              "", f"{'combination':<52}{'Hhat (J/m^2)':>14}"]
    lines.append("-" * len(lines[-1]))
    c_half = coefficients(ds, 0.5 * ds["ps"])
    c_fit = coefficients(ds, dp_fit)
    w20 = c_fit["w"]
    for label, coeff in [
            ("model Shat + half-column flux-matched 2a-1", c_half["coeff"]),
            ("model Shat + half-column mass partition 2a-1", 0.9002),
            ("model Shat + slab 2a-1 at the matched dp (CONSISTENT)",
             c_fit["coeff"])]:
        lines.append(f"{label:<52}{shat_model - L_V * coeff * w20:>14.3e}")
    lines += ["",
              f"  Observed tropical W = {w20:.1f} kg/m^2.  The consistent "
              "combination is",
              f"  POSITIVE, and puts the zero-Hhat column at W* = "
              f"{c_fit['w_star']:.1f} kg/m^2, above the",
              f"  quiescent column W_c + tau_c E_0 = "
              f"{50.0 + 14400.0 * 4.6e-5:.2f} at the V1 defaults."]

    lines += ["", "4. The literal two-slab moisture partition", ""]
    for dp in (15000.0, 19600.0, 25000.0):
        part = float(cos_weighted_mean(slab_partition(ds, dp), BAND).mean())
        c = coefficients(ds, dp)
        lines.append(f"  dp = {int(dp/100):3d} hPa: literal (W_lo-W_up)/W = "
                     f"{part:.3f} (a = {0.5*(part+1):.3f}), "
                     f"flux-matched {c['coeff']:.3f}")
    lines += ["",
              "  The flux-matched coefficient is the smaller one here, the",
              "  opposite of the half-column case.  A uniform 200 hPa slab at the",
              "  ground would carry more water than the real lower branch does,",
              "  because the real branch is deeper and slower and reaches up into",
              "  drier air; the half-column reading had the opposite bias because",
              "  it spread the transport over a 500 hPa branch."]

    lines += ["", "5. Branch speed under each reading", ""]
    ann_half = ds["v_half"].groupby("time.month").mean("time").mean("month")
    peak_half = float(np.abs(ann_half.sel(latitude=slice(30, -30))).max())
    lines += [f"  peak annual-mean |v| over |lat|<=30: {peak_half:.2f} m/s "
              "(half-column),",
              f"  {peak_half * ps20 / (2 * dp_fit):.2f} m/s (slab at dp = "
              f"{dp_fit/100:.0f} hPa), "
              f"{peak_half * ps20 / (2 * dp_shape):.2f} m/s (slab at the",
              f"  wind-derived dp = {dp_shape/100:.0f} hPa).  The model's v is a "
              "slab velocity, so",
              "  model-vs-ERA5 comparisons of max|v| must use one of the last two."]

    ann = season_mean(ds)
    eqx = season_mean(ds, months=(3, 4, 9, 10))
    lines += ["", "6. How is the time mean taken?  This is the largest "
              "uncertainty here.", "",
              f"{'time mean':<36}{'Shat':>12}{'2a-1':>9}{'a':>7}{'W*':>8}"
              f"{'Hhat/Shat':>11}"]
    lines.append("-" * len(lines[-1]))
    rows = []
    for label, src in [("every month, transport-weighted", ds),
                       ("annual mean, vbar Wbar", ann),
                       ("equinox months, vbar Wbar", eqx)]:
        c = coefficients(src, dp_fit)
        rows.append((label, c["coeff"], c["w_star"]))
        lines.append(f"{label:<36}{c['shat']:>12.3e}{c['coeff']:>9.3f}"
                     f"{0.5*(c['coeff']+1):>7.3f}{c['w_star']:>8.1f}"
                     f"{c['hhat'] / c['shat']:>11.3f}")
    # A fourth route: annual-mean fluxes, but with the moisture basis v*W
    # averaged month by month before the annual mean, so the seasonal covariance
    # of v and W is kept.  Only the moisture coefficient can differ from the
    # second row, since Shat and Hhat involve no product with W.
    c_ann = coefficients(ann, dp_fit)
    basis_mn = ((GRAV * ds["psi_ext"] / dp_fit * ds["w_total"])
                .groupby("time.month").mean("time").mean("month"))
    coeff_mn = regress(basis_mn, -ann["q_flux"], BAND)
    w_star_mn = c_ann["shat"] / (L_V * coeff_mn)
    rows.append(("annual mean, mean of v W", coeff_mn, w_star_mn))
    lines.append(f"{'annual mean, mean of v W':<36}{c_ann['shat']:>12.3e}"
                 f"{coeff_mn:>9.3f}{0.5*(coeff_mn+1):>7.3f}{w_star_mn:>8.1f}"
                 f"{1 - c_ann['w'] / w_star_mn:>11.3f}")

    lo = min(r[2] for r in rows)
    hi = max(r[2] for r in rows)
    lines += ["",
              f"  The four routes span 2a-1 = {min(r[1] for r in rows):.2f} to "
              f"{max(r[1] for r in rows):.2f} (a = "
              f"{0.5*(min(r[1] for r in rows)+1):.2f} to "
              f"{0.5*(max(r[1] for r in rows)+1):.2f}) and",
              f"  W* = {lo:.1f} to {hi:.1f} kg/m^2, a spread that straddles the "
              f"quiescent column",
              f"  {w_quiescent:.1f}: the sign of the model's gross moist "
              "stability at the V1",
              "  defaults is NOT settled by these data alone.",
              "",
              "  [plausible] The spread is seasonal covariance.  The last row "
              "keeps the",
              "  correlation between a strong lower branch and a moist column "
              "within the",
              "  seasonal cycle, so its basis is the largest and its coefficient "
              "the",
              "  smallest per unit transport; the annual-mean row removes it.  "
              "The equinox",
              "  row is the closest analogue of the model, which is steady, "
              "symmetric and",
              "  has no solstitial cell to average away.",
              "",
              f"  Recommendation: take a from the equinox row, and treat "
              f"{0.5*(min(r[1] for r in rows)+1):.2f}-"
              f"{0.5*(max(r[1] for r in rows)+1):.2f} as its",
              "  uncertainty.  The one firm statement is comparative: the model's",
              f"  default a = {a_spec:.2f} sits at the DRY end of that range and "
              "gives the lowest",
              "  W* of any of them, so it is the choice most likely to put a run "
              "in the",
              "  negative-stability corner, which is what happened.",
              "", "7. beta and the latitude mapping", ""]
    phi = np.linspace(0.0, np.deg2rad(30.0), 4001)
    beta_eq = 2 * OMEGA / EARTH_R
    beta_avg = (2 * OMEGA * np.trapezoid(np.cos(phi) ** 2, phi)
                / (EARTH_R * np.trapezoid(np.cos(phi), phi)))
    beta_15 = 2 * OMEGA * np.cos(np.deg2rad(15.0)) / EARTH_R
    lines += [f"  model beta = {cfg.beta:.3e} = 2 Omega cos(phi)/a at phi = "
              f"{np.rad2deg(np.arccos(cfg.beta * EARTH_R / (2 * OMEGA))):.1f} deg",
              f"  equatorial 2 Omega/a          = {beta_eq:.4e}",
              f"  at 15 deg                     = {beta_15:.4e}",
              f"  cos-weighted mean over 0-30   = {beta_avg:.4e}  "
              f"({100 * (beta_avg / cfg.beta - 1):+.1f}% vs the model)"]
    for y in (1.5e6, 3.0e6, 6.0e6):
        lat_cor = np.rad2deg(np.arcsin(np.clip(cfg.beta * y / (2 * OMEGA), -1, 1)))
        lines.append(f"  y = {y/1e3:5.0f} km -> {lat_cor:5.1f} deg by the "
                     f"Coriolis mapping f = beta y = 2 Omega sin(phi); "
                     f"{np.rad2deg(y / EARTH_R):5.1f} deg by arc length")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def figure_coefficients(ds: xr.Dataset, out_png: str) -> None:
    """Every transport coefficient against the slab depth that defines ``v``."""
    cfg = SWConfig()
    shat_model = gross_dry_stability(GRAV, cfg.delta, cfg.delta_z, cfg.height)
    dp_fit = dp_matching_model(ds, shat_model)
    lat, dp_up, dp_lo = branch_depths(ds)
    core = (np.abs(lat) >= 5.0) & (np.abs(lat) <= 25.0)
    dp_shape = float(np.nanmean(0.5 * (dp_up + dp_lo)[core]))
    ps20 = float(cos_weighted_mean(ds["ps"], BAND).mean())

    cs = [coefficients(ds, dp) for dp in DP_SCAN]
    shat = np.array([c["shat"] for c in cs])
    coeff = np.array([c["coeff"] for c in cs])
    hhat = np.array([c["hhat"] for c in cs])
    wstar = np.array([c["w_star"] for c in cs])
    w20 = cs[0]["w"]
    x = DP_SCAN / 100.0

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))

    def mark(ax):
        ax.axvline(dp_shape / 100, color="#1b7837", lw=1.6)
        ax.axvline(ps20 / 200, color="0.45", lw=1.2, ls=":")
        ax.set_xlabel("slab depth $dp$ (hPa)")
        ax.set_xlim(x[0], x[-1])
        ax.grid(alpha=0.25)

    ax = axes[0]
    ax.plot(x, shat / 1e6, color="#7b3294", lw=2.2)
    ax.axhline(shat_model / 1e6, color="#CC3311", lw=1.8, ls="--")
    ax.plot([dp_fit / 100], [shat_model / 1e6], "o", color="#CC3311", ms=7)
    ax.text(x[-1] - 5, shat_model / 1e6 + 10, r"model $\hat S=Cd\Delta_z/H$",
            color="#CC3311", fontsize=9, ha="right")
    ax.text(dp_fit / 100 + 12, shat_model / 1e6 - 12,
            f"matched at\n$dp={dp_fit/100:.0f}$ hPa", color="#CC3311",
            fontsize=8.5, va="top")
    k = int(np.abs(x - 350).argmin())
    ax.text(x[k], shat[k] / 1e6 + 16, r"ERA5 $\hat S$", color="#7b3294",
            fontsize=10.5, ha="center")
    ax.text(dp_shape / 100 + 7, 285, "branch depth\nfrom $[v]$ shape",
            color="#1b7837", fontsize=8.5, va="top")
    ax.text(ps20 / 200 - 7, 285, "half-column\nreading", color="0.45",
            fontsize=8.5, ha="right", va="top")
    ax.set_ylabel(r"$\hat S$ (MJ m$^{-2}$)")
    ax.set_ylim(0, 300)
    ax.set_title("gross dry stability", fontsize=11)
    mark(ax)

    ax = axes[1]
    ax.plot(x, coeff, color="#2c7fb8", lw=2.2)
    part = np.array([float(cos_weighted_mean(slab_partition(ds, dp), BAND).mean())
                     for dp in DP_SCAN])
    ax.plot(x, part, color="#d95f02", lw=1.8, ls="--")
    ax.axhline(1.0, color="0.3", lw=1.0, ls="-.")
    ax.text(x[0] + 5, 1.05, r"$2a-1=1$ ceiling", fontsize=8.5, color="0.3")
    k = int(np.abs(x - 440).argmin())
    ax.text(x[k], coeff[k] - 0.06, "flux-matched", color="#2c7fb8", fontsize=9.5,
            ha="center", va="top")
    ax.text(x[k], part[k] + 0.06, "literal two-slab", color="#d95f02",
            fontsize=9.5, ha="center")
    ax.set_ylabel(r"$2a-1$")
    ax.set_ylim(0, 1.8)
    ax.set_title("moisture transport coefficient", fontsize=11)
    mark(ax)

    ax = axes[2]
    ax.plot(x, hhat / 1e6, color="#1b7837", lw=2.2)
    ax.axhline(0, color="0.4", lw=1.0)
    k = int(np.abs(x - 380).argmin())
    ax.text(x[k], hhat[k] / 1e6 + 7, r"ERA5 $\hat H$", color="#1b7837",
            fontsize=10.5, ha="center")
    hh_mixed = shat_model - L_V * coeff * w20
    ax.plot(x, hh_mixed / 1e6, color="#CC3311", lw=1.8, ls="--")
    k = int(np.abs(x - 110).argmin())
    ax.text(x[k], hh_mixed[k] / 1e6 + 8,
            "model $\\hat S$ paired with\nthe $2a-1$ read at this $dp$",
            color="#CC3311", fontsize=8.5, ha="left")
    ax.plot([dp_fit / 100], [(shat_model - L_V * coeff[
        int(np.abs(DP_SCAN - dp_fit).argmin())] * w20) / 1e6], "o",
        color="k", ms=7)
    ax.text(dp_fit / 100 - 12, -10, "the only\nself-consistent\npairing",
            fontsize=8.5, va="top", ha="right")
    ax.set_ylabel(r"$\hat H$ (MJ m$^{-2}$)")
    ax.set_ylim(-90, 90)
    ax.set_title("gross moist stability", fontsize=11)
    mark(ax)

    fig.suptitle(f"ERA5 {ds.attrs['years']}: every transport coefficient scales "
                 r"with the slab depth that defines $v$   "
                 f"($|\\varphi|\\leq{BAND:.0f}^\\circ$)", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}   [dp_fit={dp_fit/100:.0f} hPa, "
          f"dp_shape={dp_shape/100:.0f} hPa, W*={wstar[0]:.0f}..]")


def figure_partition_map(ds: xr.Dataset, out_png: str, dp: float) -> None:
    """The slab moisture coefficient across latitude and season."""
    part = slab_partition(ds, dp).transpose("month", "latitude")
    lat = ds["latitude"].values
    months = part["month"].values

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    ax = axes[0]
    levels = np.arange(0.50, 0.721, 0.02)
    cf = ax.contourf(months, lat, part.values.T, levels=levels, cmap="YlGnBu",
                     extend="both")
    csl = ax.contour(months, lat, part.values.T, levels=levels[::2],
                     colors="0.25", linewidths=0.6)
    ax.clabel(csl, fmt="%.2f", fontsize=7)
    ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
    ax.set_ylim(-40, 40)
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                        "Sep", "Oct", "Nov", "Dec"], fontsize=8, rotation=45)
    ax.set_xlabel("month")
    ax.set_ylabel("latitude")
    ax.set_title(f"literal two-slab $2a-1$, $dp={dp/100:.0f}$ hPa", fontsize=11)
    fig.colorbar(cf, ax=ax, label="$2a-1$", pad=0.02)

    ax = axes[1]
    ann = part.mean("month").values
    ax.plot(lat, ann, color="#d95f02", lw=2.2)
    k = int(np.abs(lat - 26.0).argmin())
    ax.text(lat[k], ann[k] + 0.04, "literal two-slab", color="#d95f02",
            fontsize=9.5, ha="center")
    c = coefficients(ds, dp)
    ax.axhline(c["coeff"], color="#2c7fb8", lw=1.8, ls="--")
    ax.text(-38, c["coeff"] + 0.02,
            f"flux-matched over $|\\varphi|\\leq{BAND:.0f}^\\circ$: "
            f"{c['coeff']:.2f}", color="#2c7fb8", fontsize=9)
    ax.axhline(2 * 0.85 - 1, color="#1b7837", lw=1.4, ls=":")
    ax.text(-38, 2 * 0.85 - 1 + 0.008, "model default $a=0.85$", color="#1b7837",
            fontsize=9)
    ax.set_xlim(-40, 40)
    ax.set_ylim(0.45, 0.80)
    ax.set_xlabel("latitude")
    ax.set_ylabel("$2a-1$")
    ax.set_title("annual mean", fontsize=11)
    ax.grid(alpha=0.25)

    fig.suptitle(f"ERA5 {ds.attrs['years']}: the moisture coefficient the "
                 "two-slab geometry implies", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year-start", type=int, default=2000)
    p.add_argument("--year-end", type=int, default=2019)
    p.add_argument("--out-prefix", default="model_output/era5_normalization")
    args = p.parse_args()
    ds = build(range(args.year_start, args.year_end + 1))
    print(report(ds))
    cfg = SWConfig()
    dp_fit = dp_matching_model(
        ds, gross_dry_stability(GRAV, cfg.delta, cfg.delta_z, cfg.height))
    figure_coefficients(ds, args.out_prefix + "_coefficients.png")
    figure_partition_map(ds, args.out_prefix + "_partition.png", dp_fit)


if __name__ == "__main__":
    main()
