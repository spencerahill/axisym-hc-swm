"""Eddy moisture diffusivity D, and the RCE-vs-observed temperature gradient.

**D from the column moisture budget.**  From zonal-mean fields alone the eddy
moisture flux cannot be formed directly, because zonal averaging has already
destroyed the covariance.  The steady column budget routes around that:

    (1/(a cos p)) d/dp [cos p * F_total] = E - P
    =>  F_total(p) = (a / cos p) * int_{-pi/2}^{p} (E - P) cos p' dp'

so the *total* flux follows from evaporation and precipitation, which are
stored fields.  Subtracting the mean-meridional-circulation flux computed from
[v][q] leaves the eddy flux, here the sum of stationary and transient eddies.
Then D = -F_eddy / (dW/dy).

**RCE vs observed.**  In radiative-convective equilibrium a tropical column's
free troposphere sits on the moist adiabat through its own boundary layer, so
the RCE free-tropospheric theta tracks the boundary-layer equivalent potential
temperature.  Comparing the meridional structure of boundary-layer theta_e
against the observed free-tropospheric theta shows how much the circulation
flattens, and in which direction the RCE target should differ from what is
observed.

Run ``python scripts/era5_moisture_budget.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from era5_a_calibration import (ERA5_ROOT, GRAV, _merge_expver, column_integral,  # noqa: E402
                                cos_weighted_mean, layer_weights,
                                load_level_field, load_surface_field)

CP_DRY, L_V, KAPPA = 1004.6, 2.5e6, 287.04 / 1004.6
EARTH_R = 6371e3
RHO_W = 1000.0
SECONDS_PER_DAY = 86400.0


def zonal_mean_precip(years: range) -> xr.DataArray:
    """Zonal-mean precipitation, kg m^-2 s^-1, streamed from the lat-lon file.

    The source is the only lat-lon file in this analysis (4 GB), so it is read
    one month at a time rather than opened whole; dask is not installed here.
    """
    path = ERA5_ROOT / "pr" / "era5_pr_monthly_1979-2020.nc"
    ds = xr.open_dataset(path)
    tp = ds["tp"].sel(time=slice(f"{years[0]}-01-01", f"{years[-1]}-12-31"))
    out = np.empty((tp.sizes["time"], tp.sizes["latitude"]))
    for i in range(tp.sizes["time"]):
        out[i] = tp.isel(time=i).mean("longitude").values
    # ERA5 monthly `tp` is the mean daily accumulation in metres of water.
    return xr.DataArray(out * RHO_W / SECONDS_PER_DAY,
                        dims=("time", "latitude"),
                        coords={"time": tp["time"], "latitude": tp["latitude"]})


def eddy_diffusivity(years: range, close_budget: bool = True) -> xr.Dataset:
    """Decompose the column moisture flux; ``close_budget`` removes the global
    E-P imbalance first.

    ERA5's global mean E - P is not zero (reanalysis increments and numerics),
    and integrating an unclosed source from pole to pole accumulates the whole
    imbalance into the flux, leaving a spurious residual at the far pole that
    is comparable to the tropical signal.  Removing the area-weighted global
    mean before integrating forces the flux to vanish at both poles.  This is
    the moisture-budget counterpart of the barotropic mass adjustment applied
    to [v].
    """
    hus = load_level_field("hus", "q", years)
    va = load_level_field("va", "v", years).sel(time=hus["time"])
    ps = load_surface_field("ps", "sp").sel(time=hus["time"])
    evap = _merge_expver(
        xr.open_dataset(ERA5_ROOT / "evap" / "era5_evap_monthly_1979-2023_znl-mean.nc")
        ["mer"]).sel(time=hus["time"])
    precip = zonal_mean_precip(years).sel(time=hus["time"])

    level = hus["level"].values
    lat = hus["latitude"].values
    phi = np.deg2rad(lat)
    cosphi = np.cos(phi)

    w = layer_weights(level, np.zeros_like(ps.values), ps.values)
    v_corr = va.values - (va.values * w).sum(axis=-2, keepdims=True) / w.sum(
        axis=-2, keepdims=True)
    f_mean = xr.DataArray((v_corr * hus.values * w).sum(axis=-2) / GRAV,
                          dims=ps.dims, coords=ps.coords)
    w_col = column_integral(hus, np.zeros_like(ps.values), ps.values)

    # E - P, with evaporation stored as a downward-positive surface flux.
    e_minus_p = (-evap.values) - precip.values

    # Integrate from the south pole; latitude runs north to south in ERA5.
    order = np.argsort(phi)
    imbalance = 0.0
    if close_budget:
        imbalance = (np.trapezoid(e_minus_p[:, order] * cosphi[order], phi[order],
                                  axis=-1)
                     / np.trapezoid(cosphi[order], phi[order]))
        e_minus_p = e_minus_p - imbalance[:, np.newaxis]
    integrand = (e_minus_p[:, order] * cosphi[order])
    cum = np.concatenate(
        [np.zeros((integrand.shape[0], 1)),
         np.cumsum(0.5 * (integrand[:, 1:] + integrand[:, :-1])
                   * np.diff(phi[order])[np.newaxis, :], axis=1)], axis=1)
    f_total_sorted = EARTH_R * cum / np.maximum(cosphi[order], 1e-6)
    f_total = np.empty_like(f_total_sorted)
    f_total[:, order] = f_total_sorted

    f_total = xr.DataArray(f_total, dims=ps.dims, coords=ps.coords)
    f_eddy = f_total - f_mean
    dwdy = xr.DataArray(np.gradient(w_col.values, phi, axis=-1) / EARTH_R,
                        dims=ps.dims, coords=ps.coords)
    # Evaporation as a positive upward moisture source, and the 200 hPa
    # zonal-mean zonal wind: two more anchors for the model scorecard.
    ua = load_level_field("ua", "u", years).sel(time=hus["time"])
    return xr.Dataset({"f_mean": f_mean, "f_total": f_total, "f_eddy": f_eddy,
                       "dwdy": dwdy, "w_col": w_col,
                       "evap": -evap,
                       "u200": ua.sel(level=20000.0),
                       "e_minus_p": xr.DataArray(e_minus_p, dims=ps.dims,
                                                 coords=ps.coords)},
                      attrs={"years": f"{years[0]}-{years[-1]}"})


def rce_comparison(years: range) -> xr.Dataset:
    """Boundary-layer theta_e against free-tropospheric theta."""
    ta = load_level_field("ta", "t", years)
    hus = load_level_field("hus", "q", years).sel(time=ta["time"])
    ps = load_surface_field("ps", "sp").sel(time=ta["time"])
    level = ta["level"].values
    theta = ta.values * (1.0e5 / level[np.newaxis, :, np.newaxis]) ** KAPPA

    # Free-tropospheric theta, 200-850 hPa mass weighted.
    p_bot = np.minimum(np.full_like(ps.values, 85000.0), ps.values)
    w_ft = layer_weights(level, np.full_like(ps.values, 20000.0), p_bot)
    theta_ft = (theta * w_ft).sum(axis=-2) / w_ft.sum(axis=-2)

    # Boundary-layer theta_e, surface to 100 hPa above it.
    w_bl = layer_weights(level, ps.values - 10000.0, ps.values)
    t_bl = (ta.values * w_bl).sum(axis=-2) / w_bl.sum(axis=-2)
    q_bl = (hus.values * w_bl).sum(axis=-2) / w_bl.sum(axis=-2)
    th_bl = (theta * w_bl).sum(axis=-2) / w_bl.sum(axis=-2)
    theta_e_bl = th_bl * np.exp(L_V * q_bl / (CP_DRY * t_bl))

    dims, coords = ps.dims, ps.coords
    return xr.Dataset({"theta_ft": xr.DataArray(theta_ft, dims=dims, coords=coords),
                       "theta_e_bl": xr.DataArray(theta_e_bl, dims=dims, coords=coords)},
                      attrs={"years": f"{years[0]}-{years[-1]}"})


def report(bud: xr.Dataset, rce: xr.Dataset) -> str:
    lat = bud["latitude"].values
    ann = bud.groupby("time.month").mean("time").mean("month")
    lines = [f"Eddy moisture diffusivity, ERA5 {bud.attrs['years']}", "",
             f"{'lat':>6}{'F_total':>10}{'F_mean':>10}{'F_eddy':>10}"
             f"{'dW/dy':>12}{'D (m^2/s)':>12}"]
    lines.append("-" * len(lines[-1]))
    for L in [25, 20, 15, 10, 5, -5, -10, -15, -20, -25]:
        j = int(np.abs(lat - L).argmin())
        d = -float(ann["f_eddy"][j]) / float(ann["dwdy"][j])
        lines.append(f"{lat[j]:>6.1f}{float(ann['f_total'][j]):>10.2f}"
                     f"{float(ann['f_mean'][j]):>10.2f}"
                     f"{float(ann['f_eddy'][j]):>10.2f}"
                     f"{float(ann['dwdy'][j]):>12.2e}{d:>12.3e}")

    band = (np.abs(lat) <= 25) & (np.abs(ann["dwdy"].values) > 2e-7)
    d_all = (-ann["f_eddy"].values / ann["dwdy"].values)[band]
    lines += ["",
              f"  |lat|<=25, excluding where |dW/dy| < 2e-7 kg/m^3:",
              f"  median D = {np.median(d_all):.3e} m^2/s, "
              f"quartiles {np.percentile(d_all, 25):.3e} to "
              f"{np.percentile(d_all, 75):.3e}",
              "",
              f"  budget closure: F_total at 85N = {float(ann['f_total'][int(np.abs(lat-85).argmin())]):.2f}, "
              f"85S = {float(ann['f_total'][int(np.abs(lat+85).argmin())]):.2f} kg/m/s",
              "  (both must vanish; a large value means the E-P imbalance has",
              "   accumulated into the flux)",
              "",
              "  F_eddy here is stationary plus transient eddies together, since",
              "  the mean term comes from zonal-mean [v][q] and so is the mean",
              "  meridional circulation alone."]

    ra = rce.groupby("time.month").mean("time").mean("month")
    i0 = int(np.abs(lat).argmin())
    lines += ["", f"RCE check: boundary-layer theta_e vs free-tropospheric theta",
              "", "  (16, 24 and 33 deg are the Coriolis-mapped images of the",
              "   model's y = 2, 3 and 4 Mm; see era5_normalization.py)", "",
              f"{'lat':>6}{'theta_e(BL)':>13}{'theta(FT)':>11}"
              f"{'d theta_e':>11}{'d theta':>10}"]
    lines.append("-" * len(lines[-1]))
    for L in [0, 10, 16, 24, 33]:
        jp = int(np.abs(lat - L).argmin())
        jm = int(np.abs(lat + L).argmin())
        te = 0.5 * (float(ra["theta_e_bl"][jp]) + float(ra["theta_e_bl"][jm]))
        tf = 0.5 * (float(ra["theta_ft"][jp]) + float(ra["theta_ft"][jm]))
        lines.append(f"{L:>6d}{te:>13.1f}{tf:>11.1f}"
                     f"{float(ra['theta_e_bl'][i0]) - te:>11.2f}"
                     f"{float(ra['theta_ft'][i0]) - tf:>10.2f}")
    j24p, j24m = int(np.abs(lat - 24).argmin()), int(np.abs(lat + 24).argmin())
    d_te = float(ra["theta_e_bl"][i0]) - 0.5 * (float(ra["theta_e_bl"][j24p])
                                                + float(ra["theta_e_bl"][j24m]))
    d_tf = float(ra["theta_ft"][i0]) - 0.5 * (float(ra["theta_ft"][j24p])
                                              + float(ra["theta_ft"][j24m]))
    lines += ["",
              "  Boundary-layer theta_e is the profile a column-by-column RCE free",
              f"  troposphere would sit on.  Its equator-to-24 deg contrast is "
              f"{d_te:.1f} K",
              f"  against {d_tf:.1f} K for the observed free troposphere, a factor "
              f"of {d_te/d_tf:.1f}, so",
              "  the circulation flattens the tropical free troposphere and an RCE",
              "  target should be STEEPER than the observed temperature structure.",
              "",
              "  CAVEAT (Spencer, 2026-08-01): this proxy is contaminated.  The",
              "  boundary-layer theta_e in a reanalysis already carries the",
              "  circulation's influence, and a true latitude-by-latitude RCE has",
              "  time-mean v = w = 0 by definition, so it is not an observable",
              "  state of the real atmosphere at all.  These two numbers therefore",
              "  BOUND the radiative-convective contrast rather than determine it:",
              "  treat Delta_theta^rad as a tuning parameter, with ERA5 setting the",
              "  limits and not the value."]

    lines += ["", "Two more anchors the model can be scored against:", ""]
    for b in (10.0, 20.0, 30.0):
        lines.append(f"  E_0 (mean evaporation, |lat|<={int(b)}) = "
                     f"{float(cos_weighted_mean(bud['evap'], b).mean()):.3e} "
                     "kg/m^2/s")
    lines.append(f"  model default E_0 = 4.6e-05 kg/m^2/s")
    u200 = bud["u200"].groupby("time.month").mean("time").mean("month")
    j = int(np.abs(u200.values).argmax())
    lines.append(f"  peak 200 hPa zonal-mean [u] = {float(u200[j]):.1f} m/s at "
                 f"{float(u200['latitude'][j]):+.1f} deg")
    return "\n".join(lines)


def figure(bud: xr.Dataset, rce: xr.Dataset, out_png: str) -> None:
    lat = bud["latitude"].values
    ann = bud.groupby("time.month").mean("time").mean("month")
    ra = rce.groupby("time.month").mean("time").mean("month")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    for key, colr, label, x0, dy in [
            ("f_total", "#000000", "total (from $E-P$)", -20.0, 8.0),
            ("f_mean", "#2c7fb8", "mean ($[v][q]$)", -12.0, -9.0),
            ("f_eddy", "#d95f02", "eddy (residual)", 14.0, 9.0)]:
        ax.plot(lat, ann[key].values, color=colr, lw=2)
        k = int(np.abs(lat - x0).argmin())
        ax.text(lat[k], float(ann[key][k]) + dy, label, color=colr, fontsize=9,
                ha="center", va="bottom" if dy > 0 else "top")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xlim(-35, 35)
    ax.set_xlabel("latitude")
    ax.set_ylabel(r"column moisture flux (kg m$^{-1}$ s$^{-1}$)")
    ax.set_title("moisture budget decomposition", fontsize=11)
    ax.grid(alpha=0.25)

    # A pointwise ratio is useless here: dW/dy passes through zero at the ITCZ
    # and again at the equator, so D blows up.  Regress instead, which is what
    # a single bulk diffusivity means anyway.
    ax = axes[1]
    band = np.abs(lat) <= 35
    x = ann["dwdy"].values[band]
    y = -ann["f_eddy"].values[band]
    sc = ax.scatter(x * 1e6, y, c=lat[band], cmap="Spectral_r", s=14,
                    vmin=-35, vmax=35)
    d_fit = float((x * y).sum() / (x ** 2).sum())
    xs = np.array([x.min(), x.max()])
    ax.plot(xs * 1e6, d_fit * xs, color="k", lw=2)
    ax.plot(xs * 1e6, 1e6 * xs, color="#CC3311", lw=2, ls="--")
    ax.plot(xs * 1e6, 1.5e5 * xs, color="#1b7837", lw=2, ls=":")
    ax.text(0.03, 0.94, f"regression $D$ = {d_fit:.2e} m$^2$ s$^{{-1}}$",
            transform=ax.transAxes, fontsize=9)
    ax.text(0.03, 0.87, "model $D=10^6$", transform=ax.transAxes, fontsize=9,
            color="#CC3311")
    ax.text(0.03, 0.80, r"lat-lon estimate $1.5\times10^5$",
            transform=ax.transAxes, fontsize=9, color="#1b7837")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.axvline(0, color="0.6", lw=0.7)
    fig.colorbar(sc, ax=ax, label="latitude", pad=0.02)
    ax.set_xlabel(r"$\partial W/\partial y$ ($10^{-6}$ kg m$^{-3}$)")
    ax.set_ylabel(r"$-F_{\rm eddy}$ (kg m$^{-1}$ s$^{-1}$)")
    ax.set_title("eddy flux against the gradient it should follow", fontsize=11)
    ax.grid(alpha=0.25)

    ax = axes[2]
    i0 = int(np.abs(lat).argmin())
    for key, colr, label, x0, dy in [
            ("theta_e_bl", "#CC3311",
             "RCE proxy: BL $\\theta_e$\n(contaminated; an upper bound)", -22.0, 3.0),
            ("theta_ft", "#2c7fb8", r"observed free-tropospheric $\theta$",
             -22.0, -3.0)]:
        prof = float(ra[key][i0]) - ra[key].values
        ax.plot(lat, prof, color=colr, lw=2)
        k = int(np.abs(lat - x0).argmin())
        ax.text(lat[k], prof[k] + dy, label, color=colr, fontsize=9,
                ha="center", va="bottom" if dy > 0 else "top")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xlim(-35, 35)
    ax.set_xlabel("latitude")
    ax.set_ylabel(r"equator minus local value (K)")
    ax.set_title("how much the circulation flattens the tropics", fontsize=11)
    ax.grid(alpha=0.25)

    fig.suptitle(f"ERA5 {bud.attrs['years']}: eddy moisture transport and the "
                 "RCE temperature contrast", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year-start", type=int, default=2000)
    p.add_argument("--year-end", type=int, default=2009)
    p.add_argument("--out-prefix", default="model_output/era5_moisture_budget")
    args = p.parse_args()
    years = range(args.year_start, args.year_end + 1)
    bud = eddy_diffusivity(years)
    rce = rce_comparison(years)
    print(report(bud, rce))
    figure(bud, rce, args.out_prefix + ".png")


if __name__ == "__main__":
    main()
