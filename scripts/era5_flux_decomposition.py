"""Column moisture and MSE fluxes, split into mean-circulation and eddy parts.

Both splits use the same route.  The MEAN part is formed directly from the
zonal means, ``(1/g) int [v][x] dp``, so it is the transport by the mean
meridional circulation alone.  The TOTAL comes from the column budget, whose
source term is a stored surface or top-of-atmosphere field, so it includes
every eddy the reanalysis contains, stationary and transient.  The EDDY part is
the difference.  Zonal averaging has already destroyed the covariance that
would let the eddy flux be formed directly, which is why the budget route is
necessary.

    moisture:  d/dy F_q = E - P
    energy:    d/dy F_h = (net downward at TOA) - (net downward at the surface)

with the meridional divergence in spherical geometry,
``(1/(a cos p)) d/dp [cos p * F]``, so

    F(p) = (a / cos p) * int_{-pi/2}^{p} S cos p' dp'.

Both budgets are closed by removing their global-mean imbalance before
integrating, exactly as ``era5_moisture_budget.py`` does for moisture; the
check that this works, and that every ERA5 sign convention has been read
correctly, is that the resulting flux vanishes at both poles.  That check is
printed, and it is the reason to trust the signs.

Moisture is reported in energy units, ``L_v`` times the mass flux, so the two
panels share an axis and the cancellation between the dry-static and latent
parts of the mean MSE flux is visible.

Run ``python scripts/era5_flux_decomposition.py``.
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from era5_a_calibration import (ERA5_ROOT, GRAV, _merge_expver,  # noqa: E402
                                layer_weights, load_level_field,
                                load_surface_field)
from era5_moisture_budget import EARTH_R, zonal_mean_precip  # noqa: E402
from ss09.moist_constants import L_V  # noqa: E402

CP_DRY = 1004.6
SECONDS_PER_DAY = 86400.0


def _load_flux(var: str, name: str, times) -> xr.DataArray:
    """A stored zonal-mean surface/TOA flux, in W m^-2, positive DOWNWARD.

    ERA5 distributes these as accumulations; depending on the download they
    arrive either as a mean rate in W m^-2 or as J m^-2 accumulated over one
    day.  Which one is decided from the magnitude and reported, rather than
    assumed, because getting it wrong would rescale every flux here by 86400
    while leaving the shape of every curve unchanged and so invisible.
    """
    path = next((ERA5_ROOT / var).glob(f"era5_{var}_monthly_*_znl-mean.nc"))
    da = _merge_expver(xr.open_dataset(path)[name]).sel(time=times)
    scale = 1.0 if float(np.abs(da).max()) < 1.0e4 else 1.0 / SECONDS_PER_DAY
    print(f"  {var:6s} ({name}): max|.| = {float(np.abs(da).max()):.4g}, "
          f"treating as {'W/m^2' if scale == 1.0 else 'J/m^2 per day'}")
    return da * scale


def meridional_flux(source: xr.DataArray, lat: np.ndarray,
                    close: bool = True) -> tuple[xr.DataArray, float]:
    """Integrate a column source meridionally into a flux; returns (F, imbalance).

    An unclosed global source accumulates its whole imbalance into the flux and
    leaves a spurious residual at the far pole comparable to the tropical
    signal, so the area-weighted global mean is removed first.
    """
    phi = np.deg2rad(lat)
    cosphi = np.cos(phi)
    order = np.argsort(phi)
    src = source.values
    imbalance = (np.trapezoid(src[:, order] * cosphi[order], phi[order], axis=-1)
                 / np.trapezoid(cosphi[order], phi[order]))
    if close:
        src = src - imbalance[:, np.newaxis]
    integrand = src[:, order] * cosphi[order]
    cum = np.concatenate(
        [np.zeros((integrand.shape[0], 1)),
         np.cumsum(0.5 * (integrand[:, 1:] + integrand[:, :-1])
                   * np.diff(phi[order])[np.newaxis, :], axis=1)], axis=1)
    out_sorted = EARTH_R * cum / np.maximum(cosphi[order], 1e-6)
    out = np.empty_like(out_sorted)
    out[:, order] = out_sorted
    return (xr.DataArray(out, dims=source.dims, coords=source.coords),
            float(np.mean(imbalance)))


def build(years: range) -> xr.Dataset:
    hus = load_level_field("hus", "q", years)
    va = load_level_field("va", "v", years).sel(time=hus["time"])
    ta = load_level_field("ta", "t", years).sel(time=hus["time"])
    zg = load_level_field("zg", "z", years).sel(time=hus["time"])
    ps = load_surface_field("ps", "sp").sel(time=hus["time"])
    times = hus["time"]
    lat = hus["latitude"].values
    level = hus["level"].values

    w = layer_weights(level, np.zeros_like(ps.values), ps.values)
    v = va.values - (va.values * w).sum(axis=-2, keepdims=True) / w.sum(
        axis=-2, keepdims=True)
    dse = CP_DRY * ta.values + zg.values
    dims, coords = ps.dims, ps.coords

    def mean_flux(field):
        return xr.DataArray((v * field * w).sum(axis=-2) / GRAV, dims=dims,
                            coords=coords)

    print("Field magnitudes and unit decisions:")
    evap = _merge_expver(
        xr.open_dataset(ERA5_ROOT / "evap" / "era5_evap_monthly_1979-2023_znl-mean.nc")
        ["mer"]).sel(time=times)
    precip = zonal_mean_precip(years).sel(time=times)
    # ERA5 stores evaporation as a downward-positive surface moisture flux, so
    # evaporation into the atmosphere is -mer.
    e_minus_p = xr.DataArray((-evap.values) - precip.values, dims=dims,
                             coords=coords)

    # Net energy into the atmospheric column: TOA net downward minus surface
    # net downward.  Every ERA5 field here is positive downward, so the
    # turbulent fluxes are already negative where the surface heats the air.
    toa = _load_flux("rsnt", "tsr", times) + _load_flux("olr", "ttr", times)
    sfc = (_load_flux("rsns", "ssr", times) + _load_flux("rlns", "str", times)
           + _load_flux("hfls", "slhf", times) + _load_flux("hfss", "sshf", times))
    net_column = toa - sfc

    f_q_total, imb_q = meridional_flux(e_minus_p, lat)
    f_h_total, imb_h = meridional_flux(net_column, lat)
    f_q_mean = mean_flux(hus.values)
    f_dse_mean = mean_flux(dse)
    f_h_mean = mean_flux(dse + L_V * hus.values)

    ds = xr.Dataset({
        "lvq_total": L_V * f_q_total, "lvq_mean": L_V * f_q_mean,
        "lvq_eddy": L_V * (f_q_total - f_q_mean),
        "mse_total": f_h_total, "mse_mean": f_h_mean,
        "mse_eddy": f_h_total - f_h_mean,
        "dse_mean": f_dse_mean,
        "net_column": net_column, "e_minus_p": e_minus_p,
    })
    ds.attrs["years"] = f"{years[0]}-{years[-1]}"
    ds.attrs["imbalance_moisture"] = imb_q
    ds.attrs["imbalance_energy"] = imb_h
    return ds


def report(ds: xr.Dataset) -> str:
    lat = ds["latitude"].values
    ann = ds.groupby("time.month").mean("time").mean("month")
    jN = int(np.abs(lat - 85).argmin())
    jS = int(np.abs(lat + 85).argmin())
    lines = ["", f"ERA5 {ds.attrs['years']}: column fluxes, mean vs eddy", "",
             "Budget closure (both must vanish at the poles; this is the check",
             "that every ERA5 sign convention was read correctly):",
             f"  L_v F_q at 85N/85S = "
             f"{float(ann['lvq_total'][jN])/1e6:+.2f} / "
             f"{float(ann['lvq_total'][jS])/1e6:+.2f} MW/m",
             f"  F_MSE at 85N/85S   = "
             f"{float(ann['mse_total'][jN])/1e6:+.2f} / "
             f"{float(ann['mse_total'][jS])/1e6:+.2f} MW/m",
             f"  removed global-mean imbalance: moisture "
             f"{ds.attrs['imbalance_moisture']*SECONDS_PER_DAY:.4f} mm/day, "
             f"energy {ds.attrs['imbalance_energy']:.2f} W/m^2",
             "  The moisture budget closes to a few percent of its tropical",
             "  signal; the energy budget leaves ~20% of its peak at the poles,",
             "  a latitude-dependent bias the global-mean removal cannot fix.",
             "  So the eddy MSE flux here is the weaker of the two splits, and",
             "  its tropical values carry that caveat.", ""]

    lines += [f"{'lat':>6}{'Lv F_q mean':>13}{'Lv F_q eddy':>13}"
              f"{'F_MSE mean':>12}{'F_MSE eddy':>12}{'DSE mean':>11}"
              f"{'F_MSE tot':>11}"]
    lines.append("-" * len(lines[-1]))
    for L in [30, 24, 20, 15, 10, 5, 0, -5, -10, -15, -20, -24, -30]:
        j = int(np.abs(lat - L).argmin())
        lines.append(
            f"{lat[j]:>6.1f}"
            f"{float(ann['lvq_mean'][j])/1e6:>13.1f}"
            f"{float(ann['lvq_eddy'][j])/1e6:>13.1f}"
            f"{float(ann['mse_mean'][j])/1e6:>12.1f}"
            f"{float(ann['mse_eddy'][j])/1e6:>12.1f}"
            f"{float(ann['dse_mean'][j])/1e6:>11.1f}"
            f"{float(ann['mse_total'][j])/1e6:>11.1f}")
    j15 = int(np.abs(lat - 15).argmin())
    fac = 2 * np.pi * EARTH_R * np.cos(np.deg2rad(15.0))
    lines += ["", "  All in MW/m (per unit length along a latitude circle), the",
              "  unit the model's fluxes are in.  Multiply by 2 pi a cos(lat) for",
              f"  the total transport: at 15 deg that factor is {fac:.2e} m, so",
              f"  1 MW/m there is {1e6 * fac / 1e15:.4f} PW."]
    lines += ["",
              f"  Sanity check at 15N: total MSE flux "
              f"{float(ann['mse_total'][j15])/1e6:.1f} MW/m = "
              f"{float(ann['mse_total'][j15])*fac/1e15:.2f} PW,",
              "  against the observed peak total (atmosphere + ocean) poleward",
              "  energy transport of roughly 5 PW near 40 deg."]

    lines += ["", "The mean MSE flux is a small residual of two large opposing",
              "parts, so it must not be presented alone:", ""]
    for L in [10, 15, 20]:
        j = int(np.abs(lat - L).argmin())
        lines.append(
            f"  {lat[j]:>5.1f}N: DSE {float(ann['dse_mean'][j])/1e6:>+8.1f} "
            f"+ L_v q {float(ann['lvq_mean'][j])/1e6:>+8.1f} "
            f"= MSE {float(ann['mse_mean'][j])/1e6:>+7.1f} MW/m")
    return "\n".join(lines)


def figure(ds: xr.Dataset, out_png: str) -> None:
    lat = ds["latitude"].values
    ann = ds.groupby("time.month").mean("time").mean("month")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    bound = 40.0

    def draw(ax, keys, title, ylabel):
        for key, colr, label, x0, dy in keys:
            prof = ann[key].values / 1e6
            ax.plot(lat, prof, color=colr, lw=2)
            k = int(np.abs(lat - x0).argmin())
            ax.text(lat[k], prof[k] + dy, label, color=colr, fontsize=9,
                    ha="center", va="bottom" if dy > 0 else "top")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.set_xlim(-bound, bound)
        ax.set_xlabel("latitude")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)

    # Labels sit where the three curves separate.  Poleward of ~20 deg the mean
    # flux is small, so total and eddy nearly coincide there by construction and
    # a label placed in that region would be ambiguous between them.
    draw(axes[0],
         [("lvq_total", "#000000", "total", -34.0, -11.0),
          ("lvq_mean", "#2c7fb8", "mean circulation", -26.0, 10.0),
          ("lvq_eddy", "#d95f02", "eddies", 30.0, -13.0)],
         r"latent flux  $L_v F_q$", r"MW m$^{-1}$")

    draw(axes[1],
         [("mse_total", "#000000", "total", -37.0, 28.0),
          ("mse_mean", "#2c7fb8", "mean circulation", -17.0, 25.0),
          ("mse_eddy", "#d95f02", "eddies", 35.0, -25.0)],
         r"moist static energy flux  $F_h$", r"MW m$^{-1}$")

    draw(axes[2],
         [("dse_mean", "#7b3294", "dry static energy", -22.0, 22.0),
          ("lvq_mean", "#2c7fb8", "latent", -22.0, -22.0),
          ("mse_mean", "#1b7837", "net (their sum)", 20.0, -14.0)],
         "the mean flux is a residual of\ntwo large opposing parts",
         r"MW m$^{-1}$")

    fig.suptitle(f"ERA5 {ds.attrs['years']}: annual-mean column fluxes of "
                 "moisture and moist static energy,\nsplit into the mean "
                 "meridional circulation and the eddies (stationary plus "
                 "transient)", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year-start", type=int, default=2000)
    p.add_argument("--year-end", type=int, default=2009)
    p.add_argument("--out", default="model_output/era5_flux_decomposition.png")
    args = p.parse_args()
    ds = build(range(args.year_start, args.year_end + 1))
    print(report(ds))
    figure(ds, args.out)


if __name__ == "__main__":
    main()
