"""ERA5 calibration of the gross dry and gross moist stabilities.

The model defines both by how much energy the overturning carries per unit
branch velocity:

    mean DSE flux = Shat * v,        mean column MSE flux = Hhat * v,

with ``Hhat = Shat - L_v (2a-1) W``.  So each is a regression slope against
the same ``v_model = 2 g Psi_ext / p_s`` that ``scripts/era5_a_calibration.py``
builds, and neither requires choosing a branch interface.

Two readings of ``Shat`` are reported, exactly as for ``a``:

* **effective**, the regression slope, which absorbs the correlation between
  ``[v]`` and the transported quantity *within* each branch;
* **literal bulk**, ``M_branch * (s_upper - s_lower)`` at the half-mass
  interface, which is what a two-layer model with uniform branch velocities
  would carry.

For moisture those two differ by 56% (the flux-matched coefficient exceeds
the mass fraction's ceiling).  For dry static energy they differ by only 13%,
and in the opposite direction, because ``s`` rises with height while the
upper branch's mass transport sits below its own branch mean.

Run ``python scripts/era5_stability_calibration.py``.
"""

from __future__ import annotations

import argparse

import numpy as np
import xarray as xr

from era5_a_calibration import (GRAV, column_integral, cos_weighted_mean,
                                dynamical_interface_band, layer_weights,
                                load_level_field, load_surface_field,
                                mass_streamfunction)
from ss09.moist_constants import L_V, column_heat_capacity, gross_dry_stability
from ss09.sw_config import SWConfig

CP_DRY = 1004.6  # J/kg/K, matching ERA5's thermodynamic constants
BANDS = [10.0, 15.0, 20.0, 30.0]


def branch_velocity(v: xr.DataArray, ps: xr.DataArray) -> xr.DataArray:
    """``v_model``: the branch velocity carrying the observed mass transport."""
    p_dyn, _, _ = dynamical_interface_band(v, ps)
    edges, psi = mass_streamfunction(v, ps)
    idx = np.abs(edges[np.newaxis, :, np.newaxis]
                 - p_dyn.values[:, np.newaxis, :]).argmin(axis=-2)
    psi_ext = np.take_along_axis(psi, idx[..., np.newaxis, :], axis=-2)[..., 0, :]
    return xr.DataArray(2.0 * GRAV * psi_ext / ps.values, dims=ps.dims,
                        coords=ps.coords)


def mass_flux_corrected(v: xr.DataArray, ps: xr.DataArray) -> np.ndarray:
    """``[v]`` with its spurious net column mass flux removed."""
    w = layer_weights(v["level"].values, np.zeros_like(ps.values), ps.values)
    return v.values - (v.values * w).sum(axis=-2, keepdims=True) / w.sum(
        axis=-2, keepdims=True)


def slope(x: xr.DataArray, y: xr.DataArray, lat_bound: float) -> float:
    band = slice(lat_bound, -lat_bound)
    xv = x.sel(latitude=band).values.ravel()
    yv = y.sel(latitude=band).values.ravel()
    good = np.isfinite(xv) & np.isfinite(yv)
    return float((xv[good] * yv[good]).sum() / (xv[good] ** 2).sum())


def build(years: range) -> xr.Dataset:
    ta = load_level_field("ta", "t", years)
    zg = load_level_field("zg", "z", years).sel(time=ta["time"])
    hus = load_level_field("hus", "q", years).sel(time=ta["time"])
    va = load_level_field("va", "v", years).sel(time=ta["time"])
    ps = load_surface_field("ps", "sp").sel(time=ta["time"])
    level = ta["level"].values

    dse = CP_DRY * ta.values + zg.values          # s = c_p T + Phi
    mse = dse + L_V * hus.values                  # h = s + L_v q
    v_corr = mass_flux_corrected(va, ps)
    w_full = layer_weights(level, np.zeros_like(ps.values), ps.values)
    dims, coords = ps.dims, ps.coords

    def flux(field):
        return xr.DataArray((v_corr * field * w_full).sum(axis=-2) / GRAV,
                            dims=dims, coords=coords)

    # Literal bulk Shat at the half-mass interface: branch mass times the
    # difference of the branch-mean dry static energies.
    half = 0.5 * ps.values
    w_low = layer_weights(level, half, ps.values)
    w_up = layer_weights(level, np.zeros_like(ps.values), half)
    s_low = (dse * w_low).sum(axis=-2) / w_low.sum(axis=-2)
    s_up = (dse * w_up).sum(axis=-2) / w_up.sum(axis=-2)
    m_branch = w_low.sum(axis=-2) / GRAV

    ds = xr.Dataset({
        "v_model": branch_velocity(va, ps),
        "dse_flux": flux(dse),
        "mse_flux": flux(mse),
        "shat_bulk": xr.DataArray(m_branch * (s_up - s_low), dims=dims,
                                  coords=coords),
        "w_total": column_integral(hus, np.zeros_like(ps.values), ps.values),
    })
    ds.attrs["years"] = f"{years[0]}-{years[-1]}"
    return ds


def report(ds: xr.Dataset) -> str:
    cfg = SWConfig()
    c_col = column_heat_capacity(9.80665)
    shat_model = gross_dry_stability(9.80665, cfg.delta, cfg.delta_z, cfg.height)

    lines = [f"ERA5 {ds.attrs['years']}: gross stabilities regressed on v_model",
             "", f"{'band':<11}{'Shat_eff':>12}{'Shat_bulk':>12}"
                 f"{'Hhat_eff':>12}{'W':>8}{'Hhat/Shat':>11}"]
    lines.append("-" * len(lines[-1]))
    for b in BANDS:
        s_eff = slope(ds["v_model"], ds["dse_flux"], b)
        h_eff = slope(ds["v_model"], ds["mse_flux"], b)
        s_blk = float(cos_weighted_mean(ds["shat_bulk"], b).mean())
        w = float(cos_weighted_mean(ds["w_total"], b).mean())
        lines.append(f"|lat|<={int(b):<4d}{s_eff:>12.3e}{s_blk:>12.3e}"
                     f"{h_eff:>12.3e}{w:>8.1f}{h_eff / s_eff:>11.3f}")

    s_eff = slope(ds["v_model"], ds["dse_flux"], 20.0)
    h_eff = slope(ds["v_model"], ds["mse_flux"], 20.0)
    w20 = float(cos_weighted_mean(ds["w_total"], 20.0).mean())
    lines += ["", f"model Shat = C d Delta_z / H = {shat_model:.3e} J/m^2 "
                  f"(d={cfg.delta:.0f} m, Delta_z={cfg.delta_z:.0f} K, "
                  f"H={cfg.height:.0f} m)",
              f"observed / model = {s_eff / shat_model:.2f} (effective), "
              f"{float(cos_weighted_mean(ds['shat_bulk'], 20.0).mean()) / shat_model:.2f} (bulk)",
              "",
              "The prefactor d/H is the suspect: the moisture and momentum",
              "structure puts half the column mass in each branch, while",
              f"d/H = {cfg.delta / cfg.height:.2f} makes the theta equation act as if the",
              "branches were separated by only "
              f"{cfg.delta * cfg.delta_z / cfg.height:.0f} K.",
              f"Reproducing the observed Shat at fixed Delta_z needs "
              f"d/H = {s_eff / (c_col * cfg.delta_z):.2f}, i.e. d = "
              f"{s_eff / (c_col * cfg.delta_z) * cfg.height / 1e3:.1f} km."]

    lines += ["", "Gross moist stability Hhat = Shat - L_v(2a-1)W at the "
                  f"observed tropical W = {w20:.1f} kg/m^2:",
              f"{'Shat':<20}{'coefficient 2a-1':<26}{'Hhat':>11}{'W* (Hhat=0)':>13}"]
    for slab, s_val in [("model 7.75e7", shat_model), ("observed 1.98e8", s_eff)]:
        for alab, coeff in [("mass partition 0.900", 0.9002),
                            ("flux-matched 1.401", 1.4011)]:
            hh = s_val - L_V * coeff * w20
            lines.append(f"{slab:<20}{alab:<26}{hh:>11.3e}"
                         f"{s_val / (L_V * coeff):>13.1f}")
    lines.append(f"{'directly regressed':<46}{h_eff:>11.3e}")
    return "\n".join(lines)


def figures(ds: xr.Dataset, out_prefix: str) -> None:
    """Show the regressions themselves, not just their slopes.

    Panel a and b are the scatters the slopes come from, so a bad fit or a
    latitude-dependent slope is visible rather than hidden inside one number.
    Panel c is the local ratio flux/v_model as a function of latitude, which
    is the same quantity resolved rather than aggregated.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = SWConfig()
    shat_model = gross_dry_stability(9.80665, cfg.delta, cfg.delta_z, cfg.height)
    lat = ds["latitude"].values
    band = np.abs(lat) <= 30.0
    v = ds["v_model"].values[:, band]
    lat_b = lat[band]
    colour = np.broadcast_to(lat_b, v.shape)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, key, name, model_slope in [
            (axes[0], "dse_flux", r"mean DSE flux  $\hat S v$", shat_model),
            (axes[1], "mse_flux", r"mean column MSE flux  $\hat H v$", None)]:
        f = ds[key].values[:, band]
        sc = ax.scatter(v.ravel(), f.ravel() / 1e6, s=1.5, alpha=0.25,
                        c=colour.ravel(), cmap="Spectral_r", vmin=-30, vmax=30)
        s_obs = slope(ds["v_model"], ds[key], 30.0)
        xs = np.array([v.min(), v.max()])
        ax.plot(xs, s_obs * xs / 1e6, color="k", lw=2)
        ax.text(0.04, 0.93, f"observed slope {s_obs:.2e} J m$^{{-2}}$",
                transform=ax.transAxes, fontsize=9)
        if model_slope is not None:
            ax.plot(xs, model_slope * xs / 1e6, color="#CC3311", lw=2, ls="--")
            ax.text(0.04, 0.86, f"model slope {model_slope:.2e}",
                    transform=ax.transAxes, fontsize=9, color="#CC3311")
        ax.set_xlabel(r"$v_{\rm model}$ (m s$^{-1}$)")
        ax.set_ylabel(f"{name} (MW m$^{{-1}}$)")
        ax.axhline(0, color="0.6", lw=0.7)
        ax.axvline(0, color="0.6", lw=0.7)
        ax.grid(alpha=0.25)
        ax.set_title(name, fontsize=11)
    fig.colorbar(sc, ax=axes[1], label="latitude", pad=0.02)

    ax = axes[2]
    ann_v = ds["v_model"].groupby("time.month").mean("time").mean("month")
    strong = np.abs(ann_v) > 0.15
    for key, colr, label in [("dse_flux", "#7b3294", r"$\hat S$"),
                             ("mse_flux", "#1b7837", r"$\hat H$")]:
        ann_f = ds[key].groupby("time.month").mean("time").mean("month")
        ratio = np.where(strong, ann_f / ann_v, np.nan)
        ax.plot(lat, ratio / 1e6, color=colr, lw=2)
        k = int(np.abs(lat - 17.0).argmin())
        ax.text(lat[k], ratio[k] / 1e6 + 12, label, color=colr, fontsize=11,
                ha="center")
    ax.axhline(shat_model / 1e6, color="#CC3311", lw=2, ls="--")
    ax.text(-28, shat_model / 1e6 + 6, r"model $\hat S$", color="#CC3311",
            fontsize=9)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xlim(-30, 30)
    ax.set_xlabel("latitude")
    ax.set_ylabel(r"flux / $v_{\rm model}$ (MJ m$^{-2}$)")
    ax.set_title("local stabilities, where the overturning is resolved",
                 fontsize=11)
    ax.grid(alpha=0.25)

    fig.suptitle(f"ERA5 {ds.attrs['years']}: gross dry and gross moist "
                 "stability from the transport they carry", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_prefix + "_regression.png", dpi=130)
    print(f"Wrote {out_prefix}_regression.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year-start", type=int, default=2000)
    p.add_argument("--year-end", type=int, default=2019)
    p.add_argument("--out-prefix", default="model_output/era5_stability")
    args = p.parse_args()
    ds = build(range(args.year_start, args.year_end + 1))
    print(report(ds))
    figures(ds, args.out_prefix)


if __name__ == "__main__":
    main()
