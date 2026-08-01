"""Figures for the ERA5 calibration of the lower-branch CWV fraction ``a``.

Reads the cache written by ``scripts/era5_a_calibration.py`` and produces:

* ``_map.png``      -- a(latitude, month), the full variation Spencer asked for
* ``_profiles.png`` -- annual-mean a(latitude), its seasonal cycle, and the
                       interface pressures the two definitions pick out
* ``_vertical.png`` -- the deep-tropical [q] and [v] profiles that set a, and
                       the branch interfaces marked on them
* ``_transport.png``-- observed mean moisture flux against the strongest flux
                       the model's two-branch structure can produce

Usage: ``python scripts/era5_a_calibration_figures.py [cache.nc] [prefix]``
"""

from __future__ import annotations

import sys

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from era5_a_calibration import (INTERFACES, MONTH_NAMES, cos_weighted_mean,  # noqa: E402
                                regional_a, regress)

LADDER_COLORS = {
    "p400": "#7b3294",
    "p500": "#2c7fb8",
    "p600": "#31a354",
    "p700": "#d95f02",
    "half": "#000000",
    "dyn": "#e7298a",
}
LADDER_LABELS = dict(INTERFACES)

# The layer depth the model's own Shat = C d Delta_z / H corresponds to, Pa,
# measured by scripts/era5_normalization.py.  It sets the velocity the model's
# transport terms use, and so the scale of the coefficient they need.
SLAB_DP = 19500.0


def _month_axis(ax, short=False):
    """Month ticks. Three-letter labels: J and M each collide across the year."""
    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(MONTH_NAMES, fontsize=8 if short else 9,
                       rotation=45 if short else 0)
    ax.set_xlim(1, 12)


def _lat_axis(ax, bound=40.0):
    ax.set_xlim(-bound, bound)
    ax.set_xticks(np.arange(-bound, bound + 1, 10))
    ax.set_xlabel("latitude")


def figure_map(clim: xr.Dataset, out_png: str, years: str) -> None:
    """a(month, latitude): month on x, latitude on y (Spencer's convention).

    Each panel gets its own colour scale: at the half-mass interface ``a``
    varies by 0.04 over the whole tropics and a shared scale would render it a
    flat block of colour.  That flatness is the result, so the scale is chosen
    to show what little structure there is rather than to hide it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), sharey=True)
    lat = clim["latitude"].values
    months = clim["month"].values
    panels = [("half", np.arange(0.930, 0.9751, 0.0025), "%.3f"),
              ("p700", np.arange(0.70, 0.851, 0.01), "%.2f")]

    for ax, (key, levels, fmt) in zip(axes, panels):
        a = clim[f"a_{key}"].transpose("month", "latitude").values
        cf = ax.contourf(months, lat, a.T, levels=levels, cmap="YlGnBu",
                         extend="both")
        cs = ax.contour(months, lat, a.T, levels=levels[::2], colors="0.25",
                        linewidths=0.6)
        ax.clabel(cs, fmt=fmt, fontsize=7)
        ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
        ax.set_ylim(-40, 40)
        ax.set_yticks(np.arange(-40, 41, 10))
        _month_axis(ax, short=True)
        ax.set_xlabel("month")
        span = float(np.nanmax(a[:, np.abs(lat) <= 30])
                     - np.nanmin(a[:, np.abs(lat) <= 30]))
        ax.set_title(f"interface: {LADDER_LABELS[key]}"
                     f"   (spans {span:.3f} within $40^\\circ$)", fontsize=11)
        fig.colorbar(cf, ax=ax, label="$a$", pad=0.02)

    axes[0].set_ylabel("latitude")
    fig.suptitle(f"Lower-branch CWV fraction $a$, ERA5 {years} zonal mean"
                 "\n(a mass partition at a chosen interface; the coefficient the "
                 "model's transport term needs is in "
                 "era5_normalization_partition.png)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def figure_profiles(clim: xr.Dataset, out_png: str, years: str) -> None:
    """Annual mean a(lat), its seasonal cycle, and the interface pressures."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    lat = clim["latitude"].values

    # Label anchors chosen per curve so each sits beside its own line and
    # nearer to it than to any neighbour; the 500 hPa and half-mass curves
    # nearly coincide, so they are labelled at opposite ends of the axis.
    anchors = {"p400": (34.0, 0.011), "p500": (36.0, -0.018),
               "p600": (34.0, 0.011), "p700": (34.0, 0.011),
               "half": (10.0, 0.014), "dyn": (-14.0, -0.016)}

    ax = axes[0]
    # The observed interface is a band, not a level, so a read off from it is
    # a band too; drawing it as a line would imply precision that Psi's flat
    # mid-troposphere does not support.
    lo = clim["a_dyn_shallow"].mean("month").values
    hi = clim["a_dyn_deep"].mean("month").values
    ax.fill_between(lat, np.minimum(lo, hi), np.maximum(lo, hi),
                    color=LADDER_COLORS["dyn"], alpha=0.20, lw=0)
    for key, label in INTERFACES:
        ann = clim[f"a_{key}"].mean("month")
        ax.plot(lat, ann.values, color=LADDER_COLORS[key],
                lw=2.2 if key == "half" else 1.3,
                ls="-." if key == "dyn" else "-")
        x0, dy = anchors[key]
        k = int(np.abs(lat - x0).argmin())
        ax.text(lat[k], float(ann[k]) + dy, label, color=LADDER_COLORS[key],
                fontsize=8.5, ha="center", va="bottom" if dy > 0 else "top")
    _lat_axis(ax)
    ax.set_ylim(0.55, 1.03)
    ax.set_ylabel("$a$")
    ax.set_title("annual-mean $a$", fontsize=11)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for key, label in INTERFACES:
        seas = regional_a(clim, key, 20.0)
        ax.plot(clim["month"].values, seas.values, color=LADDER_COLORS[key],
                lw=2.2 if key == "half" else 1.3, marker="o", ms=3)
        dy = 0.014 if key == "half" else (-0.014 if key == "p500" else 0.0)
        ax.text(12.3, float(seas[-1]) + dy, label, color=LADDER_COLORS[key],
                fontsize=8.5, va="center")
    _month_axis(ax)
    ax.set_xlim(1, 16.0)
    ax.set_ylim(0.55, 1.03)
    ax.set_ylabel("$a$")
    ax.set_title(r"seasonal cycle, water-weighted $|\varphi|\leq20^\circ$",
                 fontsize=11)
    ax.grid(alpha=0.25)

    ax = axes[2]
    lo = clim["p_d_dyn_shallow"].mean("month").values / 100.0
    hi = clim["p_d_dyn_deep"].mean("month").values / 100.0
    ax.fill_between(lat, lo, hi, color=LADDER_COLORS["dyn"], alpha=0.20, lw=0)
    for key, x0, dy in [("half", 8.0, -14.0), ("dyn", -8.0, 55.0)]:
        ann = clim[f"p_d_{key}"].mean("month") / 100.0
        ax.plot(lat, ann.values, color=LADDER_COLORS[key], lw=2.0,
                ls="-." if key == "dyn" else "-")
        k = int(np.abs(lat - x0).argmin())
        ax.text(lat[k], float(ann[k]) + dy, LADDER_LABELS[key],
                color=LADDER_COLORS[key], fontsize=9, ha="center",
                va="top" if dy < 0 else "bottom")
    _lat_axis(ax)
    ax.set_ylim(900, 300)
    ax.set_ylabel("interface pressure (hPa)")
    ax.set_title(r"where the branches divide"
                 "\n"
                 r"(shading: $|\Psi|$ within 5% of its extremum)", fontsize=10)
    ax.grid(alpha=0.25)

    fig.suptitle(f"ERA5 {years}: the lower-branch CWV fraction and its interface",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def figure_vertical(clim: xr.Dataset, out_png: str, years: str) -> None:
    """The deep-tropical profiles that set a, with both interfaces marked."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    p = clim["level"].values / 100.0
    band = slice(15.0, -15.0)

    q = cos_weighted_mean(clim["q_clim"], 15.0).mean("month") * 1e3
    v = cos_weighted_mean(np.abs(clim["v_clim"]), 15.0).mean("month")
    ps_bar = float(cos_weighted_mean(clim["ps"].mean("month"), 15.0)) / 100.0
    p_dyn = float(cos_weighted_mean(clim["p_d_dyn"], 15.0).mean("month")) / 100.0

    ax = axes[0]
    ax.plot(q.values, p, color="#2c7fb8", lw=2)
    ax.set_xlabel("[q] (g/kg)")
    ax.set_title(r"specific humidity, $|\varphi|\leq15^\circ$", fontsize=11)

    ax = axes[1]
    ax.plot(v.values, p, color="#d95f02", lw=2)
    ax.set_xlabel(r"$\langle |[v]| \rangle$ (m/s)")
    ax.set_title("meridional wind magnitude", fontsize=11)

    ax = axes[2]
    cum = np.cumsum(q.values * np.gradient(clim["level"].values))
    cum = cum / cum[-1]
    ax.plot(cum, p, color="#31a354", lw=2)
    ax.set_xlabel("fraction of column water above $p$")
    ax.set_title("cumulative water", fontsize=11)
    ax.set_xlim(0, 1)

    for ax in axes:
        ax.axhline(ps_bar / 2, color="k", lw=1.4, ls="--")
        ax.axhline(p_dyn, color="#e7298a", lw=1.4, ls="-.")
        ax.set_ylim(1000, 100)
        ax.set_ylabel("pressure (hPa)")
        ax.grid(alpha=0.25)
    axes[0].text(axes[0].get_xlim()[1] * 0.55, ps_bar / 2 - 15,
                 f"half-mass {ps_bar / 2:.0f} hPa", fontsize=8.5, color="k")
    axes[0].text(axes[0].get_xlim()[1] * 0.55, p_dyn + 35,
                 f"level of nondiv. {p_dyn:.0f} hPa", fontsize=8.5, color="#e7298a")

    fig.suptitle(f"ERA5 {years}: why $a$ depends so strongly on the interface",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def figure_transport(ds: xr.Dataset, clim: xr.Dataset, out_png: str,
                     years: str, dp_slab: float = SLAB_DP) -> None:
    """Observed mean moisture flux against what the model's structure carries.

    The model's flux is the product ``(2a-1) * v``, and ``v`` is a layer
    velocity: the velocity of a slab of depth ``dp_slab``, not of a half-column
    branch.  So the transport basis here is ``v_slab W``, larger than the
    half-column ``v W`` by ``(p_s/2)/dp_slab``, and the coefficient the observed
    flux demands is correspondingly smaller.  Reading the coefficient off the
    half-column basis instead is what pushed it above its own ``2a-1 <= 1``
    ceiling and, through ``Hhat``, produced a spuriously negative gross moist
    stability.  See ``scripts/era5_normalization.py``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    lat = clim["latitude"].values

    obs_da = clim["mean_vq"].mean("month")
    # transport_basis is v_half * W; rescale it to the slab velocity the model
    # actually integrates with.
    basis_da = (clim["transport_basis"].mean("month")
                * (0.5 * clim["ps"].mean("month")) / dp_slab)
    obs, basis = obs_da.values, basis_da.values

    ax = axes[0]
    ratio = np.where(np.abs(basis) > 4.0, -obs / basis, np.nan)
    # Aggregate by regression, not by averaging the pointwise ratio: the basis
    # passes through zero at the equator and at both cell edges, where the ratio
    # is meaningless and an unweighted mean gives those points full vote.  The
    # regression is on the ANNUAL MEANS, which is what this figure plots; the
    # transport-weighted regression over all months gives a smaller coefficient
    # (era5_normalization.py section 6 reports both).
    coeff = regress(basis_da, -obs_da, 20.0)
    ax.plot(lat, obs, color="#2c7fb8", lw=2)
    ax.plot(lat, -coeff * basis, color="#d95f02", lw=1.6, ls="--")
    ax.plot(lat, -0.7 * basis, color="#31a354", lw=1.4, ls=":")
    for x0, curve, dy, txt, color in [
            (-24.0, obs, 2.2, "ERA5 mean $[v][q]$", "#2c7fb8"),
            (-11.0, -coeff * basis, 2.2,
             f"model at $2a-1={coeff:.2f}$ ($a={0.5*(coeff+1):.2f}$)", "#d95f02"),
            (14.0, -0.7 * basis, -2.4, "model at spec $a=0.85$", "#31a354")]:
        k = int(np.abs(lat - x0).argmin())
        ax.text(lat[k], curve[k] + dy, txt, color=color, fontsize=9,
                ha="center", va="bottom" if dy > 0 else "top")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylim(-30, 42)
    _lat_axis(ax, 35.0)
    ax.set_ylabel("column moisture flux (kg m$^{-1}$ s$^{-1}$)")
    ax.set_title(f"transport at $v$ = slab velocity, $dp={dp_slab/100:.0f}$ hPa",
                 fontsize=11)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(lat, ratio, color="#7b3294", lw=2)
    ann = clim["a_half"].mean("month")
    ax.plot(lat, (2 * ann - 1).values, color=LADDER_COLORS["half"], lw=1.5)
    k = int(np.abs(lat - 26.0).argmin())
    ax.text(lat[k], float(2 * ann[k] - 1) - 0.05,
            "mass partition at $p_s/2$", color=LADDER_COLORS["half"],
            fontsize=8.5, ha="center", va="top")
    k = int(np.abs(lat - 16.0).argmin())
    ax.text(lat[k], ratio[k] + 0.09, "required by the ERA5 flux", color="#7b3294",
            fontsize=9, ha="center")
    ax.axhline(1.0, color="0.3", lw=1.0, ls="--")
    ax.text(-34, 1.03, "$2a-1=1$ ceiling (all water below)", fontsize=8, color="0.3")
    _lat_axis(ax, 35.0)
    ax.set_ylim(0.0, 1.4)
    ax.set_ylabel("$2a-1$")
    ax.set_title("the coefficient the observed flux demands", fontsize=11)
    ax.grid(alpha=0.25)

    fig.suptitle(f"ERA5 {years}: annual-mean column moisture transport, with "
                 r"$v$ read as the model reads it", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def main() -> None:
    cache = sys.argv[1] if len(sys.argv) > 1 else "model_output/era5_a_calibration.nc"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "model_output/era5_a_calibration"
    ds = xr.open_dataset(cache)
    years = ds.attrs.get("years", "")
    static = [k for k in ds.data_vars if "month" in ds[k].dims]
    clim = xr.merge([ds[[k for k in ds.data_vars if "time" in ds[k].dims]]
                     .groupby("time.month").mean("time"), ds[static]])
    figure_map(clim, prefix + "_map.png", years)
    figure_profiles(clim, prefix + "_profiles.png", years)
    figure_vertical(clim, prefix + "_vertical.png", years)
    figure_transport(ds, clim, prefix + "_transport.png", years)


if __name__ == "__main__":
    main()
