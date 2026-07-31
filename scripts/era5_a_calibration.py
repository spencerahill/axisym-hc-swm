"""ERA5 calibration of the lower-branch column-water-vapor fraction ``a``.

The moist model (``guides/moist_axisymmetric_model_spec.pdf``, Eq. Q1) splits
the column into two overturning branches with equal and opposite meridional
velocities: the upper branch moves at ``+v``, the lower branch at ``-v``.  With
a fraction ``a`` of the column water vapor ``W`` residing in the lower branch,
the column moisture flux collapses to a single coefficient,

    <v q> = (1 - a) W (+v) + a W (-v) = -(2a - 1) v W,

so ``2a - 1`` is the whole mean-transport parameterization.  This script
measures the mass partition that defines ``a``,

    a(lat, month) = W_lower / W_total,

    W_lower = (1/g) int_{p_d}^{p_s} [q] dp,    W_total = (1/g) int_0^{p_s} [q] dp,

from ERA5 monthly zonal-mean specific humidity, where ``p_d`` is the pressure
of the interface separating the two branches.  Two families of interface are
computed:

* **Fixed** ``p_d`` (a ladder of pressures), the reproducible reference.
* **Dynamical** ``p_d``, the level of extremal zonal-mean mass streamfunction
  in the free troposphere, i.e. where the zonal-mean ``[v]`` reverses sign.
  This is the model-faithful interface, since the model's branches *are*
  defined by the sign of ``v``.

The vertical integral treats ``[q]`` as piecewise constant on layers bounded by
level midpoints, and clips every layer to the interval of integration, so the
lower and upper partial columns sum to the full column exactly.  Below-ground
levels get zero weight via the zonal-mean surface pressure.

Run ``python scripts/era5_a_calibration.py`` to build the climatology, write
the figures, and print the summary table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

ERA5_ROOT = Path("/Users/sah2249/Dropbox/data/ecmwf/era5/monthly")
GRAV = 9.80665  # m/s^2

# Interface pressures for the fixed-p_d ladder, in Pa.
FIXED_PD = np.array([40000.0, 50000.0, 60000.0, 70000.0])
REFERENCE_PD = 50000.0

# Pressure band within which the dynamical interface is sought, in Pa.  Wide
# enough to admit a deep-tropical interface near 500 hPa and a shallow
# subtropical one, narrow enough to exclude the boundary layer and the
# stratospheric branch of the residual circulation.
DYNAMIC_BAND = (15000.0, 85000.0)

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def _merge_expver(da: xr.DataArray) -> xr.DataArray:
    """Collapse ERA5's ``expver`` axis, preferring the finite entry."""
    if "expver" not in da.dims:
        return da
    return da.reduce(np.nanmean, dim="expver", keep_attrs=True)


def load_level_field(var: str, name: str, years: range) -> xr.DataArray:
    """Load a per-year zonal-mean pressure-level field, concatenated in time.

    Returns a DataArray with dims (time, level, latitude) and ``level`` in Pa,
    ascending (top of atmosphere first).
    """
    paths = [ERA5_ROOT / var / f"era5_{var}_monthly_{yr}_znl-mean.nc"
             for yr in years]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} files, e.g. {missing[0]}")
    da = xr.concat([_merge_expver(xr.open_dataset(p)[name]) for p in paths],
                   dim="time")
    da = da.assign_coords(level=da["level"].astype("float64") * 100.0)
    da["level"].attrs["units"] = "Pa"
    return da.sortby("level").transpose("time", "level", "latitude")


def load_surface_field(var: str, name: str) -> xr.DataArray:
    """Load a single-file zonal-mean surface field with dims (time, latitude)."""
    paths = sorted((ERA5_ROOT / var).glob(f"era5_{var}_monthly_*_znl-mean.nc"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one {var} file, found {len(paths)}")
    ds = xr.open_dataset(paths[0])
    return _merge_expver(ds[name]).load().transpose("time", "latitude")


# --------------------------------------------------------------------------
# vertical integration
# --------------------------------------------------------------------------
def layer_edges(level: np.ndarray) -> np.ndarray:
    """Pressure edges bounding each level's layer, ascending, length ``n+1``.

    The top edge is the top of the atmosphere and the bottom edge is placed
    below any realizable surface pressure; the integration interval, not the
    edge array, is what excludes below-ground air.
    """
    inner = 0.5 * (level[:-1] + level[1:])
    return np.concatenate([[0.0], inner, [1.1e5]])


def layer_weights(level: np.ndarray, p_top, p_bot) -> np.ndarray:
    """Thickness (Pa) of each level's layer clipped to ``[p_top, p_bot]``.

    ``p_top`` and ``p_bot`` broadcast against the trailing dims of the returned
    array, whose ``level`` axis is inserted at position ``-2``.
    """
    edges = layer_edges(level)
    lo = edges[:-1][:, np.newaxis]
    hi = edges[1:][:, np.newaxis]
    p_top = np.asarray(p_top)[..., np.newaxis, :]
    p_bot = np.asarray(p_bot)[..., np.newaxis, :]
    return np.clip(np.minimum(hi, p_bot) - np.maximum(lo, p_top), 0.0, None)


def column_integral(field: xr.DataArray, p_top, p_bot) -> xr.DataArray:
    """Mass-weighted column integral ``(1/g) int field dp`` over ``[p_top, p_bot]``."""
    level = field["level"].values
    w = layer_weights(level, p_top, p_bot)
    out = (field.values * w).sum(axis=-2) / GRAV
    return xr.DataArray(out, dims=("time", "latitude"),
                        coords={"time": field["time"], "latitude": field["latitude"]})


# --------------------------------------------------------------------------
# the two interface definitions
# --------------------------------------------------------------------------
def fixed_interface(shape_like: xr.DataArray, p_d: float) -> xr.DataArray:
    """A constant interface pressure broadcast onto (time, latitude)."""
    return xr.full_like(shape_like, p_d)


def half_mass_interface(ps: xr.DataArray) -> xr.DataArray:
    """The interface the model's own two-branch structure implies, ``p_s / 2``.

    Mass balance for a closed overturning requires the two branches to carry
    equal and opposite mass transports.  The model gives them equal and
    opposite *velocities*, so their masses must be equal too, which puts the
    interface at half the column mass.  Any other interface makes the model's
    two branches inconsistent with its own velocity assumption, so this is the
    interface at which ``a`` should be read off.
    """
    return 0.5 * ps


def mass_streamfunction(v: xr.DataArray, ps: xr.DataArray,
                        correct_mass_flux: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Psi on the layer-edge grid: ``(1/g) int_0^p [v] dp'``, dims (time, edge, lat).

    ERA5's zonal-mean ``[v]`` on pressure levels carries a spurious net column
    mass flux (an analysis-increment artifact), which leaves Psi non-zero at
    the surface and displaces its extremum.  With ``correct_mass_flux`` the
    mass-weighted column mean of ``[v]`` is removed first, so Psi vanishes at
    both ends and its extremum is the true level of nondivergence.
    """
    level = v["level"].values
    w = layer_weights(level, np.zeros_like(ps.values), ps.values)
    v_val = v.values
    if correct_mass_flux:
        dp = w.sum(axis=-2, keepdims=True)
        v_bar = (v_val * w).sum(axis=-2, keepdims=True) / dp
        v_val = v_val - v_bar
    psi = np.cumsum(v_val * w, axis=-2) / GRAV
    psi_edge = np.concatenate([np.zeros_like(psi[..., :1, :]), psi], axis=-2)
    return layer_edges(level), psi_edge


def dynamical_interface(v: xr.DataArray, ps: xr.DataArray,
                        band: tuple[float, float] = DYNAMIC_BAND,
                        correct_mass_flux: bool = True) -> xr.DataArray:
    """Interface where the zonal-mean mass streamfunction is extremal.

    The extremum of Psi in the free troposphere is the level of nondivergence,
    which is where ``[v]`` reverses sign and so is the observed counterpart of
    the model's branch interface.
    """
    edges, psi_edge = mass_streamfunction(v, ps, correct_mass_flux)
    in_band = (edges >= band[0]) & (edges <= band[1])
    masked = np.where(in_band[np.newaxis, :, np.newaxis], np.abs(psi_edge), -np.inf)
    idx = np.argmax(masked, axis=-2)
    return xr.DataArray(edges[idx], dims=("time", "latitude"),
                        coords={"time": v["time"], "latitude": v["latitude"]})


# --------------------------------------------------------------------------
# the calibration itself
# --------------------------------------------------------------------------
def lower_branch_fraction(q: xr.DataArray, ps: xr.DataArray,
                          p_d: xr.DataArray) -> xr.Dataset:
    """``a = W_lower / W_total`` plus the two partial columns, per (time, lat)."""
    zeros = xr.zeros_like(ps)
    p_d_clipped = np.minimum(p_d, ps)
    w_total = column_integral(q, zeros, ps)
    w_lower = column_integral(q, p_d_clipped, ps)
    return xr.Dataset({
        "a": w_lower / w_total,
        "w_lower": w_lower,
        "w_total": w_total,
        "p_d": p_d,
    })


def monthly_climatology(da: xr.DataArray) -> xr.DataArray:
    """Group by calendar month and average over years."""
    return da.groupby("time.month").mean("time")


def cos_weighted_mean(da: xr.DataArray, lat_bound: float) -> xr.DataArray:
    """Area-weighted meridional mean over ``|lat| <= lat_bound``."""
    sub = da.sel(latitude=slice(lat_bound, -lat_bound))
    wgt = np.cos(np.deg2rad(sub["latitude"]))
    return (sub * wgt).sum("latitude") / wgt.sum("latitude")


def cwv_weighted_mean(ds: xr.Dataset, lat_bound: float) -> xr.DataArray:
    """Meridional mean of ``a`` weighted by area *and* by the column water it
    partitions, i.e. the ratio of the regional integrals of W_lower and W.

    This is the number the model wants: a single coefficient reproducing the
    total moisture transport of the region, not the unweighted average of a
    ratio that is noisiest exactly where there is least water.
    """
    lo = cos_weighted_mean(ds["w_lower"], lat_bound)
    tot = cos_weighted_mean(ds["w_total"], lat_bound)
    return lo / tot


def interface_key(p_d: float) -> str:
    """Variable-name suffix for a fixed interface pressure."""
    return f"p{int(p_d / 100)}"


INTERFACES = [(interface_key(p), f"{int(p / 100)} hPa fixed") for p in FIXED_PD]
INTERFACES += [("half", "half-mass p_s/2"), ("dyn", "level of nondiv.")]


def flux_diagnostics(q: xr.DataArray, v: xr.DataArray, ps: xr.DataArray,
                     w_total: xr.DataArray) -> xr.Dataset:
    """The pieces of the flux-matched calibration of ``2a - 1``.

    ``mean_vq`` is the observed column-integrated mean meridional moisture
    flux, ``(1/g) int [v][q] dp``.  ``v_model`` is the upper-branch velocity a
    model column would need to carry the observed overturning mass transport:
    the model splits ``p_s`` into two equal-mass branches, so a branch mass
    transport of ``Psi_ext`` implies ``v_model = 2 g Psi_ext / p_s``.  The
    model's flux is then ``-(2a - 1) v_model W``, so regressing ``-mean_vq``
    on ``v_model * W`` returns the ``2a - 1`` that reproduces the observed
    moisture transport without ever choosing an interface.
    """
    level = q["level"].values
    w = layer_weights(level, np.zeros_like(ps.values), ps.values)
    dp = w.sum(axis=-2, keepdims=True)
    v_corr = v.values - (v.values * w).sum(axis=-2, keepdims=True) / dp
    mean_vq = (v_corr * q.values * w).sum(axis=-2) / GRAV

    edges, psi_edge = mass_streamfunction(v, ps)
    in_band = (edges >= DYNAMIC_BAND[0]) & (edges <= DYNAMIC_BAND[1])
    masked = np.where(in_band[np.newaxis, :, np.newaxis], np.abs(psi_edge), -np.inf)
    idx = np.argmax(masked, axis=-2)
    psi_ext = np.take_along_axis(psi_edge, idx[..., np.newaxis, :], axis=-2)[..., 0, :]

    dims, coords = ("time", "latitude"), {"time": q["time"], "latitude": q["latitude"]}
    v_model = xr.DataArray(2.0 * GRAV * psi_ext, dims=dims, coords=coords) / ps
    return xr.Dataset({
        "mean_vq": xr.DataArray(mean_vq, dims=dims, coords=coords),
        "psi_ext": xr.DataArray(psi_ext, dims=dims, coords=coords),
        "v_model": v_model,
        "transport_basis": v_model * w_total,
    })


def two_layer_flux(q: xr.DataArray, ps: xr.DataArray, p_d: xr.DataArray,
                   psi_ext: xr.DataArray) -> xr.DataArray:
    """Moisture flux of a two-layer bulk model with interface ``p_d``.

    Each branch carries the observed overturning mass transport ``psi_ext``
    spread uniformly over its own mass, so its velocity is ``psi_ext / M``
    with the sign set by the branch, and the flux is

        F = psi_ext * (qbar_upper - qbar_lower),

    with ``qbar`` the branch-mean specific humidity ``W_branch / M_branch``.
    At the half-mass interface this is algebraically identical to the model's
    ``-(2a - 1) v W``, so the gap between it and the observed flux is exactly
    what the two-layer bulk approximation throws away: the correlation between
    ``[v]`` and ``[q]`` *within* each branch.
    """
    level = q["level"].values
    p_d_c = np.minimum(p_d, ps)
    m_low = layer_weights(level, p_d_c.values, ps.values).sum(axis=-2) / GRAV
    m_up = layer_weights(level, np.zeros_like(ps.values), p_d_c.values).sum(axis=-2) / GRAV
    w_low = column_integral(q, p_d_c, ps)
    w_up = column_integral(q, xr.zeros_like(ps), p_d_c)
    return psi_ext * (w_up / m_up - w_low / m_low)


def flux_matched_coeff(ds: xr.Dataset, lat_bound: float) -> float:
    """Least-squares ``2a - 1`` over ``|lat| <= lat_bound``.

    A regression rather than a pointwise ratio because ``v_model`` passes
    through zero at the equator and at the cell edges, where a ratio is
    meaningless; the regression weights each latitude by the transport it
    actually carries.
    """
    return regress(ds["transport_basis"], -ds["mean_vq"], lat_bound)


def regress(x: xr.DataArray, y: xr.DataArray, lat_bound: float) -> float:
    """Least-squares slope of ``y`` on ``x`` over ``|lat| <= lat_bound``."""
    band = slice(lat_bound, -lat_bound)
    xv = x.sel(latitude=band).values.ravel()
    yv = y.sel(latitude=band).values.ravel()
    good = np.isfinite(xv) & np.isfinite(yv)
    return float((xv[good] * yv[good]).sum() / (xv[good] ** 2).sum())


def build(years: range) -> xr.Dataset:
    """Load ERA5 and assemble every diagnostic this calibration reports."""
    q = load_level_field("hus", "q", years)
    ps = load_surface_field("ps", "sp").sel(time=q["time"])
    tcwv = load_surface_field("cwv", "tcwv").sel(time=q["time"])
    v = load_level_field("va", "v", years).sel(time=q["time"])

    interfaces = {interface_key(p): fixed_interface(ps, p) for p in FIXED_PD}
    interfaces["half"] = half_mass_interface(ps)
    interfaces["dyn"] = dynamical_interface(v, ps)

    out = {}
    for key, p_d in interfaces.items():
        res = lower_branch_fraction(q, ps, p_d)
        out[f"a_{key}"] = res["a"]
        out[f"w_lower_{key}"] = res["w_lower"]
        out[f"p_d_{key}"] = p_d
        out["w_total"] = res["w_total"]

    flux = flux_diagnostics(q, v, ps, out["w_total"])
    out.update(flux)
    for key in ["half", "dyn"]:
        out[f"flux_2lay_{key}"] = two_layer_flux(q, ps, interfaces[key],
                                                 flux["psi_ext"])
    out["tcwv"] = tcwv
    out["ps"] = ps
    # Keep the climatological vertical structure so the figures can show where
    # the water sits and where the branches divide without re-reading ERA5.
    out["q_clim"] = monthly_climatology(q)
    out["v_clim"] = monthly_climatology(v)
    ds = xr.Dataset(out)
    ds.attrs["years"] = f"{years[0]}-{years[-1]}"
    return ds


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def validate(ds: xr.Dataset) -> str:
    """Check the pressure-level column integral against ERA5's own TCWV."""
    diff = ds["w_total"] - ds["tcwv"]
    rel = diff / ds["tcwv"]
    lines = ["Column-integral validation vs ERA5 tcwv (kg/m^2):"]
    for label, sel in [("global", slice(90.0, -90.0)), ("30S-30N", slice(30.0, -30.0))]:
        d = diff.sel(latitude=sel)
        r = rel.sel(latitude=sel)
        flat = np.abs(d.values)
        j = np.unravel_index(np.argmax(flat), flat.shape)
        lines.append(
            f"  {label:8s} mean bias {float(d.mean()):+.3f}  "
            f"max|diff| {flat[j]:.3f} at lat {float(d['latitude'][j[1]]):+.2f} "
            f"{str(d['time'].values[j[0]])[:7]}  "
            f"mean|rel| {float(np.abs(r).mean()) * 100:.2f}%")

    # The model's -(2a-1) v W and the two-layer bulk flux at the half-mass
    # interface are the same expression; confirm the code agrees.
    model_flux = -(2 * ds["a_half"] - 1) * ds["transport_basis"]
    err = np.abs(model_flux - ds["flux_2lay_half"])
    scale = float(np.abs(ds["flux_2lay_half"]).max())
    lines.append(f"Identity check -(2a-1)vW == two-layer bulk flux at p_s/2: "
                 f"max|diff| {float(err.max()):.3e} kg/m/s "
                 f"({float(err.max()) / scale * 100:.2e}% of peak)")
    return "\n".join(lines)


def regional_a(ds: xr.Dataset, key: str, lat_bound: float) -> xr.DataArray:
    """``a`` for one interface, aggregated over ``|lat| <= lat_bound``.

    The aggregation is the ratio of the regional lower-branch water to the
    regional total water, which is what makes a single ``a`` reproduce the
    region's moisture transport.  Averaging the pointwise ratio instead would
    give the dry poles the same vote as the ITCZ.
    """
    return (cos_weighted_mean(ds[f"w_lower_{key}"], lat_bound) /
            cos_weighted_mean(ds["w_total"], lat_bound))


def summary_table(ds: xr.Dataset) -> str:
    clim = ds.groupby("time.month").mean("time")
    bounds = [10.0, 20.0, 30.0, 90.0]
    lines = ["Lower-branch CWV fraction a, water-weighted regional/annual means",
             f"(ERA5 {ds.attrs['years']} monthly zonal means)", ""]
    header = (f"{'interface':<16}" +
              "".join(f"{'|lat|<=' + str(int(b)):>11}" for b in bounds[:-1]) +
              f"{'global':>11}{'2a-1 (20)':>11}")
    lines += [header, "-" * len(header)]
    for key, label in INTERFACES:
        row = label.ljust(16)
        vals = {}
        for b in bounds:
            vals[b] = float(regional_a(clim, key, b).mean("month"))
            row += f"{vals[b]:>11.4f}"
        row += f"{2 * vals[20.0] - 1:>11.4f}"
        lines.append(row)
    lines += ["", "Interface pressure (hPa), area-weighted annual mean:"]
    row = " " * 16
    for key, label in INTERFACES:
        row = f"{label:<16s}" + "".join(
            f"{float(cos_weighted_mean(clim['p_d_' + key], b).mean('month')) / 100:>11.1f}"
            for b in bounds)
        lines.append(row)

    lines += ["", "Flux-matched 2a-1 (regression of observed mean moisture flux",
              "onto the model's transport structure; no interface chosen):"]
    for b in bounds:
        coeff = flux_matched_coeff(ds, b)
        flag = "  <-- above the a<=1 ceiling" if coeff > 1.0 else ""
        lines.append(f"  |lat|<={int(b):<3d} 2a-1 = {coeff:>8.4f}"
                     f"   (a = {0.5 * (coeff + 1):.4f}){flag}")

    lines += ["", "Transport enhancement: observed flux / two-layer bulk flux",
              "(what the bulk approximation discards, at each interface):"]
    for key, label in [("half", "half-mass p_s/2"), ("dyn", "level of nondiv.")]:
        row = f"  {label:<18s}"
        for b in bounds:
            row += f"{regress(ds['flux_2lay_' + key], ds['mean_vq'], b):>11.3f}"
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year-start", type=int, default=1979)
    parser.add_argument("--year-end", type=int, default=2023)
    parser.add_argument("--out", type=Path,
                        default=Path("model_output/era5_a_calibration.nc"))
    args = parser.parse_args()

    years = range(args.year_start, args.year_end + 1)
    ds = build(years)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.out)
    print(validate(ds))
    print()
    print(summary_table(ds))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
