"""Diagnostics for the moist V2 sweep: one scalar table, built from the fields.

Every quantity here is computed from the model's own operators where the model
has one (the MUSCL-MC face value of W that the transport actually advects, the
compact half-cell divergence), because reconstructing the moisture flux at cell
centres with np.gradient leaves a residual of order the source itself.

The energetics follow SCIENCE.md 3.6:

    <h> = C theta + L_v W                       column moist static energy
    Shat = C d Delta_z / H                      gross dry stability
    Hhat = Shat - L_v (2a-1) W                  gross moist stability
    F_mean = Shat v - L_v (2a-1) v W = v Hhat    mean column MSE flux
    F_eddy = -L_v D dW/dy                       eddy column MSE flux
    d/dy(F_mean + F_eddy) = (C/tau)(theta_rad - theta) + L_v E_0 - (L_v - C Lambda) P

The last term is zero for a physical V2 run (C Lambda = L_v by construction)
and is carried explicitly because block J deliberately detunes Lambda.

Two facts the analysis leans on repeatedly:

  * At equilibrium the domain-mean precipitation MUST equal E_0, because
    transport moves no total water and the only source and sink are E_0 and P.
    So aggregation is entirely about the DISTRIBUTION of precipitation, never
    its total, and <P>/E_0 is a free equilibrium check.
  * Peak precipitation is exactly (W_max - W_c)/tau_c, so max P is a direct
    read-out of the wettest column.

Usage:
    python scripts/v2_sweep_diagnostics.py            # build the table
    python scripts/v2_sweep_diagnostics.py --seasonal # seasonal composites too
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import xarray as xr

sys.path.insert(0, "/home/shill/py/axisym-hc-swm")

from scripts.v2_sweep_manifest import (  # noqa: E402
    SWEEP_DIR, manifest, r_param, w_star,
)
from ss09.moist_constants import (  # noqa: E402
    L_V, column_heat_capacity, gross_dry_stability,
)
from ss09.read_output import load_centered  # noqa: E402
from ss09.sw_model import (  # noqa: E402
    cwv_integral, mc_face_values, v_divergence_at_centers, v_faces_to_centers,
)

SEC_DAY = 86400.0
Y_ONE = 9439e3
OMEGA = 7.292e-5  # s^-1, for the f = beta y = 2 Omega sin(phi) latitude mapping
BETA = 2e-11


def lat_to_y(deg):
    """Inverse of y_to_lat: y = 2 Omega sin(phi) / beta."""
    return 2.0 * OMEGA * np.sin(np.radians(deg)) / BETA


def y_to_lat(y):
    """Equivalent latitude from the Coriolis mapping f = beta y = 2 Omega sin(phi).

    Only defined for |y| < 2 Omega / beta = 7.29 Mm; NaN outside, rather than a
    silently clipped value.
    """
    s = BETA * np.asarray(y, dtype=float) / (2.0 * OMEGA)
    return np.where(np.abs(s) <= 1.0, np.degrees(np.arcsin(np.clip(s, -1, 1))), np.nan)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _attr(att, key, cast=float, default=None):
    """Read one global attribute. netCDF cannot hold None or bool, so the model
    writes their str() form; 'None' therefore has to come back as None rather
    than raising on float('None')."""
    if key not in att:
        return default
    val = att[key]
    if isinstance(val, str) and val == "None":
        return default
    if cast is float:
        return float(val)
    if cast is bool:
        return str(val) == "True"
    return cast(val)


def load_run(path: str, last_n: int = 1000,
             min_days: int = 0) -> Optional[Dict[str, Any]]:
    """Time-mean fields over the last last_n days, plus the energetics.

    Returns None if the file is missing, shorter than min_days, or holds
    non-finite fields. min_days matters: a run that diverges writes a file with
    a handful of days in it, and without the check those first-day transients
    load as if they were equilibria and land in the tables looking like
    ordinary numbers. Pass the run's requested length.
    """
    if not os.path.exists(path):
        return None
    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        att = dict(ds.attrs)
        n = int(ds.sizes["time"])
        if n < min_days or n < 4:
            return None
        if n < 2 * last_n:
            last_n = max(1, n // 4)
        y = ds["y"].values
        y_face = ds["y_face"].values
        sl = slice(n - last_n, n)
        prev = slice(n - 2 * last_n, n - last_n)

        def tmean(name, s=sl):
            return ds[name].isel(time=s).mean("time").values

        u = tmean("u")
        w = tmean("W")
        p = tmean("P")
        temp = tmean("T")
        v_face = tmean("v")
        if not (np.isfinite(u).all() and np.isfinite(w).all()):
            return None
        # Previous window, for the drift check
        u_prev = tmean("u", prev)
        w_prev = tmean("W", prev)
        days = int(ds["time"].values[-1])
        theta_e = ds["theta_e"].values
        if theta_e.ndim == 2:  # seasonal: theta_rad is time-varying
            theta_e = theta_e[sl].mean(axis=0)
        w_min_run = float(ds["W_min"].values.min()) if "W_min" in ds else np.nan
        # time series for the drift / spin-up figures
        ts = dict(
            time=ds["time"].values,
            w_mean=ds["W_mean"].values if "W_mean" in ds else None,
            jet=ds["north_jet_magnitude"].values if "north_jet_magnitude" in ds else None,
            jet_lat=ds["north_jet_lat"].values if "north_jet_lat" in ds else None,
            ke=(ds["steady_state_kinetic_energy"].values
                if "steady_state_kinetic_energy" in ds else None),
            pmax=(ds["P"].max("y").values * SEC_DAY) if "P" in ds else None,
        )
        # Whether the solution is steady at all, measured over the trailing
        # third of the run rather than assumed. Some solutions at large r never
        # settle: their jet magnitude and peak rain rate keep swinging while
        # their domain mean holds nearly fixed, so a time mean of the fields is
        # a mean over a fluctuating state and the MSE budget cannot close.
        nv = min(2000, max(200, n // 3))
        jm = ts["jet"][-nv:] if ts["jet"] is not None else np.array([np.nan])
        wmv = ts["w_mean"][-nv:] if ts["w_mean"] is not None else np.array([np.nan])
        pmv = ts["pmax"][-nv:] if ts["pmax"] is not None else np.array([np.nan])
        var = dict(
            window=nv,
            jet_mean=float(np.mean(jm)), jet_range=float(np.ptp(jm)),
            jet_sd=float(np.std(jm)),
            wmean_range=float(np.ptp(wmv)), wmean_sd=float(np.std(wmv)),
            pmax_mean=float(np.mean(pmv)), pmax_range=float(np.ptp(pmv)),
            pmax_sd=float(np.std(pmv)),
        )
        var["jet_range_rel"] = float(var["jet_range"]
                                     / max(abs(var["jet_mean"]), 1e-30))
        # A steady jet magnitude is necessary and not sufficient: two runs held
        # their jet to under 1% while their rain bands wandered across the
        # domain, which showed up as an MSE budget residual of order the source
        # itself. The pointwise u drift between the two trailing 1000-day means
        # catches that, so both conditions are required. The threshold is loose
        # (1%) because the two genuinely steady-but-short runs sit at 0.2 and
        # 0.4% while the wandering ones sit at 10 and 63%.
    finally:
        ds.close()

    _, _, v_all, _ = load_centered(path)
    v = v_all[n - last_n:n].mean(axis=0)

    a = _attr(att, "cwv_frac")
    d_w = _attr(att, "d_w")
    grav = _attr(att, "gravity")
    delta = _attr(att, "delta")
    delta_z = _attr(att, "delta_z")
    height = _attr(att, "height")
    tau = _attr(att, "tau")
    evap = _attr(att, "evap")
    w_crit = _attr(att, "w_crit")
    tau_c = _attr(att, "tau_c")
    latent = _attr(att, "enable_latent_heating", bool, False)
    lam = _attr(att, "lambda_conv") if latent else 0.0
    c_col = column_heat_capacity(grav)
    shat = gross_dry_stability(grav, delta, delta_z, height)
    theta = temp * 1.6
    dy = float(y[1] - y[0])

    hhat = shat - L_V * (2.0 * a - 1.0) * w
    c_f = -(2.0 * a - 1.0) * v_face
    w_face = mc_face_values(w, dy, c_f)
    dse_flux = shat * v_face
    lvq_flux = L_V * c_f * w_face
    mean_flux = dse_flux + lvq_flux
    eddy_flux = -L_V * d_w * (w[1:] - w[:-1]) / dy
    total_flux = mean_flux + eddy_flux
    source = (c_col / tau) * (theta_e - theta) + L_V * evap - (L_V - c_col * lam) * p
    residual = v_divergence_at_centers(total_flux, dy) - source

    return dict(
        path=path, att=att, y=y, y_face=y_face, dy=dy, n_time=n, last_n=last_n,
        days=days, u=u, v=v, v_face=v_face, w=w, p=p, temp=temp, theta=theta,
        theta_e=theta_e, u_prev=u_prev, w_prev=w_prev, ts=ts, var=var,
        a=a, d_w=d_w, w_crit=w_crit, tau_c=tau_c, evap=evap, tau=tau,
        delta_z=delta_z, c_col=c_col, shat=shat, lam=lam, latent=latent,
        hhat=hhat, dse_flux=dse_flux, lvq_flux=lvq_flux, mean_flux=mean_flux,
        eddy_flux=eddy_flux, total_flux=total_flux, source=source,
        residual=residual, w_min_run=w_min_run,
        w_star=shat / (L_V * (2.0 * a - 1.0)),
        w_quiescent=w_crit + tau_c * evap,
        r=(w_crit + tau_c * evap) / (shat / (L_V * (2.0 * a - 1.0))),
        v_d=_attr(att, "v_d"), delta_y_rad=_attr(att, "delta_y_rad"),
        y_0=_attr(att, "theta_e_y_0", float, 0.0),
        y_0_seas_amp=_attr(att, "theta_e_y_0_seasonal_amp", float, 0.0),
        seas_period=_attr(att, "theta_e_seasonal_period_days", float, 360.0),
        theta_e_type=str(att.get("theta_e_type", "sin2")),
    )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def precip_centroid(p, y, halfwidth=None) -> float:
    """Precipitation-weighted centroid, a sub-grid ITCZ latitude.

    Over the whole domain by default. A centroid is a location only for a
    feature that has one, so the caller should read it beside
    precip_spread(): on a two-band solution the centroid falls in the dry gap
    between the bands and means nothing.
    """
    m = np.ones_like(y, dtype=bool) if halfwidth is None else np.abs(y) < halfwidth
    wgt = np.maximum(p[m], 0.0)
    if wgt.sum() <= 0:
        return np.nan
    return float(np.sum(wgt * y[m]) / wgt.sum())


def precip_excess_centroid(p, y, evap) -> float:
    """Centroid of the excess rainfall in the band that contains the maximum.

    Three steps, each fixing a way the obvious version misleads. Subtracting
    E_0 removes the uniform drizzle, which otherwise pulls the centroid toward
    the middle of the domain and made a rain maximum that migrates 2000 km read
    as a centroid moving 500. Because transport conserves total water the
    domain integrals of P and E_0 are equal, so the excess is exactly the part
    of the rainfall the circulation has organised. And restricting to the
    connected run of excess containing the maximum keeps a second rain band in
    the opposite hemisphere from averaging against the first: with both lobes
    included the seasonal centroid came out LEADING the forcing by 38 days,
    which a damped system cannot do.
    """
    exc = p - evap
    if not (exc > 0).any():
        return np.nan
    i = int(np.argmax(p))
    if exc[i] <= 0:
        return np.nan
    lo = i
    while lo > 0 and exc[lo - 1] > 0:
        lo -= 1
    hi = i
    while hi < len(y) - 1 and exc[hi + 1] > 0:
        hi += 1
    wgt = exc[lo:hi + 1]
    return float(np.sum(wgt * y[lo:hi + 1]) / wgt.sum())


def excess_lobes(p, evap) -> int:
    """Number of separate connected regions raining above E_0. More than one
    means a single position cannot describe where the rain is."""
    m = (p - evap) > 0
    if not m.any():
        return 0
    return int(np.sum(np.diff(m.astype(int)) == 1) + (1 if m[0] else 0))


def precip_peak(p, y) -> float:
    """Latitude of maximum precipitation, refined by a parabola through the
    peak gridpoint and its two neighbours."""
    i = int(np.argmax(p))
    if i in (0, len(y) - 1):
        return float(y[i])
    a, b, c = p[i - 1], p[i], p[i + 1]
    den = a - 2 * b + c
    if den == 0:
        return float(y[i])
    return float(y[i] + 0.5 * (a - c) / den * (y[1] - y[0]))


def ascent_axis(v_face, y_face, p, y) -> float:
    """The dynamical ascent axis: where upper-level divergence is largest.

    Ascent is upper-level divergence, so the strongest ascent is the maximum of
    dv/dy, and this is the quantity that appears in the thermodynamic equation
    as the adiabatic cooling term. Taken as the parabolically refined maximum of
    the model's own compact divergence operator.

    An earlier version took the zero of v where it turns from equatorward to
    poleward. That is also a correct definition of a rising branch, and it
    picked the wrong one: plotted against the rainfall it sat 3 to 4 Mm away in
    the subtropics and swung with nearly twice the forcing's amplitude, because
    several such zeros exist and "the one nearest the rain" is not the same
    feature at every phase of a seasonal cycle. Caught by drawing it on the
    figure, not by reading the code.
    """
    dv = v_divergence_at_centers(np.asarray(v_face), float(y[1] - y[0]))
    return precip_peak(dv, y)


def precip_spread(p, y) -> float:
    """Precipitation-weighted standard deviation of latitude: the width the
    centroid is a centroid OF. Large compared with the domain means the
    centroid is not a location."""
    wgt = np.maximum(p, 0.0)
    if wgt.sum() <= 0:
        return np.nan
    c = np.sum(wgt * y) / wgt.sum()
    return float(np.sqrt(np.sum(wgt * (y - c) ** 2) / wgt.sum()))


def wet_fraction(p, y, evap) -> float:
    """Area fraction raining at more than a tenth of the mean source rate.

    Because transport conserves total water, the domain-mean P equals E_0 at
    equilibrium whatever the state; the wet fraction is therefore the whole
    content of "is this state aggregated?".
    """
    return float(np.mean(p > 0.1 * evap))


def dry_fraction(p) -> float:
    """Area fraction with precipitation identically zero (W at or below W_c)."""
    return float(np.mean(p <= 0.0))


def precip_bands(p, y, evap) -> int:
    """Number of distinct precipitation maxima.

    A peak counts if it rains at more than 1.2 times the mean source rate E_0
    and stands at least 0.25 E_0 clear of the surrounding minima. Both
    thresholds are absolute multiples of E_0 rather than fractions of the
    maximum, so the count means the same thing in a drizzling solution and in
    one with a 78 mm/day spike: a fraction-of-maximum threshold would scale
    with the spike and stop seeing the weaker bands. A count of zero means
    the domain drizzles at the evaporation rate with no organised band at all,
    which is what the passive-moisture V1 solutions do.
    """
    from scipy.signal import find_peaks
    e = evap * SEC_DAY
    pk, _ = find_peaks(p * SEC_DAY, height=1.2 * e, prominence=0.25 * e)
    return int(len(pk))


def zero_crossings(f, y, rising=None):
    """Interpolated zeros of f(y). rising=True keeps only crossings where f
    goes negative to positive with increasing y, False only the reverse."""
    out = []
    for i in range(1, len(y)):
        a, b = f[i - 1], f[i]
        if a == 0.0:
            out.append(float(y[i - 1]))
            continue
        if a * b < 0:
            up = b > a
            if rising is None or up == rising:
                out.append(float(y[i - 1] - a * (y[i] - y[i - 1]) / (b - a)))
    return out


def efe(r) -> float:
    """Energy-flux equator: the zero of the total column MSE flux nearest the
    precipitation centroid. Reported with its slope sign so a convergence zero
    is not mistaken for a divergence zero."""
    zs = zero_crossings(r["total_flux"], r["y_face"])
    if not zs:
        return np.nan
    c = precip_centroid(r["p"], r["y"])
    if not np.isfinite(c):
        c = 0.0
    return min(zs, key=lambda z: abs(z - c))


def cell_edges(r):
    """Ascending branch and the two descending edges, from the face v field.

    The ascending branch is where v diverges (dv/dy > 0, i.e. upper-level
    outflow); the descending edges are the zeros of v flanking it.
    """
    v, yf = r["v_face"], r["y_face"]
    zs = zero_crossings(v, yf)
    c = precip_centroid(r["p"], r["y"])
    if not np.isfinite(c):
        c = 0.0
    # ascending branch: the v zero nearest the ITCZ with v changing sign from
    # negative (southward, i.e. flow toward it from the north) to positive
    asc = min(zs, key=lambda z: abs(z - c)) if zs else np.nan
    north = [z for z in zs if z > (asc if np.isfinite(asc) else 0)]
    south = [z for z in zs if z < (asc if np.isfinite(asc) else 0)]
    # descending edge: the outermost v zero on each side that still bounds a
    # coherent cell, taken as the zero beyond the |v| extremum
    iN = int(np.argmax(v)) if len(v) else 0
    iS = int(np.argmin(v)) if len(v) else 0
    dn = min([z for z in north if z > yf[iN]], default=np.nan)
    dsx = max([z for z in south if z < yf[iS]], default=np.nan)
    return asc, dn, dsx


RO_BAND = (1.0e6, 4.0e6)  # |y|, about 8 to 33 degrees equivalent latitude


def rossby(r):
    """Local Rossby number du/dy / (beta y), and its mean over a fixed band.

    The band is |y| between 1 and 4 Mm, roughly 8 to 33 degrees equivalent
    latitude, in the hemisphere holding the strongest |v|: the core of the
    dominant cell. Fixed rather than tied to the cell edges, for two reasons.
    beta*y vanishes at the equator, so a band that reaches in toward it returns
    a ratio of two small numbers; an earlier version averaging from the rain
    centroid outward reported Ro = 9.7 for a solstitial run purely from that
    division. And on a solstitial solution the cell-finding picked the weak
    summer cell rather than the dominant cross-equatorial one, so it was
    answering about the wrong circulation.
    """
    y, u, dy = r["y"], r["u"], r["dy"]
    du = np.gradient(u, dy)
    ro = np.divide(du, BETA * y, out=np.full_like(du, np.nan),
                   where=np.abs(y) >= dy)
    v = r["v"]
    hemi = 1.0 if y[int(np.argmax(np.abs(v)))] > 0 else -1.0
    m = (hemi * y >= RO_BAND[0]) & (hemi * y <= RO_BAND[1])
    band = float(np.nanmean(ro[m])) if m.any() else np.nan
    return ro, band


def scalars(r, name="", block="", note="") -> Dict[str, Any]:
    """The scalar row for one equilibrium run."""
    y, u, w, p = r["y"], r["u"], r["w"], r["p"]
    i0 = int(np.argmin(np.abs(y)))
    i1 = int(np.argmin(np.abs(np.abs(y) - Y_ONE)))
    ro, ro_cell = rossby(r)
    asc, dn, dsx = cell_edges(r)
    iN = y > 0
    iS = y < 0
    # drift over the two trailing windows, the equilibration check
    drift_u = float(np.max(np.abs(u - r["u_prev"])) / max(np.max(np.abs(u)), 1e-12))
    drift_w = float(np.max(np.abs(w - r["w_prev"])) / max(np.max(np.abs(w)), 1e-12))
    ef = efe(r)
    # y = 3 Mm is 24 degrees under the Coriolis mapping, the latitude at which
    # docs/era5_calibration_report.pdf Table 8 quotes the observed column
    # fluxes, so the model number here is directly comparable.
    yfi = int(np.argmin(np.abs(r["y_face"] - 3.0e6)))
    # Potential-temperature contrasts at the three Coriolis-mapped latitudes
    # the same report's scorecard uses (y = 2, 3, 4 Mm are 16, 24, 33 degrees).
    def dtheta(ym):
        j = int(np.argmin(np.abs(y - ym)))
        return float(r["theta"][i0] - r["theta"][j])
    row = dict(
        name=name, block=block, note=note,
        a=r["a"], w_crit=r["w_crit"], tau_c=r["tau_c"], evap=r["evap"],
        d_w=r["d_w"], v_d=r["v_d"], delta_z=r["delta_z"],
        delta_y_rad=r["delta_y_rad"], lam=r["lam"], latent=r["latent"],
        y_0=r["y_0"], y_0_seas_amp=r["y_0_seas_amp"],
        theta_e_type=r["theta_e_type"], days=r["days"], ny=len(y),
        dt=int(float(r["att"].get("dt", 0))),
        r=r["r"], w_star=r["w_star"], w_quiescent=r["w_quiescent"],
        # dynamics
        jet=float(np.max(u[iN])), jet_lat=float(y[iN][np.argmax(u[iN])]),
        jet_s=float(np.max(u[iS])), jet_s_lat=float(y[iS][np.argmax(u[iS])]),
        u_eq=float(u[i0]), u_min=float(np.min(u)),
        u_min_lat=float(y[int(np.argmin(u))]),
        v_max=float(np.max(np.abs(r["v"]))),
        v_max_lat=float(y[int(np.argmax(np.abs(r["v"])))]),
        asc_edge=asc, north_edge=dn, south_edge=dsx,
        ro_cell=ro_cell, ro_max=float(np.nanmax(np.abs(ro))),
        # thermodynamics
        t_eq=float(r["temp"][i0]), dt_trop=float(r["temp"][i0] - r["temp"][i1]),
        theta_offset=float(r["theta"][0] - r["theta_e"][0]),
        dtheta16=dtheta(2.0e6), dtheta24=dtheta(3.0e6), dtheta33=dtheta(4.0e6),
        w_trop=float(np.mean(w[np.abs(y) <= lat_to_y(10)])),
        w_trop20=float(np.mean(w[np.abs(y) <= lat_to_y(20)])),
        # moisture and convection
        w_eq=float(w[i0]), w_max=float(np.max(w)), w_min=float(np.min(w)),
        w_min_run=r["w_min_run"],
        p_max=float(np.max(p)) * SEC_DAY,
        # Transport moves no total water in the finite-volume measure (half
        # cells at the walls), so at equilibrium the domain integral of P must
        # equal that of E_0 exactly. Taking the mean in that same measure makes
        # this a real check rather than an approximate one.
        p_mean_over_e0=float(
            cwv_integral(p, r["dy"]) / (r["evap"] * r["dy"] * (len(y) - 1))),
        itcz=precip_centroid(p, y), itcz_spread=precip_spread(p, y),
        itcz_excess=precip_excess_centroid(p, y, r["evap"]),
        itcz_peak=precip_peak(p, y),
        ascent_axis=ascent_axis(r["v_face"], r["y_face"], p, y),
        wet_frac=wet_fraction(p, y, r["evap"]), dry_frac=dry_fraction(p),
        n_bands=precip_bands(p, y, r["evap"]),
        # energetics
        hhat_min=float(np.min(r["hhat"])) / 1e6,
        hhat_eq=float(r["hhat"][i0]) / 1e6,
        hhat_neg_frac=float(np.mean(r["hhat"] < 0)),
        # A negative-Hhat core can only reach equilibrium if the circulation
        # dries some region below W* so that Hhat turns positive there and the
        # column can export energy. These three say whether it does: the mean
        # W and Hhat where precipitation has switched off, and how much of the
        # non-raining area has Hhat > 0.
        w_gap=(float(np.mean(w[p <= 0])) if (p <= 0).any() else np.nan),
        hhat_gap=(float(np.mean(r["hhat"][p <= 0])) / 1e6
                  if (p <= 0).any() else np.nan),
        gap_hhat_pos_frac=(float(np.mean(r["hhat"][p <= 0] > 0))
                           if (p <= 0).any() else np.nan),
        # Total water is conserved by transport, so the rain rate averaged over
        # the raining area is E_0 divided by the raining fraction. Aggregation
        # therefore forces intense rain by arithmetic, not by dynamics.
        p_wet_mean_over_e0=(
            float(np.mean(p[p > 0.1 * r["evap"]]) / r["evap"])
            if (p > 0.1 * r["evap"]).any() else np.nan),
        efe=ef,
        f_mean_ref=float(r["mean_flux"][yfi]) / 1e6,
        f_eddy_ref=float(r["eddy_flux"][yfi]) / 1e6,
        f_tot_ref=float(r["total_flux"][yfi]) / 1e6,
        f_tot_max=float(np.max(np.abs(r["total_flux"]))) / 1e6,
        # Bounded in [0, 1] deliberately. Dividing by the total instead blows
        # up wherever the mean and eddy fluxes nearly cancel, which is exactly
        # what happens on the negative-stability side, and produced a value of
        # 15.4 on a quantity read as a share.
        eddy_share=float(abs(r["eddy_flux"][yfi])
                         / max(abs(r["mean_flux"][yfi])
                               + abs(r["eddy_flux"][yfi]), 1e-30)),
        # numerics and steadiness
        resid_rel=float(np.max(np.abs(r["residual"]))
                        / max(np.max(np.abs(r["source"])), 1e-30)),
        drift_u=drift_u, drift_w=drift_w,
        steady=bool(r["var"]["jet_range_rel"] < 0.01 and drift_u < 0.01),
        var_window=r["var"]["window"],
        jet_range=r["var"]["jet_range"], jet_range_rel=r["var"]["jet_range_rel"],
        jet_sd=r["var"]["jet_sd"], wmean_range=r["var"]["wmean_range"],
        pmax_range=r["var"]["pmax_range"], pmax_sd=r["var"]["pmax_sd"],
        pmax_inst_mean=r["var"]["pmax_mean"],
    )
    return row


# ---------------------------------------------------------------------------
# Seasonal composites
# ---------------------------------------------------------------------------
def load_seasonal(path: str, n_cycles: int = 2) -> Optional[Dict[str, Any]]:
    """Composite the last n_cycles of a seasonal run onto one cycle, and
    measure repeatability as the difference between the last two cycles."""
    if not os.path.exists(path):
        return None
    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        att = dict(ds.attrs)
        period = int(float(att.get("theta_e_seasonal_period_days", 360.0)))
        amp = float(att.get("theta_e_y_0_seasonal_amp", 0.0))
        n = int(ds.sizes["time"])
        if n < 2 * period:
            return None
        y = ds["y"].values
        y_face = ds["y_face"].values
        a = float(att["cwv_frac"])
        d_w = float(att["d_w"])
        evap = float(att["evap"])
        w_crit = float(att["w_crit"])
        tau_c = float(att["tau_c"])
        grav = float(att["gravity"])
        shat = gross_dry_stability(grav, float(att["delta"]),
                                   float(att["delta_z"]), float(att["height"]))
        c_col = column_heat_capacity(grav)
        # last two complete cycles
        last = n - (n % period)
        c2 = slice(last - period, last)          # most recent
        c1 = slice(last - 2 * period, last - period)
        u2 = ds["u"].isel(time=c2).values
        u1 = ds["u"].isel(time=c1).values
        if not np.isfinite(u2).all():
            return None
        w2 = ds["W"].isel(time=c2).values
        p2 = ds["P"].isel(time=c2).values
        t2 = ds["T"].isel(time=c2).values
        v2 = ds["v"].isel(time=c2).values
        te2 = ds["theta_e"].isel(time=c2).values
        p1 = ds["P"].isel(time=c1).values
        days = int(ds["time"].values[-1])
    finally:
        ds.close()

    dy = float(y[1] - y[0])
    # forcing y_0(t) over the cycle, recovered from the theta_rad field's own
    # maximum (the SB08 profile peaks exactly at y_0), sub-grid by parabolic fit
    y0_t = np.array([_parabolic_peak(te2[k], y) for k in range(te2.shape[0])])
    # Three positions per day, because they are not interchangeable: the
    # excess-precipitation centroid (the organised rainfall), the rain maximum,
    # and the dynamical ascending branch. The plain centroid is kept only to
    # show how badly the uniform drizzle dilutes it.
    itcz_t = np.array([precip_excess_centroid(p2[k], y, evap)
                       for k in range(p2.shape[0])])
    itcz_plain_t = np.array([precip_centroid(p2[k], y)
                             for k in range(p2.shape[0])])
    itcz_peak_t = np.array([precip_peak(p2[k], y) for k in range(p2.shape[0])])
    asc_t = np.array([ascent_axis(v2[k], y_face, p2[k], y)
                      for k in range(v2.shape[0])])
    lobes_t = np.array([excess_lobes(p2[k], evap) for k in range(p2.shape[0])])
    pmax_t = p2.max(axis=1) * SEC_DAY
    wet_t = np.array([wet_fraction(p2[k], y, evap) for k in range(p2.shape[0])])
    bands_t = np.array([precip_bands(p2[k], y, evap) for k in range(p2.shape[0])])
    jetN_t = u2[:, y > 0].max(axis=1)
    jetS_t = u2[:, y < 0].max(axis=1)
    vmax_t = np.abs(v2).max(axis=1)
    hhat_t = shat - L_V * (2.0 * a - 1.0) * w2
    hmin_t = hhat_t.min(axis=1)
    # cross-equatorial mass flux at the equator face, a signed cell measure
    ieq = int(np.argmin(np.abs(y_face)))
    v_eq_t = v2[:, ieq]
    # total column MSE flux at the equator
    c_f = -(2.0 * a - 1.0) * v2
    mse_eq_t = np.array([
        (shat * v2[k] + L_V * c_f[k] * mc_face_values(w2[k], dy, c_f[k])
         - L_V * d_w * (w2[k][1:] - w2[k][:-1]) / dy)[ieq]
        for k in range(v2.shape[0])
    ])
    # Local Rossby number in the cross-equatorial cell, day by day. This is the
    # test of whether a migrating forcing reaches an angular-momentum-conserving
    # cell (Ro near 1) that a forcing held at the same latitude does not: the
    # Schneider and Bordoni monsoon-onset mechanism. Averaged over the band
    # between the rain centroid and the northern v = 0 edge, excluding the
    # gridpoints next to the equator where beta*y vanishes.
    ro_t = np.full(u2.shape[0], np.nan)
    du = np.gradient(u2, dy, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ro_all = np.where(np.abs(y) >= dy, du / (BETA * y), np.nan)
    for k in range(u2.shape[0]):
        hemi = 1.0 if y[int(np.argmax(np.abs(v_faces_to_centers(v2[k]))))] > 0 else -1.0
        m = (hemi * y >= RO_BAND[0]) & (hemi * y <= RO_BAND[1])
        if m.any():
            ro_t[k] = float(np.nanmean(ro_all[k][m]))
    repeat = float(np.max(np.abs(p2 - p1)) * SEC_DAY)
    repeat_itcz = float(np.nanmax(np.abs(
        itcz_t - np.array([precip_excess_centroid(p1[k], y, evap)
                           for k in range(p1.shape[0])])
    )))
    return dict(
        path=path, att=att, period=period, amp=amp, days=days, y=y,
        y_face=y_face, u=u2, w=w2, p=p2, temp=t2, v=v2, theta_e=te2,
        y0_t=y0_t, itcz_t=itcz_t, itcz_plain_t=itcz_plain_t,
        itcz_peak_t=itcz_peak_t, asc_t=asc_t, lobes_t=lobes_t,
        pmax_t=pmax_t, wet_t=wet_t, bands_t=bands_t, jetN_t=jetN_t,
        jetS_t=jetS_t, vmax_t=vmax_t, hmin_t=hmin_t, v_eq_t=v_eq_t,
        mse_eq_t=mse_eq_t, ro_t=ro_t, repeat_pmax=repeat,
        repeat_itcz=repeat_itcz,
        a=a, evap=evap, w_crit=w_crit, tau_c=tau_c, shat=shat, c_col=c_col,
        d_w=d_w,
    )


def _parabolic_peak(f, y) -> float:
    i = int(np.argmax(f))
    if i in (0, len(y) - 1):
        return float(y[i])
    a, b, c = f[i - 1], f[i], f[i + 1]
    den = a - 2 * b + c
    if den == 0:
        return float(y[i])
    return float(y[i] + 0.5 * (a - c) / den * (y[1] - y[0]))


def _harmonic(series, period):
    """Amplitude and phase of a series at the forcing frequency, by least
    squares against cos and sin plus a mean and a linear trend.

    Fitted rather than read off the extremes for two reasons. The peak of a
    circular cross-correlation is quantised to one day and aliases badly on a
    series that still has a spin-up trend in it, which produced a lag of -36
    days on a run whose rain band visibly followed the forcing by about 18.
    And a peak-to-trough amplitude counts one high day and one low day, where
    the fit uses every day in the cycle.

    Returns (amplitude, phase in radians, fraction of variance explained).
    NaNs are dropped rather than zero-filled, since zero-filling a position
    series moves it toward the equator and corrupts the phase.
    """
    n = len(series)
    t = np.arange(n, dtype=float)
    ok = np.isfinite(series)
    if ok.sum() < 8:
        return np.nan, np.nan, np.nan
    w = 2.0 * np.pi / period
    A = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones(n), t])[ok]
    y = np.asarray(series, dtype=float)[ok]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    amp = float(np.hypot(coef[0], coef[1]))
    phase = float(np.arctan2(coef[1], coef[0]))
    resid = y - A @ coef
    var = float(np.var(y))
    frac = float(1.0 - np.var(resid) / var) if var > 0 else np.nan
    return amp, phase, frac


def _lag_days(sig, ref, period) -> float:
    """Lag in days of sig behind ref at the forcing frequency, wrapped to
    (-period/2, period/2]. Positive means sig peaks later."""
    _, ps, _ = _harmonic(sig, period)
    _, pr, _ = _harmonic(ref, period)
    if not (np.isfinite(ps) and np.isfinite(pr)):
        return np.nan
    # A series lagging the reference by L days has phase ps = pr + omega L, so
    # the lag is (ps - pr)/omega. Verified against synthetic signals of known
    # lag; the opposite sign convention passes an amplitude check and fails
    # this one, which is why the check exists.
    d = (ps - pr) % (2.0 * np.pi)
    if d > np.pi:
        d -= 2.0 * np.pi
    return float(d * period / (2.0 * np.pi))


def seasonal_scalars(s, name="", block="", note="") -> Dict[str, Any]:
    y = s["y"]
    per = s["period"]
    amp_forcing, _, _ = _harmonic(s["y0_t"], per)
    amp_itcz, _, frac_itcz = _harmonic(s["itcz_t"], per)
    lag = _lag_days(s["itcz_t"], s["y0_t"], per)
    d_itcz = np.abs(np.diff(np.concatenate([s["itcz_t"], s["itcz_t"][:1]])))
    amp_plain, _, _ = _harmonic(s["itcz_plain_t"], per)
    amp_peak, _, _ = _harmonic(s["itcz_peak_t"], per)
    amp_asc, _, frac_asc = _harmonic(s["asc_t"], per)
    return dict(
        amp_itcz_plain_km=float(amp_plain / 1e3),
        amp_itcz_peak_km=float(amp_peak / 1e3),
        amp_asc_km=float(amp_asc / 1e3),
        gain_plain=(float(amp_plain / amp_forcing) if amp_forcing > 0 else np.nan),
        gain_peak=(float(amp_peak / amp_forcing) if amp_forcing > 0 else np.nan),
        gain_asc=(float(amp_asc / amp_forcing) if amp_forcing > 0 else np.nan),
        lag_peak_days=_lag_days(s["itcz_peak_t"], s["y0_t"], per),
        lag_asc_days=_lag_days(s["asc_t"], s["y0_t"], per),
        # How much of each series the single harmonic actually explains. A low
        # value means the response is not a sinusoid and its "amplitude" and
        # "lag" describe it only loosely.
        harm_frac_itcz=frac_itcz, harm_frac_asc=frac_asc,
        lobes_max=int(np.max(s["lobes_t"])),
        lobes_mean=float(np.mean(s["lobes_t"])),
        # Whether the cycle repeats at all. The two consecutive cycles either
        # agree to under 50 km in rain-band position or they disagree by
        # thousands, with nothing in between across these 19 runs, so the
        # threshold is not a tuned choice. For a run that fails it, "amplitude"
        # and "lag" describe one arbitrary realisation and mean nothing.
        cyclic=bool(s["repeat_itcz"] / 1e3 < 50.0),
        name=name, block=block, note=note, a=s["a"], w_crit=s["w_crit"],
        evap=s["evap"], d_w=s["d_w"], days=s["days"], ny=len(y),
        period=s["period"], amp_km=s["amp"] / 1e3,
        r=(s["w_crit"] + s["tau_c"] * s["evap"])
        / (s["shat"] / (L_V * (2.0 * s["a"] - 1.0))),
        amp_forcing_km=amp_forcing / 1e3, amp_itcz_km=amp_itcz / 1e3,
        gain=float(amp_itcz / amp_forcing) if amp_forcing > 0 else np.nan,
        lag_days=lag,
        itcz_max_km=float(np.nanmax(s["itcz_t"]) / 1e3),
        itcz_min_km=float(np.nanmin(s["itcz_t"]) / 1e3),
        itcz_rate_km_day=float(np.nanmax(d_itcz) / 1e3),
        jetN_max=float(np.max(s["jetN_t"])), jetN_min=float(np.min(s["jetN_t"])),
        jetS_max=float(np.max(s["jetS_t"])),
        jet_asym=float(np.max(s["jetN_t"]) - np.max(s["jetS_t"])),
        vmax_max=float(np.max(s["vmax_t"])), vmax_min=float(np.min(s["vmax_t"])),
        v_eq_absmax=float(np.max(np.abs(s["v_eq_t"]))),
        mse_eq_absmax=float(np.max(np.abs(s["mse_eq_t"])) / 1e6),
        pmax_max=float(np.max(s["pmax_t"])), pmax_min=float(np.min(s["pmax_t"])),
        wet_max=float(np.max(s["wet_t"])), wet_min=float(np.min(s["wet_t"])),
        bands_max=int(np.max(s["bands_t"])), bands_min=int(np.min(s["bands_t"])),
        hmin_min=float(np.min(s["hmin_t"]) / 1e6),
        ro_cell_max=float(np.nanmax(s["ro_t"])) if np.isfinite(s["ro_t"]).any() else np.nan,
        ro_cell_mean=float(np.nanmean(s["ro_t"])) if np.isfinite(s["ro_t"]).any() else np.nan,
        # A first-order relaxation on the model's own thermal timescale
        # tau = 37 d predicts these, as the null model for the two numbers
        # above them: gain = 1/sqrt(1 + (omega tau)^2) times the response a
        # forcing held still would give, and lag = arctan(omega tau)/omega.
        lag_pred_days=float(np.arctan(2 * np.pi * 37.0 / s["period"])
                            / (2 * np.pi / s["period"])),
        gain_damping_pred=float(
            1.0 / np.sqrt(1.0 + (2 * np.pi * 37.0 / s["period"]) ** 2)),
        p_mean_over_e0=float(np.mean(s["p"]) / s["evap"]),
        repeat_pmax=s["repeat_pmax"], repeat_itcz_km=s["repeat_itcz"] / 1e3,
    )


# ---------------------------------------------------------------------------
def build_table(tree=SWEEP_DIR, last_n=1000) -> Dict[str, Any]:
    runs = manifest()
    rows, seas_rows, missing = [], [], []
    for entry in runs:
        path = os.path.join(tree, entry["name"], "out.nc")
        seasonal = entry["args"].get("y0_seas_amp", 0.0) > 0
        if seasonal:
            s = load_seasonal(path)
            if s is None:
                missing.append(entry["name"])
                continue
            seas_rows.append(seasonal_scalars(s, entry["name"], entry["block"],
                                              entry["note"]))
        else:
            r = load_run(path, last_n=last_n)
            if r is None:
                missing.append(entry["name"])
                continue
            rows.append(scalars(r, entry["name"], entry["block"], entry["note"]))
    return dict(equilibrium=rows, seasonal=seas_rows, missing=missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=SWEEP_DIR)
    ap.add_argument("--last-n", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(SWEEP_DIR, "diagnostics.json"))
    args = ap.parse_args()

    table = build_table(args.tree, args.last_n)
    with open(args.out, "w") as fh:
        json.dump(table, fh, indent=1, default=float)
    print(f"{len(table['equilibrium'])} equilibrium rows, "
          f"{len(table['seasonal'])} seasonal rows, "
          f"{len(table['missing'])} missing/failed")
    if table["missing"]:
        print("missing: " + " ".join(table["missing"]))
    print(f"Wrote {args.out}")

    # A compact look at the two references and the regime transect
    hdr = ("name", "r", "jet", "v_max", "t_eq", "w_eq", "p_max", "wet_frac",
           "n_bands", "hhat_min", "f_mean_ref", "f_eddy_ref", "resid_rel",
           "drift_u")
    print("\n" + "  ".join(f"{h:>10}" for h in hdr))
    for row in table["equilibrium"]:
        if row["block"] in ("A", "B") and abs(row["a"] - 0.85) < 1e-9:
            print("  ".join(
                f"{row[h]:>10.4g}" if isinstance(row[h], float) else f"{str(row[h])[:10]:>10}"
                for h in hdr))


if __name__ == "__main__":
    main()
