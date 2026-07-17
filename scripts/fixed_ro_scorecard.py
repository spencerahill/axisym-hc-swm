"""Full-profile fixed-Ro scorecard for a suite run.

For each run, diagnose the cell-mean Rossby number, build the zero-free-
parameter theory profiles at that Ro (docs/fixed_ro_beta_plane_theory.pdf
secs. 3-5), and compare them against the simulated steady state at the
PROFILE level: max|delta| and its location for u, T, and v over the cell,
plus the equal-area lobes panel D(y) = T - T_E. Scalars (depression, v_max,
edge) fall out as by-products; the profile comparison is the confirmation
standard (suite convention, 2026-07-16).

Usage:
    python scripts/fixed_ro_scorecard.py RUN.nc [RUN2.nc ...]
        [--ndays 30] [--outdir DIR]

Writes <run>_scorecard.png beside each input (or in --outdir) and prints
the metrics table.
"""

import argparse
import os
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from ss09.read_output import load_centered
from ss09.sw_model import THETA_TO_TEMP


def diagnose(path: str, ndays: int) -> Dict:
    """Steady-state fields, diagnosed cell-mean Ro, and theory profiles."""
    y, u, v, temp = load_centered(path, ndays=ndays)
    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        a = ds.attrs
        ro_local = (
            ds["rossby_number"].isel(time=slice(-ndays, None)).mean("time").values
        )
        y_jet = float(
            ds["north_jet_lat"].isel(time=slice(-ndays, None)).mean("time")
        )
        theta_e = ds["theta_e"].values
        jet_series = ds["north_jet_lat"].isel(time=slice(-120, None)).values
        u_prev_win = (
            ds["u"].isel(time=slice(-2 * ndays, -ndays)).mean("time").values
        )
    finally:
        ds.close()

    beta, g, H, T0 = a["beta"], a["gravity"], a["height"], a["t_ref"]
    y1 = a["theta_e_y_one"]
    dT = a["theta_e_delta_y"] * THETA_TO_TEMP
    pref_v = H / (THETA_TO_TEMP * a["delta"] * a["delta_z"] * a["tau"])
    t_e = theta_e * THETA_TO_TEMP

    # Diagnosed cell-mean Ro over [y5, y_jet] (memo sec. 6): area mean of
    # local Ro, and the best-fit Ro regressing u against beta*y^2/2. The
    # inner bound is fixed at the beta-plane equivalent of Hill et al.
    # (2025)'s 5-degree cutoff (Ro ill-defined near the equator); a fixed
    # bound, unlike the earlier argmax-of-Ro anchor, is continuous in the
    # underlying fields (the anchor jumped 1.1e6 m between epochs of run 1a
    # when its two smooth Ro extrema traded rank).
    y5 = 6.371e6 * np.deg2rad(5.0)
    nh = y > 0
    band = nh & (y >= y5) & (y <= y_jet) & np.isfinite(ro_local)
    ro_area = float(np.mean(ro_local[band]))
    basis = beta * y[band] ** 2 / 2
    ro_fit = float(np.sum(u[band] * basis) / np.sum(basis**2))

    # Residual-drift gate: the slow tail rides the drag timescale 1/eps_u,
    # so a linear jet-latitude trend over the last 120 d, extrapolated as an
    # exponential tail, implies remaining drift ~ rate x tau_drag. WARN when
    # that exceeds 0.2% (the detector-stop runs 1a/1b failed this badly:
    # 0.4%/60 d at stop with tau_drag = 1157 d implies ~8% remaining).
    days = np.arange(jet_series.size, dtype=float)
    jet_slope = float(np.polyfit(days, jet_series, 1)[0])  # m/day
    jet_rate_60d = jet_slope * 60 / y_jet
    tau_drag_days = (1.0 / (a["epsilon_u"] * 86400.0)
                     if a["epsilon_u"] > 0 else np.nan)
    drift_remaining = abs(jet_slope) / y_jet * tau_drag_days
    ro_fit_prev = float(np.sum(u_prev_win[band] * basis) / np.sum(basis**2))
    drift = {"jet_rate_60d": jet_rate_60d, "tau_drag_days": tau_drag_days,
             "remaining": drift_remaining, "ndays_trend": jet_series.size,
             "ro_fit_prev": ro_fit_prev,
             "ok": bool(drift_remaining < 0.002)}

    # Theory at the fitted Ro: memo Eqs. (5)-(9) and (13).
    ro = ro_fit
    r_beta = 4 * g * H * dT / (T0 * beta**2 * y1**4)
    y_ro = y1 * np.sqrt(5 * r_beta / (3 * ro))
    d0 = 5 * r_beta * dT / (18 * ro)
    b = dT / y1**2
    c = ro * T0 * beta**2 / (8 * g * H)
    u_rce = 2 * g * H * dT / (T0 * beta * y1**2)

    in_cell = np.abs(y) <= y_ro
    u_pred = np.where(
        in_cell, ro * beta * y**2 / 2, np.where(np.abs(y) < y1, u_rce, 0.0)
    )
    d_pred = np.where(in_cell, -d0 + b * y**2 - c * y**4, 0.0)
    t_pred = t_e + d_pred
    x = y / y_ro
    v_pred = np.where(in_cell, pref_v * d0 * y_ro * (x - 2 * x**3 + x**5), 0.0)

    # Profile metrics over the NH cell interior [0, Y_Ro], plus the inner
    # half [0, Y_Ro/2] (errors concentrate at the terminus, where local Ro
    # departs furthest from the cell mean); each error is scaled by its
    # field's natural theory amplitude.
    cell = nh & in_cell
    inner = cell & (np.abs(y) <= y_ro / 2)
    scales = {"u": ro * beta * y_ro**2 / 2, "T": d0,
              "v": pref_v * d0 * y_ro * 16 / (25 * np.sqrt(5))}
    metrics = {}
    for name, sim, pred in (("u", u, u_pred), ("T", temp, t_pred),
                            ("v", v, v_pred)):
        err = sim - pred
        i = np.nanargmax(np.abs(err[cell]))
        metrics[name] = {"max_abs": float(np.abs(err[cell])[i]),
                         "at_y": float(y[cell][i]),
                         "rel": float(np.abs(err[cell])[i] / scales[name]),
                         "max_abs_inner": float(np.nanmax(np.abs(err[inner]))),
                         "rel_inner": float(np.nanmax(np.abs(err[inner]))
                                            / scales[name])}

    # Scalar by-products, for continuity with the memo's run-record tables.
    i0 = int(np.argmin(np.abs(y)))
    i_vmax = int(np.nanargmax(np.where(nh, v, -np.inf)))
    scalars = {
        "depression_meas": float(t_e[i0] - temp[i0]),
        "depression_pred": float(d0),
        "v_max_meas": float(v[i_vmax]),
        "v_max_pred": float(pref_v * d0 * y_ro * 16 / (25 * np.sqrt(5))),
        "y_v_max_meas": float(y[i_vmax]),
        "y_v_max_pred": float(y_ro / np.sqrt(5)),
    }

    return {"path": path, "ndays": ndays, "y": y, "u": u, "v": v, "T": temp,
            "t_e": t_e, "ro_local": ro_local, "y_jet": y_jet,
            "y5": y5, "ro_area": ro_area, "ro_fit": ro_fit,
            "r_beta": r_beta, "y_ro": y_ro, "d0": d0, "u_rce": u_rce,
            "u_pred": u_pred, "t_pred": t_pred, "d_pred": d_pred,
            "v_pred": v_pred, "metrics": metrics, "scalars": scalars,
            "drift": drift,
            "u_asym": float(np.nanmax(np.abs(u - u[::-1])))}


SIM = dict(color="black", lw=1.8, label="simulated")
PRED = dict(color="tab:blue", lw=2.0, ls="--", label="theory at fitted Ro")
FORC = dict(color="tab:red", lw=1.8, ls=":", label="forcing")


def _edges(ax: Axes, r: Dict) -> None:
    ax.axvline(r["y_ro"] / 1e6, color="tab:blue", lw=0.8, alpha=0.6)
    ax.axvline(r["y_jet"] / 1e6, color="gray", lw=0.8, alpha=0.8)


def make_figure(r: Dict, out_png: str) -> None:
    y6 = r["y"] / 1e6
    xmax = 4.5
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)

    ax = axs[0, 0]
    ax.plot(y6, r["u"], **SIM)
    ax.plot(y6, r["u_pred"], **PRED)
    _edges(ax, r)
    m = r["metrics"]["u"]
    ax.set_title(
        f"u: max|Δ| = {m['max_abs']:.2f} m/s at y = {m['at_y'] / 1e6:.2f}e6"
        f" ({100 * m['rel']:.1f}% of cell-edge u)", fontsize=10)
    ax.set_ylabel("u [m/s]")
    ax.legend(fontsize=8, frameon=False)

    ax = axs[0, 1]
    ax.plot(y6, r["T"], **SIM)
    ax.plot(y6, r["t_pred"], **PRED)
    ax.plot(y6, r["t_e"], **FORC)
    _edges(ax, r)
    m = r["metrics"]["T"]
    ax.set_title(
        f"T: max|Δ| = {1000 * m['max_abs']:.1f} mK at y = "
        f"{m['at_y'] / 1e6:.2f}e6 ({100 * m['rel']:.1f}% of depression)",
        fontsize=10)
    ax.set_ylabel("T [K]")
    ax.legend(fontsize=8, frameon=False)

    ax = axs[1, 0]
    ax.plot(y6, r["T"] - r["t_e"], **SIM)
    ax.plot(y6, r["d_pred"], **PRED)
    ax.axhline(0, color="gray", lw=0.6)
    _edges(ax, r)
    ax.set_title("equal-area lobes: D(y) = T − T_E", fontsize=10)
    ax.set_ylabel("T − T_E [K]")
    ax.set_xlabel("y [10⁶ m]")

    ax = axs[1, 1]
    ax.plot(y6, r["v"], **SIM)
    ax.plot(y6, r["v_pred"], **PRED)
    ax.axhline(0, color="gray", lw=0.6)
    _edges(ax, r)
    m = r["metrics"]["v"]
    ax.set_title(
        f"v: max|Δ| = {1000 * m['max_abs']:.2f} mm/s at y = "
        f"{m['at_y'] / 1e6:.2f}e6 ({100 * m['rel']:.1f}% of v_max)",
        fontsize=10)
    ax.set_ylabel("v [m/s]")
    ax.set_xlabel("y [10⁶ m]")

    for ax in axs.flat:
        ax.set_xlim(0, xmax)
        ax.grid(alpha=0.25, lw=0.5)
    axs[1, 1].set_ylim(
        -0.15 * np.nanmax(r["v_pred"]), 1.6 * np.nanmax(r["v_pred"]))

    d = r["drift"]
    fig.suptitle(
        f"{os.path.basename(r['path'])}\n"
        f"R̄o fit = {r['ro_fit']:.3f} (area mean {r['ro_area']:.3f}),  "
        f"Y_Ro = {r['y_ro'] / 1e6:.2f}e6 m,  jet = {r['y_jet'] / 1e6:.2f}e6 m,"
        f"  last {r['ndays']} d,  drift gate "
        f"{'PASS' if d['ok'] else 'WARN'} "
        f"({100 * d['remaining']:.2f}% rem)", fontsize=10)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def report(r: Dict) -> None:
    print(f"\n=== {os.path.basename(r['path'])} (last {r['ndays']} d) ===")
    print(f"diagnosed R̄o: fit {r['ro_fit']:.3f}, area mean {r['ro_area']:.3f}"
          f" over [{r['y5'] / 1e6:.2f}, {r['y_jet'] / 1e6:.2f}]e6 m"
          f" (y5 = a*5deg, HH25)")
    print(f"theory: Y_Ro {r['y_ro'] / 1e6:.3f}e6 m, depression {r['d0']:.3f} K,"
          f" u_RCE {r['u_rce']:.2f} m/s")
    for name, m in r["metrics"].items():
        print(f"  {name}: max|Δ| {m['max_abs']:.3e} at y {m['at_y'] / 1e6:.2f}e6"
              f" ({100 * m['rel']:.1f}% of scale); inner half "
              f"{m['max_abs_inner']:.3e} ({100 * m['rel_inner']:.1f}%)")
    s = r["scalars"]
    print(f"  depression: meas {s['depression_meas']:.3f} K,"
          f" pred {s['depression_pred']:.3f} K")
    print(f"  v_max: meas {1000 * s['v_max_meas']:.2f} mm/s at"
          f" {s['y_v_max_meas'] / 1e6:.2f}e6 m; pred"
          f" {1000 * s['v_max_pred']:.2f} mm/s at"
          f" {s['y_v_max_pred'] / 1e6:.2f}e6 m")
    d = r["drift"]
    print(f"  drift gate: {'PASS' if d['ok'] else 'WARN'}: jet trend "
          f"{100 * d['jet_rate_60d']:+.4f}%/60 d over last {d['ndays_trend']} d,"
          f" tau_drag {d['tau_drag_days']:.0f} d, implied remaining "
          f"{100 * d['remaining']:.2f}% (gate: <0.2%)")
    print(f"  Ro-fit stability: previous {r['ndays']}-d window "
          f"{d['ro_fit_prev']:.4f} vs current {r['ro_fit']:.4f} "
          f"({100 * (r['ro_fit'] - d['ro_fit_prev']) / r['ro_fit']:+.3f}%)")
    print(f"  u hemispheric asymmetry max|u(y)-u(-y)|: {r['u_asym']:.2e}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="model output NetCDF file(s)")
    p.add_argument("--ndays", type=int, default=30,
                   help="averaging window (days) at the end of the run")
    p.add_argument("--outdir", default=None,
                   help="figure directory (default: beside each run)")
    args = p.parse_args()

    for path in args.runs:
        r = diagnose(path, args.ndays)
        stem = os.path.splitext(os.path.basename(path))[0]
        outdir = args.outdir or os.path.dirname(os.path.abspath(path))
        out_png = os.path.join(outdir, f"{stem}_scorecard.png")
        make_figure(r, out_png)
        report(r)
        print(f"  figure: {out_png}")


if __name__ == "__main__":
    main()
