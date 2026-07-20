"""Tier-1 v_d-ladder analysis for the fixed-Ro suite.

Nine rungs: Zhang et al. (2025)'s exact v_d values {0, 0.0125, 0.025,
0.05, 0.125, 0.25, 0.5, 1.25, 2.5} m/s at the production formulation
(gate-on + mc + staggered), SS09 parabolic forcing, ny=801, dt=30,
numba. The v_d = 0 rung is the Tier-0 run 1a day-6000 record; the other
eight are slow-drift-gated runs in model_output/fixed_ro_suite/
t1_vd_ladder/ (memo run record, 2026-07-20).

Per rung: the standard full-profile scorecard (reusing
fixed_ro_scorecard.diagnose / make_figure), the local-Ro profile and its
maximum, the steady zonal momentum budget's band means (drag, EMFD,
vertical-advection shares of the non-AMC torque), and the column
momentum flux delta*v*u.

Ladder-level checks (memo sec. 7): #2 edge collapse
Y_edge (Ro/R_beta)^(1/2) / y1 = (5/3)^(1/2) flat in v_d; #4 v_max
proportional to Ro^(-3/2) with the zero-free-parameter prefactor;
#5 momentum flux increases as Ro decreases; #6 the emergent
cell-mean-Ro closure Ro = vbar/(vbar + v_d) (analysis G), plus a
drag-extended variant Ro = vbar/(vbar + v_d + eps_u*Y_Ro/4) [derived:
cell-midpoint drag] motivated by Tier 0's finding that the default drag
is leading-order at this forcing. Pre-registered Zhang et al. (2025)
Fig. 4c targets: max-Ro log-log slopes -0.26 (smaller v_d, AMC branch,
theory v_d^(-1/4)) and -0.41 (larger v_d, eddy-dominant branch, theory
v_d^(-2/5)); max-Ro at y_m (the v-max latitude); y_E = 2*y_m. Their
ladder is sin2 forcing with the gateless collocated code, so those
slopes are cross-formulation references, not pass/fail bars.

Usage:
    python scripts/fixed_ro_t1_analysis.py [--ndays 30] [--no-scorecards]

Run from the repo root. Writes figures and t1_summary.json into
model_output/fixed_ro_suite/t1_vd_ladder/.
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import brentq

from fixed_ro_scorecard import diagnose, make_figure, report

BASE = "model_output/fixed_ro_suite"
OUT = os.path.join(BASE, "t1_vd_ladder")
VD0_PATH = os.path.join(
    BASE, "tier0_amc_edge", "amc_ss09_ny801_dt30_vd0_day6000.nc"
)
VDS = [0.0, 0.0125, 0.025, 0.05, 0.125, 0.25, 0.5, 1.25, 2.5]
SMALL_BRANCH = [0.0125, 0.025, 0.05, 0.125]
LARGE_BRANCH = [0.25, 0.5, 1.25, 2.5]


def rung_path(vd: float) -> str:
    if vd == 0.0:
        return VD0_PATH
    tag = "vd" + repr(float(vd)).replace(".", "p")
    return os.path.join(OUT, tag, f"{tag}.nc")


def rung_extras(path: str, d: Dict) -> Dict:
    """Ladder diagnostics beyond the scorecard: local-Ro max, momentum
    budget band means, and the column momentum flux delta*v*u."""
    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        a = dict(ds.attrs)
        u_prev = ds["u"].isel(time=slice(-60, -30)).mean("time").values
    finally:
        ds.close()
    beta, eps_u, vd, delta = (
        a["beta"], a["epsilon_u"], a["v_d"], a["delta"]
    )
    conv_day = int(a.get("steady_state_convergence_day", -1))
    stop_day = conv_day if conv_day > 0 else int(a["total_integration_days"])

    y, u, v, temp = d["y"], d["u"], d["v"], d["T"]
    t_e, ro_local, y_jet, y5 = d["t_e"], d["ro_local"], d["y_jet"], d["y5"]

    # Local-Ro maximum over the NH cell (Zhang Fig. 4 convention: the
    # profile max; ro_local is NaN only within one grid cell of the
    # equator). Restricted to y <= y_jet: at v_d >= 0.125 the raw NH max
    # sits at y ~ 10e6 m, on the RCE-plateau front poleward of the cell,
    # a structure Zhang's domain-spanning sin2 forcing does not have.
    nh = (y > 0) & (y <= y_jet) & np.isfinite(ro_local)
    i = int(np.nanargmax(np.where(nh, ro_local, -np.inf)))
    ro_max, y_m = float(ro_local[i]), float(y[i])

    # Whole-domain steadiness beyond the gate's cell-centric diagnostics
    # (the far-field u front at large v_d is advective and could creep):
    # max |u(last 30-d window) - u(previous 30-d window)| over the domain.
    du_win = np.abs(u - u_prev)
    i_dw = int(np.nanargmax(du_win))
    u_settle = {"max": float(du_win[i_dw]), "at_y": float(y[i_dw])}

    # Steady zonal momentum budget, band means over [y5, y_jet] (memo
    # sec. 6): v*(beta*y - u_y) = H(thetaE-theta)*u*v_y + eps_u*u
    # + v_d*H(u)*sgn(y)*u_y. Shares quantify Zhang Eq. 24's neglect of
    # drag and vertical advection.
    du = np.gradient(u, y)
    dv = np.gradient(v, y)
    band = (y >= y5) & (y <= y_jet)
    lhs = float(np.mean((v * (beta * y - du))[band]))
    vert = float(np.mean(((t_e > temp) * u * dv)[band]))
    drag = float(np.mean((eps_u * u)[band]))
    emfd = float(np.mean((vd * (u > 0) * np.sign(y) * du)[band]))
    budget = {
        "lhs": lhs, "vert": vert, "drag": drag, "emfd": emfd,
        "residual": lhs - (vert + drag + emfd),
        "share_vert": vert / lhs, "share_drag": drag / lhs,
        "share_emfd": emfd / lhs,
    }

    # Column momentum flux delta*v*u (memo ms Eq. 17 analog) over the
    # cell [0, y_jet], simulated and theory-at-fitted-Ro.
    cell = (y >= 0) & (y <= y_jet)
    flux = delta * v * u
    i_f = int(np.nanargmax(np.where(cell, flux, -np.inf)))
    flux_pred = delta * d["v_pred"] * d["u_pred"]
    return {
        "vd": vd, "eps_u": eps_u, "delta": delta, "stop_day": stop_day,
        "ro_max": ro_max, "y_m": y_m, "u_settle": u_settle,
        "flux_max": float(flux[i_f]), "y_flux_max": float(y[i_f]),
        "flux_pred_max": float(np.nanmax(np.where(cell, flux_pred,
                                                  -np.inf))),
        "flux_profile": flux, "budget": budget,
    }


def closure_curve(d0_run: Dict, vds: np.ndarray, drag: bool) -> np.ndarray:
    """Emergent cell-mean-Ro closure (memo sec. 7, analysis G): solve
    Ro = vbar/(vbar + v_d [+ eps_u*Y_Ro/4]) with vbar(Ro) =
    pref_v*(5/18)*(R_beta*dT/Ro)*Y_Ro(Ro)/6, no simulation input."""
    ds = xr.open_dataset(d0_run["path"], decode_timedelta=False)
    a = dict(ds.attrs)
    ds.close()
    beta, g, H, T0 = a["beta"], a["gravity"], a["height"], a["t_ref"]
    y1, dT = a["theta_e_y_one"], a["theta_e_delta_y"] / 1.6
    eps_u = a["epsilon_u"]
    pref_v = 1.6 * H / (a["delta"] * a["delta_z"] * a["tau"])
    r_beta = 4 * g * H * dT / (T0 * beta**2 * y1**4)

    def vbar(ro: float) -> float:
        y_ro = y1 * np.sqrt(5 * r_beta / (3 * ro))
        return pref_v * (5 * r_beta * dT / (18 * ro)) * y_ro / 6

    def y_ro_of(ro: float) -> float:
        return y1 * np.sqrt(5 * r_beta / (3 * ro))

    out = []
    for vd in vds:
        def f(ro: float) -> float:
            extra = eps_u * y_ro_of(ro) / 4 if drag else 0.0
            vb = vbar(ro)
            return ro - vb / (vb + vd + extra)

        out.append(brentq(f, 1e-6, 1.0))
    return np.array(out)


def fit_slope(vds: List[float], ro_max: Dict[float, float]) -> float:
    x = np.log([v for v in vds])
    yv = np.log([ro_max[v] for v in vds])
    return float(np.polyfit(x, yv, 1)[0])


def ladder_figures(runs: List[Dict], ex: List[Dict]) -> None:
    vds = np.array([e["vd"] for e in ex])
    pos = vds > 0
    ro_fit = np.array([r["ro_fit"] for r in runs])
    ro_area = np.array([r["ro_area"] for r in runs])
    ro_max = np.array([e["ro_max"] for e in ex])
    y_m = np.array([e["y_m"] for e in ex])
    y_vmax = np.array([r["scalars"]["y_v_max_meas"] for r in runs])
    y_jet = np.array([r["y_jet"] for r in runs])
    v_max = np.array([r["scalars"]["v_max_meas"] for r in runs])
    v_max_pred = np.array([r["scalars"]["v_max_pred"] for r in runs])
    flux_max = np.array([e["flux_max"] for e in ex])
    flux_pred = np.array([e["flux_pred_max"] for e in ex])
    r_beta = runs[0]["r_beta"]
    y1 = 9.439e6
    colors = plt.cm.viridis(
        np.linspace(0.05, 0.92, int(pos.sum()))
    )

    def rung_color(k: int):
        if vds[k] == 0.0:
            return "black"
        return colors[int(np.searchsorted(vds[pos], vds[k]))]

    # Figure 1: Zhang Fig. 4 analog (local Ro profiles; max Ro vs v_d;
    # log-log slopes vs the pre-registered -0.26/-0.41).
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    ax = axs[0]
    for k, (r, e) in enumerate(zip(runs, ex)):
        ax.plot(r["y"] / 1e6, r["ro_local"], color=rung_color(k), lw=1.4,
                label=f"$v_d$={e['vd']:g}")
        ax.plot(e["y_m"] / 1e6, e["ro_max"], "o", color=rung_color(k),
                ms=4)
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.15, 1.0)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("y [10⁶ m]")
    ax.set_ylabel("local Ro")
    ax.set_title("(a) local Ro(y), dot = max", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    ax = axs[1]
    ax.plot(vds, ro_max, "o-", color="0.3", label="max local Ro")
    ax.plot(vds, ro_fit, "s--", color="tab:blue", label="fitted cell-mean Ro")
    ax.set_xlabel("$v_d$ [m/s]")
    ax.set_ylabel("Ro")
    ax.set_title("(b) Ro vs $v_d$", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axs[2]
    for series, mk, c0, lab in ((ro_max, "o", "0.3", "max"),
                                (ro_fit, "s", "tab:blue", "fit")):
        ax.loglog(vds[pos], series[pos], mk, color=c0, ms=5)
        lut = dict(zip(vds, series))
        for branch, c in ((SMALL_BRANCH, "tab:orange"),
                          (LARGE_BRANCH, "tab:red")):
            sl = fit_slope(branch, lut)
            vb = np.array(branch)
            ax.loglog(vb, lut[branch[0]] * (vb / branch[0]) ** sl, "-",
                      color=c, lw=1.5,
                      label=f"{lab} {sl:+.2f} "
                            f"({branch[0]:g}–{branch[-1]:g})")
    ax.set_xlabel("$v_d$ [m/s]")
    ax.set_ylabel("Ro (max local: circles; fitted mean: squares)")
    ax.set_title("(c) log-log; Zhang25 sin2 slopes −0.26/−0.41",
                 fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    for ax in axs:
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("T1 v_d ladder: local Rossby number "
                 "(SS09 parabolic, production formulation)", fontsize=11)
    fig.savefig(os.path.join(OUT, "t1_ro_ladder.png"), dpi=150)
    plt.close(fig)

    # Figure 2: the emergent closure (analysis G) and its drag-extended
    # variant vs the diagnosed cell-mean Ro.
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    vgrid = np.logspace(np.log10(0.004), np.log10(3.5), 60)
    ax.semilogx(vgrid, closure_curve(runs[0], vgrid, drag=False), "-",
                color="tab:blue", lw=2,
                label="closure G: Ro = v̄/(v̄+$v_d$) [no drag]")
    ax.semilogx(vgrid, closure_curve(runs[0], vgrid, drag=True), "--",
                color="tab:green", lw=2,
                label="drag-extended: +ε$_u$Y$_{Ro}$/4")
    ax.semilogx(vds[pos], ro_fit[pos], "o", color="black", ms=7,
                label="diagnosed R̄o (fit)")
    ax.semilogx(vds[pos], ro_area[pos], "o", mfc="none", color="0.4",
                ms=7, label="diagnosed R̄o (area)")
    ax.axhline(ro_fit[0], color="black", lw=1.0, ls=":",
               label=f"$v_d$=0 fit R̄o = {ro_fit[0]:.3f}")
    ax.set_xlabel("$v_d$ [m/s]")
    ax.set_ylabel("cell-mean Ro")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Emergent cell-mean-Ro closure vs the ladder",
                 fontsize=10)
    fig.savefig(os.path.join(OUT, "t1_closure.png"), dpi=150)
    plt.close(fig)

    # Figure 3: scaling checks #4, #2, #5 and the y_m/y_E geometry.
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axs[0, 0]
    for k in range(len(vds)):
        ax.loglog(ro_fit[k], 1e3 * v_max[k], "o", color=rung_color(k))
    rg = np.linspace(ro_fit.min() * 0.9, ro_fit.max() * 1.1, 40)
    pref = v_max_pred[0] * ro_fit[0] ** 1.5
    ax.loglog(rg, 1e3 * pref * rg**-1.5, "-", color="tab:blue", lw=1.5,
              label="theory: v_max ∝ Ro$^{-3/2}$ (zero-free-parameter)")
    ax.set_xlabel("fitted cell-mean Ro")
    ax.set_ylabel("v_max [mm/s]")
    ax.set_title("(a) check #4: v_max vs Ro", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axs[0, 1]
    coll = y_jet * np.sqrt(ro_fit / r_beta) / y1
    for k in range(len(vds)):
        ax.plot(max(vds[k], 0.004), coll[k], "o", color=rung_color(k))
    ax.set_xscale("log")
    ax.axhline(np.sqrt(5 / 3), color="tab:blue", lw=1.5,
               label="(5/3)$^{1/2}$ = 1.291")
    ax.set_xlabel("$v_d$ [m/s]  ($v_d$=0 at left edge)")
    ax.set_ylabel("$y_{jet}$ (Ro/R$_β$)$^{1/2}$/$y_1$")
    ax.set_title("(b) check #2: edge collapse", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axs[1, 0]
    for k in range(len(vds)):
        ax.plot(ro_fit[k], flux_max[k], "o", color=rung_color(k))
    ax.plot(ro_fit, flux_pred, "x--", color="tab:blue", lw=1,
            label="theory at fitted Ro")
    ax.set_xlabel("fitted cell-mean Ro (v_d increases leftward)")
    ax.set_ylabel("max δ·v·u [m³/s²]")
    ax.set_title("(c) check #5: momentum flux vs Ro", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axs[1, 1]
    ax.plot(np.maximum(vds, 0.004), y_m / 1e6, "o-", color="0.3",
            label="$y_m$ (local-Ro max)")
    ax.plot(np.maximum(vds, 0.004), y_vmax / 1e6, "s-", color="tab:blue",
            label="v-max latitude")
    ax.plot(np.maximum(vds, 0.004), y_jet / 2e6, "^-", color="tab:red",
            label="$y_{jet}$/2 (Zhang: $y_E$=2$y_m$)")
    ax.set_xscale("log")
    ax.set_xlabel("$v_d$ [m/s]  ($v_d$=0 at left edge)")
    ax.set_ylabel("y [10⁶ m]")
    ax.set_title("(d) Ro-max / v-max / half-edge latitudes", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for ax in axs.flat:
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("T1 scaling checks (theory curves at diagnosed R̄o, "
                 "no free parameters)", fontsize=11)
    fig.savefig(os.path.join(OUT, "t1_scalings.png"), dpi=150)
    plt.close(fig)

    # Figure 5: full-NH-domain u and v (whole-domain inspection: the RCE
    # plateau, its poleward front, and how EMFD reshapes both).
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for k, r in enumerate(runs):
        axs[0].plot(r["y"] / 1e6, r["u"], color=rung_color(k), lw=1.2,
                    label=f"$v_d$={vds[k]:g}")
        axs[1].plot(r["y"] / 1e6, 1e3 * r["v"], color=rung_color(k),
                    lw=1.2)
    for ax, ylab in ((axs[0], "u [m/s]"), (axs[1], "v [mm/s]")):
        ax.axvline(9.439, color="tab:red", lw=0.8, ls=":",
                   label="$y_1$ (forcing edge)" if ax is axs[0] else None)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xlim(0, 15.751)
        ax.set_xlabel("y [10⁶ m]")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, lw=0.5)
    axs[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("T1 ladder, full NH domain: cell, RCE plateau, and the "
                 "EMFD-reshaped plateau front", fontsize=11)
    fig.savefig(os.path.join(OUT, "t1_farfield.png"), dpi=150)
    plt.close(fig)

    # Figure 4: momentum-budget shares of the non-AMC torque.
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    for key, label, c in (("share_emfd", "EMFD $v_d$H(u)sgn(y)u$_y$",
                           "tab:blue"),
                          ("share_drag", "drag ε$_u$u", "tab:orange"),
                          ("share_vert", "vert. advec. H(θ$_E$−θ)uv$_y$",
                           "tab:green")):
        ax.plot(np.maximum(vds, 0.004),
                [e["budget"][key] for e in ex], "o-", color=c, label=label)
    ax.plot(np.maximum(vds, 0.004),
            [e["budget"]["residual"] / e["budget"]["lhs"] for e in ex],
            ":", color="0.5", label="residual")
    ax.set_xscale("log")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("$v_d$ [m/s]  ($v_d$=0 at left edge)")
    ax.set_ylabel("band-mean share of v(βy − u$_y$)")
    ax.set_title("Zonal momentum budget over [$y_5$, $y_{jet}$]: "
                 "what balances the non-AMC torque", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(os.path.join(OUT, "t1_budget.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ndays", type=int, default=30)
    p.add_argument("--no-scorecards", action="store_true",
                   help="skip regenerating per-run scorecard PNGs")
    args = p.parse_args()

    runs, ex = [], []
    for vd in VDS:
        path = rung_path(vd)
        d = diagnose(path, args.ndays)
        e = rung_extras(path, d)
        runs.append(d)
        ex.append(e)
        report(d)
        print(f"  stop day {e['stop_day']}, max local Ro {e['ro_max']:.3f}"
              f" at y {e['y_m'] / 1e6:.2f}e6 m; budget shares"
              f" emfd {e['budget']['share_emfd']:+.2f}"
              f" drag {e['budget']['share_drag']:+.2f}"
              f" vert {e['budget']['share_vert']:+.2f}"
              f" resid {e['budget']['residual'] / e['budget']['lhs']:+.2f}")
        print(f"  domain u settling: max|Δu| {e['u_settle']['max']:.2e} m/s"
              f" between last two 30-d windows, at y"
              f" {e['u_settle']['at_y'] / 1e6:.2f}e6 m")
        if not args.no_scorecards and vd > 0:
            stem = os.path.splitext(os.path.basename(path))[0]
            make_figure(d, os.path.join(os.path.dirname(path),
                                        f"{stem}_scorecard.png"))

    lut_max = {e["vd"]: e["ro_max"] for e in ex}
    lut_fit = {e["vd"]: r["ro_fit"] for r, e in zip(runs, ex)}
    sl_small = fit_slope(SMALL_BRANCH, lut_max)
    sl_large = fit_slope(LARGE_BRANCH, lut_max)
    sl_small_fit = fit_slope(SMALL_BRANCH, lut_fit)
    sl_large_fit = fit_slope(LARGE_BRANCH, lut_fit)
    print(f"\nmax-Ro log-log slopes: small branch {sl_small:+.3f}"
          f" (Zhang25 sin2: -0.26, theory -1/4), large branch"
          f" {sl_large:+.3f} (Zhang25 sin2: -0.41, theory -2/5)")
    print(f"fitted-cell-mean-Ro slopes: small branch {sl_small_fit:+.3f},"
          f" large branch {sl_large_fit:+.3f}")

    ladder_figures(runs, ex)

    vpos = np.array([v for v in VDS if v > 0])
    cl_g = closure_curve(runs[0], vpos, drag=False)
    cl_drag = closure_curve(runs[0], vpos, drag=True)
    closure = {f"{v:g}": {"closure_g": float(g), "closure_drag": float(cd)}
               for v, g, cd in zip(vpos, cl_g, cl_drag)}
    print("\nclosure vs diagnosed R̄o (fit):")
    for r, e in zip(runs, ex):
        if e["vd"] > 0:
            c = closure[f"{e['vd']:g}"]
            print(f"  vd {e['vd']:g}: diagnosed {r['ro_fit']:.3f},"
                  f" G {c['closure_g']:.3f}"
                  f" ({100 * (c['closure_g'] / r['ro_fit'] - 1):+.1f}%),"
                  f" drag-ext {c['closure_drag']:.3f}"
                  f" ({100 * (c['closure_drag'] / r['ro_fit'] - 1):+.1f}%)")

    summary = {
        "closure": closure,
        "config": "SS09 parabolic, ny=801, dt=30, numba, production "
                  "formulation (gate-on+mc+staggered), slow-drift-gated "
                  "(window 30 d/1e-4 + slow gate 0.2%/auto); vd=0 rung = "
                  "tier0 run 1a day-6000 fixed-length record",
        "slopes": {"max_ro_small_branch": sl_small,
                   "max_ro_large_branch": sl_large,
                   "ro_fit_small_branch": sl_small_fit,
                   "ro_fit_large_branch": sl_large_fit,
                   "zhang_sin2_reference": [-0.26, -0.41],
                   "zhang_theory": [-0.25, -0.4]},
        "rungs": [
            {
                "vd": e["vd"], "path": rung_path(e["vd"]),
                "stop_day": e["stop_day"], "ro_fit": r["ro_fit"],
                "ro_area": r["ro_area"], "ro_max": e["ro_max"],
                "y_m": e["y_m"],
                "y_vmax": r["scalars"]["y_v_max_meas"],
                "y_jet": r["y_jet"],
                "v_max": r["scalars"]["v_max_meas"],
                "v_max_pred": r["scalars"]["v_max_pred"],
                "depression": r["scalars"]["depression_meas"],
                "depression_pred": r["scalars"]["depression_pred"],
                "edge_collapse": r["y_jet"]
                * float(np.sqrt(r["ro_fit"] / r["r_beta"])) / 9.439e6,
                "flux_max": e["flux_max"],
                "flux_pred_max": e["flux_pred_max"],
                "budget": {k: v for k, v in e["budget"].items()},
                "drift_gate_ok": r["drift"]["ok"],
                "drift_remaining": r["drift"]["remaining"],
                "u_settle_max": e["u_settle"]["max"],
                "u_settle_at_y": e["u_settle"]["at_y"],
                "u_asym": r["u_asym"],
                "profile_metrics": {
                    f: {k: m[k] for k in ("rel", "rel_inner", "at_y")}
                    for f, m in r["metrics"].items()
                },
            }
            for r, e in zip(runs, ex)
        ],
    }
    out_json = os.path.join(OUT, "t1_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nsummary: {out_json}")
    print("figures: t1_ro_ladder.png, t1_closure.png, t1_scalings.png,"
          " t1_budget.png in", OUT)


if __name__ == "__main__":
    main()
