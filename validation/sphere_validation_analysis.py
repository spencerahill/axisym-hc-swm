"""Scores items (i) to (iii) of the spherical-geometry plan's validation runs,
and draws the inspection figures of every run, item (iv)'s included.

The runs are those of validation/sphere_validation_runs.py, one directory
per run under --run-root. Every quantity here is computed from the output
files with this script's own finite differences, written from the plan's
formulas and independent of the model's operator functions. All time means
are over the final 30 days; a tendency is the difference of the means of the
last 10 and the first 10 of those days, divided by 20 days.

Item (i), the drag-limited angular-momentum identity (runs i_amc_*: v_d = 0,
no vertical advection). The steady u equation is v (f + zeta) = eps_u u, so
1 - Ro_loc = eps_u u / (f v), with Ro_loc = -zeta/f and zeta = -d(cos u)/dy
computed here by first-order upwind differences on the sign of v. The misfit
of the identity, (1 - Ro_loc) - eps_u u / (f v), equals the tendency divided
by f v in a model whose u equation is the spherical one. The band is each
hemisphere's centers between 5 degrees and the jet (the maximum of u) where
|v| is at least 10% of that hemisphere's max|v|. Pass: on every point of the
band, |misfit - tendency/(f v)| <= 0.1 S, where S = u tan(lat) / (a f) is the
residual a model missing the metric part of the advection would leave. The
same identity with zeta = -cos(lat) du/dy, the form without the metric part,
is printed as the positive control and should miss by about S.

Item (ii), steady budgets of the 50 K pair (iv_sphere_dy50, iv_twin_dy50).
The u, v and theta budgets are evaluated from the 30-day means with
second-order centered differences (compact face differences for the
divergence and the pressure gradient). Each budget's residual is the
tendency minus the sum of its terms, and its size is max|residual| / max
over terms of max|term|, both taken within 40 degrees of the equator (the
twin's latitude is y/a). Pass: for each budget, the sphere's size is no
larger than the twin's. The same size poleward of 40 degrees, away from the
two outermost centers or faces at each end, is printed separately. (The
40-degree restriction, the same as item (iii)'s, was set on 2026-11-08 before
the validation runs, after 200- and 400-day cold starts showed a
near-grid-scale easterly notch at the poleward end of the westerlies, near 75
to 81 degrees on the sphere.)

Item (iii), resolution (the 50 K pair at ny = 325 and 509 against
ny = 649 and 1017). The finer grids contain every center of the coarser ones,
so u, v (reconstructed at the centers) and theta are compared at the shared
centers. Within 40 degrees of the equator, max|fine - coarse| is scaled by
the coarse field's max|field| over the domain. Pass: for each field, the
sphere's scaled difference is no larger than the twin's. Poleward of 40
degrees the same quantity is printed separately (the polar cells).

Item (iv) is not scored here: its criterion needs the next-order term from
theory and is evaluated in the science project.

Usage:
    python validation/sphere_validation_analysis.py --run-root DIR --fig-dir DIR
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from puffins.plotting import plt_rc_params_custom  # noqa: E402

plt.rcParams.update(plt_rc_params_custom)

WINDOW = 30
EDGE = 10  # days at each end of the window for the tendency
DAY = 86400.0
THETA_TO_TEMP = 1 / 1.6
COLORS = {"sphere": "#1f5fa8", "twin": "#c4512b", "control": "0.45"}


class Run:
    """A run's 30-day means, tendencies and geometry, from its output file."""

    def __init__(self, path):
        ds = xr.open_dataset(path, decode_times=False)
        self.path = path
        self.attrs = dict(ds.attrs)
        self.ndays = ds.sizes["time"]
        self.y = ds.y.values
        self.y_face = ds.y_face.values
        self.dy = float(self.attrs["dy"])
        w = slice(-WINDOW, None)
        self.u = ds.u.isel(time=w).mean("time").values
        self.v_face = ds.v.isel(time=w).mean("time").values
        self.T = ds["T"].isel(time=w).mean("time").values
        self.theta = self.T / THETA_TO_TEMP
        te = ds.theta_e
        self.theta_e = (te.isel(time=w).mean("time") if "time" in te.dims else te).values
        span = (WINDOW - EDGE) * DAY

        def tend(name):
            da = ds[name].isel(time=w)
            return ((da.isel(time=slice(-EDGE, None)).mean("time")
                     - da.isel(time=slice(0, EDGE)).mean("time")).values / span)

        self.du_dt, self.dv_dt = tend("u"), tend("v")
        self.dtheta_dt = tend("T") / THETA_TO_TEMP
        self.finite = all(bool(np.all(np.isfinite(ds[n].values))) for n in ("u", "v", "T"))
        self.u_all = ds.u.values
        ds.close()
        self.sphere = self.attrs["geometry"] == "sphere"
        self.a = float(self.attrs["radius"])
        if self.sphere:
            omega = float(self.attrs["rot_rate"])
            s, s_f = self.y / self.a, self.y_face / self.a
            self.c = np.sqrt((1 - s) * (1 + s))
            self.c_f = np.sqrt((1 - s_f) * (1 + s_f))
            self.f = 2 * omega * s
            self.f_f = 2 * omega * s_f
            self.lat = np.rad2deg(np.arcsin(np.clip(s, -1, 1)))
            self.lat_f = np.rad2deg(np.arcsin(s_f))
            self.metric_f = np.tan(np.deg2rad(self.lat_f)) / self.a
        else:
            beta = float(self.attrs["beta"])
            self.c, self.c_f = np.ones_like(self.y), np.ones_like(self.y_face)
            self.f, self.f_f = beta * self.y, beta * self.y_face
            # the twin's latitude: its walls are at +-pi a / 2, phi = y / a
            self.lat, self.lat_f = np.rad2deg(self.y / self.a), np.rad2deg(self.y_face / self.a)
            self.metric_f = np.zeros_like(self.y_face)
        self.v = faces_to_centers(self.v_face)

    def p(self, name):
        return float(self.attrs[name])


def faces_to_centers(g):
    out = np.zeros(g.shape[0] + 1)
    out[1:-1] = 0.5 * (g[:-1] + g[1:])
    return out


def compact_divergence(g, dy):
    """d(g)/dy at the centers from face values, half cells at the ends."""
    d = np.empty(g.shape[0] + 1)
    d[1:-1] = (g[1:] - g[:-1]) / dy
    d[0], d[-1] = 2 * g[0] / dy, -2 * g[-1] / dy
    return d


def upwind_ddy(q, vc, dy):
    out = np.zeros_like(q)
    back = (q[1:-1] - q[:-2]) / dy
    fwd = (q[2:] - q[1:-1]) / dy
    out[1:-1] = np.where(vc[1:-1] > 0, back, np.where(vc[1:-1] < 0, fwd, 0.0))
    return out


def lat_axis(ax, lim=90):
    """sin(latitude) spacing, so equal lengths cover equal area."""
    ticks = [t for t in (-90, -60, -30, -15, 0, 15, 30, 60, 90) if abs(t) <= lim]
    ax.set_xticks(np.sin(np.deg2rad(ticks)))
    ax.set_xticklabels([f"{t}" for t in ticks])
    ax.set_xlim(-np.sin(np.deg2rad(lim)), np.sin(np.deg2rad(lim)))
    ax.set_xlabel("latitude (degrees, spaced by sine)")


def sx(lat):
    return np.sin(np.deg2rad(lat))


# --- item (i) ---------------------------------------------------------------

def item_i(run):
    eps = run.p("epsilon_u")
    u, vc, c, f, lat = run.u, run.v, run.c, run.f, run.lat
    zeta = -upwind_ddy(c * u, vc, run.dy)
    zeta_nometric = -c * upwind_ddy(u, vc, run.dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        lhs = 1 + zeta / f
        rhs = eps * u / (f * vc)
        tend = run.du_dt / (f * vc)
        S = u * np.tan(np.deg2rad(lat)) / (run.a * f)
        err = np.abs(lhs - rhs - tend) / np.abs(S)
        err_nm = np.abs(1 + zeta_nometric / f - rhs - tend) / np.abs(S)
    band = np.zeros(run.y.shape, bool)
    for hemi in (lat > 0, lat < 0):
        idx = np.where(hemi)[0]
        jet = idx[np.argmax(u[idx])]
        vmax = np.max(np.abs(vc[idx]))
        inside = (np.abs(lat) >= 5) & (np.abs(lat) <= abs(lat[jet]))
        band |= hemi & inside & (np.abs(vc) >= 0.1 * vmax)
    res = dict(
        run=os.path.basename(os.path.dirname(run.path)), eps_u=eps, days=run.ndays,
        finite=run.finite, band_points=int(band.sum()),
        band_lat_range=[float(np.min(np.abs(lat[band]))), float(np.max(np.abs(lat[band])))],
        max_err_over_S=float(np.max(err[band])),
        median_control_over_S=float(np.median(err_nm[band])),
        max_abs_lhs=float(np.max(np.abs(lhs[band]))),
        max_abs_tendency_term=float(np.max(np.abs(tend[band]))),
        min_S=float(np.min(np.abs(S[band]))), max_S=float(np.max(np.abs(S[band]))),
    )
    res["pass"] = bool(res["max_err_over_S"] <= 0.1 and res["finite"])
    return res, dict(lat=lat, u=u, lhs=lhs, rhs=rhs, err=err, err_nm=err_nm, S=S,
                     band=band, a=run.a, omega=run.p("rot_rate"))


def fig_item_i(results, path):
    fig, axes = plt.subplots(len(results), 3, figsize=(11, 3.2 * len(results)),
                             squeeze=False, constrained_layout=True)
    for row, (res, d) in zip(axes, results):
        lat, band = d["lat"], d["band"]
        x = sx(lat)
        ax = row[0]
        amc = d["omega"] * d["a"] * np.sin(np.deg2rad(lat)) ** 2 / np.cos(np.deg2rad(lat))
        ax.plot(x, d["u"], color=COLORS["sphere"])
        ok = np.abs(lat) < 45
        ax.plot(x[ok], amc[ok], color="0.6", ls="--")
        ax.text(0.03, 0.92, f"(a) u, eps_u = {res['eps_u']:.0e} s$^{{-1}}$",
                transform=ax.transAxes)
        ax.text(0.03, 0.82, "dashed: angular-momentum-\nconserving wind", color="0.45",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.set_ylabel("m s$^{-1}$")
        lat_axis(ax, 60)
        ax = row[1]
        ax.plot(x[band], d["lhs"][band], ".", color=COLORS["sphere"], ms=3)
        ax.plot(x[band], d["rhs"][band], ".", color=COLORS["twin"], ms=3)
        ax.text(0.03, 0.92, "(b) the identity's two sides, in the band",
                transform=ax.transAxes)
        ax.text(0.03, 0.82, "1 - Ro_loc", color=COLORS["sphere"], transform=ax.transAxes)
        ax.text(0.03, 0.74, "eps_u u / (f v)", color=COLORS["twin"], transform=ax.transAxes)
        lat_axis(ax, 60)
        ax = row[2]
        ax.semilogy(x[band], d["err"][band], ".", color=COLORS["sphere"], ms=3)
        ax.semilogy(x[band], d["err_nm"][band], ".", color=COLORS["control"], ms=3)
        ax.axhline(0.1, color="k", lw=0.8)
        ax.text(0.03, 0.92, "(c) misfit beyond the tendency, / S", transform=ax.transAxes)
        ax.text(0.03, 0.82, "spherical zeta", color=COLORS["sphere"], transform=ax.transAxes)
        ax.text(0.03, 0.74, "zeta without the metric part (control)",
                color=COLORS["control"], transform=ax.transAxes)
        ax.text(0.97, 0.1, "pass threshold 0.1", ha="right", transform=ax.transAxes,
                fontsize=8)
        lat_axis(ax, 60)
    fig.savefig(path)
    plt.close(fig)


# --- item (ii) --------------------------------------------------------------

def budgets(run):
    u, v, vf, th, te = run.u, run.v, run.v_face, run.theta, run.theta_e
    c, c_f, f, f_f, dy = run.c, run.c_f, run.f, run.f_f, run.dy
    div = compact_divergence(c_f * vf, dy)
    gate_name = run.attrs.get("vert_advec_gate", "theta")
    gate = (np.heaviside(div, 0.5) if gate_name == "kinematic"
            else np.heaviside(te - th, 0.5))
    with np.errstate(divide="ignore", invalid="ignore"):
        emfd = -(run.p("v_d") * np.heaviside(u, 0.5) * np.sign(run.y)
                 * np.gradient(c**2 * u, dy) / c)
    emfd[c == 0] = 0.0
    u_terms = {
        "Coriolis": f * v,
        "meridional advection": -v * np.gradient(c * u, dy),
        "vertical advection": -gate * u * div if run.attrs.get(
            "include_vert_advec_u", "True") == "True" else 0 * u,
        "drag": -run.p("epsilon_u") * u,
        "eddy momentum flux divergence": emfd,
    }
    uf = 0.5 * (u[:-1] + u[1:])
    dT = (run.T[1:] - run.T[:-1]) / dy
    v_terms = {
        "Coriolis and metric": -(f_f * uf + run.metric_f * uf**2) / 2,
        "pressure gradient": -(run.p("gravity") * run.p("height") / run.p("t_ref")) * c_f * dT / 2,
        "smoothing": run.p("k_v") * c_f * (div[1:] - div[:-1]) / dy / 2,
    }
    th_terms = {
        "adiabatic": -(run.p("delta") * run.p("delta_z") / run.p("height")) * div,
        "Newtonian cooling": (te - th) / run.p("tau"),
    }
    out = {}
    for name, terms, tend, lat in (
        ("u", u_terms, run.du_dt, run.lat),
        ("v", v_terms, run.dv_dt, run.lat_f),
        ("theta", th_terms, run.dtheta_dt, run.lat),
    ):
        resid = tend - sum(terms.values())
        inner = np.abs(lat) <= 40
        outer = ~inner
        outer[:2] = outer[-2:] = False
        sizes = {}
        for region, mask in (("within_40", inner), ("poleward_of_40", outer)):
            largest = max(float(np.max(np.abs(np.asarray(t) * np.ones_like(lat))[mask]))
                          for t in terms.values())
            sizes[region] = float(np.max(np.abs(resid[mask]))) / largest
            sizes[f"{region}_largest_term"] = largest
        sizes["tendency_within_40"] = (float(np.max(np.abs(tend[inner])))
                                       / sizes["within_40_largest_term"])
        out[name] = sizes
    return out, u_terms


def fig_item_ii(sph, twin, sph_terms, twin_terms, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True, constrained_layout=True)
    colors = ["#1f5fa8", "#c4512b", "#2e8b57", "0.35", "#8e44ad"]
    for ax, run, terms, title in ((axes[0], sph, sph_terms, "(a) sphere"),
                                  (axes[1], twin, twin_terms, "(b) beta-plane twin")):
        x = sx(run.lat)
        for (name, term), col in zip(terms.items(), colors):
            ax.plot(x, term, color=col, lw=1.2, label=name)
        ax.text(0.02, 0.95, title + ": u budget terms, 30-day means",
                transform=ax.transAxes)
        lat_axis(ax, 90)
    axes[0].set_ylabel("m s$^{-2}$")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


# --- item (iii) -------------------------------------------------------------

def resolution(coarse, fine):
    sub = slice(None, None, 2)
    if not np.allclose(fine.y[sub], coarse.y, rtol=0, atol=1e-3):
        raise ValueError("the fine grid does not contain the coarse centers")
    within = np.abs(coarse.lat) <= 40
    out = {}
    for name in ("u", "v", "theta"):
        a, b = getattr(coarse, name), getattr(fine, name)[sub]
        scale = float(np.max(np.abs(a)))
        d = np.abs(b - a)
        out[name] = dict(
            within_40=float(np.max(d[within])) / scale,
            poleward_of_40=float(np.max(d[~within])) / scale,
            scale=scale,
            within_40_over_range=float(np.max(d[within])) / float(np.ptp(a)),
        )
    return out


def fig_item_iii(pairs, path):
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True, constrained_layout=True)
    for col, name, unit in ((0, "u", "m s$^{-1}$"), (1, "v", "m s$^{-1}$"), (2, "theta", "K")):
        for label, (coarse, fine) in pairs.items():
            x = sx(coarse.lat)
            a = getattr(coarse, name)
            b = getattr(fine, name)[::2]
            axes[0, col].plot(x, a, color=COLORS[label], lw=1.2)
            axes[1, col].plot(x, b - a, color=COLORS[label], lw=1.0)
        axes[0, col].set_ylabel(f"{name} ({unit})")
        axes[1, col].set_ylabel(f"fine - coarse ({unit})")
        axes[1, col].axvspan(-1, -np.sin(np.deg2rad(40)), color="0.93", lw=0)
        axes[1, col].axvspan(np.sin(np.deg2rad(40)), 1, color="0.93", lw=0)
        lat_axis(axes[1, col], 90)
    axes[0, 0].text(0.03, 0.9, "sphere", color=COLORS["sphere"], transform=axes[0, 0].transAxes)
    axes[0, 0].text(0.03, 0.8, "beta-plane twin", color=COLORS["twin"],
                    transform=axes[0, 0].transAxes)
    axes[1, 0].text(0.03, 0.9, "shaded: poleward of 40 degrees", color="0.45",
                    transform=axes[1, 0].transAxes, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


# --- item (iv): inspection only -------------------------------------------

def fig_item_iv(runs, path):
    rungs = [r for r in ("12.5", "25", "50") if f"sphere_dy{r}" in runs]
    fig, axes = plt.subplots(3, len(rungs), figsize=(3.8 * len(rungs), 8),
                             squeeze=False, sharex=True, constrained_layout=True)
    for j, r in enumerate(rungs):
        for label in ("sphere", "twin"):
            run = runs.get(f"{label}_dy{r}")
            if run is None:
                continue
            x = sx(run.lat)
            axes[0, j].plot(x, run.u, color=COLORS[label], lw=1.2)
            axes[1, j].plot(x, run.v, color=COLORS[label], lw=1.2)
            axes[2, j].plot(x, run.theta, color=COLORS[label], lw=1.2)
            axes[2, j].plot(x, run.theta_e, color=COLORS[label], lw=0.8, ls="--")
        axes[0, j].set_title(f"Delta_y = {r} K", fontsize=10)
        lat_axis(axes[2, j], 90)
    axes[0, 0].set_ylabel("u (m s$^{-1}$)")
    axes[1, 0].set_ylabel("v at centers (m s$^{-1}$)")
    axes[2, 0].set_ylabel("theta, theta_E dashed (K)")
    axes[0, 0].text(0.03, 0.9, "sphere", color=COLORS["sphere"], transform=axes[0, 0].transAxes)
    axes[0, 0].text(0.03, 0.8, "beta-plane twin", color=COLORS["twin"],
                    transform=axes[0, 0].transAxes)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-root", required=True)
    p.add_argument("--fig-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    def load(label):
        path = os.path.join(args.run_root, label, "output.nc")
        return Run(path) if os.path.exists(path) else None

    ledger = {}
    res_i = [item_i(r) for r in (load("i_amc_eps1e-8"), load("i_amc_eps1e-9")) if r]
    for res, _ in res_i:
        print(f"(i) {res['run']}: {res['days']} days; band {res['band_points']} points "
              f"at |lat| {res['band_lat_range'][0]:.1f} to {res['band_lat_range'][1]:.1f}; "
              f"max |misfit - tendency term| / S = {res['max_err_over_S']:.4f}; "
              f"median of the no-metric control / S = {res['median_control_over_S']:.3f}; "
              f"max|1 - Ro_loc| = {res['max_abs_lhs']:.4f}; S from {res['min_S']:.4f} "
              f"to {res['max_S']:.4f} -> {'PASS' if res['pass'] else 'FAIL'}")
    if res_i:
        ledger["i"] = [r for r, _ in res_i]
        fig_item_i(res_i, os.path.join(args.fig_dir, "sphere_item_i_am_identity.pdf"))

    sph, twin = load("iv_sphere_dy50"), load("iv_twin_dy50")
    if sph and twin:
        b_s, terms_s = budgets(sph)
        b_t, terms_t = budgets(twin)
        ledger["ii"] = dict(sphere=b_s, twin=b_t)
        for name in ("u", "v", "theta"):
            rs, rt = b_s[name]["within_40"], b_t[name]["within_40"]
            print(f"(ii) {name} budget within 40 deg: residual / largest term = {rs:.2e} "
                  f"on the sphere, {rt:.2e} on the twin -> {'PASS' if rs <= rt else 'FAIL'}; "
                  f"poleward of 40 deg {b_s[name]['poleward_of_40']:.2e} and "
                  f"{b_t[name]['poleward_of_40']:.2e}")
        fig_item_ii(sph, twin, terms_s, terms_t,
                    os.path.join(args.fig_dir, "sphere_item_ii_u_budget.pdf"))

    sph_f, twin_f = load("iii_sphere_ny649_dy50"), load("iii_twin_ny1017_dy50")
    if sph and twin and sph_f and twin_f:
        r_s, r_t = resolution(sph, sph_f), resolution(twin, twin_f)
        ledger["iii"] = dict(sphere=r_s, twin=r_t)
        for name in ("u", "v", "theta"):
            a, b = r_s[name]["within_40"], r_t[name]["within_40"]
            print(f"(iii) {name}: max|fine - coarse| within 40 deg / max|field| = "
                  f"{a:.2e} on the sphere, {b:.2e} on the twin -> "
                  f"{'PASS' if a <= b else 'FAIL'}; poleward of 40 deg "
                  f"{r_s[name]['poleward_of_40']:.2e} and {r_t[name]['poleward_of_40']:.2e}")
        fig_item_iii({"sphere": (sph, sph_f), "twin": (twin, twin_f)},
                     os.path.join(args.fig_dir, "sphere_item_iii_resolution.pdf"))

    runs_iv = {}
    for r in ("12.5", "25", "50"):
        for label in ("sphere", "twin"):
            run = load(f"iv_{label}_dy{r}")
            if run:
                runs_iv[f"{label}_dy{r}"] = run
                print(f"(iv) {label} Delta_y = {r} K: {run.ndays} days, finite {run.finite}, "
                      f"max u {np.max(run.u):.2f} m/s at {run.lat[np.argmax(run.u)]:.2f} deg, "
                      f"max|v| {np.max(np.abs(run.v)):.3f} m/s")
    if runs_iv:
        fig_item_iv(runs_iv, os.path.join(args.fig_dir, "sphere_item_iv_profiles.pdf"))

    with open(os.path.join(args.fig_dir, "sphere_validation_ledger.json"), "w") as f:
        json.dump(ledger, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
