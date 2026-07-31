"""Analyze the moist V2 equilibria: the attribution chain and the
Delta_theta^rad ladder.

The attribution chain changes exactly one thing per step, so a difference
between consecutive members is attributable to that thing:

  M1 V1        moisture on, H(theta_E - theta) gate, Delta_theta = 50 K
  M2 V1-kin    + kinematic H(dv/dy) gate
  M3 V2_0      + retarget to theta_rad at Delta_theta^rad = 75 K (Lambda = 0)
  M4 V2        + latent heating Lambda P

The ladder then varies Delta_theta^rad over 55-95 K at full coupling.

Everything is evaluated on the last-N-day mean of a slow-drift-gated run.
Column constants come from ss09.moist_constants, so C*Lambda = L_v exactly
and the MSE budget residual below is meaningful (the V1 scripts hard-code a
rounded C = 5.2e6, which is 0.7% off the derived value).

Diagnostics, per run:
  Hhat(y) = Shat - L_v (2a-1) W          gross moist stability
  F_mean  = Shat v - L_v(2a-1) v W       mean MSE flux and its two parts
  F_eddy  = -L_v D dW/dy                 eddy MSE flux
  budget residual: d/dy(F_mean + F_eddy) - source, which vanishes at
  equilibrium and is the check that the run converged.

The fluxes are built ON THE FACES with the model's own operators: the face v
straight from the file (not reconstructed to centers), the MUSCL-MC face
value of W that the transport actually advects, and the compact half-cell
divergence. Reconstructing at cell centers with np.gradient instead leaves a
residual of order the source itself, because the MUSCL face value departs
from the center value by up to 0.64 MW/m at the ITCZ front (diagnosed
2026-07-30; the model's own identity closes to 1.4e-13 relative).

The source carries -L_v P for a run WITHOUT latent heating: there the theta
equation has no Lambda P to cancel the moisture sink, so precipitation is a
genuine export of column MSE. With latent heating it cancels exactly.

Usage:
    python scripts/moist_v2_analysis.py [run_dir] [out_prefix]
"""

import os
import sys

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ss09.moist_constants import L_V, column_heat_capacity, gross_dry_stability
from ss09.read_output import load_centered
from ss09.sw_model import mc_face_values, v_divergence_at_centers

CHAIN = [
    ("V1", "m1_v1", "#666666"),
    ("V1-kin", "m2_v1kin", "#4477AA"),
    ("V2$_0$", "m3_v20", "#EE6677"),
    ("V2", "m4_v2_dyr75", "#228833"),
]
LADDER = [
    ("55 K", "l55", "#332288"),
    ("65 K", "l65", "#88CCEE"),
    ("75 K", "m4_v2_dyr75", "#228833"),
    ("85 K", "l85", "#DDCC77"),
    ("95 K", "l95", "#CC6677"),
]
# The gross-moist-stability regime test: W_c straddles the Hhat = 0 crossover
# at W = Shat/(L_v(2a-1)) = 44.25 kg/m^2, since the quiescent column sits at
# W_c + tau_c E_0. W_c = 35 and 40 are positive-GMS, W_c = 50 negative.
WC_REGIME = [
    (r"$W_c$=35", "wc35", "#4477AA"),
    (r"$W_c$=40", "wc40", "#DDAA33"),
    (r"$W_c$=50", "m4_v2_dyr75", "#CC3311"),
]


def load_equilibrium(path, last_n=10):
    """Last-N-day mean fields plus the derived energetics for one run."""
    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        y = ds["y"].values
        y_face = ds["y_face"].values
        n = ds.sizes["time"]
        sl = slice(max(0, n - last_n), n)
        w = ds["W"].isel(time=sl).mean("time").values
        p = ds["P"].isel(time=sl).mean("time").values
        u = ds["u"].isel(time=sl).mean("time").values
        temp = ds["T"].isel(time=sl).mean("time").values
        v_face = ds["v"].isel(time=sl).mean("time").values  # native face grid
        theta_e = ds["theta_e"].values  # static: theta_E in V1, theta_rad in V2
        att = ds.attrs
        days = int(ds["time"].values[-1])
        conv_day = att.get("convergence_day", None)
    finally:
        ds.close()
    _, _, v_all, _ = load_centered(path)
    v = v_all[max(0, n - last_n):n].mean(axis=0)  # centers, for reporting max|v|

    a = float(att["cwv_frac"])
    d_w = float(att["d_w"])
    gravity = float(att["gravity"])
    latent = str(att.get("enable_latent_heating", "False")) == "True"
    c_col = column_heat_capacity(gravity)
    shat = gross_dry_stability(gravity, float(att["delta"]),
                               float(att["delta_z"]), float(att["height"]))
    theta = temp * 1.6
    dy = y[1] - y[0]

    hhat = shat - L_V * (2.0 * a - 1.0) * w
    # Fluxes on the faces, with the model's own operators (see the module
    # docstring): the face transport velocity, the MUSCL-MC face value of W
    # the transport actually advects, and the compact face gradient.
    c_f = -(2.0 * a - 1.0) * v_face
    w_face = mc_face_values(w, dy, c_f)
    dse_flux = shat * v_face
    lvq_flux = L_V * c_f * w_face
    mean_flux = dse_flux + lvq_flux
    eddy_flux = -L_V * d_w * (w[1:] - w[:-1]) / dy
    total_flux = mean_flux + eddy_flux
    source = (c_col / float(att["tau"])) * (theta_e - theta) + L_V * float(att["evap"])
    # Precipitation is internal only to the extent that the heating it adds to
    # theta matches the latent energy it removes from W. The uncancelled part
    # is (L_v - C*Lambda) P: zero for a full V2 run by construction, the whole
    # -L_v P for V1 and for the Lambda = 0 bridge, where the sink exports
    # column MSE with no compensating heating.
    lam = float(att["lambda_conv"]) if latent else 0.0
    source = source - (L_V - c_col * lam) * p
    residual = v_divergence_at_centers(total_flux, dy) - source

    return dict(
        y=y, y_face=y_face, dy=dy, w=w, p=p, u=u, v=v, v_face=v_face,
        temp=temp, theta=theta, theta_e=theta_e,
        a=a, d_w=d_w, w_crit=float(att["w_crit"]), tau_c=float(att["tau_c"]),
        evap=float(att["evap"]), tau=float(att["tau"]), c_col=c_col, shat=shat,
        hhat=hhat, dse_flux=dse_flux, lvq_flux=lvq_flux, mean_flux=mean_flux,
        eddy_flux=eddy_flux, total_flux=total_flux, source=source,
        residual=residual, days=days, conv_day=conv_day, latent=latent,
        lam=att.get("lambda_conv", None), gate=att.get("vert_advec_gate", "theta"),
        delta_y_rad=att.get("delta_y_rad", None),
        w_plateau=float(att["w_crit"]) + float(att["tau_c"]) * float(att["evap"]),
        w_hhat_zero=shat / (L_V * (2.0 * a - 1.0)),
    )


def efe(r):
    """Energy-flux-equator latitude: the zero of the total column MSE flux
    nearest the equator, linearly interpolated between gridpoints."""
    f, y = r["total_flux"], r["y_face"]
    i0 = int(np.argmin(np.abs(y)))
    best = np.nan
    for i in range(1, len(y)):
        if f[i - 1] == 0.0:
            cand = y[i - 1]
        elif f[i - 1] * f[i] < 0:
            cand = y[i - 1] - f[i - 1] * (y[i] - y[i - 1]) / (f[i] - f[i - 1])
        else:
            continue
        if np.isnan(best) or abs(cand - y[i0]) < abs(best - y[i0]):
            best = cand
    return best


def itcz(r, halfwidth=6e6):
    """Precipitation-weighted centroid over |y| < halfwidth, a sub-grid ITCZ
    latitude less jumpy than the gridpoint argmax."""
    m = np.abs(r["y"]) < halfwidth
    wgt = np.maximum(r["p"][m], 0.0)
    if wgt.sum() <= 0:
        return np.nan
    return float(np.sum(wgt * r["y"][m]) / wgt.sum())


def scorecard(labels, runs, title):
    print(f"\n=== {title} ===")
    print(f"{'run':8} {'day':>5} {'gate':>9} {'jet':>7} {'max|v|':>7} "
          f"{'T_eq':>7} {'dT_trop':>8} {'off_col':>8} {'W_eq':>7} "
          f"{'Pmax':>8} {'ITCZ':>7} {'EFE':>7} {'Hmin':>7} {'resid':>8}")
    for lab, r in zip(labels, runs):
        y = r["y"]
        i0 = int(np.argmin(np.abs(y)))
        i1 = int(np.argmin(np.abs(np.abs(y) - 9439e3)))  # |y| = y_1
        jet = float(np.max(r["u"]))
        # equilibrium offset of the column above its relaxation target, in the
        # quiescent collar (predicted tau*Lambda*E_0 for a latent run)
        off = float(r["theta"][0] - r["theta_e"][0])
        rel = np.max(np.abs(r["residual"])) / np.max(np.abs(r["source"]))
        print(f"{lab:8} {r['days']:>5} {str(r['gate']):>9} {jet:>7.2f} "
              f"{np.max(np.abs(r['v'])):>7.3f} {r['temp'][i0]:>7.2f} "
              f"{r['temp'][i0] - r['temp'][i1]:>8.2f} {off:>8.2f} "
              f"{r['w'][i0]:>7.2f} {np.max(r['p']) * 86400:>8.3f} "
              f"{itcz(r) / 1e6:>7.3f} {efe(r) / 1e6:>7.3f} "
              f"{np.min(r['hhat']) / 1e6:>7.2f} {rel:>8.1e}")
    print("jet = max u (m/s); dT_trop = T(0) - T(|y|=y_1) (K); off_col = "
          "theta - theta_target at the wall (K); Pmax mm/day; ITCZ = precip "
          "centroid (Mm); EFE = zero of the total MSE flux (Mm); Hmin = min "
          "Hhat (MJ/m^2); resid = max|d(flux)/dy - source| / max|source|.")


def profile_figure(labels, runs, colors, out_png, title, direct_labels=True):
    """Six-panel profile overlay. direct_labels places color-matched text
    beside each curve (the house style); pass False when curves overlap too
    much for that to read, as in the attribution chain, where V1, V1-kin and
    V2_0 have all but identical W."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    panels = [
        ("u (m s$^{-1}$)", lambda r: r["u"], "y"),
        ("v (m s$^{-1}$)", lambda r: r["v_face"], "y_face"),
        ("T (K)", lambda r: r["temp"], "y"),
        ("W (kg m$^{-2}$)", lambda r: r["w"], "y"),
        ("P (mm day$^{-1}$)", lambda r: r["p"] * 86400, "y"),
        (r"$\hat H$ (MJ m$^{-2}$)", lambda r: r["hhat"] / 1e6, "y"),
    ]
    for ax, (label, getter, grid) in zip(axes.ravel(), panels):
        for lab, r, c in zip(labels, runs, colors):
            ax.plot(r[grid] / 1e6, getter(r), color=c, lw=1.3, label=lab)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("y (Mm)")
    axes[1, 2].axhline(0, color="k", lw=0.6)
    if direct_labels:
        # Labels beside each curve on the W panel, placed at the run's own
        # equatorial peak so each sits nearer its curve than any other.
        ax = axes[1, 0]
        for k, (lab, r, c) in enumerate(zip(labels, runs, colors)):
            j = len(r["y"]) // 2 + 12 + 22 * k
            ax.text(r["y"][j] / 1e6, r["w"][j], lab, color=c, fontsize=10,
                    fontweight="bold", va="bottom", ha="left")
    else:
        axes[0, 2].legend(fontsize=9, loc="lower center")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def budget_figure(r, out_png, title):
    """The MSE flux resolved into its large opposing parts, and the budget
    closure. Never present the net alone: it is the residual of a poleward
    DSE flux and an equatorward latent flux several times its size."""
    yf = r["y_face"] / 1e6
    yc = r["y"] / 1e6
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(yf, r["dse_flux"] / 1e6, color="#CC6677", label=r"DSE $\hat S v$")
    axes[0].plot(yf, r["lvq_flux"] / 1e6, color="#4477AA",
                 label=r"latent $-L_v(2a{-}1)vW$")
    axes[0].plot(yf, r["mean_flux"] / 1e6, color="k", lw=2,
                 label=r"mean MSE $v\hat H$ (net)")
    axes[0].plot(yf, r["eddy_flux"] / 1e6, color="#999933", ls="--",
                 label=r"eddy $-L_vD\,\partial_yW$")
    axes[0].plot(yf, r["total_flux"] / 1e6, color="#EE6677", ls=":", lw=2,
                 label="total")
    axes[0].set_ylabel("northward flux (MW m$^{-1}$)")
    axes[0].set_title("Column MSE flux and its parts")
    axes[1].plot(yc, v_divergence_at_centers(r["total_flux"], r["dy"]) * 1e3,
                 color="k", label=r"$\partial_y$(total flux)")
    axes[1].plot(yc, r["source"] * 1e3, color="#EE6677", ls="--",
                 label=r"radiation + $L_vE_0$")
    axes[1].plot(yc, r["residual"] * 1e3, color="#4477AA",
                 label="residual")
    axes[1].set_ylabel("column energy source (mW m$^{-3}$)")
    axes[1].set_title("MSE budget closure at equilibrium")
    for ax in axes:
        ax.axhline(0, ls=":", c="gray", lw=1)
        ax.set_xlabel("y (Mm)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def ladder_figure(runs, out_png):
    """Delta_theta^rad dependence of the quantities a calibration would target
    (all gradient-sensitive, unlike the absolute temperature level, which the
    cosmetic theta_00 sets)."""
    d = np.array([float(r["delta_y_rad"]) for r in runs])
    metrics = [
        ("max u (m s$^{-1}$)", [float(np.max(r["u"])) for r in runs]),
        (r"max $|v|$ (m s$^{-1}$)", [float(np.max(np.abs(r["v"]))) for r in runs]),
        (r"$T(0)-T(y_1)$ (K)",
         [float(r["temp"][np.argmin(np.abs(r["y"]))]
                - r["temp"][np.argmin(np.abs(np.abs(r["y"]) - 9439e3))])
          for r in runs]),
        ("max P (mm day$^{-1}$)", [float(np.max(r["p"])) * 86400 for r in runs]),
        (r"min $\hat H$ (MJ m$^{-2}$)",
         [float(np.min(r["hhat"])) / 1e6 for r in runs]),
        ("W at equator (kg m$^{-2}$)",
         [float(r["w"][np.argmin(np.abs(r["y"]))]) for r in runs]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (label, vals) in zip(axes.ravel(), metrics):
        ax.plot(d, vals, "o-", color="#4477AA")
        ax.set_ylabel(label)
        ax.set_xlabel(r"$\Delta_\theta^{\rm rad}$ (K)")
        ax.grid(alpha=0.3)
    fig.suptitle(r"Moist V2: sensitivity to the radiative contrast "
                 r"$\Delta_\theta^{\rm rad}$", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "model_output/moist_v2"
    prefix = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run_dir, "moist_v2")

    chain = [load_equilibrium(os.path.join(run_dir, d, "out.nc"))
             for _, d, _ in CHAIN]
    scorecard([c[0] for c in CHAIN], chain, "Attribution chain")
    profile_figure([c[0] for c in CHAIN], chain, [c[2] for c in CHAIN],
                   prefix + "_attribution.png",
                   "Moist V2 attribution: one change per step, at equilibrium",
                   direct_labels=False)
    budget_figure(chain[-1], prefix + "_budget.png",
                  r"Moist V2 ($\Delta_\theta^{\rm rad}=75$ K) column MSE budget")

    wc = [load_equilibrium(os.path.join(run_dir, d, "out.nc"))
          for _, d, _ in WC_REGIME]
    scorecard([w[0] for w in WC_REGIME], wc, "Gross-moist-stability regime")
    profile_figure([w[0] for w in WC_REGIME], wc, [w[2] for w in WC_REGIME],
                   prefix + "_gms_regime.png",
                   "Moist V2 across the gross-moist-stability crossover "
                   r"($\hat H=0$ at $W$=44.25 kg m$^{-2}$)")
    budget_figure(wc[1], prefix + "_budget_wc40.png",
                  r"Moist V2, $W_c$=40 (positive $\hat H$): column MSE budget")

    ladder = [load_equilibrium(os.path.join(run_dir, d, "out.nc"))
              for _, d, _ in LADDER]
    scorecard([l[0] for l in LADDER], ladder, "Delta_theta^rad ladder")
    profile_figure([l[0] for l in LADDER], ladder, [l[2] for l in LADDER],
                   prefix + "_ladder_profiles.png",
                   r"Moist V2: $\Delta_\theta^{\rm rad}$ ladder at equilibrium")
    ladder_figure(ladder, prefix + "_ladder.png")


if __name__ == "__main__":
    main()
