"""Figures for the moist V2 sweep.

House style: direct colour-matched labels beside each curve rather than a
legend box, each label nearer its own curve than any other; a legend only where
the curves overlap too much for that to read. Palette is Okabe-Ito
(colour-vision-deficiency safe), which is the documented fallback on this
machine because the dataviz skill's palette validator cannot run here.

Usage:
    python scripts/v2_sweep_figures.py [--tree DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

sys.path.insert(0, "/home/shill/py/axisym-hc-swm")

from scripts.v2_sweep_diagnostics import (  # noqa: E402
    load_run, load_seasonal, precip_centroid, scalars, seasonal_scalars,
    y_to_lat, Y_ONE,
)
from scripts.v2_sweep_manifest import SWEEP_DIR, manifest, r_param, w_star  # noqa: E402
from ss09.moist_constants import L_V  # noqa: E402
from ss09.sw_model import v_divergence_at_centers  # noqa: E402

OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", verm="#D55E00", purple="#CC79A7")
SEQ = [OI["blue"], OI["sky"], OI["green"], OI["yellow"], OI["orange"],
       OI["verm"], OI["purple"], OI["black"]]
MM = 1e-6  # metres to Mm


# ---------------------------------------------------------------------------
def load_all(tree=SWEEP_DIR, last_n=1000, complete_frac=0.98):
    """{name: run} for every non-seasonal entry, {name: composite} for the
    seasonal ones, plus the manifest entry beside each.

    A run has to have reached complete_frac of its requested days to count.
    Without that gate a diverged run's first-day transient loads silently and
    its numbers are indistinguishable from an equilibrium's in the tables.
    """
    eq, seas, meta, missing = {}, {}, {}, []
    for entry in manifest():
        name = entry["name"]
        meta[name] = entry
        path = os.path.join(tree, name, "out.nc")
        want = int(complete_frac * entry["args"]["ndays"])
        if entry["args"].get("y0_seas_amp", 0.0) > 0:
            s = load_seasonal(path)
            if s is None or s["days"] < want:
                missing.append(name)
            else:
                seas[name] = s
        else:
            r = load_run(path, last_n=last_n, min_days=want)
            if r is None:
                missing.append(name)
            else:
                eq[name] = r
    return eq, seas, meta, missing


def label_curve(ax, x, y, text, color, dx=0.0, dy=0.0, **kw):
    """Colour-matched text beside a curve at (x, y), nudged by the minimum
    needed to clear it."""
    ax.text(x + dx, y + dy, text, color=color, fontsize=9, fontweight="bold",
            **kw)


def _panel_grid(nrow, ncol, figsize, sharex=True):
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=sharex)
    return fig, np.atleast_1d(axes).ravel()


def _lat_axis(ax):
    """Secondary axis in equivalent latitude from f = beta y = 2 Omega sin(phi).
    The mapping saturates at |y| = 7.29 Mm, so ticks stop there rather than
    being extrapolated."""
    sec = ax.secondary_xaxis("top", functions=(lambda v: v, lambda v: v))
    ticks = [-6, -3, 0, 3, 6]
    sec.set_xticks(ticks)
    sec.set_xticklabels([f"{y_to_lat(t / MM):.0f}°" for t in ticks],
                        fontsize=7)
    return sec


# ---------------------------------------------------------------------------
# 1. The regime map in the (a, W_c) plane
# ---------------------------------------------------------------------------
def fig_regime_map(rows, out):
    plane = [r for r in rows if r["block"] in ("A", "B")]
    fields = [
        ("wet_frac", "fraction of the domain raining", "viridis"),
        ("p_max", "peak precipitation (mm day$^{-1}$)", "magma_r"),
        ("n_bands", "number of rain bands", "cividis"),
        ("v_max", r"peak $|v|$ (m s$^{-1}$)", "cividis"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
    for ax, (key, title, cmap) in zip(axes, fields):
        xs = [r["a"] for r in plane]
        ys = [r["w_crit"] for r in plane]
        cs = [r[key] for r in plane]
        # Thick white ring marks a solution that never settles, so a reader can
        # see at a glance which cells of the map are time means of a
        # fluctuating state rather than equilibria.
        vmin, vmax = float(np.min(cs)), float(np.max(cs))
        for steady, mk, size in ((True, "o", 230), (False, "s", 200)):
            keep = [k for k, r in enumerate(plane) if r["steady"] is steady]
            if not keep:
                continue
            sc = ax.scatter([xs[k] for k in keep], [ys[k] for k in keep],
                            c=[cs[k] for k in keep], s=size, cmap=cmap,
                            marker=mk, edgecolor="k", linewidth=0.6, zorder=3,
                            vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, label=title)
        # the r = 1 line: W_c = W*(a) - tau_c E_0
        aa = np.linspace(0.735, 0.885, 200)
        ax.plot(aa, [w_star(a) - 14400 * 4.6e-5 for a in aa], color=OI["verm"],
                lw=2.2, zorder=2)
        ax.plot(aa, [0.85 * w_star(a) - 14400 * 4.6e-5 for a in aa],
                color=OI["verm"], lw=1.0, ls=":", zorder=2)
        ax.set_xlim(0.735, 0.885)
        ax.set_ylim(26, 64)
        ax.set_xlabel("lower-branch CWV fraction $a$")
        ax.grid(alpha=0.25, zorder=0)
    axes[0].set_ylabel(r"precipitation threshold $W_c$ (kg m$^{-2}$)")
    axes[0].text(0.03, 0.05, "circles settle; squares never do", fontsize=7.5,
                 color="0.25", transform=axes[0].transAxes)
    label_curve(axes[0], 0.80, w_star(0.80) - 0.66 + 1.4, r"$r=1$: $\hat H=0$",
                OI["verm"], rotation=-38)
    label_curve(axes[0], 0.795, 0.85 * w_star(0.795) - 0.66 - 3.4, r"$r=0.85$",
                OI["verm"], rotation=-32)
    fig.suptitle(r"Moist V2 in the plane of $a$ and $W_c$, the two parameters "
                 r"that set the sign of $\hat H$ "
                 r"($\Delta_\theta^{\rm rad}=75$ K, everything else at default)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# 2. Collapse onto r = W_q / W*
# ---------------------------------------------------------------------------
COLLAPSE_GROUPS = [
    ("$a$ varied", lambda r: r["block"] == "A", OI["blue"], "o"),
    ("$W_c$ varied", lambda r: r["block"] == "B", OI["sky"], "s"),
    ("matched $r$", lambda r: r["block"] == "C", OI["black"], "*"),
    (r"$\tau_c$ varied", lambda r: r["block"] == "E", OI["green"], "^"),
    (r"$\Delta_z$ varied", lambda r: r["block"] == "I", OI["orange"], "v"),
    ("$E_0$ varied", lambda r: r["block"] == "G", OI["purple"], "D"),
]


def fig_collapse(rows, out):
    metrics = [
        ("wet_frac", "fraction of the domain raining"),
        ("p_max", "peak precipitation (mm day$^{-1}$)"),
        ("hhat_min", r"min $\hat H$ (MJ m$^{-2}$)"),
        ("v_max", r"peak $|v|$ (m s$^{-1}$)"),
        ("jet", "subtropical jet (m s$^{-1}$)"),
        ("t_eq", "equatorial temperature (K)"),
        ("eddy_share", r"eddy share, $|F_e|/(|F_m|+|F_e|)$ at 24$^\circ$"),
        ("f_mean_ref", r"mean MSE flux at 24$^\circ$ (MW m$^{-1}$)"),
        ("hhat_neg_frac", r"fraction of the domain with $\hat H<0$"),
    ]
    fig, axes = _panel_grid(3, 3, (14.5, 10.5), sharex=True)
    for ax, (key, ylab) in zip(axes, metrics):
        for lab, sel, col, mk in COLLAPSE_GROUPS:
            sub = [r for r in rows if sel(r)]
            if not sub:
                continue
            # Filled markers are steady solutions; open markers are solutions
            # that never settle, whose plotted value is a 1000-day time mean of
            # a fluctuating state rather than an equilibrium. Mixing the two
            # without saying so would present a time mean as a fixed point.
            for steady, kw in ((True, dict(edgecolor="none")),
                               (False, dict(facecolor="none", edgecolor=col,
                                            linewidth=1.1))):
                ss = [r for r in sub if bool(r["steady"]) is steady]
                if not ss:
                    continue
                ax.scatter([r["r"] for r in ss], [r[key] for r in ss],
                           color=col, marker=mk, s=48, alpha=0.9,
                           label=lab if steady else None, **kw)
        ax.axvline(1.0, color=OI["verm"], lw=1.6)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.25)
    for ax in axes[-3:]:
        ax.set_xlabel(r"$r=W_q/W^*$  (quiescent CWV / $\hat H=0$ crossover)")
    axes[0].legend(fontsize=8, loc="lower left", framealpha=0.9)
    axes[1].text(0.03, 0.86, "filled: steady solution\nopen: never settles, so the\n"
                 "plotted value is a 1000-day mean",
                 transform=axes[1].transAxes, fontsize=7.5, color="0.25",
                 va="top")
    axes[2].text(1.02, 0.55, r"$\hat H<0$" + "\nin an\nundisturbed\ncolumn",
                 transform=axes[2].transAxes, color=OI["verm"], fontsize=8)
    fig.suptitle(r"Four of the six swept parameters collapse onto "
                 r"$r=L_v(2a{-}1)(W_c+\tau_cE_0)/(Cd\Delta_z/H)$; "
                 r"$\tau_c$ and $E_0$ do not," + "\n"
                 r"because each also sets something else: how fast convection "
                 r"removes water, and how much water there is",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# 3. Profiles across the transition
# ---------------------------------------------------------------------------
def profile_panels(eq, names, labels, out, title, colors=None,
                   label_x=None, extra=None, label_panel=3, label_key="w"):
    colors = colors or SEQ
    panels = [
        (r"$u$ (m s$^{-1}$)", lambda r: (r["y"], r["u"])),
        (r"$v$ (m s$^{-1}$)", lambda r: (r["y_face"], r["v_face"])),
        ("$T$ (K)", lambda r: (r["y"], r["temp"])),
        (r"$W$ (kg m$^{-2}$)", lambda r: (r["y"], r["w"])),
        ("$P$ (mm day$^{-1}$)", lambda r: (r["y"], r["p"] * 86400)),
        (r"$\hat H$ (MJ m$^{-2}$)", lambda r: (r["y"], r["hhat"] / 1e6)),
    ]
    fig, axes = _panel_grid(2, 3, (15, 8))
    for k, (ax, (ylab, getter)) in enumerate(zip(axes, panels)):
        for j, (nm, lab) in enumerate(zip(names, labels)):
            if nm not in eq:
                continue
            x, v = getter(eq[nm])
            ax.plot(np.asarray(x) * MM, v, color=colors[j % len(colors)],
                    lw=1.4)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
        if ylab.startswith(r"$\hat H$"):
            ax.axhline(0, color="k", lw=0.7)
    for ax in axes[3:]:
        ax.set_xlabel("$y$ (Mm)")
    _lat_axis(axes[0])
    _lat_axis(axes[1])
    _lat_axis(axes[2])
    # Direct labels on the W panel. Placed in the quiescent collar at the far
    # right, where W is flat at W_c + tau_c E_0 and the curves are as far apart
    # as they ever get, so each label is nearer its own curve than any other.
    ax = axes[label_panel]
    scale = 86400.0 if label_key == "p" else 1.0
    for j, (nm, lab) in enumerate(zip(names, labels)):
        if nm not in eq:
            continue
        r = eq[nm]
        arr = np.asarray(r[label_key]) * scale
        grid = r["y_face"] if label_key == "v_face" else r["y"]
        if label_x is None:
            # Place at the curve's own extremum, where a family of curves is
            # furthest apart. Placing every label at one fixed abscissa puts
            # them all at the same point wherever the curves coincide, which is
            # what happened on the off-equatorial series, whose quiescent W is
            # identical for every member.
            i = int(np.argmax(np.abs(arr)))
        else:
            i = min(int(label_x * len(arr)), len(arr) - 1)
        ax.text(grid[i] * MM, arr[i], " " + lab,
                color=colors[j % len(colors)], fontsize=8, fontweight="bold",
                va="center", ha="left")
    ax.margins(x=0.16, y=0.10)
    if extra:
        extra(axes)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# 4. MSE budget for one run: never the net alone
# ---------------------------------------------------------------------------
def fig_budget(eq, names, labels, out, title):
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(6.4 * n, 8.6), squeeze=False)
    for j, (nm, lab) in enumerate(zip(names, labels)):
        if nm not in eq:
            continue
        r = eq[nm]
        yf, yc = r["y_face"] * MM, r["y"] * MM
        ax = axes[0][j]
        ax.plot(yf, r["dse_flux"] / 1e6, color=OI["verm"], lw=1.5)
        ax.plot(yf, r["lvq_flux"] / 1e6, color=OI["blue"], lw=1.5)
        ax.plot(yf, r["mean_flux"] / 1e6, color=OI["black"], lw=2.2)
        ax.plot(yf, r["eddy_flux"] / 1e6, color=OI["green"], lw=1.5, ls="--")
        ax.plot(yf, r["total_flux"] / 1e6, color=OI["purple"], lw=1.8, ls=":")
        i = int(0.62 * len(yf))
        label_curve(ax, yf[i], r["dse_flux"][i] / 1e6, r"dry static $\hat Sv$",
                    OI["verm"], dy=0.6)
        label_curve(ax, yf[i], r["lvq_flux"][i] / 1e6,
                    r"latent $-L_v(2a{-}1)vW$", OI["blue"], dy=-1.8)
        label_curve(ax, yf[int(0.34 * len(yf))],
                    r["mean_flux"][int(0.34 * len(yf))] / 1e6,
                    r"mean $v\hat H$", OI["black"], dy=0.7)
        label_curve(ax, yf[int(0.78 * len(yf))],
                    r["eddy_flux"][int(0.78 * len(yf))] / 1e6,
                    r"eddy $-L_vD\partial_yW$", OI["green"], dy=0.7)
        label_curve(ax, yf[int(0.5 * len(yf)) + 40],
                    r["total_flux"][int(0.5 * len(yf)) + 40] / 1e6, "total",
                    OI["purple"], dy=-1.6)
        ax.axhline(0, color="gray", ls=":", lw=0.9)
        ax.set_title(lab, fontsize=11)
        ax.set_ylabel("northward column MSE flux (MW m$^{-1}$)")
        ax.grid(alpha=0.25)
        ax2 = axes[1][j]
        ax2.plot(yc, v_divergence_at_centers(r["total_flux"], r["dy"]) * 1e3,
                 color=OI["black"], lw=1.6)
        ax2.plot(yc, r["source"] * 1e3, color=OI["verm"], lw=1.4, ls="--")
        ax2.plot(yc, r["residual"] * 1e3, color=OI["blue"], lw=1.2)
        label_curve(ax2, yc[int(0.66 * len(yc))],
                    (v_divergence_at_centers(r["total_flux"], r["dy"])
                     * 1e3)[int(0.66 * len(yc))], r"$\partial_y$(flux)",
                    OI["black"], dy=0.5)
        label_curve(ax2, yc[int(0.2 * len(yc))],
                    (r["source"] * 1e3)[int(0.2 * len(yc))],
                    r"radiation + $L_vE_0$", OI["verm"], dy=-1.1)
        label_curve(ax2, yc[int(0.42 * len(yc))], 0.4, "residual", OI["blue"])
        ax2.axhline(0, color="gray", ls=":", lw=0.9)
        ax2.set_xlabel("$y$ (Mm)")
        ax2.set_ylabel("column energy source (mW m$^{-3}$)")
        ax2.grid(alpha=0.25)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# 4b. Anatomy of one aggregated state: how a negative-Hhat solution closes
#     its own energy budget by drying part of the domain back above Hhat = 0
# ---------------------------------------------------------------------------
def fig_anatomy(eq, name, out, title, xlim=(-9, 9)):
    if name not in eq:
        print(f"skipped {out}: {name} unavailable")
        return
    r = eq[name]
    y, yf = r["y"] * MM, r["y_face"] * MM
    wstar = r["w_star"]
    fig, axes = plt.subplots(4, 1, figsize=(8.6, 11.2), sharex=True)

    ax = axes[0]
    ax.plot(y, r["w"], color=OI["blue"], lw=1.8)
    ax.axhline(wstar, color=OI["verm"], lw=1.4, ls="--")
    ax.axhline(r["w_crit"], color=OI["green"], lw=1.4, ls=":")
    label_curve(ax, xlim[0] + 0.3, wstar, r"$W^*$: $\hat H=0$", OI["verm"],
                dy=0.9)
    label_curve(ax, xlim[0] + 0.3, r["w_crit"], r"$W_c$: rain starts",
                OI["green"], dy=-1.8)
    label_curve(ax, 0.4, float(np.max(r["w"])) * 0.98, "$W$", OI["blue"])
    ax.set_ylabel(r"$W$ (kg m$^{-2}$)")

    ax = axes[1]
    ax.plot(y, r["p"] * 86400, color=OI["purple"], lw=1.8)
    ax.axhline(r["evap"] * 86400, color=OI["black"], lw=1.0, ls="--")
    label_curve(ax, xlim[0] + 0.3, r["evap"] * 86400,
                r"$E_0$, the uniform source", OI["black"], dy=1.4)
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    ax.set_ylim(0, max(2.0, 1.6 * float(np.max(r["p"]) * 86400)))
    ax.set_ylabel("$P$ (mm day$^{-1}$)")

    ax = axes[2]
    ax.plot(y, r["hhat"] / 1e6, color=OI["orange"], lw=1.8)
    ax.axhline(0, color="k", lw=0.9)
    dry = r["p"] <= 0
    if dry.any():
        ax.fill_between(y, 0, 1, where=dry, transform=ax.get_xaxis_transform(),
                        color=OI["sky"], alpha=0.25, lw=0, zorder=0)
        ax.text(0.015, 0.06, "blue shading: no rain at all",
                transform=ax.transAxes, color=OI["sky"], fontsize=9,
                fontweight="bold")
    ax.set_ylabel(r"$\hat H$ (MJ m$^{-2}$)")

    ax = axes[3]
    ax.plot(yf, r["mean_flux"] / 1e6, color=OI["black"], lw=1.8)
    ax.plot(yf, r["eddy_flux"] / 1e6, color=OI["green"], lw=1.5, ls="--")
    ax.plot(yf, r["total_flux"] / 1e6, color=OI["purple"], lw=1.5, ls=":")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    # Label each curve where the three are furthest apart, which is at their
    # own extremum, so each label sits nearer its own curve than the others.
    m = np.abs(yf) < abs(xlim[0]) * 1e6
    span = float(np.max(np.abs(r["mean_flux"][m]))) / 1e6 or 1.0
    for arr, lab, col, sgn in ((r["mean_flux"], r"mean $v\hat H$", OI["black"], 1),
                               (r["eddy_flux"], r"eddy $-L_vD\partial_yW$",
                                OI["green"], -1)):
        j = int(np.flatnonzero(m)[np.argmax(np.abs(arr[m]))])
        label_curve(ax, yf[j], arr[j] / 1e6, "  " + lab, col,
                    dy=sgn * 0.10 * span, va="center")
    # The total is small and flat next to the two that nearly cancel, so its
    # label goes in the empty upper-left rather than on the curve.
    ax.annotate("total (the two above nearly cancel)",
                xy=(xlim[0] * 0.72, float(r["total_flux"][
                    int(np.argmin(np.abs(yf - xlim[0] * 0.72e6)))]) / 1e6),
                xytext=(0.03, 0.88), textcoords="axes fraction",
                color=OI["purple"], fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=OI["purple"], lw=0.9))
    ax.set_ylabel("northward MSE flux (MW m$^{-1}$)")
    ax.set_xlabel("$y$ (Mm)")
    ax.margins(y=0.20)

    for ax in axes:
        ax.set_xlim(*xlim)
        ax.grid(alpha=0.25)
    _lat_axis(axes[0])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# 5. Generic ladder figure: metrics against one swept parameter
# ---------------------------------------------------------------------------
LADDER_METRICS = [
    ("jet", "subtropical jet (m s$^{-1}$)"),
    ("v_max", r"peak $|v|$ (m s$^{-1}$)"),
    ("dt_trop", r"$T(0)-T(y_1)$ (K)"),
    ("p_max", "peak $P$ (mm day$^{-1}$)"),
    ("wet_frac", "fraction raining"),
    ("hhat_min", r"min $\hat H$ (MJ m$^{-2}$)"),
    ("w_eq", r"$W$ at equator (kg m$^{-2}$)"),
    ("eddy_share", "eddy share of MSE flux"),
]


def fig_ladder(rows, out, xkey, xlabel, series, title, logx=False,
               metrics=None, xdiv=1.0):
    """series: list of (label, selector, colour). Each becomes one curve.

    xdiv rescales the abscissa so matplotlib does not add a shared exponent at
    the right end of the axis, which collides with the axis label and made one
    figure read "(m$^2$ s$^{-1}$1e6". Fold the factor into xlabel instead.
    """
    metrics = metrics or LADDER_METRICS
    nrow = int(np.ceil(len(metrics) / 4))
    fig, axes = _panel_grid(nrow, 4, (16, 3.4 * nrow), sharex=True)
    for ax, (key, ylab) in zip(axes, metrics):
        for lab, sel, col in series:
            sub = sorted([r for r in rows if sel(r)], key=lambda r: r[xkey])
            if not sub:
                continue
            x = [r[xkey] / xdiv for r in sub]
            v = [r[key] for r in sub]
            ax.plot(x, v, "-", color=col, lw=1.5)
            # Hollow markers are runs that never settle, so their value is a
            # 1000-day mean of a fluctuating state rather than an equilibrium.
            for xi, vi, st in zip(x, v, [bool(r["steady"]) for r in sub]):
                ax.plot([xi], [vi], "o", color=col, ms=5,
                        mfc=col if st else "white", mew=1.2)
            ax.text(x[-1], v[-1], "  " + lab, color=col, fontsize=9,
                    fontweight="bold", va="center", ha="left")
        if logx:
            ax.set_xscale("log")
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.25)
        ax.margins(x=0.22)
    for ax in axes[-4:]:
        ax.set_xlabel(xlabel)
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# 6. Off-equatorial forcing
# ---------------------------------------------------------------------------
def fig_offeq_summary(rows, out):
    fig, axes = _panel_grid(2, 3, (15, 8), sharex=True)
    series = [
        ("sin$^2$, positive $\\hat H$", lambda r: r["block"] == "L"
         and r["theta_e_type"] == "sin2" and abs(r["a"] - 0.79) < 1e-9, OI["blue"]),
        ("sin$^2$, negative $\\hat H$", lambda r: r["block"] == "L"
         and r["theta_e_type"] == "sin2" and abs(r["a"] - 0.85) < 1e-9, OI["sky"]),
        ("SB08, positive $\\hat H$", lambda r: r["block"] == "L"
         and r["theta_e_type"] == "SB08" and abs(r["a"] - 0.79) < 1e-9, OI["verm"]),
        ("SB08, negative $\\hat H$", lambda r: r["block"] == "L"
         and r["theta_e_type"] == "SB08" and abs(r["a"] - 0.85) < 1e-9, OI["orange"]),
    ]
    metrics = [
        ("itcz_peak", "rain maximum (Mm)", MM),
        ("efe", "energy flux equator (Mm)", MM),
        ("u_eq", r"equatorial $u$ (m s$^{-1}$)", 1),
        ("jet", "summer-side jet (m s$^{-1}$)", 1),
        ("jet_s", "winter-side jet (m s$^{-1}$)", 1),
        ("v_max", r"peak $|v|$ (m s$^{-1}$)", 1),
    ]
    for ax, (key, ylab, scale) in zip(axes, metrics):
        for lab, sel, col in series:
            sub = sorted([r for r in rows if sel(r)], key=lambda r: r["y_0"])
            # include the y_0 = 0 reference from block A
            if not sub:
                continue
            x = [r["y_0"] * MM for r in sub]
            v = [r[key] * scale for r in sub]
            ax.plot(x, v, "-", color=col, lw=1.5)
            for xi, vi, st in zip(x, v, [bool(r["steady"]) for r in sub]):
                ax.plot([xi], [vi], "o", color=col, ms=5,
                        mfc=col if st else "white", mew=1.2)
            ax.text(x[-1], v[-1], "  " + lab, color=col, fontsize=8,
                    fontweight="bold", va="center", ha="left")
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.25)
        ax.margins(x=0.35)
    for ax in axes[3:]:
        ax.set_xlabel("forcing maximum $y_0$ (Mm)")
    axes[0].plot([0, 2.8], [0, 2.8], color="gray", ls=":", lw=1)
    axes[0].text(1.9, 2.3, "1:1", color="gray", fontsize=8)
    axes[0].text(0.03, 0.90, "hollow markers never settle", fontsize=7.5,
                 color="0.25", transform=axes[0].transAxes)
    fig.suptitle("Time-invariant off-equatorial forcing: where the rain, the "
                 "energy flux equator and the jets go", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# 7. Seasonal: Hovmoller of the composite cycle
# ---------------------------------------------------------------------------
def fig_seasonal_hov(seas, names, labels, out, title, field="p",
                     ylim=(-6, 6)):
    keep = [(nm, lb) for nm, lb in zip(names, labels) if nm in seas]
    if not keep:
        print(f"skipped {out}: no seasonal composites available")
        return
    names = [k[0] for k in keep]
    labels = [k[1] for k in keep]
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.0), sharey=True,
                             squeeze=False)
    axes = axes[0]
    vmax = 0.0
    for nm in names:
        if nm in seas:
            arr = seas[nm][field] * (86400 if field == "p" else 1)
            vmax = max(vmax, float(np.percentile(np.abs(arr), 99.5)))
    for ax, nm, lab in zip(axes, names, labels):
        s = seas[nm]
        arr = s[field] * (86400 if field == "p" else 1)
        t = np.arange(arr.shape[0]) / s["period"]
        if field == "p":
            im = ax.pcolormesh(t, s["y"] * MM, arr.T, cmap="magma_r",
                               vmin=0, vmax=vmax, shading="auto")
        else:
            im = ax.pcolormesh(t, s["y"] * MM, arr.T, cmap="RdBu_r",
                               norm=TwoSlopeNorm(0, -vmax, vmax),
                               shading="auto")
        ax.plot(t, s["y0_t"] * MM, color=OI["green"], lw=1.8)
        ax.plot(t, np.array(s["itcz_peak_t"]) * MM, color=OI["sky"], lw=1.5,
                ls="--")
        ax.plot(t, np.array(s["itcz_t"]) * MM, color=OI["blue"], lw=1.2,
                ls="-.")
        ax.plot(t, np.array(s["asc_t"]) * MM, color=OI["verm"], lw=1.2,
                ls=":")
        ax.set_ylim(*ylim)
        ax.set_xlabel("phase of the cycle")
        ax.set_title(lab, fontsize=10)
    axes[0].set_ylabel("$y$ (Mm)")
    for k, (txt, col) in enumerate((("forcing max $y_0$", OI["green"]),
                                    ("rain maximum", OI["sky"]),
                                    ("rain-band centroid", OI["blue"]),
                                    ("ascending branch", OI["verm"]))):
        axes[0].text(0.04, ylim[1] * (0.88 - 0.10 * k), txt, color=col,
                     fontsize=8.5, fontweight="bold",
                     bbox=dict(fc="white", ec="none", alpha=0.78, pad=1.0))
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("$P$ (mm day$^{-1}$)" if field == "p" else field)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _seas_amp_set(srows):
    return [r for r in srows if r["period"] == 360 and "_p360" in r["name"]
            and "tanh" not in r["name"] and "square" not in r["name"]]


def _seas_per_set(srows):
    return [r for r in srows if "amp1400" in r["name"]
            and "tanh" not in r["name"] and "square" not in r["name"]]


def fig_seasonal_summary(srows, out):
    if not srows:
        print(f"skipped {out}: no seasonal composites available")
        return
    sub = _seas_amp_set(srows)
    if not sub:
        print(f"skipped {out}: no 360-day amplitude series")
        return
    metrics = [("amp_itcz_km", "rain-centroid amplitude (km)"),
               ("gain", "centroid amplitude / forcing amplitude"),
               ("lag_days", "rain-centroid lag (days)"),
               ("jet_asym", r"summer minus winter jet (m s$^{-1}$)"),
               ("vmax_max", r"largest $|v|$ in the cycle (m s$^{-1}$)"),
               ("pmax_max", "largest peak $P$ (mm day$^{-1}$)"),
               ("wet_min", "smallest raining fraction"),
               ("ro_cell_max", r"largest cell-mean Ro")]
    fig, axes = _panel_grid(2, 4, (16.5, 7.6), sharex=True)
    for ax, (key, ylab) in zip(axes, metrics):
        for lab, col, af in (("positive $\\hat H$", OI["blue"], 0.79),
                             ("negative $\\hat H$", OI["verm"], 0.85)):
            ss = sorted([r for r in sub if abs(r["a"] - af) < 1e-9],
                        key=lambda r: r["amp_km"])
            if not ss:
                continue
            x = [r["amp_km"] for r in ss]
            v = [r[key] for r in ss]
            # A dashed line with open markers means the cycle does not repeat
            # from one year to the next, so "amplitude" and "lag" describe one
            # arbitrary realisation. Plotting those as a result would be wrong.
            cyc = all(r["cyclic"] for r in ss)
            ax.plot(x, v, "o-" if cyc else "o--", color=col, lw=1.5, ms=5,
                    mfc=col if cyc else "none")
            ax.text(x[-1], v[-1], "  " + lab + ("" if cyc else "\n  (no two cycles alike)"),
                    color=col, fontsize=8, fontweight="bold", va="center",
                    ha="left")
        if key == "ro_cell_max":
            ax.axhline(1.0, color=OI["green"], lw=1.2, ls="--")
            label_curve(ax, 400, 1.0, "angular momentum conserving",
                        OI["green"], dy=0.03)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.25)
        ax.margins(x=0.34)
    for ax in axes[-4:]:
        ax.set_xlabel("forcing migration amplitude (km)", fontsize=9)
    fig.suptitle("Seasonal response against the amplitude of the forcing "
                 "migration, at a 360-day period", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def fig_seasonal_period(srows, out):
    """The period dependence against the first-order relaxation null model.

    A single relaxation on the model's thermal timescale tau = 37 d predicts
    gain proportional to 1/sqrt(1+(omega tau)^2) and a lag of
    arctan(omega tau)/omega. Both are plotted as the grey line, so a departure
    from them is visible rather than asserted.
    """
    sub = _seas_per_set(srows)
    if not sub:
        print(f"skipped {out}: no period series")
        return
    fig, axes = _panel_grid(1, 4, (17.5, 4.6), sharex=True)
    metrics = [("amp_itcz_km", "rain-band amplitude (km)", None),
               ("gain", "band amplitude / forcing amplitude",
                "gain_damping_pred"),
               ("lag_days", "rain-band lag (days)", "lag_pred_days"),
               ("ro_cell_max", "largest cell-mean Ro", None)]
    for ax, (key, ylab, pred) in zip(axes, metrics):
        for lab, col, af in (("positive $\\hat H$", OI["blue"], 0.79),
                             ("negative $\\hat H$", OI["verm"], 0.85)):
            ss = sorted([r for r in sub if abs(r["a"] - af) < 1e-9],
                        key=lambda r: r["period"])
            if not ss:
                continue
            cyc = all(r["cyclic"] for r in ss)
            ax.plot([r["period"] for r in ss], [r[key] for r in ss],
                    "o-" if cyc else "o--", color=col, lw=1.5, ms=5,
                    mfc=col if cyc else "none")
            ax.text(ss[-1]["period"], ss[-1][key],
                    "  " + lab + ("" if cyc else "\n  (no two cycles alike)"),
                    color=col, fontsize=8, fontweight="bold", va="center",
                    ha="left")
        if pred:
            ss = sorted([r for r in sub if abs(r["a"] - 0.79) < 1e-9],
                        key=lambda r: r["period"])
            if len(ss) >= 3:
                # For the gain the null model gives only the damping factor, so
                # it is anchored on the longest period, where the damping is
                # negligible and the response is the quasi-steady one.
                scale = (ss[-1]["gain"] / ss[-1]["gain_damping_pred"]
                         if pred == "gain_damping_pred" else 1.0)
                ax.plot([r["period"] for r in ss],
                        [scale * r[pred] for r in ss], color="gray", lw=1.4,
                        ls="--", zorder=1)
                j = 1
                label_curve(ax, ss[j]["period"], scale * ss[j][pred],
                            "  37-day relaxation", "gray", va="top")
        if key == "ro_cell_max":
            ax.axhline(1.0, color=OI["green"], lw=1.2, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("forcing period (days)", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.25)
        ax.margins(x=0.3)
    fig.suptitle("How fast the forcing migrates: the same 1400 km amplitude "
                 "at five periods, against a single-relaxation prediction",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# 8. Seasonal vs perpetual
# ---------------------------------------------------------------------------
def fig_seasonal_vs_perpetual(eq, seas, out):
    """The solstitial state of a migrating cycle against the time-invariant run
    at the same instantaneous forcing."""
    pairs = [
        ("M_P_amp0700_p360", "L_P_sb08_y0700", "700 km, positive $\\hat H$"),
        ("M_P_amp1400_p360", "L_P_sb08_y1400", "1400 km, positive $\\hat H$"),
        ("M_P_amp2100_p360", "L_P_sb08_y2100", "2100 km, positive $\\hat H$"),
        ("M_N_amp1400_p360", "L_N_sb08_y1400", "1400 km, negative $\\hat H$"),
    ]
    if not any(sn in seas for sn, _, _ in pairs):
        print(f"skipped {out}: no seasonal composites available")
        return
    fig, axes = _panel_grid(2, 4, (16.5, 7.5), sharex=True)
    for j, (sn, pn, lab) in enumerate(pairs):
        axP, axU = axes[j], axes[j + 4]
        if sn in seas:
            s = seas[sn]
            k = int(np.argmax(s["y0_t"]))
            axP.plot(s["y"] * MM, s["p"][k] * 86400, color=OI["verm"], lw=1.6)
            axU.plot(s["y"] * MM, s["u"][k], color=OI["verm"], lw=1.6)
        if pn in eq:
            r = eq[pn]
            axP.plot(r["y"] * MM, r["p"] * 86400, color=OI["blue"], lw=1.6,
                     ls="--")
            axU.plot(r["y"] * MM, r["u"], color=OI["blue"], lw=1.6, ls="--")
        axP.set_title(lab, fontsize=10)
        for ax in (axP, axU):
            ax.grid(alpha=0.25)
            ax.set_xlim(-8, 8)
        axU.set_xlabel("$y$ (Mm)")
    axes[0].set_ylabel("$P$ (mm day$^{-1}$)")
    axes[4].set_ylabel("$u$ (m s$^{-1}$)")
    label_curve(axes[0], -7.5, axes[0].get_ylim()[1] * 0.85,
                "seasonal, at peak $y_0$", OI["verm"])
    label_curve(axes[0], -7.5, axes[0].get_ylim()[1] * 0.70,
                "time-invariant, same $y_0$", OI["blue"])
    fig.suptitle("A migrating forcing at its solstice against a forcing held "
                 "there forever", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# 9. Spin-up / equilibration
# ---------------------------------------------------------------------------
def fig_spinup(eq, names, labels, out):
    fig, axes = _panel_grid(1, 3, (15, 4.2), sharex=True)
    keys = [("jet", "subtropical jet (m s$^{-1}$)"),
            ("w_mean", r"domain-mean $W$ (kg m$^{-2}$)"),
            ("ke", r"domain-mean KE ((m s$^{-1}$)$^2$)")]
    for ax, (key, ylab) in zip(axes, keys):
        for j, (nm, lab) in enumerate(zip(names, labels)):
            if nm not in eq:
                continue
            ts = eq[nm]["ts"]
            if ts.get(key) is None:
                continue
            ax.plot(ts["time"], ts[key], color=SEQ[j % len(SEQ)], lw=1.0)
            ax.text(ts["time"][-1], ts[key][-1], "  " + lab,
                    color=SEQ[j % len(SEQ)], fontsize=8, fontweight="bold",
                    va="center")
        ax.set_xlabel("day")
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.25)
        ax.margins(x=0.18)
    fig.suptitle("Equilibration: the fixed 5800-day integration against the "
                 "1157-day drag timescale", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=SWEEP_DIR)
    ap.add_argument("--out", default=os.path.join(SWEEP_DIR, "figs"))
    ap.add_argument("--last-n", type=int, default=1000)
    ap.add_argument("--complete-frac", type=float, default=0.98,
                    help="fraction of its requested days a run must have "
                         "reached to be analysed; lower it only to exercise "
                         "the code against a smoke tree")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    eq, seas, meta, missing = load_all(args.tree, args.last_n,
                                       args.complete_frac)
    print(f"loaded {len(eq)} equilibrium runs, {len(seas)} seasonal, "
          f"{len(missing)} missing")
    if missing:
        print("missing: " + " ".join(missing))
    rows = [scalars(eq[n], n, meta[n]["block"], meta[n]["note"]) for n in eq]
    srows = [seasonal_scalars(seas[n], n, meta[n]["block"], meta[n]["note"])
             for n in seas]
    with open(os.path.join(args.out, "..", "diagnostics.json"), "w") as fh:
        json.dump(dict(equilibrium=rows, seasonal=srows, missing=missing), fh,
                  indent=1, default=float)

    o = lambda f: os.path.join(args.out, f)  # noqa: E731

    fig_regime_map(rows, o("01_regime_map.png"))
    fig_collapse(rows, o("02_collapse_on_r.png"))

    transect = ["A_a085_wc30", "B_a085_wc35", "A_a085_wc40", "B_a085_wc44",
                "A_a085_wc50", "B_a085_wc55", "A_a085_wc60"]
    tlab = [f"$W_c$={w}" for w in (30, 35, 40, 44, 50, 55, 60)]
    profile_panels(eq, transect, tlab, o("03_transect_profiles.png"),
                   r"Crossing $\hat H=0$ by raising $W_c$ at $a=0.85$: "
                   r"$r$ from 0.69 to 1.37", label_x=0.74)
    profile_panels(eq, ["A_a085_wc40", "B_a085_wc42", "B_a085_wc44",
                        "B_a085_wc46", "A_a085_wc50"],
                   [f"$W_c$={w}" for w in (40, 42, 44, 46, 50)],
                   o("03b_transition_zoom.png"),
                   r"The transition resolved: $r$ from 0.92 to 1.14 at $a=0.85$",
                   label_x=0.74)
    fig_budget(eq, ["A_a079_wc40", "A_a085_wc50"],
               [r"$a$=0.79, $W_c$=40 ($r$=0.76, positive $\hat H$)",
                r"$a$=0.85, $W_c$=50 ($r$=1.14, negative $\hat H$)"],
               o("04_mse_budget.png"),
               "Column MSE flux resolved into its large opposing parts, and "
               "the budget it has to close")

    fig_anatomy(eq, "A_a085_wc50", o("04b_anatomy_negative.png"),
                "How a negative-stability solution closes its energy budget: "
                r"$a$=0.85, $W_c$=50, $r$=1.14")
    fig_anatomy(eq, "A_a079_wc40", o("04c_anatomy_positive.png"),
                "The same four panels for a positive-stability solution: "
                r"$a$=0.79, $W_c$=40, $r$=0.76")

    fig_ladder(rows, o("05_ladder_dyrad.png"), "delta_y_rad",
               r"$\Delta_\theta^{\rm rad}$ (K)",
               [("positive $\\hat H$", lambda r: r["block"] in ("A", "D")
                 and abs(r["a"] - 0.79) < 1e-9 and r["w_crit"] == 40.0,
                 OI["blue"]),
                ("negative $\\hat H$", lambda r: r["block"] in ("A", "D")
                 and abs(r["a"] - 0.85) < 1e-9 and r["w_crit"] == 50.0,
                 OI["verm"])],
               "Forcing strength: the radiative-equilibrium contrast")
    fig_ladder(rows, o("06_ladder_D.png"), "d_w",
               r"eddy moisture diffusivity $D$ ($10^6$ m$^2$ s$^{-1}$)",
               [("positive $\\hat H$", lambda r: r["block"] in ("A", "F")
                 and abs(r["a"] - 0.79) < 1e-9 and r["w_crit"] == 40.0,
                 OI["blue"]),
                ("negative $\\hat H$", lambda r: r["block"] in ("A", "F")
                 and abs(r["a"] - 0.85) < 1e-9 and r["w_crit"] == 50.0,
                 OI["verm"])],
               "The eddy moisture flux, the only brake on aggregation",
               xdiv=1e6)
    fig_ladder(rows, o("07_ladder_tauc.png"), "tau_c",
               r"convective timescale $\tau_c$ (hours)",
               [(r"$a$=0.85, $W_c$=40", lambda r: r["block"] in ("A", "E")
                 and abs(r["a"] - 0.85) < 1e-9 and r["w_crit"] == 40.0,
                 OI["green"]),
                ("positive $\\hat H$", lambda r: r["block"] in ("A", "E")
                 and abs(r["a"] - 0.79) < 1e-9 and r["w_crit"] == 40.0,
                 OI["blue"]),
                ("negative $\\hat H$", lambda r: r["block"] in ("A", "E")
                 and abs(r["a"] - 0.85) < 1e-9 and r["w_crit"] == 50.0,
                 OI["verm"])],
               "Convective timescale", logx=True, xdiv=3600.0)
    fig_ladder(rows, o("08_ladder_vd.png"), "v_d",
               r"eddy momentum flux velocity $v_d$ (m s$^{-1}$)",
               [("positive $\\hat H$", lambda r: r["block"] in ("A", "H")
                 and abs(r["a"] - 0.79) < 1e-9 and r["w_crit"] == 40.0,
                 OI["blue"]),
                ("negative $\\hat H$", lambda r: r["block"] in ("A", "H")
                 and abs(r["a"] - 0.85) < 1e-9 and r["w_crit"] == 50.0,
                 OI["verm"])],
               "Eddy momentum forcing, including the axisymmetric limit")
    fig_ladder(rows, o("09_ladder_lambda.png"), "lam",
               r"latent heating coefficient $\Lambda$ (K per kg m$^{-2}$ s$^{-1}$)",
               [("positive $\\hat H$", lambda r: r["block"] in ("A", "J")
                 and abs(r["a"] - 0.79) < 1e-9 and r["w_crit"] == 40.0,
                 OI["blue"]),
                ("negative $\\hat H$", lambda r: r["block"] in ("A", "J")
                 and abs(r["a"] - 0.85) < 1e-9 and r["w_crit"] == 50.0,
                 OI["verm"])],
               "Feedback strength, from the severed bridge to the physical value")
    fig_ladder(rows, o("10_ladder_evap.png"), "evap",
               r"evaporation $E_0$ (mm day$^{-1}$)",
               [("positive $\\hat H$", lambda r: r["block"] in ("A", "G")
                 and abs(r["a"] - 0.79) < 1e-9 and r["w_crit"] == 40.0,
                 OI["blue"]),
                ("negative $\\hat H$", lambda r: r["block"] in ("A", "G")
                 and abs(r["a"] - 0.85) < 1e-9 and r["w_crit"] == 50.0,
                 OI["verm"])],
               "The water source", xdiv=1.0 / 86400.0)

    fig_offeq_summary(rows, o("11_offequatorial.png"))
    for ref, lab in (("P", "positive $\\hat H$"), ("N", "negative $\\hat H$")):
        names = [f"L_{ref}_sb08_y{v:04d}" for v in (0, 700, 1400, 2100, 2800)]
        profile_panels(eq, names,
                       [f"$y_0$={v} km" for v in (0, 700, 1400, 2100, 2800)],
                       o(f"12_offeq_profiles_{ref}.png"),
                       f"Time-invariant off-equatorial SB08 forcing, {lab}",
                       label_panel=1, label_key="v_face")

    seas_names = ["M_P_amp0350_p360", "M_P_amp0700_p360", "M_P_amp1400_p360",
                  "M_P_amp2100_p360", "M_P_amp2800_p360"]
    fig_seasonal_hov(seas, seas_names,
                     [f"amplitude {v} km" for v in (350, 700, 1400, 2100, 2800)],
                     o("13_seasonal_hov_P.png"),
                     "Precipitation through the seasonal cycle, positive "
                     r"$\hat H$ ($a$=0.79, $W_c$=40)")
    fig_seasonal_hov(seas, ["M_N_amp0350_p360", "M_N_amp0700_p360",
                            "M_N_amp1400_p360", "M_N_amp2100_p360",
                            "M_N_amp2800_p360"],
                     [f"amplitude {v} km" for v in (350, 700, 1400, 2100, 2800)],
                     o("14_seasonal_hov_N.png"),
                     "Precipitation through the seasonal cycle, negative "
                     r"$\hat H$ ($a$=0.85, $W_c$=50)")
    fig_seasonal_hov(seas, ["M_P_amp1400_p0090", "M_P_amp1400_p0180",
                            "M_P_amp1400_p360", "M_P_amp1400_p0720",
                            "M_P_amp1400_p1440"],
                     [f"{v}-day period" for v in (90, 180, 360, 720, 1440)],
                     o("15_seasonal_hov_period.png"),
                     "Migration rate: the same 1400 km amplitude at five periods")
    fig_seasonal_summary(srows, o("16_seasonal_summary.png"))
    fig_seasonal_period(srows, o("16b_seasonal_period.png"))
    fig_seasonal_vs_perpetual(eq, seas, o("17_seasonal_vs_perpetual.png"))

    fig_spinup(eq, ["A_a079_wc40", "A_a085_wc44", "A_a085_wc50",
                    "A_a085_wc60", "L_P_sb08_y1400"],
               [r"$r$=0.76", r"$r$=1.01", r"$r$=1.14", r"$r$=1.37",
                r"$y_0$=1400 km"],
               o("18_spinup.png"))

    profile_panels(eq, ["A_a085_wc44", "K_wc44_wi59", "A_a085_wc46",
                        "K_wc46_wi61", "A_a085_wc40", "K_wc40_wi55"],
                   [r"$W_c$44, $W_0$=44", r"$W_c$44, $W_0$=59",
                    r"$W_c$46, $W_0$=46", r"$W_c$46, $W_0$=61",
                    r"$W_c$40, $W_0$=40", r"$W_c$40, $W_0$=55"],
                   o("19_initial_condition.png"),
                   "Does the initial moisture pick the attractor?")

    numerics = [("O_P_ny401", "$n_y$=401, $dt$=60"),
                ("A_a079_wc40", "$n_y$=801, $dt$=30"),
                ("O_P_dt15", "$n_y$=801, $dt$=15"),
                ("O_P_ny1601", "$n_y$=1601, $dt$=15")]
    profile_panels(eq, [n for n, _ in numerics], [l for _, l in numerics],
                   o("20_numerics_P.png"),
                   r"Resolution and timestep twins, positive $\hat H$ "
                   r"reference ($a$=0.79, $W_c$=40)")
    profile_panels(eq, ["O_N_ny401", "A_a085_wc50", "O_N_dt15", "O_N_ny1601"],
                   ["$n_y$=401, $dt$=60", "$n_y$=801, $dt$=30",
                    "$n_y$=801, $dt$=15", "$n_y$=1601, $dt$=15"],
                   o("21_numerics_N.png"),
                   r"Resolution and timestep twins, negative $\hat H$ "
                   r"reference ($a$=0.85, $W_c$=50)")
    print("figures done")


if __name__ == "__main__":
    main()
