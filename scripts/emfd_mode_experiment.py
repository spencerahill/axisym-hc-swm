"""Screen two mitigations of the EMFD jet-flank 2Δy mode (staggered+RK4 scheme).

The numerics rewrite left one defect: a standing 2Δy computational mode in ``u``
at the poleward jet flanks, sitting on the flank ``u = 0`` crossing, excited by
the hard ``H(u)`` gate of the eddy-momentum flux divergence. It is
damping-limited and resolution-dependent (large at ny<=100, gone by ny~400),
which is unacceptable because convergence is tested across resolution.

This driver screens two opt-in levers (both implemented default-off in the model):

  - **tanh gate** ``emfd_gate_width = u_w``: smooth the EMFD gate
    ``g(u) = 0.5(1 + tanh(u/u_w))`` so the forcing no longer steps at ``u = 0``.
  - **hyperdiffusion** ``k_u4``: biharmonic ``-k_u4 d^4u/dy^4``, scale-selective
    damping of the 2Δy mode.

Matrix at ny=50, dt=900, 1000 days: gate {hard, tanh(u_w=5)} x hyper {off,
k_u4=1e17} x case {on-eq sin2 y0=0, off-eq SB08 y0=1e6} (8 runs). The ny=200
subset (for the winner) is run by ``subset`` mode and compared to the persisted
baselines in model_output/mode_experiment/.

In-process (no NetCDF round-trip), following scripts/numerics_investigation.py.
Diagnostics reuse scripts/cmp_utils.py. Usage:

    python scripts/emfd_mode_experiment.py matrix      # ny=50 2x2x2 + plots + npz
    python scripts/emfd_mode_experiment.py subset      # ny=200 winner(s) vs baseline
"""

import logging
import os
import sys

import numpy as np

from ss09.sw_config import SWConfig
from ss09.theta_e import ThetaEConfig, Sin2Profile, SB08Profile
from ss09.sw_model import SWModel
from ss09 import rhs as rhs_module

BETA = 2e-11  # config default; none of these runs change it

sys.path.insert(0, os.path.dirname(__file__))
import cmp_utils as cu  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)  # silence per-day INFO spam

# Output dir for npz + figures; override with EMFD_SCRATCH (no repo writes).
SCRATCH = os.environ.get("EMFD_SCRATCH", "/tmp/emfd_mode_experiment")
os.makedirs(SCRATCH, exist_ok=True)
BASELINE_DIR = "./model_output/mode_experiment"

# Screening lever values (derived in the plan; not final tunings).
U_W = 5.0      # tanh gate width [m/s]
K_U4 = 1e17    # biharmonic hyperdiffusion [m^4/s]
FLANK_MIN = 7e6  # |y| >= this is "flank"; below is "resolved climate"

# (case label, profile, y_0)
CASES = [("on-eq", "sin2", 0.0), ("off-eq", "SB08", 1.0e6)]
# (config label, emfd_gate_width, k_u4)
CONFIGS = [
    ("hard,         off ", 0.0, 0.0),
    ("tanh(uw=5),   off ", U_W, 0.0),
    ("hard,    k4=1e17  ", 0.0, K_U4),
    ("tanh(uw=5),k4=1e17", U_W, K_U4),
]
_PROFILES = {"sin2": Sin2Profile, "SB08": SB08Profile}

# divisors of 86400 (SECONDS_PER_DAY), descending, for clean daily averaging.
# Spans down from 900 so base_dt (900 at ny=50, 225 at ny=200) is reachable.
_DT_DIVISORS = [900, 800, 720, 675, 640, 600, 576, 540, 480, 450, 432, 400,
                384, 360, 320, 300, 288, 270, 240, 225, 192, 180, 160, 144, 120,
                108, 100, 96, 90, 80, 72, 60, 48, 45, 40, 36, 32, 30, 25, 24,
                20, 18, 16, 15, 12, 10, 9, 8]


def stable_dt(k_u4, ny, base_dt, safety=2.0):
    """Largest dt (<= base_dt, divisor of 86400) that keeps explicit biharmonic
    hyperdiffusion stable: dt * k_u4 * 16/dy^4 <= safety (< RK4 limit 2.79)."""
    if k_u4 == 0.0:
        return base_dt
    dy = (15751e3 * 2) / ny
    dt_lim = min(base_dt, safety / (k_u4 * 16.0 / dy ** 4))
    for d in _DT_DIVISORS:
        if d <= dt_lim:
            return d
    return _DT_DIVISORS[-1]


def rossby(y, u, dy):
    """Local Rossby number Ro = (du/dy)/(beta y), matching the model's
    hadley_diagnostics convention exactly (np.gradient, no sign flip). NaN within
    dy of the equator to avoid the 1/y singularity. Centered differences do not
    see the 2Δy mode (it cancels), so this is a smooth diagnostic of the resolved
    shear, not of the grid mode."""
    ro = np.gradient(u, dy) / (BETA * y)
    ro[np.abs(y) < dy] = np.nan
    return ro


def run(profile, y_0, gate_width, k_u4, ny=50, dt=900, ndays=1000, v_d=2.5,
        k_u=1e5, epsilon_u=1e-8):
    """Run one config; return (y, u_mean over last 200 days, final SWModel).

    v_d is the eddy-momentum-flux-divergence coefficient; v_d=0 is the
    axisymmetric (no parameterized eddy momentum forcing) limit. k_u is the
    explicit eddy viscosity on u (SS09 have none). epsilon_u is the Rayleigh drag
    coefficient (SS09's sole u-dissipation; their range 1e-10..1e-7).
    """
    cfg = SWConfig(
        total_integration_days=ndays, ny=ny, dt=dt, k_u=k_u, v_d=v_d,
        epsilon_u=epsilon_u, emfd_gate_width=gate_width, k_u4=k_u4,
        domain_size=15751e3 * 2,
        output_path=f"{SCRATCH}/emfd_exp_tmp.nc", restart_output_dir=SCRATCH,
    )
    tcfg = ThetaEConfig(theta_e_type=profile, y_0=y_0)
    model = SWModel(cfg, _PROFILES[profile](tcfg))
    model.run_sim()
    u = model.results.u
    finite = np.array([np.all(np.isfinite(u[d])) and np.any(u[d] != 0)
                       for d in range(u.shape[0])])
    blew = not finite.all()
    u_mean = u[-200:].mean(axis=0)
    return cfg.y, u_mean, model, blew


def field_means(model):
    """Daily-mean v and theta over the last 200 days (both on centers)."""
    v = model.results.v[-200:].mean(axis=0)
    th = model.results.theta[-200:].mean(axis=0)
    return v, th


def diag(y, u, u_base):
    """Per-config diagnostics dict."""
    fm = cu.flank_mode(y, u, flank_min_m=FLANK_MIN)
    core = np.abs(y) < FLANK_MIN
    dmax = float(np.max(np.abs(u[core] - u_base[core])))  # off-flank climate change
    return {
        "jet": float(np.max(u)),
        "u_eq": float(u[np.argmin(np.abs(y))]),
        "sawtooth": fm["sawtooth_max"],
        "saw_at_Mm": fm["at_m"] / 1e6,
        "mode": fm["alternating"],          # True = 2Δy mode still present
        "dmax_core": dmax,
    }


def run_matrix():
    print("=== EMFD jet-flank mode mitigation: ny=50, dt=900, 1000 days ===")
    print(f"  levers: tanh u_w={U_W} m/s, hyperdiff k_u4={K_U4:.0e} m^4/s; k_u=1e5 fixed\n")
    results = {}  # (case, cfg_label) -> dict(y,u,diag); also store models for budget
    for case, profile, y_0 in CASES:
        # baseline (hard, off) first for the off-flank delta reference
        y, u_base, m_base, blew0 = run(profile, y_0, 0.0, 0.0)
        results[(case, CONFIGS[0][0])] = dict(y=y, u=u_base, model=m_base, blew=blew0)
        for label, gw, k4 in CONFIGS[1:]:
            yy, uu, mm, blew = run(profile, y_0, gw, k4)
            results[(case, label)] = dict(y=yy, u=uu, model=mm, blew=blew)

    # table
    hdr = (f"{'case':6s} {'config':19s} | {'jet':>6s} {'u_eq':>6s} | "
           f"{'saw':>6s} @{'Mm':>5s} {'mode?':>5s} | {'d|u|core':>8s} {'':4s}")
    print(hdr)
    print("-" * len(hdr))
    for case, profile, y_0 in CASES:
        u_base = results[(case, CONFIGS[0][0])]["u"]
        for label, gw, k4 in CONFIGS:
            r = results[(case, label)]
            d = diag(r["y"], r["u"], u_base)
            r["diag"] = d
            flag = "BLEW" if r["blew"] else ("MODE" if d["mode"] else "ok")
            print(f"{case:6s} {label:19s} | {d['jet']:6.2f} {d['u_eq']:6.2f} | "
                  f"{d['sawtooth']:6.3f} @{d['saw_at_Mm']:+5.1f} {str(d['mode']):>5s} | "
                  f"{d['dmax_core']:8.3f} {flag:4s}")
        print()

    _save_npz(results)
    _make_plots(results)
    return results


def _save_npz(results):
    out = {}
    for (case, label), r in results.items():
        key = f"{case}|{label}".replace(" ", "")
        out[f"{key}|y"] = r["y"]
        out[f"{key}|u"] = r["u"]
    np.savez(f"{SCRATCH}/emfd_matrix_ny50.npz", **out)
    print(f"saved arrays -> {SCRATCH}/emfd_matrix_ny50.npz")


def _load_npz():
    """Reconstruct the results dict (y, u, diag) from the saved ny=50 matrix."""
    d = np.load(f"{SCRATCH}/emfd_matrix_ny50.npz")
    results = {}
    for case, profile, y_0 in CASES:
        for label, gw, k4 in CONFIGS:
            key = f"{case}|{label}".replace(" ", "")
            results[(case, label)] = dict(y=d[f"{key}|y"], u=d[f"{key}|u"])
    for case, profile, y_0 in CASES:
        u_base = results[(case, CONFIGS[0][0])]["u"]
        for label, gw, k4 in CONFIGS:
            r = results[(case, label)]
            r["diag"] = diag(r["y"], r["u"], u_base)
    return results


def _make_plots(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {CONFIGS[0][0]: "k", CONFIGS[1][0]: "C0",
              CONFIGS[2][0]: "C1", CONFIGS[3][0]: "C3"}

    # (1) flank-zoom small multiples 2x4 (case x config), at the worst-mode flank
    # (the SH winter flank for off-eq); plotted vs |y| so both cases read L->R.
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
    for r_i, (case, profile, y_0) in enumerate(CASES):
        # hemisphere of the baseline's largest sawtooth
        hemi = np.sign(results[(case, CONFIGS[0][0])]["diag"]["saw_at_Mm"]) or 1.0
        for c_i, (label, gw, k4) in enumerate(CONFIGS):
            ax = axes[r_i, c_i]
            r = results[(case, label)]
            y, u = r["y"], r["u"]
            sel = (np.sign(y) == hemi) & (np.abs(y) > 6.5e6)
            order = np.argsort(np.abs(y[sel]))
            ax.plot(np.abs(y[sel])[order] / 1e6, u[sel][order], "-o", ms=3,
                    color=colors[label])
            ax.axhline(0, color="grey", lw=0.5)
            d = r["diag"]
            ax.set_title(f"{case} | {label.strip()}\nmode={d['mode']} saw={d['sawtooth']:.2f}",
                         fontsize=8)
            if r_i == 1:
                ax.set_xlabel("|y| [Mm]")
            if c_i == 0:
                ax.set_ylabel("u [m/s]")
    fig.suptitle("Worst-flank zoom (ny=50): the 2Δy mode and its mitigations "
                 "(on-eq=NH, off-eq=SH winter jet)")
    fig.tight_layout()
    fig.savefig(f"{SCRATCH}/fig1_flank_zoom_ny50.png", dpi=130)
    plt.close(fig)

    # (2) full-domain overlay per case, all configs
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (case, profile, y_0) in zip(axes, CASES):
        for label, gw, k4 in CONFIGS:
            r = results[(case, label)]
            ax.plot(r["y"] / 1e6, r["u"], "-", lw=1.2, color=colors[label],
                    label=label.strip())
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{case}  (ny=50, 1000 d mean)")
        ax.set_xlabel("y [Mm]")
        ax.set_ylabel("u [m/s]")
        ax.legend(fontsize=7)
    fig.suptitle("Full-domain zonal wind: whole-domain spurious-feature check")
    fig.tight_layout()
    fig.savefig(f"{SCRATCH}/fig2_full_domain_ny50.png", dpi=130)
    plt.close(fig)

    # (3) EMFD term + gate across the off-eq SH winter flank: hard vs tanh (hyper off)
    case = "off-eq"
    r_hard = results[(case, CONFIGS[0][0])]
    r_tanh = results[(case, CONFIGS[1][0])]
    y = r_hard["y"]
    sel = y < -6.5e6
    cfg_hard = SWConfig(ny=50, total_integration_days=1, emfd_gate_width=0.0)
    cfg_tanh = SWConfig(ny=50, total_integration_days=1, emfd_gate_width=U_W)
    emfd_hard = rhs_module.emfd_u(r_hard["u"], cfg_hard)
    emfd_tanh = rhs_module.emfd_u(r_tanh["u"], cfg_tanh)
    g_hard = rhs_module._heaviside(r_hard["u"])
    g_tanh = 0.5 * (1 + np.tanh(r_tanh["u"] / U_W))
    yp = -y[sel] / 1e6  # plot vs |y| (SH flank)
    o = np.argsort(yp)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(yp[o], r_hard["u"][sel][o], "-o", ms=3, color="k", label="hard")
    ax[0].plot(yp[o], r_tanh["u"][sel][o], "-o", ms=3, color="C0", label="tanh")
    ax[0].axhline(0, color="grey", lw=0.5); ax[0].set_title("u (daily-mean)")
    ax[0].legend(fontsize=8)
    ax[1].plot(yp[o], g_hard[sel][o], "-o", ms=3, color="k", label="H(u)")
    ax[1].plot(yp[o], g_tanh[sel][o], "-o", ms=3, color="C0", label="0.5(1+tanh)")
    ax[1].set_title("EMFD gate g(u)"); ax[1].legend(fontsize=8)
    ax[2].plot(yp[o], emfd_hard[sel][o], "-o", ms=3, color="k", label="hard")
    ax[2].plot(yp[o], emfd_tanh[sel][o], "-o", ms=3, color="C0", label="tanh")
    ax[2].axhline(0, color="grey", lw=0.5)
    ax[2].set_title("EMFD term  S = v_d g(u) sgn(y) du/dy"); ax[2].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("|y| [Mm]")
    fig.suptitle("off-eq SH winter flank: the hard H(u) gate toggles where moded u crosses 0; tanh smooths the forcing")
    fig.tight_layout()
    fig.savefig(f"{SCRATCH}/fig3_emfd_flank_budget.png", dpi=130)
    plt.close(fig)
    print(f"saved fig1/fig2/fig3 -> {SCRATCH}/")


def run_subset(winners):
    """Re-run winner config(s) at ny=200/dt=225/1000d, compare to scan baselines."""
    print("=== ny=200 subset: confirm the fix scales (dt=225, 1000 days) ===\n")
    base = {
        "on-eq": f"{BASELINE_DIR}/scan_oneq_ny200.nc",
        "off-eq": f"{BASELINE_DIR}/scan_offeq_ny200.nc",
    }
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(winners), 2, figsize=(14, 4.5 * len(winners)),
                             squeeze=False)
    for w_i, (wlabel, gw, k4) in enumerate(winners):
        for c_i, (case, profile, y_0) in enumerate(CASES):
            yb, ub = cu.steady(base[case], "u", 200)
            y, u, m, blew = run(profile, y_0, gw, k4, ny=200, dt=225, ndays=1000)
            fm = cu.flank_mode(y, u, flank_min_m=FLANK_MIN)
            fb = cu.flank_mode(yb, ub, flank_min_m=FLANK_MIN)
            d = cu.report_diff(yb, ub, y, u, name=f"{case} {wlabel.strip()} vs (hard,off)")
            print(f"  {case:6s} {wlabel.strip():18s} | baseline mode={fb['alternating']} "
                  f"saw={fb['sawtooth_max']:.3f}  ->  fix mode={fm['alternating']} "
                  f"saw={fm['sawtooth_max']:.3f} | jet {np.max(ub):.2f}->{np.max(u):.2f}"
                  f" | max|Δ|={d['max_abs']:.2f}@{d['at_m']/1e6:+.1f}Mm"
                  f"{'  BLEW' if blew else ''}")
            ax = axes[w_i, c_i]
            selb = yb > 6.5e6
            sel = y > 6.5e6
            ax.plot(yb[selb] / 1e6, ub[selb], "-o", ms=3, color="k",
                    label=f"hard,off (mode={fb['alternating']})")
            ax.plot(y[sel] / 1e6, u[sel], "-o", ms=3, color="C0",
                    label=f"{wlabel.strip()} (mode={fm['alternating']})")
            ax.axhline(0, color="grey", lw=0.5)
            ax.set_title(f"{case}  ny=200 NH flank", fontsize=9)
            ax.set_xlabel("y [Mm]"); ax.set_ylabel("u [m/s]"); ax.legend(fontsize=7)
        print()
    fig.suptitle("ny=200: winner mitigation vs (hard,off) baseline at the NH jet flank")
    fig.tight_layout()
    fig.savefig(f"{SCRATCH}/fig4_ny200_flank.png", dpi=130)
    plt.close(fig)
    print(f"saved fig4 -> {SCRATCH}/fig4_ny200_flank.png")


def run_field_matrix():
    """Re-run the 8 ny=50 configs and plot the daily-mean v and theta fields,
    the companions to the u plots (fig1/fig2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {CONFIGS[0][0]: "k", CONFIGS[1][0]: "C0",
              CONFIGS[2][0]: "C1", CONFIGS[3][0]: "C3"}

    F = {}
    for case, profile, y_0 in CASES:
        for label, gw, k4 in CONFIGS:
            y, u, m, blew = run(profile, y_0, gw, k4)
            v, th = field_means(m)
            F[(case, label)] = dict(y=y, v=v, theta=th)
            sw_v = np.nanmax(cu.sawtooth(v)[np.abs(y) >= FLANK_MIN])
            sw_t = np.nanmax(cu.sawtooth(th)[np.abs(y) >= FLANK_MIN])
            print(f"{case:6s} {label:19s} | flank sawtooth: v={sw_v:7.4f} m/s  "
                  f"theta={sw_t:8.5f} K{'  BLEW' if blew else ''}")
        print()

    for fld, fname, unit in [("v", "fig5_v", "m/s"), ("theta", "fig6_theta", "K")]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for c_i, (case, profile, y_0) in enumerate(CASES):
            # top: full domain; bottom: worst flank (SH for off-eq), vs |y|
            hemi = -1.0 if case == "off-eq" else 1.0
            for label, gw, k4 in CONFIGS:
                r = F[(case, label)]
                axes[0, c_i].plot(r["y"] / 1e6, r[fld], "-", lw=1.1,
                                  color=colors[label], label=label.strip())
                sel = (np.sign(r["y"]) == hemi) & (np.abs(r["y"]) > 6.5e6)
                o = np.argsort(np.abs(r["y"][sel]))
                axes[1, c_i].plot(np.abs(r["y"][sel])[o] / 1e6, r[fld][sel][o],
                                  "-o", ms=3, color=colors[label], label=label.strip())
            axes[0, c_i].set_title(f"{case}  full domain"); axes[0, c_i].set_xlabel("y [Mm]")
            axes[1, c_i].set_title(f"{case}  worst flank zoom"); axes[1, c_i].set_xlabel("|y| [Mm]")
            for rr in (0, 1):
                axes[rr, c_i].set_ylabel(f"{fld} [{unit}]")
                axes[rr, c_i].axhline(0, color="grey", lw=0.5)
            axes[0, c_i].legend(fontsize=7)
        fig.suptitle(f"daily-mean {fld} field (ny=50, 1000 d): "
                     f"companion to the u plots")
        fig.tight_layout()
        fig.savefig(f"{SCRATCH}/{fname}_ny50.png", dpi=130)
        plt.close(fig)
    print(f"saved fig5_v / fig6_theta -> {SCRATCH}/")


def _sweep(values, gate_of, k4_of, vary_name):
    """Run a 1-D parameter sweep for both cases; print table, return data."""
    print(f"=== sensitivity to {vary_name} (ny=50, dt=900, 1000 days) ===")
    hdr = f"{vary_name:>12s} | {'case':6s} | {'jet':>6s} {'u_eq':>6s} {'flank saw':>9s} {'mode?':>5s}"
    print(hdr); print("-" * len(hdr))
    data = {case: {"x": [], "saw": [], "jet": [], "u_eq": []} for case, _, _ in CASES}
    for val in values:
        for case, profile, y_0 in CASES:
            y, u, m, blew = run(profile, y_0, gate_of(val), k4_of(val))
            fm = cu.flank_mode(y, u, flank_min_m=FLANK_MIN)
            jet = float(np.max(u)); ueq = float(u[np.argmin(np.abs(y))])
            data[case]["x"].append(val); data[case]["saw"].append(fm["sawtooth_max"])
            data[case]["jet"].append(jet); data[case]["u_eq"].append(ueq)
            print(f"{val:12.4g} | {case:6s} | {jet:6.2f} {ueq:6.2f} {fm['sawtooth_max']:9.3f} "
                  f"{str(fm['alternating']):>5s}{'  BLEW' if blew else ''}")
        print()
    return data


def _plot_sweep(data, values, vary_name, fname, logx=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for case in data:
        m = "o-" if case == "on-eq" else "s-"
        ax[0].plot(data[case]["x"], data[case]["saw"], m, label=case)
        ax[1].plot(data[case]["x"], data[case]["jet"], m, label=case)
    ax[0].set_ylabel("flank sawtooth [m/s]  (mode amplitude)")
    ax[1].set_ylabel("jet strength max(u) [m/s]")
    for a in ax:
        a.set_xlabel(vary_name)
        if logx:
            a.set_xscale("symlog" if 0 in values else "log")
        a.legend(fontsize=8); a.grid(alpha=0.3)
    ax[0].set_title("mode amplitude vs " + vary_name)
    ax[1].set_title("resolved climate (jet) vs " + vary_name)
    fig.suptitle(f"Sensitivity to {vary_name} (ny=50, hard/hyper as labeled)")
    fig.tight_layout()
    fig.savefig(f"{SCRATCH}/{fname}.png", dpi=130)
    plt.close(fig)
    print(f"saved {fname} -> {SCRATCH}/{fname}.png")


def _profiles_figure(configs, vary_name, fname, ny=50, base_dt=900, v_d=2.5):
    """Plot full-domain u, v, theta, Ro for every value in a 1-D sweep, both cases.

    configs: list of (label, gate_width, k_u4). Each value gets one color; the
    values overlay in each (case x field) panel. NO zero line on theta (it lives
    at 250-330 K; a y=0 reference squishes it); u, v, Ro keep the zero line. Ro =
    (du/dy)/(beta y) follows the model's convention; it is autoscaled (the
    centered-difference Ro is smooth, O(0.1-0.5), since the 2Δy mode cancels).
    Per-config dt is auto-reduced for biharmonic stability at high ny. v_d is the
    EMFD coefficient (v_d=0 is the axisymmetric limit). Reports whole-domain
    diagnostics before returning.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = [plt.cm.viridis(t) for t in np.linspace(0, 0.92, len(configs))]

    data = {}
    for case, profile, y_0 in CASES:
        for label, gw, k4 in configs:
            dt = stable_dt(k4, ny, base_dt)
            print(f"  running {case} {label.strip()} (ny={ny}, dt={dt}, v_d={v_d}) ...",
                  flush=True)
            y, u, m, blew = run(profile, y_0, gw, k4, ny=ny, dt=dt, v_d=v_d)
            v, th = field_means(m)
            dy = (15751e3 * 2) / ny
            data[(case, label)] = dict(y=y, u=u, v=v, theta=th, ro=rossby(y, u, dy),
                                       blew=blew, dt=dt)

    # ---- inspect the whole domain and report ----
    print(f"=== full-domain profiles vs {vary_name} (ny={ny}, 1000 d) ===")
    hdr = (f"{vary_name:>12s} {'case':6s} {'dt':>4s} | {'max u':>6s} {'min u':>6s} "
           f"{'u_eq':>6s} {'uSAW':>6s} | {'max|v|':>6s} | {'minT':>6s} {'maxT':>6s} "
           f"| {'max|Ro|':>7s} {'':4s}")
    print(hdr); print("-" * len(hdr))
    for label, gw, k4 in configs:
        for case, profile, y_0 in CASES:
            r = data[(case, label)]
            y, u, v, th, ro = r["y"], r["u"], r["v"], r["theta"], r["ro"]
            sw = cu.flank_mode(y, u, flank_min_m=FLANK_MIN)["sawtooth_max"]
            print(f"{label:>12s} {case:6s} {r['dt']:4d} | {np.max(u):6.2f} {np.min(u):6.2f} "
                  f"{u[np.argmin(np.abs(y))]:6.2f} {sw:6.2f} | {np.max(np.abs(v)):6.3f} | "
                  f"{np.min(th):6.1f} {np.max(th):6.1f} | {np.nanmax(np.abs(ro)):7.2f} "
                  f"{'BLEW' if r['blew'] else '':4s}")
        print()

    # ---- plot ----
    fields = [("u", "m/s", True), ("v", "m/s", True),
              ("theta", "K", False), ("ro", "Ro = du/dy /(beta y)", True)]
    fig, axes = plt.subplots(2, 4, figsize=(21, 9))
    for r_i, (case, profile, y_0) in enumerate(CASES):
        for c_i, (fld, unit, zline) in enumerate(fields):
            ax = axes[r_i, c_i]
            for i, (label, gw, k4) in enumerate(configs):
                ax.plot(data[(case, label)]["y"] / 1e6, data[(case, label)][fld],
                        "-", lw=1.2, color=cmap[i], label=label)
            if zline:
                ax.axhline(0, color="grey", lw=0.5)
            ax.set_xlabel("y [Mm]")
            ax.set_ylabel(unit if fld == "ro" else f"{fld} [{unit}]")
            ax.set_title(f"{case}: {fld}")
            if c_i == 0:
                ax.legend(fontsize=7, title=vary_name)
    fig.suptitle(f"Full solutions (u, v, theta, Ro) across the {vary_name} sweep "
                 f"(ny={ny}, v_d={v_d}, 1000 d mean)")
    fig.tight_layout()
    fig.savefig(f"{SCRATCH}/{fname}.png", dpi=130)
    plt.close(fig)
    print(f"saved {fname} -> {SCRATCH}/{fname}.png")


def axisym_ku_test(ny=50, dt=900):
    """Confirm the v_d=0 superrotation source: sweep k_u (explicit eddy
    viscosity) and compare to the analytical AMC wind u_M=(beta/2)(y^2-y_asc^2).

    Hypothesis [derived]: k_u's down-gradient diffusion fills the AMC equatorial
    minimum, so u_eq grows with k_u. As k_u->0 the model should recover AMC
    (u_eq=0 on-eq; easterly off-eq). The original SS09 (Asselin time-filter, no
    spatial u-diffusion) follows AMC.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k_us = [0.0, 1e3, 1e4, 3e4, 1e5]
    cmap = [plt.cm.plasma(t) for t in np.linspace(0, 0.85, len(k_us))]
    # (label, profile, y_0, y_asc): y_asc = AMC ascent latitude (u_M=0 there)
    cases = [("on-eq", "sin2", 0.0, 0.0), ("off-eq", "SB08", 1e6, 1e6)]

    print(f"=== v_d=0 axisymmetric: u_eq vs k_u (ny={ny}, dt={dt}; "
          f"AMC u_eq=-(beta/2)y_asc^2) ===")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, (label, prof, y_0, y_asc) in zip(axes, cases):
        print(f"\n {label}: AMC u_eq = {-0.5 * BETA * y_asc ** 2:7.2f} m/s "
              f"(y_asc={y_asc / 1e6:.1f} Mm)")
        umax = 0.0
        for c, ku in zip(cmap, k_us):
            y, u, m, blew = run(prof, y_0, 0.0, 0.0, ny=ny, dt=dt, v_d=0.0, k_u=ku)
            umax = max(umax, float(np.max(np.abs(u))))
            print(f"   k_u={ku:8.0e} | u_eq={u[np.argmin(np.abs(y))]:7.2f} | "
                  f"jet={np.max(u):6.2f} | min u={np.min(u):6.2f}"
                  f"{'  BLEW' if blew else ''}")
            ax.plot(y / 1e6, u, "-", color=c, lw=1.3, label=f"k_u={ku:.0e}")
        # analytical AMC overlay, clipped to the model's u-range
        yy = np.linspace(y.min(), y.max(), 400)
        uM = 0.5 * BETA * (yy ** 2 - y_asc ** 2)
        uM = np.where(np.abs(uM) <= 1.3 * umax, uM, np.nan)
        ax.plot(yy / 1e6, uM, "k--", lw=2.2, label="AMC (beta/2)(y^2 - y_asc^2)")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{label}  (v_d=0, ny={ny})")
        ax.set_xlabel("y [Mm]"); ax.set_ylabel("u [m/s]"); ax.legend(fontsize=8)
    fig.suptitle(f"Axisymmetric limit (ny={ny}): explicit k_u down-gradient "
                 "diffusion fills the AMC equatorial minimum -> superrotation")
    fig.tight_layout()
    fname = f"{SCRATCH}/fig15_axisym_ku_ny{ny}.png"
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"\nsaved -> {fname}")


def rayleigh_sweep(ny=50, dt=900):
    """Reproduce SS09's Rayleigh-drag sensitivity in OUR model with k_u=0 (no
    explicit u-diffusion, as in SS09): sweep epsilon_u at v_d=0, overlay the
    analytical AMC wind. Should recover SS09 Figs 3-5: near-AMC for small drag,
    stronger circulation / departures for larger drag; equatorial easterlies
    off-equator.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eps = [1e-10, 1e-9, 1e-8, 1e-7]
    cmap = [plt.cm.viridis(t) for t in np.linspace(0, 0.85, len(eps))]
    cases = [("on-eq", "sin2", 0.0, 0.0), ("off-eq", "SB08", 1e6, 1e6)]

    print(f"=== SS09 Rayleigh-drag sensitivity, k_u=0, v_d=0 (ny={ny}, dt={dt}; "
          f"AMC u_eq=-(b/2)y_asc^2) ===")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for col, (label, prof, y_0, y_asc) in enumerate(cases):
        print(f"\n {label}: AMC u_eq = {-0.5 * BETA * y_asc ** 2:7.2f} m/s")
        umax = 0.0
        for c, e in zip(cmap, eps):
            y, u, m, blew = run(prof, y_0, 0.0, 0.0, ny=ny, dt=dt, v_d=0.0,
                                k_u=0.0, epsilon_u=e)
            v, _ = field_means(m)
            umax = max(umax, float(np.max(np.abs(u))))
            print(f"   eps_u={e:8.0e} | u_eq={u[np.argmin(np.abs(y))]:7.2f} | "
                  f"jet={np.max(u):6.2f} | max|v|={np.max(np.abs(v)):.3f}"
                  f"{'  BLEW' if blew else ''}")
            axes[0, col].plot(y / 1e6, u, "-", color=c, lw=1.3, label=f"eps={e:.0e}")
            axes[1, col].plot(y / 1e6, v, "-", color=c, lw=1.3, label=f"eps={e:.0e}")
        yy = np.linspace(y.min(), y.max(), 400)
        uM = 0.5 * BETA * (yy ** 2 - y_asc ** 2)
        uM = np.where(np.abs(uM) <= 1.3 * umax, uM, np.nan)
        axes[0, col].plot(yy / 1e6, uM, "k--", lw=2.2, label="AMC")
        for r, lab in [(0, "u [m/s]"), (1, "v [m/s]")]:
            axes[r, col].axhline(0, color="grey", lw=0.5)
            axes[r, col].set_xlabel("y [Mm]"); axes[r, col].set_ylabel(lab)
            axes[r, col].legend(fontsize=8)
        axes[0, col].set_title(f"{label}: zonal wind (k_u=0, ny={ny})")
        axes[1, col].set_title(f"{label}: meridional wind")

    fig.suptitle(f"SS09 Rayleigh-drag sensitivity reproduced (k_u=0, ny={ny}): "
                 "u tracks AMC; drag sets circulation strength")
    fig.tight_layout()
    fname = f"{SCRATCH}/fig16_rayleigh_sweep_ny{ny}.png"
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"\nsaved -> {fname}")


def production_ku(ny=200, dt=225, ndays=3000, v_d=2.5):
    """DECISIVE test: does the EMFD-driven equatorial superrotation / jet-creep
    that motivated the explicit k_u actually appear at k_u=0?

    Runs the production forcing (v_d=2.5, EMFD active) for on-eq (sin2, y_0=0)
    and off-eq (SB08, y_0=1e6) at k_u=0 vs k_u=1e5, tracking the equatorial u
    *time series* (the creep detector) and the final u(y) profile. Unlike the
    axisymmetric v_d=0 case there is no analytical AMC benchmark here (EMFD is
    on), so the decision criteria are behavioural:

      (a) k_u=0 is STEADY and non-superrotating (u_eq stays small, flat time
          series, no secular climb) -> clean rollback of k_u; align with SS09's
          drag-only scheme; the flank-mode (tanh/hyperdiff) work is moot.
      (b) k_u=0 SUPERROTATES or CREEPS (u_eq grows large-westerly and/or keeps
          climbing at end of run) -> the EMFD source is real and needs an
          AMC-preserving, non-diffusive stabilizer (Asselin-style *time*
          damping), NOT the spatial-diffusion k_u.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K_US = [0.0, 1e5]  # test (drag-only, SS09-like) vs current default
    colors = {0.0: "C3", 1e5: "k"}
    cases = [("on-eq", "sin2", 0.0), ("off-eq", "SB08", 1e6)]

    print(f"=== PRODUCTION v_d={v_d}: k_u=0 (SS09 drag-only) vs k_u=1e5 "
          f"(ny={ny}, dt={dt}, {ndays} d) ===")
    print("  creep slope = OLS d(u_eq)/dt over the last half of the run "
          "[m/s per 1000 d]\n")
    hdr = (f"{'case':6s} {'k_u':>6s} | {'u_eq':>7s} {'jet':>6s} {'min u':>6s} "
           f"{'flankSAW':>8s} | {'creep/1kd':>9s} {'':4s}")
    print(hdr); print("-" * len(hdr))

    data = {}
    for case, profile, y_0 in cases:
        for ku in K_US:
            y, u_mean, m, blew = run(profile, y_0, 0.0, 0.0, ny=ny, dt=dt,
                                     ndays=ndays, v_d=v_d, k_u=ku)
            eqidx = int(np.argmin(np.abs(y)))
            u_eq_t = m.results.u[:, eqidx]          # daily-mean equatorial u(t)
            days = np.arange(u_eq_t.size)
            half = u_eq_t.size // 2
            slope = np.polyfit(days[half:], u_eq_t[half:], 1)[0] * 1000.0
            saw = cu.flank_mode(y, u_mean, flank_min_m=FLANK_MIN)["sawtooth_max"]
            data[(case, ku)] = dict(y=y, u=u_mean, u_eq_t=u_eq_t, slope=slope,
                                    blew=blew)
            print(f"{case:6s} {ku:6.0e} | {u_eq_t[-1]:7.2f} {np.max(u_mean):6.2f} "
                  f"{np.min(u_mean):6.2f} {saw:8.3f} | {slope:9.3f} "
                  f"{'BLEW' if blew else '':4s}")
        print()

    # top: equatorial u(t) creep detector; bottom: final u(y) profile
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for c_i, (case, profile, y_0) in enumerate(cases):
        for ku in K_US:
            r = data[(case, ku)]
            axes[0, c_i].plot(np.arange(r["u_eq_t"].size), r["u_eq_t"], "-",
                              color=colors[ku], lw=1.3, label=f"k_u={ku:.0e}")
            axes[1, c_i].plot(r["y"] / 1e6, r["u"], "-", color=colors[ku],
                              lw=1.3, label=f"k_u={ku:.0e}")
        axes[0, c_i].axhline(0, color="grey", lw=0.5)
        axes[0, c_i].set_title(f"{case}: equatorial u(t)  (creep detector)")
        axes[0, c_i].set_xlabel("day"); axes[0, c_i].set_ylabel("u_eq [m/s]")
        axes[0, c_i].legend(fontsize=8)
        axes[1, c_i].axhline(0, color="grey", lw=0.5)
        axes[1, c_i].set_title(f"{case}: final u(y)  ({ndays} d, last-200-d mean)")
        axes[1, c_i].set_xlabel("y [Mm]"); axes[1, c_i].set_ylabel("u [m/s]")
        axes[1, c_i].legend(fontsize=8)
    fig.suptitle(f"Production v_d={v_d}, ny={ny}: does k_u=0 superrotate / creep "
                 "at the equator?  (k_u=1e5 = current default)")
    fig.tight_layout()
    fname = f"{SCRATCH}/fig17_production_ku_ny{ny}.png"
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"saved -> {fname}")


def axisym_ku4_clean(ny=200, dt=225, ndays=500):
    """AMC-safety of biharmonic hyperdiffusion, ISOLATED (k_u=0).

    The earlier axisym_ku4 sweep left k_u=1e5 on as backdrop, so it never tested
    k_u4 alone. Here k_u=0, v_d=0: sweep k_u4 and overlay the analytical AMC wind
    u_M=(beta/2)(y^2-y_asc^2). Prediction [derived]: biharmonic diffusion vanishes
    on the AMC parabola (d^4/dy^4 of a quadratic = 0), so every k_u4 should track
    AMC (u_eq=0 on-eq), UNLIKE k_u (Laplacian d^2u=beta != 0 fills the minimum).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k_u4s = [0.0, 1e16, 1e17]  # all dt=225-stable at ny=200 (1e18 needs dt=72)
    cmap = [plt.cm.plasma(t) for t in np.linspace(0, 0.85, len(k_u4s))]
    cases = [("on-eq", "sin2", 0.0, 0.0), ("off-eq", "SB08", 1e6, 1e6)]

    print(f"=== v_d=0 axisymmetric, k_u=0: u_eq vs k_u4 (ny={ny}); "
          f"AMC u_eq=-(beta/2)y_asc^2 ===")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, (label, prof, y_0, y_asc) in zip(axes, cases):
        print(f"\n {label}: AMC u_eq = {-0.5 * BETA * y_asc ** 2:7.2f} m/s "
              f"(y_asc={y_asc / 1e6:.1f} Mm)")
        umax = 0.0
        for c, k4 in zip(cmap, k_u4s):
            d = stable_dt(k4, ny, dt)
            y, u, m, blew = run(prof, y_0, 0.0, k4, ny=ny, dt=d, ndays=ndays,
                                v_d=0.0, k_u=0.0)
            umax = max(umax, float(np.max(np.abs(u))))
            saw = cu.flank_mode(y, u, flank_min_m=FLANK_MIN)["sawtooth_max"]
            print(f"   k_u4={k4:8.0e} dt={d:4d} | u_eq={u[np.argmin(np.abs(y))]:7.2f} | "
                  f"jet={np.max(u):6.2f} | flankSAW={saw:7.3f}"
                  f"{'  BLEW' if blew else ''}")
            ax.plot(y / 1e6, u, "-", color=c, lw=1.3, label=f"k_u4={k4:.0e}")
        yy = np.linspace(y.min(), y.max(), 400)
        uM = 0.5 * BETA * (yy ** 2 - y_asc ** 2)
        uM = np.where(np.abs(uM) <= 1.3 * umax, uM, np.nan)
        ax.plot(yy / 1e6, uM, "k--", lw=2.2, label="AMC")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{label}  (v_d=0, k_u=0, ny={ny})")
        ax.set_xlabel("y [Mm]"); ax.set_ylabel("u [m/s]"); ax.legend(fontsize=8)
    fig.suptitle(f"Does biharmonic hyperdiffusion preserve AMC? (k_u=0, ny={ny}): "
                 "d^4/dy^4 of the AMC parabola = 0 -> should track AMC")
    fig.tight_layout()
    fname = f"{SCRATCH}/fig18_axisym_ku4_clean_ny{ny}.png"
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"\nsaved -> {fname}")


def production_ku4(ny=200, dt=225, ndays=500, v_d=2.5):
    """Does biharmonic hyperdiffusion tame the k_u=0 EMFD flank mode while
    preserving the core climate? Production v_d=2.5, k_u=0, sweep k_u4. Companion
    to axisym_ku4_clean (the AMC side). k_u4=0 is the ~345 m/s flank catastrophe.
    Plot y-limited to the climate range so the mitigated curves are legible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k_u4s = [0.0, 1e16, 1e17]  # all dt=225-stable at ny=200 (1e18 needs dt=72)
    cmap = [plt.cm.viridis(t) for t in np.linspace(0, 0.85, len(k_u4s))]
    cases = [("on-eq", "sin2", 0.0), ("off-eq", "SB08", 1e6)]

    print(f"=== production v_d={v_d}, k_u=0: flank mode vs k_u4 "
          f"(ny={ny}, {ndays} d) ===")
    hdr = (f"{'k_u4':>8s} {'case':6s} {'dt':>4s} | {'u_eq':>7s} {'core jet':>8s} "
           f"{'flankSAW':>8s} {'':4s}")
    print(hdr); print("-" * len(hdr))
    data = {}
    for k4 in k_u4s:
        for case, profile, y_0 in cases:
            d = stable_dt(k4, ny, dt)
            y, u, m, blew = run(profile, y_0, 0.0, k4, ny=ny, dt=d, ndays=ndays,
                                v_d=v_d, k_u=0.0)
            core = np.abs(y) < FLANK_MIN
            saw = cu.flank_mode(y, u, flank_min_m=FLANK_MIN)["sawtooth_max"]
            data[(k4, case)] = dict(y=y, u=u)
            print(f"{k4:8.0e} {case:6s} {d:4d} | {u[np.argmin(np.abs(y))]:7.2f} "
                  f"{np.max(u[core]):8.2f} {saw:8.3f} {'BLEW' if blew else '':4s}")
        print()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, (case, profile, y_0) in zip(axes, cases):
        for c, k4 in zip(cmap, k_u4s):
            r = data[(k4, case)]
            ax.plot(r["y"] / 1e6, r["u"], "-", color=c, lw=1.3,
                    label=f"k_u4={k4:.0e}")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylim(-80, 90)  # k_u4=0 spikes to ~320 clip off-scale (intended)
        ax.set_title(f"{case}  (v_d={v_d}, k_u=0, ny={ny})")
        ax.set_xlabel("y [Mm]"); ax.set_ylabel("u [m/s]"); ax.legend(fontsize=8)
    fig.suptitle(f"Biharmonic hyperdiffusion vs the k_u=0 EMFD flank mode "
                 f"(v_d={v_d}, ny={ny}): k_u4=0 spikes to ~320 m/s (off-scale)")
    fig.tight_layout()
    fname = f"{SCRATCH}/fig19_production_ku4_ny{ny}.png"
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"\nsaved -> {fname}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "matrix"
    if mode == "matrix":
        run_matrix()
    elif mode == "axisym_ku":
        axisym_ku_test()
    elif mode == "axisym_ku_ny200":
        axisym_ku_test(ny=200, dt=225)
    elif mode == "rayleigh_sweep":
        rayleigh_sweep()
    elif mode == "rayleigh_sweep_ny200":
        rayleigh_sweep(ny=200, dt=225)
    elif mode == "production_ku":  # DECISIVE: v_d=2.5, k_u=0 vs 1e5, creep check
        production_ku()
    elif mode == "production_ku_short":  # quick timing/smoke check (500 d)
        production_ku(ndays=500)
    elif mode == "axisym_ku4_clean":  # AMC-safety of hyperdiff, ISOLATED (k_u=0)
        axisym_ku4_clean()
    elif mode == "production_ku4":  # does hyperdiff tame the k_u=0 flank mode?
        production_ku4()
    elif mode == "vtheta":
        run_field_matrix()
    elif mode == "prof_uw":
        cfgs = [(f"u_w={v:g}", float(v), 0.0) for v in [0, 2, 5, 10, 20, 40]]
        _profiles_figure(cfgs, "u_w [m/s]", "fig9_prof_uw")
    elif mode == "prof_ku4":
        cfgs = [(f"k4={v:.0e}", 0.0, float(v))
                for v in [0, 1e16, 3e16, 1e17, 3e17, 1e18]]
        _profiles_figure(cfgs, "k_u4 [m^4/s]", "fig10_prof_ku4")
    elif mode == "prof_uw_ny200":
        cfgs = [(f"u_w={v:g}", float(v), 0.0) for v in [0, 10, 40]]
        _profiles_figure(cfgs, "u_w [m/s]", "fig11_prof_uw_ny200",
                         ny=200, base_dt=225)
    elif mode == "prof_ku4_ny200":
        cfgs = [(f"k4={v:.0e}", 0.0, float(v)) for v in [0, 1e17, 1e18]]
        _profiles_figure(cfgs, "k_u4 [m^4/s]", "fig12_prof_ku4_ny200",
                         ny=200, base_dt=225)
    elif mode == "axisym_uw":  # tanh sweep in the v_d=0 axisymmetric limit
        cfgs = [(f"u_w={v:g}", float(v), 0.0) for v in [0, 5, 10, 20, 40]]
        _profiles_figure(cfgs, "u_w [m/s]", "fig13_axisym_uw", v_d=0.0)
    elif mode == "axisym_ku4":  # hyperdiff sweep in the v_d=0 axisymmetric limit
        cfgs = [(f"k4={v:.0e}", 0.0, float(v)) for v in [0, 1e16, 1e17, 1e18]]
        _profiles_figure(cfgs, "k_u4 [m^4/s]", "fig14_axisym_ku4", v_d=0.0)
    elif mode == "sweep_uw":
        vals = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]  # u_w [m/s]; 0 = hard gate
        data = _sweep(vals, gate_of=lambda v: v, k4_of=lambda v: 0.0,
                      vary_name="u_w [m/s]")
        _plot_sweep(data, vals, "u_w [m/s]", "fig7_sweep_uw", logx=False)
    elif mode == "sweep_ku4":
        vals = [0.0, 1e16, 3e16, 1e17, 3e17, 1e18]  # k_u4 [m^4/s]; 0 = off
        data = _sweep(vals, gate_of=lambda v: 0.0, k4_of=lambda v: v,
                      vary_name="k_u4 [m^4/s]")
        _plot_sweep(data, vals, "k_u4 [m^4/s]", "fig8_sweep_ku4", logx=True)
    elif mode == "plots":
        _make_plots(_load_npz())  # regenerate figures from the saved ny=50 npz
        print(f"regenerated figures -> {SCRATCH}/")
    elif mode == "subset":
        # winner from the ny=50 matrix: hyperdiffusion alone (hard gate, k4=1e17)
        winners = [(CONFIGS[2][0], 0.0, K_U4)]
        run_subset(winners)
    print("\ndone.")


if __name__ == "__main__":
    main()
