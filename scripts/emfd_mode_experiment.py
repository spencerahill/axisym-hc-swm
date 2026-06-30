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


def run(profile, y_0, gate_width, k_u4, ny=50, dt=900, ndays=1000):
    """Run one config; return (y, u_mean over last 200 days, final SWModel)."""
    cfg = SWConfig(
        total_integration_days=ndays, ny=ny, dt=dt, k_u=1e5,
        emfd_gate_width=gate_width, k_u4=k_u4,
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


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "matrix"
    if mode == "matrix":
        run_matrix()
    elif mode == "vtheta":
        run_field_matrix()
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
