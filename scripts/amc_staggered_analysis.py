"""AMC (v_d=0) verdict for the staggered-v patch, vs the vd0_vert reference.

Builds a center-grid climatology npz from the staggered run (v averaged
from faces back to centers, the field u actually feels), evaluates the
pre-registered targets against the validated collocated vd0_vert
reference and the analytical AMC parabola u_amc = beta*y^2/2, and prints
PASS/FAIL per criterion. The standard anchor table then comes from
anchor_compare on the saved npz.

Pre-registered targets (task #20, set before the run):
  |u_eq| <= 0.05 (ref exactly 0.0; equatorial superrotation is the
  historical failure mode); jet within 1% of 49.94 and position within
  0.5 Mm; v_absmax within 5% of 0.0408; |dT_eq| <= 0.5 K; exponent
  within 0.1 of 1.926; interior (|y|<=8 Mm) max|du| <= 0.5 m/s; parity
  <= 1e-6; |drift| <= 0.05 per 500 d; no NaN; no super-AMC (report
  max(u - u_amc) inside the cell).

Usage:
    python scripts/amc_staggered_analysis.py [--run DIR]
"""
import argparse
import pathlib
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from anchor_compare import M_PER_DEG, near_eq_powerlaw  # noqa: E402
from cmp_utils import sawtooth  # noqa: E402

Mm = 1e6
BETA = 2e-11
DAYS = 1825
RUN = pathlib.Path(
    "model_output/formulation_suite/mc_stencil/amc_staggered_vd0")
REF = pathlib.Path("model_output/validation_20260709/runs/vd0_vert")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=str(RUN))
    p.add_argument("--fig", action="store_true",
                   help="also save the u overlay / difference figure")
    args = p.parse_args()
    run = pathlib.Path(args.run)

    ds = xr.open_dataset(run / "output.nc", decode_timedelta=False)
    nt = ds.sizes["time"]
    avg = slice(nt - min(nt, DAYS), nt)
    y = ds["y"].values
    u = ds["u"].values[avg].mean(axis=0)
    T = ds["T"].values[avg].mean(axis=0)
    f = ds["v"].values[avg].mean(axis=0)  # faces, padding at [-1]
    umax_t = np.abs(ds["u"].values).max(axis=1)
    nan_days = int(np.isnan(ds["u"].values).any(axis=1).sum())
    ds.close()

    # v back to centers (the average u feels); ends are wall zeros
    vc = np.zeros_like(f)
    vc[1:-1] = 0.5 * (f[:-2] + f[1:-1])
    np.savez(run / "climatology_centers.npz", y=y, u=u, v=vc, T=T)

    ref = np.load(REF / "climatology.npz")
    yr, ur, vr, Tr = ref["y"], ref["u"], ref["v"], ref["T"]

    ieq = int(np.argmin(np.abs(y)))
    checks = []

    def check(name, value, ok, detail=""):
        checks.append((name, value, bool(ok), detail))

    check("|u_eq| <= 0.05 m/s", f"{u[ieq]:+.6f}", abs(u[ieq]) <= 0.05,
          "ref 0.0 exactly; superrotation = failure mode")
    ij, ijr = int(np.argmax(u)), int(np.argmax(ur))
    check("jet within 1%", f"{u[ij]:.3f} vs {ur[ijr]:.3f}",
          abs(u[ij] / ur[ijr] - 1) <= 0.01,
          f"dev {100 * (u[ij] / ur[ijr] - 1):+.2f}%")
    check("jet position within 0.5 Mm",
          f"{y[ij] / Mm:+.2f} vs {yr[ijr] / Mm:+.2f} Mm",
          abs(abs(y[ij]) - abs(yr[ijr])) <= 0.5 * Mm)
    vmax, vmaxr = np.abs(vc).max(), np.abs(vr).max()
    check("v_absmax within 5%", f"{vmax:.5f} vs {vmaxr:.5f}",
          abs(vmax / vmaxr - 1) <= 0.05,
          f"dev {100 * (vmax / vmaxr - 1):+.2f}% (centers-reconstructed)")
    check("|dT_eq| <= 0.5 K", f"{T[ieq]:.3f} vs {Tr[int(np.argmin(np.abs(yr)))]:.3f}",
          abs(T[ieq] - Tr[int(np.argmin(np.abs(yr)))]) <= 0.5)
    pt, _ = near_eq_powerlaw(y / M_PER_DEG, u)
    check("exponent within 0.1 of 1.926", f"{pt:.3f}",
          abs(pt - 1.9262) <= 0.1, "AMC parabola regime")
    inter = np.abs(y) <= 8 * Mm
    du = u - np.interp(y, yr, ur)
    check("interior max|du| <= 0.5 m/s", f"{np.abs(du[inter]).max():.4f}",
          np.abs(du[inter]).max() <= 0.5,
          f"at {y[inter][int(np.argmax(np.abs(du[inter])))] / Mm:+.2f} Mm")
    asym = np.max(np.abs(u - u[::-1]))
    check("parity max|u(y)-u(-y)| <= 1e-6", f"{asym:.3g}", asym <= 1e-6)
    nd = min(500, nt - 1)
    drift = float(umax_t[-1] - umax_t[-1 - nd])
    check("|drift| <= 0.05 per 500 d", f"{drift:+.5f}", abs(drift) <= 0.05)
    check("no NaN days", str(nan_days), nan_days == 0)

    print("=== AMC v_d=0 staggered test, pre-registered criteria ===")
    npass = 0
    for name, value, ok, detail in checks:
        tag = "PASS" if ok else "FAIL"
        npass += ok
        print(f"[{tag}] {name:>34}: {value}  {detail}")
    print(f"--- {npass}/{len(checks)} criteria pass")

    # AMC parabola report (no pre-set numeric limit; super-AMC = red flag)
    u_amc = 0.5 * BETA * y ** 2
    excess = u - u_amc
    cell = np.abs(y) <= 9.439 * Mm  # inside the forcing width y1
    k = int(np.argmax(excess[cell]))
    print("\n=== AMC parabola comparison (report) ===")
    print(f"max(u - beta*y^2/2) inside |y|<=y1: {excess[cell].max():+.4f} "
          f"m/s at {y[cell][k] / Mm:+.2f} Mm")
    print(f"ref (collocated) same metric:       "
          f"{(ur - 0.5 * BETA * yr ** 2)[np.abs(yr) <= 9.439 * Mm].max():+.4f} m/s")
    print(f"u_min (easterlies; ref -3e-8): {u.min():+.3g} m/s")
    su, sv = sawtooth(u), sawtooth(vc)
    print(f"sawtooth u max {np.nanmax(su):.4f} (ref 0.0771); "
          f"v max {np.nanmax(sv):.3g} (ref 4.07e-5)")
    print(f"\nnpz saved: {run / 'climatology_centers.npz'}")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gray, orange, purple = "#52514e", "#bf7300", "#4a3aa7"
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        ax = axes[0]
        ax.plot(yr / Mm, ur, color=gray, lw=2.4,
                label="collocated reference (vd0_vert)")
        ax.plot(y / Mm, u, color=orange, lw=1.4, ls="--",
                label="staggered-v, cold 15-yr")
        ax.plot(y / Mm, u_amc, color="#3d3d3a", lw=1.0, ls=":",
                label="AMC parabola beta*y^2/2")
        ax.set_ylim(-2, 60)
        ax.set_xlabel("y [Mm]")
        ax.set_ylabel("u [m/s]")
        ax.set_title("time-mean u at v_d = 0: staggered on the reference,\n"
                     "both under the AMC parabola")
        ax.legend(fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, color="#e1e0d9", linewidth=0.7)
        ax.set_axisbelow(True)

        ax = axes[1]
        ax.plot(y / Mm, du, color=purple, lw=1.4)
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        k2 = int(np.argmax(np.abs(du)))
        ax.annotate(f"max|du| = {np.abs(du).max():.3f} m/s "
                    f"at {y[k2] / Mm:+.1f} Mm",
                    xy=(0.03, 0.95), xycoords="axes fraction", va="top",
                    fontsize=8, color="#3d3d3a")
        ax.set_xlabel("y [Mm]")
        ax.set_ylabel("u difference [m/s]")
        ax.set_title("staggered minus collocated reference")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, color="#e1e0d9", linewidth=0.7)
        ax.set_axisbelow(True)
        fig.suptitle("AMC limit check (v_d = 0, 15 yr, ny=801)", y=1.02)
        fig.tight_layout()
        fig.savefig(run / "fig_amc_check.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"figure saved: {run / 'fig_amc_check.png'}")

        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
        for row, (name, ft, fr, unit) in enumerate(
            [("v", vc, vr, "m/s"), ("T", T, Tr, "K")]
        ):
            ax = axes[row, 0]
            ax.plot(yr / Mm, fr, color=gray, lw=2.4,
                    label="collocated reference")
            ax.plot(y / Mm, ft, color=orange, lw=1.4, ls="--",
                    label="staggered-v (centers)")
            ax.set_xlabel("y [Mm]")
            ax.set_ylabel(f"{name} [{unit}]")
            ax.set_title(f"time-mean {name}")
            ax.legend(fontsize=8)
            ax = axes[row, 1]
            dd = ft - np.interp(y, yr, fr)
            ax.plot(y / Mm, dd, color=purple, lw=1.2)
            ax.axhline(0, color="#c3c2b7", lw=0.8)
            kk = int(np.argmax(np.abs(dd)))
            ax.annotate(f"max|d{name}| = {np.abs(dd).max():.2e} {unit} "
                        f"at {y[kk] / Mm:+.1f} Mm",
                        xy=(0.03, 0.95), xycoords="axes fraction",
                        va="top", fontsize=8, color="#3d3d3a")
            ax.set_xlabel("y [Mm]")
            ax.set_ylabel(f"{name} difference [{unit}]")
            ax.set_title("staggered minus reference")
        for ax in axes.flat:
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(True, color="#e1e0d9", linewidth=0.7)
            ax.set_axisbelow(True)
        fig.suptitle("AMC limit check, v and T profiles (v_d = 0)", y=1.0)
        fig.tight_layout()
        fig.savefig(run / "fig_amc_check_vT.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"figure saved: {run / 'fig_amc_check_vT.png'}")


if __name__ == "__main__":
    main()
