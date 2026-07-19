"""One-panel synthesis of the Moist V1 scans in the (W_q, W*) plane.

Every scanned run is placed at x = W_q = W_c + tau_c E_0 (the quiescent
plateau the moist parameters set) and y = W* = Shat/(L_v(2a-1)) (the
crossover W the stratification sets). Because the equilibrium W field is
within ~0.01-1.4 kg/m^2 of W_q everywhere, the sign of the gross moist
stability Hhat = Shat - L_v(2a-1)W is decided by which side of the diagonal
W_q = W* a run sits on: one boundary, four dials (W_c and tau_c move x,
Delta_z moves y, D moves neither). Markers distinguish the scans; fills give
the measured regime. The D ladder (D = 0, 1, 2e6) collapses onto the shared
center point (all its runs sit at the default W_q, W*).

Usage:
    python scripts/moist_v1_regime_figure.py [out_png]

Reads the standard run locations (model_output/moist_v1_validation,
moist_v1_wc_scan, moist_v1_tauc_dz); writes out_png (default
model_output/moist_v1_tauc_dz/moist_v1_regime_plane.png).
"""

import sys
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moist_v1_analysis import load_equilibrium

VAL = "model_output/moist_v1_validation"
WCS = "model_output/moist_v1_wc_scan"
TDZ = "model_output/moist_v1_tauc_dz"

GROUPS = (
    ("$W_c$ scan (35-55)", "o",
     [f"{WCS}/wc35/out.nc", f"{WCS}/wc40/out.nc", f"{WCS}/wc44/out.nc",
      f"{VAL}/D1/out.nc", f"{WCS}/wc55/out.nc"]),
    (r"$\tau_c$ ladder (0.5 h-2 d)", "s",
     [f"{TDZ}/tc1800/out.nc", f"{TDZ}/tc7200/out.nc", f"{VAL}/D1/out.nc",
      f"{TDZ}/tc57600/out.nc", f"{TDZ}/tc172800/out.nc"]),
    (r"$W_c$=40 $\tau_c$ pair", "^",
     [f"{TDZ}/wc40tc1800/out.nc", f"{TDZ}/wc40tc172800/out.nc"]),
    (r"$\Delta_z$ ladder (30-90 K)", "D",
     [f"{TDZ}/dz30/out.nc", f"{TDZ}/dz45/out.nc", f"{VAL}/D1/out.nc",
      f"{TDZ}/dz68p2/out.nc", f"{TDZ}/dz75/out.nc", f"{TDZ}/dz90/out.nc"]),
)

C_NEG, C_POS = "#CC6677", "#0077BB"


def fill_of(r):
    frac = float(np.mean(r["hhat"] < 0))
    return C_NEG if frac == 1.0 else C_POS if frac == 0.0 else "none"


def main():
    out_png = (sys.argv[1] if len(sys.argv) > 1
               else os.path.join(TDZ, "moist_v1_regime_plane.png"))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    lo, hi = 20.0, 70.0
    diag = np.array([lo, hi])
    ax.fill_between(diag, diag, hi, color=C_POS, alpha=0.06)
    ax.fill_between(diag, lo, diag, color=C_NEG, alpha=0.06)
    ax.plot(diag, diag, "-", c="k", lw=1.5)
    ax.annotate(r"$\hat H > 0$ (dry-side): mean flow exports MSE from ITCZ",
                xy=(0.04, 0.93), xycoords="axes fraction", fontsize=9,
                color=C_POS)
    ax.annotate(r"$\hat H < 0$ (moisture-mode): mean MSE flux up-gradient",
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=9,
                color=C_NEG)
    ax.annotate(r"$W_q = W^*$", xy=(66.3, 67.3), fontsize=9, rotation=45,
                ha="center", va="center")

    for _, marker, paths in GROUPS:
        for p in paths:
            r = load_equilibrium(p)
            ax.plot(r["w_plateau"], r["w_hhat_zero"], marker, mfc=fill_of(r),
                    mec="k", ms=8, zorder=3)

    ax.annotate(r"$\Delta_z$=30", xy=(50.66, 22.29), xytext=(8, 0),
                textcoords="offset points", fontsize=8, va="center")
    ax.annotate(r"$\Delta_z$=90", xy=(50.66, 66.86), xytext=(8, 0),
                textcoords="offset points", fontsize=8, va="center")
    ax.annotate(r"$W_c$=35", xy=(35.66, 44.57), xytext=(-2, 8),
                textcoords="offset points", fontsize=8, ha="center")
    ax.annotate(r"$\tau_c$=2 d", xy=(57.95, 44.57), xytext=(2, 8),
                textcoords="offset points", fontsize=8, ha="center")
    ax.annotate(r"$W_c$=40 pair", xy=(40.08, 44.57), xytext=(-2, -14),
                textcoords="offset points", fontsize=8, ha="center")

    handles = [Line2D([], [], marker=m, ls="", mfc="#BBBBBB", mec="k", ms=8,
                      label=lab) for lab, m, _ in GROUPS]
    handles += [
        Line2D([], [], marker="o", ls="", mfc=C_NEG, mec="k", ms=8,
               label=r"run: $\hat H<0$ everywhere"),
        Line2D([], [], marker="o", ls="", mfc=C_POS, mec="k", ms=8,
               label=r"run: $\hat H>0$ everywhere"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="k", ms=8,
               label="run: mixed sign"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.set_xlim(lo + 13, hi - 7)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$W_q = W_c + \tau_c E_0$ (kg m$^{-2}$): "
                  "set by the moist closure")
    ax.set_ylabel(r"$W^* = \hat S/(L_v(2a{-}1))$ (kg m$^{-2}$): "
                  "set by the stratification")
    ax.set_title("All Moist V1 scans cross one GMS boundary")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
