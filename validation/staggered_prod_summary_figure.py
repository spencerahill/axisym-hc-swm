"""Validation summary: the production staggered model across the (v_d, y_0) box.

Left: u(y) for the four validated regimes (AMC v_d=0, weak-eddy v_d=0.125,
equinoctial v_d=2.5, off-equatorial armA), spanning jet 28-50 m/s.
Right: grid-scale v sawtooth for each, all sitting on the gateless noise floor
(the collocated equinoctial ripple, 24-91x higher, shown grey for scale).
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ss09.read_output import load_centered  # noqa: E402
from cmp_utils import sawtooth  # noqa: E402

Mm = 1e6
BASE = pathlib.Path("model_output/formulation_suite/staggered_v_prod")
B1 = pathlib.Path("model_output/formulation_suite/mc_stencil/b1_y0p0000_gateon_mc")
RUNS = [
    ("AMC (v_d=0)", BASE / "v2_amc" / "output.nc", 1825, "C3"),
    ("weak-eddy (v_d=0.125)", BASE / "v3_vd0125" / "output.nc", 300, "C2"),
    ("equinoctial (v_d=2.5)", BASE / "output.nc", 100, "C0"),
    ("off-equatorial (armA)", BASE / "v4_armA" / "output.nc", 300, "C1"),
]


def main():
    fig, (axu, axs) = plt.subplots(1, 2, figsize=(12, 4.6))

    for label, path, nd, c in RUNS:
        y, u, v, T = load_centered(str(path), ndays=nd)
        axu.plot(y / Mm, u, color=c, lw=1.1, label=label)
        band = np.abs(y) < 8 * Mm
        axs.semilogy(y[band] / Mm, sawtooth(v)[band], color=c, lw=0.9, label=label)

    # collocated equinoctial ripple, for scale
    yb, ub, vb, Tb = load_centered(str(B1 / "output.nc"), ndays=1825)
    bb = np.abs(yb) < 8 * Mm
    axs.semilogy(yb[bb] / Mm, sawtooth(vb)[bb], color="0.6", lw=0.8,
                 label="collocated v_d=2.5 (ripple)")
    axs.axhline(8.8e-5, color="k", lw=0.7, ls=":", label="gateless floor")

    axu.set_xlabel("y (Mm)"); axu.set_ylabel("u (m/s)")
    axu.set_title("u profile across the validated (v_d, y_0) box")
    axu.axhline(0, color="k", lw=0.4); axu.legend(fontsize=8)

    axs.set_xlabel("y (Mm)"); axs.set_ylabel("sawtooth(v) (m/s)")
    axs.set_title("grid-scale v ripple: on the floor in every regime")
    axs.legend(fontsize=7)

    fig.tight_layout()
    out = BASE / "validation_summary.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
