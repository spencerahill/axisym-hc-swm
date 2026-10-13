"""V1 verification figure: production staggered vs patch vs collocated B1.

Two panels the claim can be checked at a glance:
  (left) interior v: production (staggered) sits on the gateless noise floor
         while the collocated B1 baseline carries the standing 2*dy ripple;
  (right) whole-profile u overlay, production on top of the patch and B1, with
          the notch and jets.
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
SUITE = pathlib.Path("model_output/formulation_suite")
PROD = SUITE / "staggered_v_prod" / "output.nc"
PATCH = SUITE / "mc_stencil" / "staggered_v" / "output.nc"
B1 = SUITE / "mc_stencil" / "b1_y0p0000_gateon_mc" / "output.nc"


def main():
    y, u, v, T = load_centered(str(PROD), ndays=100)
    yp, up, vp, Tp = load_centered(str(PATCH), ndays=100)
    yb, ub, vb, Tb = load_centered(str(B1), ndays=1825)

    fig, (axv, axu) = plt.subplots(1, 2, figsize=(12, 4.5))

    # left: grid-scale v sawtooth, production (floor) vs B1 (ripple).
    # Raw v hides the ~1e-3 ripple against the ~0.4 m/s curve, so plot the
    # sawtooth |v_i - 0.5(v_{i-1}+v_{i+1})| that isolates the 2*dy component.
    band = np.abs(y) < 8 * Mm
    bandb = np.abs(yb) < 8 * Mm
    axv.semilogy(yb[bandb] / Mm, sawtooth(vb)[bandb], color="0.6", lw=0.8,
                 label="collocated B1 (ripple)")
    axv.semilogy(y[band] / Mm, sawtooth(v)[band], color="C0", lw=1.0,
                 label="production staggered")
    axv.axhline(8.8e-5, color="C2", lw=0.8, ls=":", label="gateless floor")
    axv.set_xlabel("y (Mm)")
    axv.set_ylabel("sawtooth(v) (m/s)")
    axv.set_title("grid-scale v ripple removed (24-91x, to the floor)")
    axv.legend(fontsize=8)

    # right: whole-profile u overlay
    axu.plot(yb / Mm, ub, color="0.6", lw=1.5, label="collocated B1")
    axu.plot(yp / Mm, up, color="C1", lw=1.0, ls="--", label="patch staggered")
    axu.plot(y / Mm, u, color="C0", lw=1.0, label="production staggered")
    axu.set_xlabel("y (Mm)")
    axu.set_ylabel("u (m/s)")
    axu.set_title("u profile: production on the patch, notch + jets")
    axu.legend(fontsize=8)
    axu.axhline(0, color="k", lw=0.4)

    fig.tight_layout()
    outpath = PROD.parent / "v1_figure.png"
    fig.savefig(outpath, dpi=130)
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
