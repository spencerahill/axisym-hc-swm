"""V5 figure: notch and ripple convergence, ny=801 vs ny=1601 (staggered mc).

Three panels:
  (a) whole-domain u(y) overlay - the two resolutions coincide except at the
      terminus, confirming no new domain-wide artifact at finer resolution;
  (b) terminus-notch zoom - depth/width convergence;
  (c) banded max sawtooth(v) on faces vs the gateless noise floor - the ripple
      stays on the floor at doubled resolution.

Usage:
    python scripts/v5_figure.py [--test DIR] [--ref DIR] [--days N] [--out PNG]
"""
import argparse
import pathlib
import sys

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ss09.read_output import load_centered  # noqa: E402
from cmp_utils import sawtooth  # noqa: E402

Mm = 1e6
BANDS = [(0, 2), (2, 5), (5, 8)]
GATELESS_FLOOR = 8.8e-5
SUITE = pathlib.Path("model_output/formulation_suite/staggered_v_prod")


def load_u(run, days):
    y, u, _, _ = load_centered(str(pathlib.Path(run) / "output.nc"), ndays=days)
    return y, u


def ripple(run, days):
    ds = xr.open_dataset(pathlib.Path(run) / "output.nc", decode_timedelta=False)
    y = ds["y"].values
    yf = 0.5 * (y[:-1] + y[1:])
    vf = ds["v"].values[-days:].mean(axis=0)
    ds.close()
    sv = sawtooth(vf)
    return [float(np.nanmax(np.where(
        (np.abs(yf) >= a * Mm) & (np.abs(yf) < b * Mm), sv, np.nan)))
        for a, b in BANDS]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", default=str(SUITE / "v5_ny1601"))
    p.add_argument("--ref", default=str(SUITE))
    p.add_argument("--days", type=int, default=100)
    p.add_argument("--out", default=str(SUITE / "v5_ny1601/v5_figure.png"))
    args = p.parse_args()

    yr, ur = load_u(args.ref, args.days)
    yt, ut = load_u(args.test, args.days)
    nyr, nyt = yr.size, yt.size
    rr, rt = ripple(args.ref, args.days), ripple(args.test, args.days)

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.2))

    a.plot(yr / Mm, ur, lw=1.4, label=f"ny={nyr}", color="#1b9e77")
    a.plot(yt / Mm, ut, lw=0.9, label=f"ny={nyt}", color="#7570b3")
    a.axhline(0, color="0.7", lw=0.6)
    a.set_xlabel("y (Mm)")
    a.set_ylabel("u (m/s)")
    a.set_title("(a) whole-domain u: overlap except at terminus")
    a.legend(frameon=False)

    for y, u, ny, col, lw in ((yr, ur, nyr, "#1b9e77", 1.8),
                              (yt, ut, nyt, "#7570b3", 1.2)):
        m = (y >= 7.5 * Mm) & (y <= 10.0 * Mm)
        b.plot(y[m] / Mm, u[m], "-o", ms=3, lw=lw, color=col,
               label=f"ny={ny}")
    b.axhline(0, color="0.7", lw=0.6)
    b.axhline(-5, color="0.8", lw=0.6, ls="--")
    b.set_xlabel("y (Mm)")
    b.set_ylabel("u (m/s)")
    b.set_title("(b) NH terminus notch")
    b.legend(frameon=False)

    x = np.arange(len(BANDS))
    w = 0.38
    c.bar(x - w / 2, rr, w, label=f"ny={nyr}", color="#1b9e77")
    c.bar(x + w / 2, rt, w, label=f"ny={nyt}", color="#7570b3")
    c.axhline(GATELESS_FLOOR, color="k", ls="--", lw=1,
              label=f"gateless floor {GATELESS_FLOOR:.1e}")
    c.set_xticks(x)
    c.set_xticklabels([f"[{lo},{hi}) Mm" for lo, hi in BANDS])
    c.set_ylabel("max sawtooth(v) (m/s)")
    c.set_title("(c) standing v ripple stays on the floor")
    c.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    print(f"ripple ny={nyr}: {rr}")
    print(f"ripple ny={nyt}: {rt}")


if __name__ == "__main__":
    main()
