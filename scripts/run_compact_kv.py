"""Compact-Laplacian k_v patch test: is the standing 2dy v ripple k_v-blindness?

Continues the equilibrated Tier-1 gate-on+upwind state (restart_day5475.nc)
with diffusion_v replaced in-process by the compact 3-point Laplacian
(v_{i+1} - 2 v_i + v_{i-1}) / dy^2, instead of the published code's
np.gradient applied twice (which is exactly blind to the 2dy checkerboard).
Same formal truncation order on smooth fields; the only new physics is that
k_v can now damp grid-scale v structure (rate 4 k_v / dy^2, e-fold ~8 min at
k_v = 778600). No model code is modified.

Prediction [derived, 2026-07-10]: the standing interior v sawtooth
(0.0017-0.008 m/s, sourced at the terminus) collapses by orders of
magnitude, the coupled interior u sawtooth (0.010) falls with it, and the
resolved anchors (jets, v extrema, T_eq) move negligibly.

Usage:
    python scripts/run_compact_kv.py [--extend-days 200]
        [--restart model_output/formulation_suite/tier1_y0p0000_gateon_upwind/restart_day5475.nc]
        [--outdir model_output/formulation_suite/tier1_compact_kv]
"""
import argparse
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ss09.sw_config import SWConfig  # noqa: E402
from ss09.sw_model import SWModel  # noqa: E402
from ss09.theta_e import ThetaEConfig, Sin2Profile  # noqa: E402


class CompactKvModel(SWModel):
    """SWModel with the compact 3-point Laplacian in diffusion_v."""

    def diffusion_v(self):
        v = self.state.v
        lap = np.zeros_like(v)
        lap[1:-1] = (v[2:] - 2.0 * v[1:-1] + v[:-2]) / self.config.dy ** 2
        return lap * self.config.k_v


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extend-days", type=int, default=200)
    p.add_argument(
        "--restart",
        default="model_output/formulation_suite/tier1_y0p0000_gateon_upwind/"
                "restart_day5475.nc")
    p.add_argument("--outdir",
                   default="model_output/formulation_suite/tier1_compact_kv")
    args = p.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = SWConfig(
        total_integration_days=5475 + args.extend_days,
        ny=801,
        dt=30,
        emfd_heaviside_gate=True,
        emfd_upwind=True,
        output_path=str(outdir / "output.nc"),
        restart_output_dir=str(outdir),
    )
    model = CompactKvModel(config, Sin2Profile(ThetaEConfig()))
    model.restart_day = model.load_from_restart(args.restart)
    model.run_sim()
    model.save_results()


if __name__ == "__main__":
    main()
