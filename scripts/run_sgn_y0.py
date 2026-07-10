"""R11: sgn(y) vs sgn(y - y0) fork test (H4), continuing from arm A.

Loads the equilibrated arm A state (y0 = 1000 km, SB08, gate-on + upwind,
restart_day5475.nc) and continues it with the EMFD's poleward-advection
pole moved from the equator to y0: sgn(y) -> sgn(y - y0), with the upwind
split moved consistently (the advection velocity is v_d*sgn(y - y0), so
points north of y0 difference backward, south of y0 forward). No model
code is modified; the EMFD method is overridden in-process.

Prediction [derived, 2026-07-10]: the two forms differ only in
0 < y < y0, which is easterly at arm A's equilibrium, where the H(u)
gate zeroes the EMFD either way -- so max|du| <~ 0.1 m/s outside the
terminus band after re-equilibration.

Usage:
    python scripts/run_sgn_y0.py [--extend-days 730] [--y-split 1.0e6]
        [--restart model_output/gate_y0_experiment/armA_gateon_upwind/restart_day5475.nc]
        [--outdir model_output/formulation_suite/tier3_r11_sgn_fork]
"""
import argparse
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ss09.sw_config import SWConfig  # noqa: E402
from ss09.sw_model import SWModel  # noqa: E402
from ss09.theta_e import ThetaEConfig, SB08Profile  # noqa: E402


class SgnShiftModel(SWModel):
    """Gate-on + upwind EMFD with the advection pole at y_split, not 0."""

    y_split = 1.0e6

    def edd_mom_flux_div_u(self):
        gate = np.heaviside(self.state.u, 0.5)
        u = self.state.u
        diff = (u[1:] - u[:-1]) / self.config.dy
        backward = np.zeros_like(u)
        backward[1:] = diff
        forward = np.zeros_like(u)
        forward[:-1] = diff
        du_dy = np.where(self.config.y > self.y_split, backward, forward)
        return (self.config.v_d * gate
                * np.sign(self.config.y - self.y_split) * du_dy)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extend-days", type=int, default=730)
    p.add_argument("--y-split", type=float, default=1.0e6, dest="y_split")
    p.add_argument(
        "--restart",
        default="model_output/gate_y0_experiment/armA_gateon_upwind/"
                "restart_day5475.nc")
    p.add_argument("--outdir",
                   default="model_output/formulation_suite/tier3_r11_sgn_fork")
    args = p.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    restart = pathlib.Path(args.restart)
    config = SWConfig(
        total_integration_days=5475 + args.extend_days,
        ny=801,
        dt=30,
        emfd_heaviside_gate=True,
        emfd_stencil="upwind",
        output_path=str(outdir / "output.nc"),
        restart_output_dir=str(outdir),
    )
    theta_cfg = ThetaEConfig(theta_e_type="SB08", y_0=1.0e6)
    model = SgnShiftModel(config, SB08Profile(theta_cfg))
    model.y_split = args.y_split
    model.restart_day = model.load_from_restart(str(restart))
    model.run_sim()
    model.save_results()


if __name__ == "__main__":
    main()
