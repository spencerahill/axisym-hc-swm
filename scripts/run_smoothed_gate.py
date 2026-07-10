"""R6: smoothed-gate (candidate 2) empirical falsifier.

Runs the published equinoctial protocol (sin2, y0=0, ny=801, dt=30,
v_d=2.5, all other parameters at repo defaults) with the EMFD H(u) gate
replaced in-process by the smooth ramp G(u) = 0.5*(1 + tanh(u/u_w)),
u_w = 5 m/s, and the published CENTERED du/dy stencil (np.gradient) --
no hard gate, no upwind. This is candidate 2 of the unified-formulation
question taken at face value; no model code is modified.

Early stop: at the end of any day with domain max|u| > 100 m/s the run
aborts, the day and amplitude are logged to status.json, and the daily
output accumulated so far is still saved.

Either outcome rejects candidate 2: runaway means the smooth ramp does
not regularize the terminus shock (the characteristics argument), while
survival means the ramp throttles the EMFD in the near-equator
westerlies (u < 5 m/s inside |y| < 2 Mm), corrupting the u ~ y^3 regime
-- report G(u(y)) and the exponent/amp shift vs the gateless anchors.

Usage:
    python scripts/run_smoothed_gate.py [--ndays 5475] [--u-w 5.0]
        [--outdir model_output/formulation_suite/tier2_r6_smoothed_gate]
"""
import argparse
import json
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ss09.sw_config import SWConfig  # noqa: E402
from ss09.sw_model import SWModel  # noqa: E402
from ss09.theta_e import ThetaEConfig, Sin2Profile  # noqa: E402


class Runaway(Exception):
    def __init__(self, day, umax):
        self.day, self.umax = day, umax
        super().__init__(f"day {day}: domain max|u| = {umax:.1f} m/s")


class SmoothedGateModel(SWModel):
    """SWModel with G(u) = 0.5*(1+tanh(u/u_w)) EMFD gate, centered du/dy."""

    u_w = 5.0
    umax_stop = 100.0

    def edd_mom_flux_div_u(self):
        gate = 0.5 * (1.0 + np.tanh(self.state.u / self.u_w))
        du_dy = np.gradient(self.state.u, self.config.dy)
        return self.config.v_d * gate * np.sign(self.config.y) * du_dy

    def store_daily_avgs(self, day):
        super().store_daily_avgs(day)
        umax = float(np.abs(self.state.u).max())
        if umax > self.umax_stop:
            raise Runaway(day, umax)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ndays", type=int, default=5475)
    p.add_argument("--u-w", type=float, default=5.0, dest="u_w")
    p.add_argument("--outdir",
                   default="model_output/formulation_suite/tier2_r6_smoothed_gate")
    args = p.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = SWConfig(
        total_integration_days=args.ndays,
        ny=801,
        dt=30,
        output_path=str(outdir / "output.nc"),
        restart_output_dir=str(outdir),
    )
    model = SmoothedGateModel(config, Sin2Profile(ThetaEConfig()))
    model.u_w = args.u_w

    status = {"u_w": args.u_w, "ndays_requested": args.ndays,
              "runaway": False}
    try:
        model.run_sim()
    except Runaway as e:
        logging.warning(f"EARLY STOP (runaway): {e}")
        status.update(runaway=True, runaway_day=e.day,
                      runaway_umax=e.umax)
    model.save_results()
    status["nan_in_output"] = bool(np.isnan(np.asarray(model.results.u)).any())
    (outdir / "status.json").write_text(json.dumps(status, indent=2))
    logging.info(f"status: {status}")


if __name__ == "__main__":
    main()
