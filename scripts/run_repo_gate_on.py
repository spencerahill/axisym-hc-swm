"""Run the repo SWModel stock (EMFD H(u) gate ON) at the original's config.

Long-horizon companion to equivalence_test.py: documents what the current
main-branch model does at ny=801/dt=30/v_d=2.5 over many days, to characterize
the gate-driven flank pathology against the gateless original.
Usage: run_repo_gate_on.py OUTDIR NDAYS
"""
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ss09.sw_config import SWConfig
from ss09.sw_model import SWModel
from ss09.theta_e import Sin2Profile, ThetaEConfig

outdir = pathlib.Path(sys.argv[1]).resolve()
ndays = int(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)

config = SWConfig(
    total_integration_days=ndays, ny=801, dt=30, v_d=2.5,
    output_path=str(outdir / "output.nc"),
    restart_output_dir=str(outdir),
)
model = SWModel(config, Sin2Profile(ThetaEConfig()))
model.run_sim()
model.save_results()
