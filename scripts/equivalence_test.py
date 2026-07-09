"""Short-horizon equivalence test: repo SWModel vs Pengcheng's original SS_Model.py.

Runs three 30-day integrations at the original's config (ny=801, dt=30, v_d=2.5,
sin2 theta_E, y_0=0) and quantifies pairwise differences of the daily-mean fields:
  A. original SS_Model.py (via run_original_ss_model.py patching machinery)
  B. repo SWModel, EMFD Heaviside H(u) gate ON (the pre-2026-07-09 default)
  C. repo SWModel with the H(u) gate removed (matching the original code)

Claim under test: B differs from A only through the H(u) gate, so C should match
A to near-roundoff while B may diverge where u < 0 pockets exist.
"""
import pathlib
import subprocess
import sys

import numpy as np
import xarray as xr

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from ss09.sw_config import SWConfig
from ss09.sw_model import SWModel
from ss09.theta_e import Sin2Profile, ThetaEConfig

DAYS = 30
OUTBASE = pathlib.Path(sys.argv[1]).resolve()
ORIG_SRC = pathlib.Path(sys.argv[2]).resolve()
PYEXE = sys.executable


def run_repo(outdir, gate_on):
    config = SWConfig(
        total_integration_days=DAYS, ny=801, dt=30, v_d=2.5,
        emfd_heaviside_gate=gate_on,
        output_path=str(outdir / "output.nc"),
        restart_output_dir=str(outdir),
    )
    model = SWModel(config, Sin2Profile(ThetaEConfig()))
    if not gate_on:
        def emfd_no_gate():
            return (config.v_d * np.sign(config.y)
                    * np.gradient(model.state.u, config.dy))
        model.edd_mom_flux_div_u = emfd_no_gate
    model.run_sim()
    model.save_results()


def main():
    dir_a = OUTBASE / "orig30"
    dir_b = OUTBASE / "repo30_gate_on"
    dir_c = OUTBASE / "repo30_gate_off"

    subprocess.run(
        [PYEXE, "-u", str(REPO / "scripts" / "run_original_ss_model.py"),
         "--src", str(ORIG_SRC), "--outdir", str(dir_a),
         "--vd", "2.5", "--days", str(DAYS)],
        check=True, capture_output=True,
    )
    dir_b.mkdir(parents=True, exist_ok=True)
    dir_c.mkdir(parents=True, exist_ok=True)
    run_repo(dir_b, gate_on=True)
    run_repo(dir_c, gate_on=False)

    a = xr.open_dataset(dir_a / "output.nc")
    b = xr.open_dataset(dir_b / "output.nc")
    c = xr.open_dataset(dir_c / "output.nc")

    for name, ds in [("repo_gate_on_vs_orig", b), ("repo_gate_off_vs_orig", c)]:
        print(f"\n--- {name} ---")
        for var in ["u", "v", "T"]:
            n = min(a.sizes["time"], ds.sizes["time"])
            d = np.abs(ds[var].values[:n] - a[var].values[:n])
            imax = np.unravel_index(np.argmax(d), d.shape)
            print(f"{var}: max|D|={d.max():.3e} at day {imax[0]+1}, "
                  f"y={a['y'].values[imax[1]]/1e6:+.2f} Mm; "
                  f"final-day max|D|={d[-1].max():.3e}")


if __name__ == "__main__":
    main()
