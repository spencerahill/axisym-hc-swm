"""Tests for restart/checkpoint continuation of the integration.

The key correctness requirement: a run that is stopped, checkpointed, and
resumed must reproduce the trajectory of an uninterrupted run. This exercises
the full save_restart_file -> load_from_restart -> run_sim path, which restores
the instantaneous state at timesteps n and n-1 needed to seed the leapfrog
scheme exactly.
"""

import numpy as np

from ss09.sw_model import SWModel
from ss09.sw_config import SWConfig
from ss09.theta_e import ThetaEConfig, SS09Profile
from ss09.output_path_utils import generate_restart_filename


def _make_config(tmp_path, total_days):
    return SWConfig(
        total_integration_days=total_days,
        output_path=str(tmp_path / "run_output.nc"),
        restart_output_dir=str(tmp_path),
        ny=21,
        dt=3600,
    )


def _theta_e_config():
    return ThetaEConfig(theta_00=310.0, y_0=0.0, y_one=9000e3, delta_y=45)


def test_restart_reproduces_continuous_trajectory(tmp_path):
    """A checkpointed-and-resumed run must match an uninterrupted run."""
    total_days = 6
    split_day = 3

    # Uninterrupted reference run.
    full = SWModel(_make_config(tmp_path / "full", total_days), SS09Profile(_theta_e_config()))
    (tmp_path / "full").mkdir()
    full.run_sim()

    # Partial run that checkpoints at split_day, then a resumed continuation.
    part_dir = tmp_path / "part"
    part_dir.mkdir()
    part = SWModel(_make_config(part_dir, split_day), SS09Profile(_theta_e_config()))
    part.run_sim()

    restart_file = generate_restart_filename(part.config.output_path, split_day)

    cont = SWModel(_make_config(part_dir, total_days), SS09Profile(_theta_e_config()))
    cont.restart_day = cont.load_from_restart(restart_file)
    cont.run_sim()

    # Compare the days that both runs computed (split_day .. total_days-1).
    for day in range(split_day, total_days):
        np.testing.assert_allclose(
            cont.results.u[day], full.results.u[day], rtol=1e-10, atol=1e-12,
            err_msg=f"u mismatch on resumed day {day}",
        )
        np.testing.assert_allclose(
            cont.results.v[day], full.results.v[day], rtol=1e-10, atol=1e-12,
            err_msg=f"v mismatch on resumed day {day}",
        )
        np.testing.assert_allclose(
            cont.results.theta[day], full.results.theta[day], rtol=1e-10, atol=1e-12,
            err_msg=f"theta mismatch on resumed day {day}",
        )
