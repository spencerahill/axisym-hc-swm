"""Diagnostic: the polar westerly jet on the sphere under the three spherical
forms of the eddy term that the plan compared (--emfd-form).

  (a) vorticity closure, "vorticity":   S = v_d H(u) sgn(y) d(c u)/dy
  (b) literal form, "literal":          S = v_d H(u) sgn(y) c du/dy
  (c) divergence of the eddy angular-momentum flux, "angular_momentum" (the
      default):                         S = v_d H(u) sgn(y) (1/c) d(c^2 u)/dy

with c = cos(lat). Each is a 200-day cold start at ny=325, dt=30, the
production parameters and the spherical forcing at Delta_y = 50 K, on the
numpy backend. The hypothesis it tests (2026-11-08): the factor c^2 in form
(c) makes the poleward eddy flux of westerly momentum converge toward the
pole, so (c) puts the westerly maximum near the pole and (b), with no such
factor, does not. Until 2026-11-08 the script swapped forms (a) and (b) in by
subclassing the model; the emfd_form option that replaced the subclasses
reproduces them bit for bit (3-day runs at ny=81, checked that day). The
5,800-day resolution runs of the three forms are in sphere_validation_runs.py.

Usage:
    python validation/sphere_emfd_forms.py
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ss09.sw_config import SWConfig
from ss09.sw_model import SWModel
from ss09.theta_e import SphericalProfile, ThetaEConfig

FORMS = (("(a) d(c u)/dy", "vorticity"), ("(b) c du/dy", "literal"),
         ("(c) (1/c) d(c^2 u)/dy", "angular_momentum"))


def run(item):
    name, form = item
    cfg = SWConfig(geometry="sphere", ny=325, dt=30, total_integration_days=200,
                   emfd_form=form)
    m = SWModel(cfg, SphericalProfile(ThetaEConfig(theta_e_type="sphere")))
    m.save_restart_file = lambda day: None  # nothing written to disk
    m.run_sim()
    return name, m


def main() -> int:
    logging.disable(logging.INFO)
    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(run, FORMS))
    for name, m in results:
        lat = m.config.geom.lat
        u = m.results.u[199]
        th = m.results.theta[199] - m.current_theta_e()
        nh = np.where(lat > 0)[0]
        j = nh[np.argmax(u[nh])]
        jump = np.abs(np.diff(u))
        k = np.argmax(np.where(lat[:-1] > 50, jump, 0))
        print(f"{name:24s}: day 200 northern max u {u[j]:6.2f} m/s at {lat[j]:5.2f} deg; "
              f"u at 30, 50, 70 deg {np.interp([30, 50, 70], lat, u).round(1)}; largest "
              f"jump in u between neighboring centers north of 50 deg {jump[k]:6.2f} m/s at "
              f"{lat[k]:.2f} deg; min(theta - theta_E) north of 50 deg "
              f"{th[lat > 50].min():6.2f} K")
    return 0


if __name__ == "__main__":
    sys.exit(main())
