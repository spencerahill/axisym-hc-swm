"""Validation harness for the staggered-grid / RK4 numerics.

Originally written for the 2026-06-26 numerics review of the old collocated /
leapfrog model; updated for the rewrite. It documents, by direct experiment, the
three findings that shaped the new scheme:

  - **stability**: the off-equatorial / solstitial regime (constant y0>0 from
    rest) is now stable at the default dt=3600 (the old scheme went to NaN in
    ~2 steps).
  - **superrotation**: with the explicit momentum diffusion turned off (k_u=0),
    the eddy-momentum flux divergence -- up-gradient near a westerly maximum --
    drives a slow equatorial superrotation. This is the long-documented
    "spurious momentum source"; the old leapfrog/Asselin damping had masked it.
    The default k_u suppresses it and reproduces the original climate.
  - **budget**: the domain-integrated zonal-momentum budget at the equilibrated
    state, term by term.

Run in-process (no NetCDF round-trip) for speed. Not a unit test; a throwaway
investigation tool.

Usage: python scripts/numerics_investigation.py [stability|superrotation|budget|climate]
"""

import logging
import sys
import numpy as np

from ss09.sw_config import SWConfig
from ss09.theta_e import ThetaEConfig, Sin2Profile, SB08Profile, SS09Profile
from ss09.sw_model import SWModel
from ss09 import rhs as rhs_module

logging.getLogger().setLevel(logging.WARNING)  # silence per-day INFO spam

SCRATCH = "/tmp"
_PROFILES = {"sin2": Sin2Profile, "SB08": SB08Profile, "SS09": SS09Profile}


def build(ndays, ny=50, dt=3600, k_u=1e5, v_d=2.5, k_v=7786 * 100,
          domain=15751e3 * 2, include_merid=True, include_vert=True,
          profile="sin2", y_0=0.0):
    cfg = SWConfig(
        total_integration_days=ndays, ny=ny, dt=dt, k_u=k_u, v_d=v_d, k_v=k_v,
        domain_size=domain, include_merid_advec_u=include_merid,
        include_vert_advec_u=include_vert,
        output_path=f"{SCRATCH}/inv_output.nc", restart_output_dir=SCRATCH,
    )
    tcfg = ThetaEConfig(theta_e_type=profile, y_0=y_0)
    return SWModel(cfg, _PROFILES[profile](tcfg))


def blowup_diag(model):
    """Return (blew_up, last_good_day, max_abs_u_seen)."""
    u = model.results.u
    finite = np.array([np.all(np.isfinite(u[d])) and np.any(u[d] != 0)
                       for d in range(u.shape[0])])
    last_good = int(np.max(np.where(finite)[0])) if finite.any() else -1
    blew = (not finite.all()) or last_good < u.shape[0] - 1
    maxu = float(np.nanmax(np.abs(u[finite]))) if finite.any() else float("nan")
    return blew, last_good, maxu


def jet_and_equator(model):
    """(winter-jet magnitude, equatorial u) of the last good daily mean."""
    u, y = model.results.u, model.config.y
    _, lg, _ = blowup_diag(model)
    if lg < 0:
        return float("nan"), float("nan")
    return float(np.max(u[lg])), float(u[lg][np.argmin(np.abs(y))])


def momentum_budget(model):
    """Domain-integrated zonal-momentum tendency, term by term, at end state."""
    terms = rhs_module.momentum_budget_terms(
        model.state, model.config, model.theta_e_profile
    )
    out = {k: float(np.trapezoid(v, dx=model.config.dy)) for k, v in terms.items()}
    du, _, _ = model.rhs(model.state)
    out["NET du/dt (integral)"] = float(np.trapezoid(du, dx=model.config.dy))
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "stability"

    if mode == "stability":
        print("=== OFF-EQUATORIAL STABILITY at dt=3600 (old scheme: NaN @ ~day 2) ===")
        for label, kw in [
            ("sym  y0=0    400d", dict(ndays=400, profile="sin2")),
            ("off  y0=1000 1500d", dict(ndays=1500, profile="SB08", y_0=1000e3)),
            ("off  y0=1500 1500d", dict(ndays=1500, profile="SB08", y_0=1500e3)),
            ("off  y0=1000 N=100", dict(ndays=1000, ny=100, profile="SB08", y_0=1000e3)),
        ]:
            m = build(**kw)
            m.run_sim()
            blew, lg, maxu = blowup_diag(m)
            jet, ueq = jet_and_equator(m)
            tag = f"BLEW UP @day {lg}" if blew else "stable"
            print(f"  {label:20s} | {tag:18s} | jet={jet:7.2f}  u_eq={ueq:7.2f}  max|u|={maxu:.1f}")

    elif mode == "superrotation":
        nd = 1500
        print(f"=== EMFD SUPERROTATION vs k_u (symmetric, {nd} days) ===")
        print("  (k_u=0 reproduces the 'spurious momentum source'; default 1e5 suppresses it)")
        for k_u in [0.0, 3e4, 1e5, 3e5]:
            m = build(nd, k_u=k_u)
            m.run_sim()
            jet, ueq = jet_and_equator(m)
            y = m.config.y
            jlat = y[np.argmax(m.results.u[blowup_diag(m)[1]])] / 1e3
            print(f"  k_u={k_u:7.0e} | jet={jet:7.2f} @ {jlat:6.0f}km | u_eq={ueq:7.2f}")

    elif mode == "budget":
        nd = 2000
        for prof, y0 in [("sin2", 0.0), ("SB08", 1000e3)]:
            m = build(nd, profile=prof, y_0=y0)
            m.run_sim()
            jet, ueq = jet_and_equator(m)
            print(f"\n=== {prof} y0={y0/1e3:.0f}km, {nd} days: jet={jet:.2f}, u_eq={ueq:.2f} ===")
            print("domain-integrated zonal-momentum budget (m^2/s^2):")
            for k, v in momentum_budget(m).items():
                print(f"   {k:28s} {v:+.4e}")

    elif mode == "climate":
        nd = 2000
        print(f"=== STEADY CLIMATE ({nd} days) vs original-model reference ===")
        print("  reference: symmetric jet 28.7 m/s; off-eq y0=1000km winter jet ~41.7 m/s")
        for label, kw in [
            ("sym  y0=0",     dict(profile="sin2")),
            ("off  y0=1000",  dict(profile="SB08", y_0=1000e3)),
            ("off  y0=1500",  dict(profile="SB08", y_0=1500e3)),
        ]:
            m = build(nd, **kw)
            m.run_sim()
            jet, ueq = jet_and_equator(m)
            y = m.config.y
            jlat = y[np.argmax(m.results.u[blowup_diag(m)[1]])] / 1e3
            print(f"  {label:14s} jet={jet:7.2f} @ {jlat:6.0f}km  u_eq={ueq:7.2f}")

    print("\ndone.")


if __name__ == "__main__":
    main()
