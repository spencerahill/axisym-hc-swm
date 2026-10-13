"""Diagnostic harness for the SS09 numerics review (2026-06-26).

Reproduces the documented spurious-momentum symptoms (jet creep over many
thousands of days; u > u_amc across the cell) and localizes the cause by
(a) decomposing the domain-integrated zonal-momentum tendency term by term at
the end state, and (b) a one-knob-at-a-time sensitivity sweep.

Run in-process (no NetCDF round-trip) for speed and direct access to the RHS
term methods. Not a unit test; a throwaway investigation tool.
"""

import logging
import sys
import numpy as np

from ss09.sw_config import SWConfig
from ss09.theta_e import ThetaEConfig, Sin2Profile, SB08Profile, SS09Profile
from ss09.sw_model import SWModel, SECONDS_PER_DAY

logging.getLogger().setLevel(logging.WARNING)  # silence per-day INFO spam

SCRATCH = (
    "/Users/sah2249/.claude/tmp/claude-501/"
    "-Users-sah2249-Library-CloudStorage-Dropbox-py-pengcheng-ss09/"
    "0e2fadbb-eb8c-4aad-9395-68fbb071358e/scratchpad"
)


_PROFILES = {"sin2": Sin2Profile, "SB08": SB08Profile, "SS09": SS09Profile}


def build(ndays, ny=51, dt=3600, asselin=0.04, v_d=2.5, k_v=7786 * 100,
          domain=15751e3 * 2, include_merid=True, include_vert=True,
          profile="sin2", y_0=0.0):
    cfg = SWConfig(
        total_integration_days=ndays, ny=ny, dt=dt, asselin_filt_coef=asselin,
        v_d=v_d, k_v=k_v, domain_size=domain,
        include_merid_advec_u=include_merid, include_vert_advec_u=include_vert,
        output_path=f"{SCRATCH}/inv_output.nc", restart_output_dir=SCRATCH,
    )
    tcfg = ThetaEConfig(theta_e_type=profile, y_0=y_0)
    return SWModel(cfg, _PROFILES[profile](tcfg))


def abs_vort_diag(model):
    """Absolute vorticity eta = beta*y - du/dy; flag inertial instability."""
    u, y, dy, beta = model.state.u, model.config.y, model.config.dy, model.config.beta
    eta = beta * y - np.gradient(u, dy)
    cell = np.abs(y) < 6e6
    # inertially unstable where sign(y)*eta < 0
    unstable = (np.sign(y) * eta < 0) & cell
    return np.mean(unstable[cell]), eta


def jet_series(model):
    """NH subtropical-jet magnitude per day = max over y>0 of daily-mean u."""
    u = model.results.u  # (ndays, ny)
    y = model.config.y
    nh = y > 0
    return np.max(u[:, nh], axis=1)


def creep_rate(js, day_lo, day_hi):
    """Linear slope of jet magnitude (m/s per 1000 days) over [day_lo, day_hi)."""
    d = np.arange(day_lo, day_hi)
    p = np.polyfit(d, js[day_lo:day_hi], 1)
    return p[0] * 1000.0


def momentum_budget(model):
    """Domain-integrated zonal-momentum tendency, term by term, at end state."""
    m, y, dy = model, model.config.y, model.config.dy
    terms = {
        "coriolis (beta*y*v)": m.coriolis_term(m.state.v),
        "merid_advec (-v du/dy)": -m.merid_advec_u(),
        "vert_advec (-u dv/dy H)": -(m.vert_advec_u() if m.config.include_vert_advec_u else np.zeros_like(y)),
        "rayleigh (-eps u)": -m.rayleigh_drag_u(),
        "emfd (-S)": -m.edd_mom_flux_div_u(),
    }
    out = {k: np.trapezoid(v, dx=dy) for k, v in terms.items()}
    out["NET du/dt (integral)"] = np.trapezoid(m.du_dt(), dx=dy)
    return out, terms


def u_vs_amc(model):
    u, y, beta = model.state.u, model.config.y, model.config.beta
    u_amc = 0.5 * beta * y ** 2
    excess = u - u_amc
    cell = np.abs(y) < 4e6  # within ~36 deg, the cell region
    return np.max(excess[cell]), np.mean(excess[cell] > 0)


def summarize(label, model):
    js = jet_series(model)
    nd = len(js)
    lo = min(1000, nd // 3)
    cr = creep_rate(js, lo, nd)
    max_exc, frac_exc = u_vs_amc(model)
    print(f"{label:38s} | jet[final]={js[-1]:6.2f}  creep={cr:+7.4f} m/s/kday "
          f"| u-u_amc max={max_exc:+6.2f}  frac(u>amc)={frac_exc:4.2f}")
    return js


def asym_report(label, model):
    js = jet_series(model)
    nd = len(js)
    lo = min(1000, nd // 3)
    cr = creep_rate(js, lo, nd)
    frac_unstable, _ = abs_vort_diag(model)
    print(f"{label:34s} | jet[final]={js[-1]:6.2f}  creep={cr:+8.5f} m/s/kday "
          f"| frac inertially-unstable in cell={frac_unstable:4.2f}")
    return js


def blowup_diag(model):
    """Return (blew_up, last_good_day, max_abs_u_seen)."""
    u = model.results.u
    finite_day = np.array([np.all(np.isfinite(u[d])) and np.any(u[d] != 0)
                           for d in range(u.shape[0])])
    last_good = int(np.max(np.where(finite_day)[0])) if finite_day.any() else -1
    blew = (not finite_day.all()) or last_good < u.shape[0] - 1
    maxu = float(np.nanmax(np.abs(u[finite_day]))) if finite_day.any() else float("nan")
    return blew, last_good, maxu


def instab_line(label, model):
    blew, last_good, maxu = blowup_diag(model)
    tag = f"BLEW UP @day {last_good}" if blew else "stable"
    print(f"{label:40s} | {tag:20s} | max|u| seen = {maxu:.2f} m/s")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"

    if mode == "noise":
        nd = 2600
        print(f"=== INSTANTANEOUS-NOISE vs SECULAR-SOURCE (SB08 y0=1000, dt=1800), "
              f"{nd} days ===")
        m = build(nd, profile="SB08", y_0=1000e3, dt=1800)
        m.run_sim()
        y, dy = m.config.y, m.config.dy
        # (1) secular tendency of the daily MEAN, last two days
        ud = m.results.u
        secular = np.trapezoid((ud[-1] - ud[-2]) / SECONDS_PER_DAY, dx=dy)
        # (2) instantaneous tendency integral (snapshot)
        instant = np.trapezoid(m.du_dt(), dx=dy)
        # (3) computational-mode amplitude: neighbor corr of instantaneous v
        from ss09.steady_state import SteadyStateDetector
        sm = SteadyStateDetector().compute_v_smoothness(m.state.v, dy)
        # (4) instantaneous vs daily-mean field difference (the wiggle)
        u_wiggle = np.max(np.abs(m.state.u - ud[-1]))
        v_wiggle = np.max(np.abs(m.state.v - m.results.v[-1]))
        print(f"\n  domain-integral d/dt of DAILY-MEAN u (secular source): {secular:+.4e} m^2/s^2")
        print(f"  domain-integral of INSTANTANEOUS du/dt (snapshot):     {instant:+.4e} m^2/s^2")
        print(f"\n  instantaneous v neighbor-correlation (1=smooth):       {sm['neighbor_correlation']:.4f}")
        print(f"  max|u_instant - u_dailymean|:                          {u_wiggle:.4f} m/s")
        print(f"  max|v_instant - v_dailymean|:                          {v_wiggle:.6f} m/s")
        print("\n  => if secular ~0 but snapshot is O(10s), the budget imbalance is")
        print("     instantaneous computational-mode noise, not a secular source.")

    elif mode == "longasym":
        nd = 15000
        print(f"=== STABLE OFF-EQ (SB08 y0=1000km, dt=1800), {nd} days ===")
        m = build(nd, profile="SB08", y_0=1000e3, dt=1800)
        m.run_sim()
        blew, lg, _ = blowup_diag(m)
        print(f"(blew_up={blew}, last_good_day={lg})")
        u = m.results.u; y = m.config.y
        gmax = np.max(u, axis=1)          # strongest westerly anywhere (winter jet)
        gmin = np.min(u, axis=1)          # strongest easterly
        print("\nstrong (winter) jet = max_y u(y), per day:")
        for d in [200, 500, 1000, 2000, 4000, 8000, 12000, nd - 1]:
            if d <= lg:
                print(f"   day {d:5d}: max u={gmax[d]:7.3f}  min u={gmin[d]:7.3f} m/s")
        cr = creep_rate(gmax, 2000, lg + 1)
        print(f"\ncreep of max u over days [2000,{lg}]: {cr:+.5f} m/s / 1000 days")
        # u vs AMC references at final state
        beta = m.config.beta
        u_amc_eq = 0.5 * beta * y ** 2                 # ascent at equator
        u_amc_y0 = 0.5 * beta * (y ** 2 - (1000e3) ** 2)  # ascent at y0
        uf = m.state.u
        cell = np.abs(y) < 6e6
        print(f"\nu vs AMC limits in cell at final state:")
        print(f"   max(u - 0.5*beta*y^2)         = {np.max((uf-u_amc_eq)[cell]):+.3f} m/s")
        print(f"   max(u - 0.5*beta*(y^2-y0^2))   = {np.max((uf-u_amc_y0)[cell]):+.3f} m/s")
        fr_unstable, eta = abs_vort_diag(m)
        print(f"   frac of cell with sign(y)*eta<0 (inertially unstable) = {fr_unstable:.2f}")
        print("\nDomain-integrated zonal-momentum budget at final state (m^2/s^2):")
        bud, _ = momentum_budget(m)
        for k, v in bud.items():
            print(f"   {k:28s} {v:+.4e}")
        # dt convergence check
        print(f"\n-- dt convergence (3000 days) --")
        for ddt in [1800, 900, 450]:
            mm = build(3000, profile="SB08", y_0=1000e3, dt=ddt); mm.run_sim()
            blew2, lg2, _ = blowup_diag(mm)
            jf = np.max(mm.results.u[lg2]) if lg2 >= 0 else float("nan")
            print(f"   dt={ddt:5d}: blew_up={blew2}  max u @last good day {lg2} = {jf:.3f} m/s")

    elif mode == "instab":
        nd = 600
        print(f"=== OFF-EQ INSTABILITY BISECTION (SB08 y0=1000km unless noted), "
              f"{nd} days ===")
        cases = [
            ("baseline dt3600 ny51 ass.04",  dict()),
            ("dt=1800",                       dict(dt=1800)),
            ("dt=900",                        dict(dt=900)),
            ("dt=450",                        dict(dt=450)),
            ("ny=101 dt3600",                 dict(ny=101)),
            ("ny=101 dt900",                  dict(ny=101, dt=900)),
            ("asselin=0.1",                   dict(asselin=0.1)),
            ("asselin=0.2",                   dict(asselin=0.2)),
            ("v_d=0 (no EMFD)",               dict(v_d=0.0)),
            ("k_v x4",                        dict(k_v=7786 * 400)),
            ("no merid advec u",              dict(include_merid=False)),
            ("no vert advec u",               dict(include_vert=False)),
        ]
        for label, kw in cases:
            m = build(nd, profile="SB08", y_0=1000e3, **kw)
            m.run_sim()
            instab_line(label, m)
        print("\n-- profile comparison at y0=1000km, dt3600 ny51 --")
        for prof in ["sin2", "SB08", "SS09"]:
            m = build(nd, profile=prof, y_0=1000e3); m.run_sim()
            instab_line(f"profile={prof}", m)

    elif mode == "asym":
        nd = 8000
        print(f"=== OFF-EQUATORIAL (SB08, y0=1000 km), {nd} days ===")
        m = build(nd, profile="SB08", y_0=1000e3)
        m.run_sim()
        js = asym_report("SB08 y0=1000", m)
        print("\njet magnitude vs day:")
        for d in [200, 500, 1000, 2000, 4000, 6000, nd - 1]:
            print(f"   day {d:5d}: {js[d]:.4f} m/s")
        print("\nDomain-integrated zonal-momentum budget at end (m^2/s^2):")
        bud, _ = momentum_budget(m)
        for k, v in bud.items():
            print(f"   {k:28s} {v:+.4e}")
        dudt = m.du_dt(); y = m.config.y
        ip, ineg = np.argmax(dudt), np.argmin(dudt)
        print(f"\n   du/dt max {dudt[ip]:+.3e} at y={y[ip]/1e3:+.0f} km; "
              f"min {dudt[ineg]:+.3e} at y={y[ineg]/1e3:+.0f} km")
        # compare across y0 values, shorter
        print(f"\n=== y0 sensitivity, 4000 days ===")
        for y0 in [0.0, 500e3, 1000e3, 1500e3, 2000e3]:
            mm = build(4000, profile="SB08", y_0=y0); mm.run_sim()
            asym_report(f"SB08 y0={y0/1e3:.0f}km", mm)

    elif mode == "quick":
        nd = 3000
        print(f"=== BASELINE (default config), {nd} days ===")
        m = build(nd)
        m.run_sim()
        js = summarize("baseline", m)
        print("\njet magnitude at days [500,1000,1500,2000,2500,2999]:")
        for d in [500, 1000, 1500, 2000, 2500, 2999]:
            print(f"   day {d:5d}: {js[d]:.4f} m/s")
        print("\nDomain-integrated zonal-momentum budget at end state "
              "(m^2/s^2, should ~cancel at steady state):")
        bud, _ = momentum_budget(m)
        for k, v in bud.items():
            print(f"   {k:28s} {v:+.4e}")
        # where is du/dt nonzero?
        dudt = m.du_dt()
        y = m.config.y
        ipos = np.argmax(dudt)
        ineg = np.argmin(dudt)
        print(f"\n   du/dt max {dudt[ipos]:+.3e} at y={y[ipos]/1e3:+.0f} km; "
              f"min {dudt[ineg]:+.3e} at y={y[ineg]/1e3:+.0f} km")
        print(f"   (u_amc at jet lat, final u there): "
              f"see u-u_amc above")

    elif mode == "sweep":
        nd = 4000
        print(f"=== SENSITIVITY SWEEP, {nd} days each (creep over days [1000,{nd}]) ===")
        print(f"{'config':38s} | {'metrics'}")
        m = build(nd); m.run_sim(); summarize("baseline (ny51 dt3600 ass.04)", m)
        m = build(nd, asselin=0.01); m.run_sim(); summarize("asselin=0.01", m)
        m = build(nd, asselin=0.005); m.run_sim(); summarize("asselin=0.005", m)
        m = build(nd, dt=1800); m.run_sim(); summarize("dt=1800", m)
        m = build(nd, v_d=0.0); m.run_sim(); summarize("v_d=0 (no EMFD)", m)
        m = build(nd, include_merid=False); m.run_sim(); summarize("no merid advec u", m)
        m = build(nd, include_vert=False); m.run_sim(); summarize("no vert advec u", m)
        m = build(nd, ny=101, dt=1800); m.run_sim(); summarize("ny=101 dt=1800", m)
        m = build(nd, ny=151, dt=900); m.run_sim(); summarize("ny=151 dt=900", m)

    print("\ndone.")


if __name__ == "__main__":
    main()
