"""The moist V2 parameter sweep: one manifest, one source of truth.

Every run in this sweep is the production formulation (staggered grid, EMFD
H(u) gate on, MUSCL-MC EMFD stencil, kinematic vertical-advection gate) with
moisture and latent heating on, and differs from the two reference runs only
in the parameters its block names.

The organising physics is the gross moist stability. The column MSE budget
(SCIENCE.md 3.6) is

    d<h>/dt + d/dy[v Hhat - L_v D dW/dy] = (C/tau)(theta_rad - theta) + L_v E_0

with <h> = C theta + L_v W and Hhat = Shat - L_v (2a-1) W. Precipitation is
internal and cancels. Hhat changes sign at

    W* = Shat / (L_v (2a-1)),      Shat = C d Delta_z / H,

and a column with no transport sits at

    W_q = W_c + tau_c E_0

because there E_0 = P = (W - W_c)/tau_c. So the single dimensionless number

    r = W_q / W* = L_v (2a-1) (W_c + tau_c E_0) / (C d Delta_z / H)

says whether an undisturbed column has positive (r < 1) or negative (r > 1)
gross moist stability, and FIVE separate model parameters move it: a, W_c,
tau_c, E_0 and Delta_z. The sweep crosses r = 1 along a, along W_c, along
tau_c and along Delta_z, which is what makes "does the model collapse onto r?"
a falsifiable question rather than a restatement of the definition. It fails
if, say, two runs at matched r but different a give different circulations.

Blocks
------
A  GMS plane            a x W_c, the 2-D regime map
B  fine W_c transect    a = 0.85, W_c resolving r = 0.80 to 1.26
C  matched-r pairs      r fixed at 0.85 and 1.05, a varied: the collapse test
D  radiative contrast   Delta_theta^rad 50-110 K, forcing strength
E  convective timescale tau_c 1800-172800 s (crosses r = 1 at a = 0.85, W_c = 40)
F  moisture diffusivity D 0-4e6 m^2/s, the eddy brake on aggregation
G  evaporation          E_0 0.5x-2x, the water source
H  eddy momentum        v_d 0-7.5 m/s, including the axisymmetric v_d = 0 limit
I  static stability     Delta_z 30-90 K (moves Shat, hence W*, hence r)
J  feedback strength    Lambda/Lambda_0 0-1.25, interpolating V2_0 to V2
K  initial moisture     W_init above W_c: is the aggregated state a separate attractor?
L  off-equatorial       time-invariant y_0, sin^2 and SB08 profiles
M  seasonal             SB08 migrating y_0: amplitude, period, waveform
N  V1 twins             latent heating off, for the attribution chain
O  numerics             resolution and timestep twins

Usage:
    python -c "from scripts.v2_sweep_manifest import manifest; print(len(manifest()))"
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List

from ss09.moist_constants import (
    L_V,
    column_heat_capacity,
    gross_dry_stability,
    latent_heating_coeff,
)

# ---------------------------------------------------------------------------
# Where the runs live. model_output/ is a symlink to /data3 on berimbau.
# ---------------------------------------------------------------------------
REPO = "/home/shill/py/axisym-hc-swm"
SWEEP_DIR = os.path.join(REPO, "model_output", "moist_v2_sweep")
SMOKE_DIR = os.path.join(REPO, "model_output", "moist_v2_sweep_smoke")
RUN_SW_MODEL = "/home/shill/miniconda3/envs/claude-swm/bin/run-sw-model"

# ---------------------------------------------------------------------------
# Production numerics. dt = 30 s at ny = 801 is the documented V2 ceiling
# (CLAUDE.md, moist V2 stability); ndays = 5800 is 5.01 drag timescales
# (1/epsilon_u = 1e8 s = 1157.4 d), the fixed-length alternative to the
# slow-drift gate recommended for suite runs. Fixed length, not a convergence
# gate, so every run is compared on the same footing and the drift is
# something the analysis measures rather than something a detector asserts.
# ---------------------------------------------------------------------------
NY = 801
DT = 30
NDAYS = 5800

# ---------------------------------------------------------------------------
# Measured timestep requirements (100-day and 300-day stability probes,
# berimbau, 2026-08-09). dt = 30 s at ny = 801 is the documented V2 ceiling at
# the reference parameters, and it is NOT enough everywhere: 22 of the 149
# configurations diverged within 100 days of a cold start, all of them cases
# with an unusually strong circulation (deeply negative gross moist stability,
# a steep radiative contrast, a small static stability) or a strongly
# off-equatorial forcing maximum, where the adjustment from the rest state is
# violent. dt = 15 s rescued 16 of the 22; the remaining 6 needed dt = 6 s.
#
# The failures are cold-start transients rather than instabilities of the
# equilibrium: a seasonal run whose forcing WALKS out to y_0 = 1400 km survives
# at dt = 30 where a run started with the forcing already there does not. Even
# so, every seasonal run is given dt = 15 s, both because one of them
# (amplitude 2800 km on the negative-stability reference) diverged at day 800
# and because holding dt fixed across the seasonal block removes dt as a
# confounder when amplitudes and periods are compared. Block O measures what
# halving dt does to the reference solutions, so the cost of mixing timesteps
# across the sweep is a number this sweep reports rather than an assumption.
DT_15 = {
    "A_a083_wc60", "A_a085_wc60", "A_a087_wc60",
    "D_N_dyr100", "D_N_dyr110", "I_N_dz45", "J_N_lam125",
    "L_P_sin2_y1400", "L_P_sin2_y2100", "L_P_sb08_y1400", "L_P_sb08_y2100",
    "L_N_sin2_y0700", "L_N_sin2_y1400", "L_N_sin2_y2100",
    "L_N_sb08_y0700", "L_N_sb08_y1400",
}
DT_6 = {
    "I_P_dz30", "I_N_dz30",
    "L_P_sb08_y2800", "L_N_sb08_y2100", "L_N_sb08_y2800",
}

# Model defaults that the sweep varies, kept here so r() and the manifest
# cannot disagree about what "default" means.
GRAVITY = 9.81
DELTA = 4.0e3
HEIGHT = 16.0e3
DELTA_Z = 60.0
A_DEF = 0.85
WC_DEF = 50.0
TAUC_DEF = 14400.0
EVAP_DEF = 4.6e-5
DW_DEF = 1.0e6
VD_DEF = 2.5
DYR_DEF = 75.0
LAMBDA_0 = latent_heating_coeff(GRAVITY)
C_COL = column_heat_capacity(GRAVITY)


def w_star(a: float, delta_z: float = DELTA_Z) -> float:
    """W at which the gross moist stability Hhat vanishes."""
    return gross_dry_stability(GRAVITY, DELTA, delta_z, HEIGHT) / (L_V * (2.0 * a - 1.0))


def w_quiescent(w_crit: float, tau_c: float = TAUC_DEF, evap: float = EVAP_DEF) -> float:
    """Steady W of a column with no transport: E_0 = P fixes W = W_c + tau_c E_0."""
    return w_crit + tau_c * evap


def r_param(a=A_DEF, w_crit=WC_DEF, tau_c=TAUC_DEF, evap=EVAP_DEF,
            delta_z=DELTA_Z) -> float:
    """r = W_q / W*: > 1 means an undisturbed column has negative Hhat."""
    return w_quiescent(w_crit, tau_c, evap) / w_star(a, delta_z)


def wc_for_r(r: float, a: float, tau_c: float = TAUC_DEF,
             evap: float = EVAP_DEF) -> float:
    """The W_c that puts a column at a chosen r, given a."""
    return r * w_star(a) - tau_c * evap


# ---------------------------------------------------------------------------
# The two reference runs. Everything else is described as a departure from one.
# P: positive gross moist stability, at the ERA5-recommended a ~ 0.79.
# N: negative, the documented aggregating case (a = 0.85, W_c = 50).
# ---------------------------------------------------------------------------
REF = {
    "P": dict(cwv_frac=0.79, w_crit=40.0),
    "N": dict(cwv_frac=0.85, w_crit=50.0),
}


def _base() -> Dict[str, Any]:
    """CLI arguments common to every run in the sweep."""
    return dict(
        ny=NY, dt=DT, ndays=NDAYS, backend="numba",
        theta_e_type="sin2", y0=0.0,
        enable_moisture=True, enable_latent_heating=True,
    )


def _run(name: str, block: str, note: str, **over) -> Dict[str, Any]:
    args = _base()
    args.update(over)
    return dict(name=name, block=block, note=note, args=args)


def _ref_args(ref: str) -> Dict[str, Any]:
    return dict(REF[ref])


# ---------------------------------------------------------------------------
def manifest() -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    seen = set()

    def add(name, block, note, **over):
        if name in seen:
            raise ValueError(f"duplicate run name {name!r}")
        seen.add(name)
        runs.append(_run(name, block, note, **over))

    # -- A: the (a, W_c) gross-moist-stability plane -------------------------
    for a in (0.75, 0.79, 0.83, 0.85, 0.87):
        for wc in (30.0, 40.0, 50.0, 60.0):
            add(f"A_a{int(a*100):03d}_wc{int(wc):02d}", "A",
                f"GMS plane: a={a}, W_c={wc}, r={r_param(a, wc):.3f}",
                cwv_frac=a, w_crit=wc)

    # -- B: fine W_c transect through r = 1 at a = 0.85 ----------------------
    for wc in (35.0, 42.0, 44.0, 46.0, 55.0):
        add(f"B_a085_wc{int(wc):02d}", "B",
            f"fine W_c transect: W_c={wc}, r={r_param(0.85, wc):.3f}",
            cwv_frac=0.85, w_crit=wc)

    # -- C: matched r, different a (the collapse test) -----------------------
    for r in (0.85, 1.05):
        for a in (0.75, 0.79, 0.83, 0.87):
            wc = wc_for_r(r, a)
            add(f"C_r{int(r*100):03d}_a{int(a*100):03d}", "C",
                f"matched r={r}: a={a} needs W_c={wc:.2f}",
                cwv_frac=a, w_crit=round(wc, 4))

    # -- D: radiative contrast Delta_theta^rad -------------------------------
    # Truncated below at 50 K deliberately: delta_y = 50 K is the
    # radiative-convective contrast, and a radiative target flatter than it
    # leaves convective heating inside the relaxation that Lambda P then adds
    # a second time (the model warns). So the ladder is asymmetric about 75 by
    # design, -25/+35, with the low side stopped by physics rather than cost.
    for ref in ("P", "N"):
        for dyr in (50.0, 60.0, 90.0, 100.0, 110.0):
            add(f"D_{ref}_dyr{int(dyr):03d}", "D",
                f"radiative contrast {dyr} K on the {ref} reference",
                delta_y_rad=dyr, **_ref_args(ref))

    # -- E: convective timescale tau_c ---------------------------------------
    # At a = 0.85, W_c = 40 the ladder crosses r = 1 through tau_c alone
    # (W_q = W_c + tau_c E_0), which is the second independent crossing.
    for tc in (1800.0, 7200.0, 43200.0, 86400.0, 172800.0):
        add(f"E_a085_wc40_tc{int(tc):06d}", "E",
            f"tau_c={tc:.0f} s at a=0.85, W_c=40, r={r_param(0.85, 40.0, tc):.3f}",
            cwv_frac=0.85, w_crit=40.0, tau_c=tc)
    for ref in ("P", "N"):
        for tc in (1800.0, 172800.0):
            add(f"E_{ref}_tc{int(tc):06d}", "E",
                f"tau_c={tc:.0f} s on the {ref} reference", tau_c=tc,
                **_ref_args(ref))

    # -- F: eddy moisture diffusivity D --------------------------------------
    for ref in ("P", "N"):
        for dw in (0.0, 2.5e5, 5.0e5, 2.0e6, 4.0e6):
            add(f"F_{ref}_D{dw:.2e}".replace("+", "").replace(".", "p"), "F",
                f"D={dw:.3g} m^2/s on the {ref} reference", dw=dw,
                **_ref_args(ref))

    # -- G: evaporation E_0 --------------------------------------------------
    for ref in ("P", "N"):
        for ev in (2.3e-5, 3.45e-5, 6.9e-5, 9.2e-5):
            add(f"G_{ref}_E{ev:.2e}".replace("-", "m").replace(".", "p"), "G",
                f"E_0={ev:.3g} kg/m2/s ({ev*86400:.2f} mm/day) on {ref}",
                evap=ev, **_ref_args(ref))

    # -- H: eddy momentum flux velocity v_d ----------------------------------
    for ref in ("P", "N"):
        for vd in (0.0, 1.25, 5.0, 7.5):
            add(f"H_{ref}_vd{vd:.2f}".replace(".", "p"), "H",
                f"v_d={vd} m/s on the {ref} reference", vd=vd, **_ref_args(ref))

    # -- I: static stability Delta_z (moves Shat and so W*) -------------------
    for ref in ("P", "N"):
        for dz in (30.0, 45.0, 75.0, 90.0):
            a = REF[ref]["cwv_frac"]
            wc = REF[ref]["w_crit"]
            add(f"I_{ref}_dz{int(dz):02d}", "I",
                f"Delta_z={dz} K on {ref}, r={r_param(a, wc, delta_z=dz):.3f}",
                delta_z=dz, **_ref_args(ref))

    # -- J: latent-heating strength Lambda -----------------------------------
    # Lambda != L_v/C deliberately breaks C*Lambda = L_v, so precipitation
    # stops cancelling in the MSE budget; the analysis carries the
    # (L_v - C Lambda) P term explicitly. Lambda = 0 is the V2_0 bridge.
    for ref in ("P", "N"):
        for frac in (0.0, 0.25, 0.5, 0.75, 1.25):
            add(f"J_{ref}_lam{int(frac*100):03d}", "J",
                f"Lambda = {frac} x L_v/C on the {ref} reference",
                lambda_conv=round(frac * LAMBDA_0, 10), **_ref_args(ref))

    # -- K: initial moisture, testing for a second attractor -----------------
    for wc in (40.0, 42.0, 44.0, 46.0):
        for bump in (15.0, 30.0):
            add(f"K_wc{int(wc):02d}_wi{int(wc+bump):02d}", "K",
                f"a=0.85, W_c={wc}, W(t=0)={wc+bump} (r={r_param(0.85, wc):.3f})",
                cwv_frac=0.85, w_crit=wc, w_init=wc + bump)

    # -- L: off-equatorial, time-invariant forcing ---------------------------
    for ref in ("P", "N"):
        for y0 in (700e3, 1400e3, 2100e3):
            add(f"L_{ref}_sin2_y{int(y0/1e3):04d}", "L",
                f"sin^2 forcing peak at y_0={y0/1e3:.0f} km on {ref}",
                y0=y0, **_ref_args(ref))
        for y0 in (0.0, 700e3, 1400e3, 2100e3, 2800e3):
            add(f"L_{ref}_sb08_y{int(y0/1e3):04d}", "L",
                f"SB08 forcing peak at y_0={y0/1e3:.0f} km on {ref}",
                theta_e_type="SB08", y0=y0, **_ref_args(ref))

    # -- M: seasonal (SB08 with migrating y_0) -------------------------------
    # Length: 2880 d of spin-up plus at least 6 full cycles, so the last two
    # cycles can be differenced as the repeatability check.
    def seas_days(period):
        return int(max(5760, 2880 + 6 * period))

    for ref in ("P", "N"):
        for amp in (350e3, 700e3, 1400e3, 2100e3, 2800e3):
            add(f"M_{ref}_amp{int(amp/1e3):04d}_p360", "M",
                f"seasonal amplitude {amp/1e3:.0f} km, 360-day period, {ref}",
                theta_e_type="SB08", y0_seas_amp=amp, seas_period=360.0,
                ndays=seas_days(360), **_ref_args(ref))
    for period in (90.0, 180.0, 720.0, 1440.0):
        add(f"M_P_amp1400_p{int(period):04d}", "M",
            f"seasonal period {period:.0f} days at amplitude 1400 km, P",
            theta_e_type="SB08", y0_seas_amp=1400e3, seas_period=period,
            ndays=seas_days(period), **_ref_args("P"))
    for period in (90.0, 1440.0):
        add(f"M_N_amp1400_p{int(period):04d}", "M",
            f"seasonal period {period:.0f} days at amplitude 1400 km, N",
            theta_e_type="SB08", y0_seas_amp=1400e3, seas_period=period,
            ndays=seas_days(period), **_ref_args("N"))
    for cyc, extra in (("tanh", {}), ("square", {})):
        add(f"M_P_amp1400_p360_{cyc}", "M",
            f"{cyc} seasonal waveform, amplitude 1400 km, 360-day period, P",
            theta_e_type="SB08", y0_seas_amp=1400e3, seas_period=360.0,
            seas_cycle_type=cyc, ndays=seas_days(360), **_ref_args("P"),
            **extra)

    # -- N: V1 twins (latent heating off) ------------------------------------
    for ref in ("P", "N"):
        add(f"N_{ref}_v1", "N",
            f"moist V1 twin of the {ref} reference: no latent heating, "
            "no retarget, thermodynamic gate",
            enable_latent_heating=False, **_ref_args(ref))

    # -- O: numerics ---------------------------------------------------------
    for ref in ("P", "N"):
        add(f"O_{ref}_ny1601", "O", f"{ref} reference at ny=1601, dt=15",
            ny=1601, dt=15, **_ref_args(ref))
        add(f"O_{ref}_ny401", "O", f"{ref} reference at ny=401, dt=60",
            ny=401, dt=60, **_ref_args(ref))
        add(f"O_{ref}_dt15", "O", f"{ref} reference at ny=801, dt=15",
            dt=15, **_ref_args(ref))
    add("O_crit_ny1601", "O", "near-critical a=0.85, W_c=44 at ny=1601, dt=15",
        ny=1601, dt=15, cwv_frac=0.85, w_crit=44.0)
    # y_0 = 700 km rather than 1400: the 1400 km cold start needs dt = 6 s even
    # at ny = 801, so at ny = 1601 it would have cost 2.4 h on its own and set
    # the critical path for the whole sweep. 700 km answers the same question
    # (does an off-equatorial solution move with resolution?) for 58 min.
    add("O_P_sb08_y0700_ny1601", "O",
        "off-equatorial SB08 y_0=700 km, P, at ny=1601, dt=15",
        ny=1601, dt=15, theta_e_type="SB08", y0=700e3, **_ref_args("P"))
    add("O_P_amp1400_p360_ny1601", "O",
        "seasonal amplitude 1400 km, P, at ny=1601, dt=15",
        ny=1601, dt=15, theta_e_type="SB08", y0_seas_amp=1400e3,
        seas_period=360.0, ndays=5760, **_ref_args("P"))
    # An off-equatorial resolution pair. Measured: the y_0 = 700 km cold start
    # diverges at ny=1601 even at dt=15, so both members of this pair run at
    # dt=6 and both are cut to 2500 days (2.2 drag timescales) to keep the
    # finer one off the critical path. Neither member is fully equilibrated,
    # but they are equally unequilibrated, so their paired difference still
    # measures grid sensitivity.
    for ny_, tag in ((801, "ny0801"), (1601, "ny1601")):
        add(f"O_P_sb08_y0700_{tag}_dt6", "O",
            f"off-equatorial SB08 y_0=700 km, P, at ny={ny_}, dt=6, 2500 d",
            ny=ny_, dt=6, ndays=2500, theta_e_type="SB08", y0=700e3,
            **_ref_args("P"))

    # Apply the measured timestep requirements. Seasonal runs all take dt = 15;
    # see the DT_15 / DT_6 comment above.
    for r in runs:
        if r["name"] in DT_6:
            r["args"]["dt"] = 6
        elif r["name"] in DT_15:
            r["args"]["dt"] = 15
        elif r["args"].get("y0_seas_amp", 0.0) > 0:
            r["args"]["dt"] = min(r["args"]["dt"], 15)
    return runs


# ---------------------------------------------------------------------------
# Turning a manifest entry into a command line
# ---------------------------------------------------------------------------
_FLAG = {
    "ndays": "--ndays", "ny": "--ny", "dt": "--dt", "backend": "--backend",
    "theta_e_type": "--theta-e-type", "y0": "--y0", "delta_y": "--delta-y",
    "delta_z": "--delta-z", "vd": "--vd", "tau": "--tau", "kv": "--kv",
    "y0_seas_amp": "--y0-seas-amp", "seas_period": "--seas-period",
    "seas_cycle_type": "--seas-cycle-type", "tanh_steepness": "--tanh-steepness",
    "cwv_frac": "--cwv-frac", "dw": "--dw", "w_crit": "--w-crit",
    "tau_c": "--tau-c", "evap": "--evap", "w_init": "--w-init",
    "delta_y_rad": "--delta-y-rad", "lambda_conv": "--lambda-conv",
    "theta00": "--theta00",
}
_SWITCH = {
    "enable_moisture": "--enable-moisture",
    "enable_latent_heating": "--enable-latent-heating",
}


def command(run: Dict[str, Any], out_dir: str, ndays: int | None = None) -> List[str]:
    """The argv for one manifest entry, writing into out_dir/out.nc."""
    args = dict(run["args"])
    if ndays is not None:
        args["ndays"] = ndays
    cmd = [RUN_SW_MODEL, "--output-path", os.path.join(out_dir, "out.nc")]
    for key, val in args.items():
        if key in _SWITCH:
            if val:
                cmd.append(_SWITCH[key])
            continue
        flag = _FLAG.get(key)
        if flag is None:
            raise KeyError(f"no CLI flag known for manifest key {key!r}")
        cmd += [flag, repr(val) if isinstance(val, float) else str(val)]
    return cmd


def cost_days(run: Dict[str, Any]) -> float:
    """Cost of a run in ny=801/dt=30-equivalent model days, for scheduling.

    Per-step cost is roughly linear in ny and inversely proportional to dt, so
    the equivalent is ndays * (ny/801) * (30/dt).
    """
    a = run["args"]
    return a["ndays"] * (a["ny"] / NY) * (DT / a["dt"])


if __name__ == "__main__":
    runs = manifest()
    by_block: Dict[str, int] = {}
    for r in runs:
        by_block[r["block"]] = by_block.get(r["block"], 0) + 1
    print(f"{len(runs)} runs")
    for b in sorted(by_block):
        print(f"  block {b}: {by_block[b]:3d}")
    total = sum(cost_days(r) for r in runs)
    # 0.180 s/day is the per-run rate measured with 16 runs in flight
    # (CLAUDE.md, moist numba ny=801/dt=30), so 16 runs advance 16 days per
    # 0.180 s and the wall time of the whole sweep is sum/16 * 0.180.
    print(f"total cost: {total:,.0f} equivalent model-days "
          f"({total/1e3:.1f}k); at 16-way 0.180 s/day that is "
          f"{total * 0.180 / 16 / 3600:.2f} h wall [UNVERIFIED: pending "
          f"a 16-way timing measured on this machine tonight]")
    print(f"\nW* table: " + "  ".join(
        f"a={a}:{w_star(a):.2f}" for a in (0.75, 0.79, 0.83, 0.85, 0.87)))
    print(f"Lambda_0 = {LAMBDA_0:.6f} K per kg/m2/s;  C = {C_COL:.4e} J/m2/K")
    print("\nfirst three commands:")
    for r in runs[:3]:
        print("  " + " ".join(command(r, os.path.join(SWEEP_DIR, r["name"]))))
