from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np

from .moist_constants import latent_heating_coeff

SECONDS_PER_DAY = 86400

# Moist V1 parameter fields, all inert unless enable_moisture. Used by the
# validation that rejects a non-default moist parameter on a dry config
# (repo style: a hard error, never a silent no-op).
MOIST_PARAMS = ("cwv_frac", "d_w", "w_crit", "tau_c", "evap", "w_init")
# Moist V2 (latent heating) parameter fields, inert unless
# enable_latent_heating; same hard-error treatment.
LATENT_PARAMS = ("delta_y_rad", "lambda_conv")
# Spec's provisional radiative-equilibrium contrast (75-100 K); the low end,
# closest to the V1 RCE contrast of 50 K.
DEFAULT_DELTA_Y_RAD = 75.0


@dataclass
class SWConfig:
    """Configuration for the Shallow Water Model."""

    total_integration_days: int = 250
    gravity: float = 9.81
    height: float = 16e3
    beta: float = 2e-11
    t_ref: float = 300.0
    output_path: str = "./model_output/output.nc"
    k_v: float = 7786 * 100
    epsilon_u: float = 1e-8
    delta_z: float = 60
    delta: float = 4e3
    tau: float = 37.0 * SECONDS_PER_DAY
    v_d: float = 2.5
    dt: int = 3600
    ny: int = 51
    domain_size: float = 15751e3 * 2
    dy: float = field(init=False)
    y: np.ndarray = field(init=False)
    # v-grid layout. "staggered" (default): v lives on the ny-1 interior cell
    # faces (Arakawa C-grid), the production formulation adopted 2026-07-11;
    # removes the standing interior 2*dy v ripple the centered dv/dy stencil is
    # blind to. "collocated": legacy layout with v on the same ny centers as u,
    # the Zhang et al. (2025)-lineage reproduction path.
    grid: str = "staggered"
    nv: int = field(init=False)  # length of the v array (ny or ny-1)
    y_v: np.ndarray = field(init=False)  # v-grid coordinate (centers or faces)
    # Integration backend. "numpy": the reference implementation (default).
    # "numba": JIT-compiled fused day kernel, bitwise-identical to numpy;
    # staggered grid only (the collocated layout is the bit-exact Zhang et
    # al. 2025 reproduction path and stays on the reference implementation).
    backend: str = "numpy"
    asselin_filt_coef: float = 0.04
    coeff_eddy_heat_diff: float = 0.0  # values <1e4 make little difference
    include_vert_advec_u: bool = True
    include_merid_advec_u: bool = True  # Toggle for v*du/dy meridional advection term
    # Gate on the vertical momentum advection u*(dv/dy). "theta": the
    # production H(theta_E - theta) gate (convecting columns), the default for
    # dry and V1 runs and the formulation every regression baseline was
    # generated with. "kinematic": SS09 Eq. 2.1's own H(dv/dy), validated as an
    # equivalent dry formulation by the 2026-07-18 steady-state A/B twins and
    # adopted for V2, where the theta gate would read the retargeted
    # theta_rad and switch off exactly where convection is strongest (spec
    # item 11). None resolves by version: kinematic under latent heating,
    # theta otherwise.
    vert_advec_gate: Optional[str] = None
    # H(u) gate on the EMFD, per SS09 eq. (2.5) / Zhang et al. (2025) eq. (5).
    # On by default (2026-07-12): the production formulation gates the EMFD.
    # Set False (--no-emfd-heaviside-gate) for the published Zhang et al. (2025)
    # code, which omits the gate.
    emfd_heaviside_gate: bool = True
    # Spatial stencil for the EMFD du/dy. The EMFD is advective in form with
    # poleward velocity v_d*sgn(y). "mc" (default): MUSCL with monotonized-
    # central limited slopes, second-order in smooth regions, reverting toward
    # upwind at extrema and discontinuities; the production stencil, needed for
    # a stable gate-on integration. "upwind": first-order one-sided from the
    # equatorward (upstream) side, per SS09 section 2b. "centered": np.gradient,
    # matching the published Zhang et al. (2025) code (pair with
    # --no-emfd-heaviside-gate for the gateless reproduction path).
    emfd_stencil: str = "mc"
    # Steady-state detection parameters
    enable_steady_state: bool = False
    steady_state_window_size: int = 10
    steady_state_threshold: float = 0.001
    steady_state_check_both: bool = True
    smoothness_threshold: float = 0.5  # Neighbor correlation threshold for v field smoothness
    # Slow-drift gate: opt-in additional convergence criterion requiring the
    # slow diagnostics (jet latitude, max |v|, equatorial depression) to have
    # a relative range below slow_drift_thresh over a trailing
    # slow_drift_window days. The KE/Tvar metrics are nearly blind to the
    # slow jet-position mode (runs 1a-1b, 2026-07-17: the detector fired at
    # day ~960 with the jet still moving -0.69%/60 d on an oscillatory
    # ~400-d tail), and the range criterion over a drag-timescale window
    # sees the ringing amplitude a trend test misses at turning points.
    # Non-seasonal runs only (the slow diagnostics oscillate with the
    # seasonal cycle; enforced by SWModel).
    slow_drift_gate: bool = False
    slow_drift_window: int = 0  # days; 0 = auto: ceil(1/epsilon_u) in days
    slow_drift_thresh: float = 0.002
    # Seasonal convergence parameters (for seasonally-varying forcing)
    seasonal_convergence_enabled: bool = False  # Disabled by default - user must opt-in
    seasonal_convergence_window: int = 30  # Days that must match year-to-year
    seasonal_convergence_threshold: float = 0.01  # 1% year-to-year change threshold
    # Restart/checkpoint parameters
    save_restart_every: int = 0  # Save restart file every N days (0 = only at end)
    restart_output_dir: str = "./model_output"  # Directory for restart files
    # Moist V1: prognostic passive column water vapor W(y, t), one-way coupled
    # to the circulation (guides/moist_axisymmetric_model_spec.pdf Eq. Q1).
    # Defaults are the V1 plan's placeholders, to be replaced by the
    # ERA5-calibrated values when they land. Staggered grid only; both the
    # numpy and numba backends are supported (validated below).
    enable_moisture: bool = False
    cwv_frac: float = 0.85  # a: lower-layer CWV fraction; mean transport ~ (2a-1)
    d_w: float = 1.0e6  # D: eddy moisture diffusivity (m^2/s)
    w_crit: float = 50.0  # W_c: critical CWV for precipitation onset (kg/m^2)
    tau_c: float = 14400.0  # convective relaxation time (s)
    evap: float = 4.6e-5  # E_0: uniform evaporation (kg m^-2 s^-1)
    w_init: Optional[float] = None  # uniform W(y, 0); None means w_crit
    # Moist V2: the latent-heating feedback (spec Eq. T2). The single
    # structural change is thermodynamic: the relaxation retargets from the
    # warm RCE profile to the steeper radiative one (contrast delta_y_rad in
    # place of the theta_e profile's delta_y), and explicit latent heating
    # lambda_conv * P is added. Keeping convective heating inside the
    # relaxation AND adding lambda_conv * P would double-count it. Both
    # parameters stay None on a dry or V1 config (not applicable) and resolve
    # to their defaults when latent heating is on.
    enable_latent_heating: bool = False
    delta_y_rad: Optional[float] = None  # Delta_theta^rad (K); None -> 75
    lambda_conv: Optional[float] = None  # Lambda = L_v/C; None -> derived

    def __post_init__(self):
        self.dy = self.domain_size / (self.ny - 1)
        self.y = np.linspace(-self.domain_size / 2, self.domain_size / 2, self.ny)

        valid_stencils = ("centered", "upwind", "mc")
        if self.emfd_stencil not in valid_stencils:
            raise ValueError(
                f"emfd_stencil must be one of {valid_stencils}, "
                f"got {self.emfd_stencil!r}"
            )

        valid_grids = ("staggered", "collocated")
        if self.grid not in valid_grids:
            raise ValueError(
                f"grid must be one of {valid_grids}, got {self.grid!r}"
            )
        if self.grid == "staggered":
            # v on the ny-1 interior cell faces at the midpoints between
            # adjacent u centers. The average form (rather than y[:-1] + dy/2)
            # is exactly antisymmetric about the equator whenever y is, which
            # the mirror-parity invariant of the integration depends on.
            self.nv = self.ny - 1
            self.y_v = 0.5 * (self.y[:-1] + self.y[1:])
        else:
            self.nv = self.ny
            self.y_v = self.y

        valid_backends = ("numpy", "numba")
        if self.backend not in valid_backends:
            raise ValueError(
                f"backend must be one of {valid_backends}, got {self.backend!r}"
            )
        if self.backend == "numba":
            if self.grid != "staggered":
                raise ValueError(
                    "backend='numba' supports only the staggered grid; the "
                    "collocated layout is the bit-exact reproduction path and "
                    "stays on the numpy reference implementation"
                )
            # Fractional dt is numba-specific: a value like 4.5 divides 86400
            # exactly in float arithmetic (so the all-backend divisibility
            # check below passes it) but would be silently truncated by the
            # kernel's integer-t bookkeeping.
            if not float(self.dt).is_integer():
                raise ValueError(
                    "backend='numba' requires an integer dt (its time "
                    f"bookkeeping is exact integer seconds); got dt={self.dt}"
                )

        # Any backend: for a dt not dividing 86400, the total-step count
        # int(86400*ndays/dt), the per-"day" storage cycle of int(86400/dt)
        # steps, and the restart resume step int(day*86400/dt) are mutually
        # inconsistent (trailing sub-day steps belong to no stored day, and a
        # continuation duplicates or skips a step relative to the
        # uninterrupted run).
        if SECONDS_PER_DAY % self.dt != 0:
            raise ValueError(
                f"dt must divide {SECONDS_PER_DAY} s (daily-average storage "
                "and restart resume assume an integer number of steps per "
                f"day); got dt={self.dt}"
            )

        if self.enable_moisture:
            # Guard the two moist parameters whose bad values are not merely
            # unphysical but catastrophic: tau_c divides the precipitation
            # closure (tau_c <= 0 is an immediate divide-to-NaN), and a
            # negative diffusivity is anti-diffusion (a guaranteed grid-scale
            # blow-up). d_w = 0 stays valid: the advection-only member of the
            # V1 D ladder.
            if self.tau_c <= 0:
                raise ValueError(
                    f"tau_c must be positive (it is the precipitation "
                    f"relaxation timescale and a divisor); got tau_c={self.tau_c}"
                )
            if self.d_w < 0:
                raise ValueError(
                    f"d_w must be non-negative (a negative moisture "
                    f"diffusivity is anti-diffusion and blows up at the grid "
                    f"scale); got d_w={self.d_w}"
                )
            if self.grid == "collocated":
                raise ValueError(
                    "enable_moisture=True requires the staggered grid (the "
                    "moisture fluxes live on the v faces); the collocated "
                    "layout is the frozen Zhang et al. (2025) reproduction "
                    "path"
                )
        else:
            # A moist parameter on a dry config would be silently inert;
            # refuse it instead (repo style, like --seas-conv without
            # --stop-at-steady-state).
            for name in MOIST_PARAMS:
                default = type(self).__dataclass_fields__[name].default
                if getattr(self, name) != default:
                    raise ValueError(
                        f"{name}={getattr(self, name)} has no effect without "
                        "enable_moisture=True; pass --enable-moisture (or set "
                        "enable_moisture=True) to run with moisture"
                    )

        if self.enable_latent_heating:
            # Latent heating is Lambda * P, and P is a function of the
            # prognostic W: V2 is V1 plus the feedback, never a standalone.
            if not self.enable_moisture:
                raise ValueError(
                    "enable_latent_heating=True requires enable_moisture=True "
                    "(the latent heating is Lambda * P, and P comes from the "
                    "prognostic column water vapor); pass --enable-moisture "
                    "alongside --enable-latent-heating"
                )
            if self.delta_y_rad is None:
                self.delta_y_rad = DEFAULT_DELTA_Y_RAD
            if self.lambda_conv is None:
                # Lambda = L_v/C with the same column heat capacity C that
                # sets the gross dry stability, so C*Lambda = L_v and
                # precipitation cancels exactly in the column MSE budget.
                self.lambda_conv = latent_heating_coeff(self.gravity)
        else:
            for name in LATENT_PARAMS:
                if getattr(self, name) is not None:
                    raise ValueError(
                        f"{name}={getattr(self, name)} has no effect without "
                        "enable_latent_heating=True; pass "
                        "--enable-latent-heating (or set "
                        "enable_latent_heating=True) to run with the latent "
                        "heating feedback"
                    )

        valid_gates = ("theta", "kinematic")
        if self.vert_advec_gate is None:
            self.vert_advec_gate = (
                "kinematic" if self.enable_latent_heating else "theta"
            )
        elif self.vert_advec_gate not in valid_gates:
            raise ValueError(
                f"vert_advec_gate must be one of {valid_gates}, got "
                f"{self.vert_advec_gate!r}"
            )
        if self.enable_latent_heating and self.vert_advec_gate == "theta":
            # Reachable only on explicit request, and left reachable so the
            # pathology can be demonstrated; it is never a V2 default.
            warnings.warn(
                "vert_advec_gate='theta' under latent heating gates the "
                "vertical momentum advection on H(theta_rad - theta), which "
                "switches off wherever the column precipitates (the spec's "
                "item 11); use the kinematic gate for V2 physics"
            )

        # The seasonal year-to-year convergence check reads the daily history
        # the steady-state detector records, and the detector records only
        # when enabled: without enable_steady_state the history stays empty
        # and seasonal convergence can never trigger (a former silent no-op).
        if self.seasonal_convergence_enabled and not self.enable_steady_state:
            raise ValueError(
                "seasonal_convergence_enabled=True requires "
                "enable_steady_state=True (the steady-state detector records "
                "the daily history the seasonal convergence check compares "
                "year-to-year); pass --stop-at-steady-state alongside "
                "--seas-conv"
            )

        if self.slow_drift_gate and not self.enable_steady_state:
            raise ValueError(
                "slow_drift_gate=True requires enable_steady_state=True "
                "(the gate is an additional criterion of the steady-state "
                "detector); pass --stop-at-steady-state alongside "
                "--slow-drift-gate"
            )
        if self.slow_drift_window < 0:
            raise ValueError(
                f"slow_drift_window must be >= 0 (0 = auto); got "
                f"{self.slow_drift_window}"
            )
        if self.slow_drift_gate and self.slow_drift_window == 0:
            if self.epsilon_u <= 0:
                raise ValueError(
                    "the slow-drift window defaults to the drag timescale "
                    "1/epsilon_u, which needs epsilon_u > 0; pass "
                    "--slow-drift-window explicitly for a drag-free run"
                )
            self.slow_drift_window = int(
                np.ceil(1.0 / (self.epsilon_u * SECONDS_PER_DAY))
            )

        # Validate steady-state parameters
        if self.enable_steady_state and self.steady_state_window_size > self.total_integration_days:
            warnings.warn(
                f"Steady-state window size ({self.steady_state_window_size}) exceeds "
                f"total integration days ({self.total_integration_days}). "
                f"Convergence detection will not trigger."
            )
