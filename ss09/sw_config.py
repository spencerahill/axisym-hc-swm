from dataclasses import dataclass, field
import numpy as np

SECONDS_PER_DAY = 86400


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
    asselin_filt_coef: float = 0.04
    coeff_eddy_heat_diff: float = 0.0  # values <1e4 make little difference
    include_vert_advec_u: bool = True
    include_merid_advec_u: bool = True  # Toggle for v*du/dy meridional advection term
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
    # Seasonal convergence parameters (for seasonally-varying forcing)
    seasonal_convergence_enabled: bool = False  # Disabled by default - user must opt-in
    seasonal_convergence_window: int = 30  # Days that must match year-to-year
    seasonal_convergence_threshold: float = 0.01  # 1% year-to-year change threshold
    # Restart/checkpoint parameters
    save_restart_every: int = 0  # Save restart file every N days (0 = only at end)
    restart_output_dir: str = "./model_output"  # Directory for restart files

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

        # Validate steady-state parameters
        if self.enable_steady_state and self.steady_state_window_size > self.total_integration_days:
            import warnings
            warnings.warn(
                f"Steady-state window size ({self.steady_state_window_size}) exceeds "
                f"total integration days ({self.total_integration_days}). "
                f"Convergence detection will not trigger."
            )
