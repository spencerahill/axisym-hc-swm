from dataclasses import dataclass, field
import numpy as np

from .grid import StaggeredGrid

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
    k_u: float = 1e5  # eddy viscosity on u (momentum diffusion); replaces the old Asselin damping
    emfd_gate_width: float = 0.0  # tanh smoothing width [m/s] for the EMFD H(u) gate; 0 = hard Heaviside
    k_u4: float = 0.0  # biharmonic hyperdiffusion on u (-k_u4 d^4u/dy^4) [m^4/s]; 0 = off
    epsilon_u: float = 1e-8
    delta_z: float = 60
    delta: float = 4e3
    tau: float = 37.0 * SECONDS_PER_DAY
    v_d: float = 2.5
    dt: int = 3600
    ny: int = 50  # number of cell centers N (even recommended for exact symmetry)
    domain_size: float = 15751e3 * 2
    dy: float = field(init=False)
    y: np.ndarray = field(init=False)   # cell centers (carry u, theta)
    yf: np.ndarray = field(init=False)  # cell faces (carry v)
    coeff_eddy_heat_diff: float = 0.0  # values <1e4 make little difference
    include_vert_advec_u: bool = True
    include_merid_advec_u: bool = True  # Toggle for v*du/dy meridional advection term
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
        # Staggered Arakawa C-grid: ny cell centers carry u/theta, ny+1 faces
        # carry v (boundary faces fixed at v=0). dy = domain_size / ny.
        grid = StaggeredGrid(n=self.ny, domain_size=self.domain_size)
        self.dy = grid.dy
        self.y = grid.yc
        self.yf = grid.yf

        # Validate steady-state parameters
        if self.enable_steady_state and self.steady_state_window_size > self.total_integration_days:
            import warnings
            warnings.warn(
                f"Steady-state window size ({self.steady_state_window_size}) exceeds "
                f"total integration days ({self.total_integration_days}). "
                f"Convergence detection will not trigger."
            )
