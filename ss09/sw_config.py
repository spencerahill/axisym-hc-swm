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
    asselin_filt_coef: float = 0.04
    coeff_eddy_heat_diff: float = 0.0  # values <1e4 make little difference
    include_vert_advec_u: bool = True
    # Steady-state detection parameters
    enable_steady_state: bool = False
    steady_state_window_size: int = 10
    steady_state_threshold: float = 0.001
    steady_state_check_both: bool = True
    smoothness_threshold: float = 0.5  # Neighbor correlation threshold for v field smoothness

    def __post_init__(self):
        self.dy = self.domain_size / (self.ny - 1)
        self.y = np.linspace(-self.domain_size / 2, self.domain_size / 2, self.ny)

        # Validate steady-state parameters
        if self.enable_steady_state and self.steady_state_window_size > self.total_integration_days:
            import warnings
            warnings.warn(
                f"Steady-state window size ({self.steady_state_window_size}) exceeds "
                f"total integration days ({self.total_integration_days}). "
                f"Convergence detection will not trigger."
            )
