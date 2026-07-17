import argparse
import logging
from .theta_e import ThetaEConfig, SS09Profile, Sin2Profile, SB08Profile
from .sw_config import SWConfig
from .sw_model import SECONDS_PER_DAY, SWModel
from .output_path_utils import generate_descriptive_path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the S-S model simulation.")
    parser.add_argument(
        "--ndays",
        type=int,
        default=None,  # None to detect if user explicitly provided it
        dest="total_integration_days",
        help="Total number of integration days (default: 250, or 200000 with --stop-at-steady-state)",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=9.81,
        help="Gravitational acceleration (default: 9.81 m/s^2)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=16e3,
        help="Height of the model (default: 16000 m)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2e-11,
        help="Beta parameter (default: 2e-11)",
    )
    parser.add_argument(
        "--t-ref",
        type=float,
        default=300.0,
        dest="t_ref",
        help="Reference temperature (default: 300 K)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./model_output/output.nc",
        dest="output_path",
        help="Path to save the output NetCDF file",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=51,
        help="Number of grid points in the y-direction (default: 51)",
    )
    parser.add_argument(
        "--dt",
        type=int,
        default=3600,
        help="Time step size in seconds (default: 3600)",
    )
    parser.add_argument(
        "--theta-e-type",
        type=str,
        choices=["SS09", "sin2", "SB08"],
        default="sin2",
        dest="theta_e_type",
        help="Profile to use for theta_e calculation (default: sin2)",
    )
    parser.add_argument(
        "--y0",
        type=float,
        default=0.0,
        dest="y_0",
        help="Position in meters north of the domain center where the theta_e profile peaks (default: 0.0)",
    )
    parser.add_argument(
        "--delta-y",
        type=float,
        default=50.0,
        dest="delta_y",
        help="Delta y for theta_e profile (default: 50.0)",
    )
    parser.add_argument(
        "--theta00",
        type=float,
        default=330.0,
        dest="theta_00",
        help="Peak temperature of the θₑ profile (default: 330.0 K)",
    )
    parser.add_argument(
        "--y1",
        type=float,
        default=9439e3,
        dest="y_one",
        help="Width parameter for the θₑ profile (default: 9439e3 m)",
    )
    # Seasonal cycle arguments (ITCZ migration)
    parser.add_argument(
        "--y0-seas-amp",
        type=float,
        default=0.0,
        dest="y_0_seasonal_amp",
        help="Amplitude of y_0 (ITCZ) migration in m (default: 0, no seasonal cycle)",
    )
    parser.add_argument(
        "--seas-period",
        type=float,
        default=360.0,
        dest="seasonal_period_days",
        help="Seasonal period in days (default: 360)",
    )
    parser.add_argument(
        "--seas-phase",
        type=float,
        default=0.0,
        dest="seasonal_phase_days",
        help="Phase offset in days (default: 0)",
    )
    parser.add_argument(
        "--seas-cycle-type",
        type=str,
        choices=["sin", "square", "tanh"],
        default="sin",
        dest="seasonal_cycle_type",
        help="Shape of seasonal cycle: 'sin' (sinusoidal, default), 'square' (instant flip), or 'tanh' (smoothed square wave)",
    )
    parser.add_argument(
        "--tanh-steepness",
        type=float,
        default=4.0,
        dest="tanh_steepness",
        help="Steepness of tanh smoothing for seasonal cycle (default: 4.0, only used with --seas-cycle-type tanh)",
    )
    parser.add_argument(
        "--coeff-eddy-heat-diff",
        type=float,
        default=0.0,
        dest="coeff_eddy_heat_diff",
        help="Diffusivity constant for eddy heat flux (default: 0.0, inactive)",
    )
    parser.add_argument(
        "--kv",
        type=float,
        default=7786 * 100,
        dest="k_v",
        help="Vertical eddy viscosity (default: 778600)",
    )
    parser.add_argument(
        "--eps-u",
        type=float,
        default=1e-8,
        dest="epsilon_u",
        help="Rayleigh drag coefficient for u (default: 1e-8)",
    )
    parser.add_argument(
        "--delta-z",
        type=float,
        default=60,
        dest="delta_z",
        help="Vertical potential temperature gradient (default: 60 K)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=4e3,
        help="Height of upper-tropospheric layer (default: 4000 m)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=37.0 * SECONDS_PER_DAY,
        help="Newtonian cooling timescale (default: 37 days)",
    )
    parser.add_argument(
        "--vd",
        type=float,
        default=2.5,
        dest="v_d",
        help="Eddy momentum flux divergence coefficient (default: 2.5)",
    )
    parser.add_argument(
        "--domain-size",
        type=float,
        default=15751e3 * 2,
        dest="domain_size",
        help="Size of the domain (default: 31502000 m)",
    )
    parser.add_argument(
        "--asselin-coef",
        type=float,
        default=0.04,
        dest="asselin_filt_coef",
        help="Asselin filter coefficient (default: 0.04)",
    )
    parser.add_argument(
        "--no-vert-advec-u",
        action="store_false",
        dest="include_vert_advec_u",
        help="Disable vertical advection of zonal momentum (default: enabled)",
    )
    parser.add_argument(
        "--no-merid-advec-u",
        action="store_false",
        dest="include_merid_advec_u",
        help="Disable meridional advection of zonal momentum (v*du/dy) (default: enabled)",
    )
    parser.add_argument(
        "--emfd-heaviside-gate",
        action="store_true",
        default=True,
        dest="emfd_heaviside_gate",
        help=(
            "Apply the H(u) gate to the EMFD, per the papers' written "
            "equations (SS09 Eq. 2.5) (the production default)"
        ),
    )
    parser.add_argument(
        "--no-emfd-heaviside-gate",
        action="store_false",
        dest="emfd_heaviside_gate",
        help=(
            "Disable the H(u) gate, for the published Zhang et al. (2025) "
            "code (pair with --emfd-stencil centered)"
        ),
    )
    parser.add_argument(
        "--emfd-stencil",
        type=str,
        choices=["centered", "upwind", "mc"],
        default=None,  # None to detect conflicts with the --emfd-upwind alias
        dest="emfd_stencil",
        help=(
            "Spatial stencil for the EMFD du/dy: 'mc' (MUSCL with "
            "monotonized-central limited slopes, the production default), "
            "'upwind' (first-order one-sided per SS09 section 2b), or "
            "'centered' (np.gradient, the published Zhang et al. (2025) code; "
            "pair with --no-emfd-heaviside-gate for the gateless path)"
        ),
    )
    parser.add_argument(
        "--emfd-upwind",
        action="store_true",
        default=False,
        dest="emfd_upwind",
        help="Deprecated alias for --emfd-stencil upwind",
    )
    parser.add_argument(
        "--grid",
        type=str,
        choices=["staggered", "collocated"],
        default="staggered",
        dest="grid",
        help=(
            "v-grid layout: 'staggered' (v on the ny-1 interior cell faces, "
            "the production default) or 'collocated' (legacy v on the same "
            "ny centers as u, the Zhang et al. (2025) reproduction path)"
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["numpy", "numba"],
        default="numpy",
        dest="backend",
        help=(
            "Integration backend: 'numpy' (the reference implementation, "
            "default) or 'numba' (JIT-compiled fused day kernel, bitwise-"
            "identical to the reference; requires numba installed, the "
            "staggered grid, and a dt that divides 86400)"
        ),
    )
    parser.add_argument(
        "--migrate-restart",
        action="store_true",
        default=False,
        dest="migrate_restart",
        help=(
            "Permit continuing a collocated restart file under a staggered "
            "run (or vice versa) by interpolating v between the center and "
            "face grids once at load time. Without this flag, a grid mismatch "
            "between the restart file and the run is refused."
        ),
    )
    # Steady-state detection arguments
    parser.add_argument(
        "--stop-at-steady-state",
        action="store_true",
        dest="enable_steady_state",
        help="Stop simulation early when convergence criteria are met",
    )
    parser.add_argument(
        "--steady-state-window",
        type=int,
        default=10,
        dest="steady_state_window_size",
        help="Number of days to use for steady-state convergence check (default: 10)",
    )
    parser.add_argument(
        "--steady-state-thresh",
        type=float,
        default=0.001,
        dest="steady_state_threshold",
        help="Relative change threshold for convergence, e.g., 0.001 = 0.1%% (default: 0.001)",
    )
    parser.add_argument(
        "--steady-state-either",
        action="store_false",
        dest="steady_state_check_both",
        help="Require only one metric (KE or Tvar) to converge instead of both (default: require both)",
    )
    parser.add_argument(
        "--v-smooth-thresh",
        type=float,
        default=0.5,
        dest="smoothness_threshold",
        help="Neighbor correlation threshold for v field smoothness warning (default: 0.5)",
    )
    parser.add_argument(
        "--slow-drift-gate",
        action="store_true",
        default=False,
        dest="slow_drift_gate",
        help=(
            "Additionally require the slow diagnostics (jet latitude, "
            "max |v|, equatorial depression) to have a relative range below "
            "--slow-drift-thresh over the trailing --slow-drift-window days "
            "before stopping; requires --stop-at-steady-state, non-seasonal "
            "runs only (default: disabled)"
        ),
    )
    parser.add_argument(
        "--slow-drift-window",
        type=int,
        default=0,
        dest="slow_drift_window",
        help=(
            "Trailing window in days for the slow-drift gate; 0 = auto: "
            "the drag timescale 1/epsilon_u in days (default: 0)"
        ),
    )
    parser.add_argument(
        "--slow-drift-thresh",
        type=float,
        default=0.002,
        dest="slow_drift_thresh",
        help="Relative range threshold for the slow-drift gate (default: 0.002 = 0.2%%)",
    )
    # Seasonal convergence arguments (for seasonally-varying forcing)
    parser.add_argument(
        "--seas-conv",
        action="store_true",
        dest="seasonal_convergence_enabled",
        help=(
            "Stop when seasonal cycle converges; requires "
            "--stop-at-steady-state (default: disabled)"
        ),
    )
    parser.add_argument(
        "--seas-conv-window",
        type=int,
        default=30,
        dest="seasonal_convergence_window",
        help="Number of days that must match previous year for convergence (default: 30)",
    )
    parser.add_argument(
        "--seas-conv-thresh",
        type=float,
        default=0.01,
        dest="seasonal_convergence_threshold",
        help="Relative change threshold for year-to-year comparison (default: 0.01 = 1%%)",
    )
    # Restart/checkpoint arguments
    parser.add_argument(
        "--restart-from",
        type=str,
        default=None,
        dest="restart_file",
        help="Path to restart file to continue from (default: None, fresh start)",
    )
    parser.add_argument(
        "--restart-every",
        type=int,
        default=0,
        dest="save_restart_every",
        help="Save restart file every N days (default: 0, only save at end)",
    )
    parser.add_argument(
        "--restart-dir",
        type=str,
        default="./model_output",
        dest="restart_output_dir",
        help="Directory for restart files (default: ./model_output)",
    )
    args = parser.parse_args()

    # Resolve the deprecated --emfd-upwind alias against --emfd-stencil
    if args.emfd_upwind:
        if args.emfd_stencil is not None and args.emfd_stencil != "upwind":
            raise SystemExit(
                "Error: --emfd-upwind is an alias for --emfd-stencil upwind "
                f"and conflicts with --emfd-stencil {args.emfd_stencil}."
            )
        args.emfd_stencil = "upwind"
    elif args.emfd_stencil is None:
        args.emfd_stencil = "mc"

    # Handle mutual exclusivity: --ndays and --stop-at-steady-state
    ndays_provided = args.total_integration_days is not None
    if ndays_provided and args.enable_steady_state:
        raise SystemExit(
            "Error: Cannot specify both --ndays and --stop-at-steady-state. "
            "Use --ndays for fixed-length runs, or --stop-at-steady-state "
            "for convergence-based termination."
        )

    # --seas-conv reads the daily history the steady-state detector records,
    # and the detector records only when --stop-at-steady-state is on; alone,
    # --seas-conv was a silent no-op (the run completed its full length).
    if args.seasonal_convergence_enabled and not args.enable_steady_state:
        raise SystemExit(
            "Error: --seas-conv requires --stop-at-steady-state (the "
            "steady-state detector records the daily history the seasonal "
            "convergence check compares year-to-year)."
        )

    # Apply conditional default for ndays
    if args.total_integration_days is None:
        if args.enable_steady_state:
            args.total_integration_days = 200000  # Large default for convergence runs
        else:
            args.total_integration_days = 250  # Original default

    return args


def _resolve_emfd_stencil(args) -> str:
    """Resolve the EMFD stencil, honoring the deprecated emfd_upwind flag on
    args objects that predate emfd_stencil (hand-built test/script args)."""
    stencil = getattr(args, "emfd_stencil", None)
    if stencil is None:
        return "upwind" if getattr(args, "emfd_upwind", False) else "mc"
    return stencil


def setup_sw_config(args, theta_e_config: ThetaEConfig) -> SWConfig:
    """
    Setup SWConfig, generating descriptive output path if using defaults.

    Args:
        args: Parsed command-line arguments
        theta_e_config: ThetaE configuration (needed for path generation)

    Returns:
        SWConfig instance
    """
    # Check if user provided custom output_path or restart_output_dir
    user_provided_output = args.output_path != "./model_output/output.nc"
    user_provided_restart_dir = args.restart_output_dir != "./model_output"

    config = SWConfig(
        total_integration_days=args.total_integration_days,
        gravity=args.gravity,
        height=args.height,
        beta=args.beta,
        t_ref=args.t_ref,
        output_path=args.output_path,
        ny=args.ny,
        dt=args.dt,
        coeff_eddy_heat_diff=args.coeff_eddy_heat_diff,
        k_v=args.k_v,
        epsilon_u=args.epsilon_u,
        delta_z=args.delta_z,
        delta=args.delta,
        tau=args.tau,
        v_d=args.v_d,
        domain_size=args.domain_size,
        asselin_filt_coef=args.asselin_filt_coef,
        include_vert_advec_u=args.include_vert_advec_u,
        include_merid_advec_u=args.include_merid_advec_u,
        emfd_heaviside_gate=getattr(args, "emfd_heaviside_gate", False),
        emfd_stencil=_resolve_emfd_stencil(args),
        grid=getattr(args, "grid", "staggered"),
        backend=getattr(args, "backend", "numpy"),
        enable_steady_state=args.enable_steady_state,
        steady_state_window_size=args.steady_state_window_size,
        steady_state_threshold=args.steady_state_threshold,
        steady_state_check_both=args.steady_state_check_both,
        smoothness_threshold=args.smoothness_threshold,
        slow_drift_gate=getattr(args, "slow_drift_gate", False),
        slow_drift_window=getattr(args, "slow_drift_window", 0),
        slow_drift_thresh=getattr(args, "slow_drift_thresh", 0.002),
        # Seasonal convergence parameters (use getattr for backward compatibility with tests)
        seasonal_convergence_enabled=getattr(args, 'seasonal_convergence_enabled', False),
        seasonal_convergence_window=getattr(args, 'seasonal_convergence_window', 30),
        seasonal_convergence_threshold=getattr(args, 'seasonal_convergence_threshold', 0.01),
        # Restart/checkpoint parameters
        save_restart_every=getattr(args, 'save_restart_every', 0),
        restart_output_dir=args.restart_output_dir,
    )

    # Fill in descriptive paths for any field the user left at its default.
    # generate_descriptive_path only reads ny/total_integration_days from
    # config; output_path/restart_output_dir derive nothing in __post_init__,
    # so mutating them here is safe.
    if not user_provided_output or not user_provided_restart_dir:
        output_path, restart_dir = generate_descriptive_path(
            config, theta_e_config, base_dir="./model_output"
        )
        if not user_provided_output:
            config.output_path = output_path
        if not user_provided_restart_dir:
            config.restart_output_dir = restart_dir

    return config


def setup_theta_e_config(args) -> ThetaEConfig:
    return ThetaEConfig(
        theta_00=args.theta_00,
        y_0=args.y_0,
        y_one=args.y_one,
        delta_y=args.delta_y,
        theta_e_type=args.theta_e_type,
        # Seasonal cycle parameters (use getattr for backward compatibility with tests)
        y_0_seasonal_amp=getattr(args, 'y_0_seasonal_amp', 0.0),
        seasonal_period_days=getattr(args, 'seasonal_period_days', 360.0),
        seasonal_phase_days=getattr(args, 'seasonal_phase_days', 0.0),
        seasonal_cycle_type=getattr(args, 'seasonal_cycle_type', "sin"),
        tanh_steepness=getattr(args, 'tanh_steepness', 4.0),
    )


def main():
    import os

    args = parse_arguments()

    # Create theta_e_config first (needed for path generation in setup_sw_config)
    theta_e_config = setup_theta_e_config(args)

    # Create sw_config (will generate descriptive paths if defaults are used)
    config = setup_sw_config(args, theta_e_config)

    # Log the generated paths for user visibility
    logging.info(f"Output path: {config.output_path}")
    logging.info(f"Restart directory: {config.restart_output_dir}")

    # Instantiate the appropriate ThetaEProfile
    theta_e_profile_class = {
        "SS09": SS09Profile,
        "sin2": Sin2Profile,
        "SB08": SB08Profile,
    }[theta_e_config.theta_e_type]
    theta_e_profile = theta_e_profile_class(theta_e_config)

    model = SWModel(config, theta_e_profile)

    # Load from restart if specified
    if args.restart_file:
        if not os.path.exists(args.restart_file):
            raise FileNotFoundError(f"Restart file not found: {args.restart_file}")
        model.restart_day = model.load_from_restart(
            args.restart_file, migrate=getattr(args, "migrate_restart", False)
        )

    model.run_sim()
    model.save_results()


if __name__ == "__main__":
    main()
