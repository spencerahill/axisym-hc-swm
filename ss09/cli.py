import argparse
import logging
from .theta_e import ThetaEConfig, SS09Profile, Sin2Profile, SB08Profile
from .sw_config import SWConfig
from .sw_model import SECONDS_PER_DAY, SWModel
from .output_path_utils import generate_descriptive_path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the S-S model simulation.")
    parser.add_argument(
        "--total_integration_days",
        type=int,
        default=250,
        help="Total number of integration days (default: 250)",
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
        "--t_ref",
        type=float,
        default=300.0,
        help="Reference temperature (default: 300 K)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./model_output/output.nc",
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
        "--theta_e_type",
        type=str,
        choices=["SS09", "sin2", "SB08"],
        default="sin2",
        help="Profile to use for theta_e calculation (default: sin2)",
    )
    parser.add_argument(
        "--y_0",
        type=float,
        default=0.0,
        help="Position in meters north of the domain center where the theta_e profile peaks (default: 0.0)",
    )
    parser.add_argument(
        "--delta_y",
        type=float,
        default=50.0,
        help="Delta y for theta_e profile (default: 50.0)",
    )
    parser.add_argument(
        "--theta_00",
        type=float,
        default=330.0,
        help="Peak temperature of the θₑ profile (default: 330.0 K)",
    )
    parser.add_argument(
        "--y_one",
        type=float,
        default=9439e3,
        help="Width parameter for the θₑ profile (default: 9439e3 m)",
    )
    # Seasonal cycle arguments (ITCZ migration)
    parser.add_argument(
        "--y0-seasonal-amp",
        type=float,
        default=0.0,
        dest="y_0_seasonal_amp",
        help="Amplitude of y_0 (ITCZ) migration in m (default: 0, no seasonal cycle)",
    )
    parser.add_argument(
        "--seasonal-period",
        type=float,
        default=360.0,
        dest="seasonal_period_days",
        help="Seasonal period in days (default: 360)",
    )
    parser.add_argument(
        "--seasonal-phase",
        type=float,
        default=0.0,
        dest="seasonal_phase_days",
        help="Phase offset in days (default: 0)",
    )
    parser.add_argument(
        "--seasonal-cycle-type",
        type=str,
        choices=["sin", "square"],
        default="sin",
        dest="seasonal_cycle_type",
        help="Shape of seasonal cycle: 'sin' (sinusoidal, default) or 'square' (instant flip at half-period)",
    )
    parser.add_argument(
        "--coeff_eddy_heat_diff",
        type=float,
        default=0.0,
        help="Diffusivity constant for eddy heat flux (default: 0.0, inactive)",
    )
    parser.add_argument(
        "--k_v",
        type=float,
        default=7786 * 100,
        help="Vertical eddy viscosity (default: 778600)",
    )
    parser.add_argument(
        "--epsilon_u",
        type=float,
        default=1e-8,
        help="Rayleigh drag coefficient for u (default: 1e-8)",
    )
    parser.add_argument(
        "--delta_z",
        type=float,
        default=60,
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
        "--v_d",
        type=float,
        default=2.5,
        help="Eddy momentum flux divergence coefficient (default: 2.5)",
    )
    parser.add_argument(
        "--domain_size",
        type=float,
        default=15751e3 * 2,
        help="Size of the domain (default: 31502000 m)",
    )
    parser.add_argument(
        "--asselin_filt_coef",
        type=float,
        default=0.04,
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
    # Steady-state detection arguments
    parser.add_argument(
        "--enable-steady-state",
        action="store_true",
        dest="enable_steady_state",
        help="Enable steady-state detection to stop simulation early when convergence criteria are met",
    )
    parser.add_argument(
        "--steady-state-window",
        type=int,
        default=10,
        dest="steady_state_window_size",
        help="Number of days to use for steady-state convergence check (default: 10)",
    )
    parser.add_argument(
        "--steady-state-threshold",
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
        "--smoothness-threshold",
        type=float,
        default=0.5,
        dest="smoothness_threshold",
        help="Neighbor correlation threshold for v field smoothness warning (default: 0.5)",
    )
    # Seasonal convergence arguments (for seasonally-varying forcing)
    parser.add_argument(
        "--enable-seasonal-convergence",
        action="store_true",
        dest="seasonal_convergence_enabled",
        help="Enable automatic stopping when seasonal cycle converges (default: disabled)",
    )
    parser.add_argument(
        "--seasonal-convergence-window",
        type=int,
        default=30,
        dest="seasonal_convergence_window",
        help="Number of days that must match previous year for convergence (default: 30)",
    )
    parser.add_argument(
        "--seasonal-convergence-threshold",
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
        "--save-restart-every",
        type=int,
        default=0,
        dest="save_restart_every",
        help="Save restart file every N days (default: 0, only save at end)",
    )
    parser.add_argument(
        "--restart-output-dir",
        type=str,
        default="./model_output",
        dest="restart_output_dir",
        help="Directory for restart files (default: ./model_output)",
    )
    return parser.parse_args()


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

    # Generate descriptive paths if using defaults
    if not user_provided_output or not user_provided_restart_dir:
        # Create a temporary SWConfig to get ny and total_integration_days
        # (we need these for path generation)
        temp_config = SWConfig(
            total_integration_days=args.total_integration_days,
            ny=args.ny,
            gravity=args.gravity,
            height=args.height,
            beta=args.beta,
            t_ref=args.t_ref,
            output_path="",  # placeholder
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
            enable_steady_state=args.enable_steady_state,
            steady_state_window_size=args.steady_state_window_size,
            steady_state_threshold=args.steady_state_threshold,
            steady_state_check_both=args.steady_state_check_both,
            smoothness_threshold=args.smoothness_threshold,
            seasonal_convergence_enabled=getattr(args, 'seasonal_convergence_enabled', False),
            seasonal_convergence_window=getattr(args, 'seasonal_convergence_window', 30),
            seasonal_convergence_threshold=getattr(args, 'seasonal_convergence_threshold', 0.01),
            save_restart_every=getattr(args, 'save_restart_every', 0),
            restart_output_dir="",  # placeholder
        )

        output_path, restart_dir = generate_descriptive_path(
            temp_config, theta_e_config, base_dir="./model_output"
        )

        # Use generated paths unless user overrode them
        if not user_provided_output:
            args.output_path = output_path
        if not user_provided_restart_dir:
            args.restart_output_dir = restart_dir

    return SWConfig(
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
        enable_steady_state=args.enable_steady_state,
        steady_state_window_size=args.steady_state_window_size,
        steady_state_threshold=args.steady_state_threshold,
        steady_state_check_both=args.steady_state_check_both,
        smoothness_threshold=args.smoothness_threshold,
        # Seasonal convergence parameters (use getattr for backward compatibility with tests)
        seasonal_convergence_enabled=getattr(args, 'seasonal_convergence_enabled', False),
        seasonal_convergence_window=getattr(args, 'seasonal_convergence_window', 30),
        seasonal_convergence_threshold=getattr(args, 'seasonal_convergence_threshold', 0.01),
        # Restart/checkpoint parameters
        save_restart_every=getattr(args, 'save_restart_every', 0),
        restart_output_dir=args.restart_output_dir,
    )


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
        model.restart_day = model.load_from_restart(args.restart_file)

    model.run_sim()
    model.save_results()


if __name__ == "__main__":
    main()
