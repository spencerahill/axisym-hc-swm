import argparse
from .theta_e import ThetaEConfig, SS09Profile, Sin2Profile, SB08Profile
from .sw_config import SWConfig
from .sw_model import SECONDS_PER_DAY, SWModel


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
    return parser.parse_args()


def setup_sw_config(args) -> SWConfig:
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
    )


def setup_theta_e_config(args) -> ThetaEConfig:
    return ThetaEConfig(
        theta_00=args.theta_00,
        y_0=args.y_0,
        y_one=args.y_one,
        delta_y=args.delta_y,
        theta_e_type=args.theta_e_type,
    )


def main():
    args = parse_arguments()
    config = setup_sw_config(args)
    theta_e_config = setup_theta_e_config(args)

    # Instantiate the appropriate ThetaEProfile
    theta_e_profile_class = {
        "SS09": SS09Profile,
        "sin2": Sin2Profile,
        "SB08": SB08Profile,
    }[theta_e_config.theta_e_type]
    theta_e_profile = theta_e_profile_class(theta_e_config)

    model = SWModel(config, theta_e_profile)
    model.run_sim()
    model.save_results()


if __name__ == "__main__":
    main()
