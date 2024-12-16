import argparse
from .sw_model import SWConfig, SWModel


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
        choices=["SS09", "sin2"],
        default="sin2",
        help="Profile to use for theta_e calculation (default: sin2)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = SWConfig(
        total_integration_days=args.total_integration_days,
        gravity=args.gravity,
        height=args.height,
        beta=args.beta,
        t_ref=args.t_ref,
        output_path=args.output_path,
        ny=args.ny,
        dt=args.dt,
        theta_e_type=args.theta_e_type,
    )
    model = SWModel(config)
    model.run_sim()
    model.save_results()


if __name__ == "__main__":
    main()
