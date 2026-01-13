from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .sw_config import SWConfig
    from .theta_e import ThetaEConfig


def generate_descriptive_path(
    config: "SWConfig",
    theta_e_config: "ThetaEConfig",
    base_dir: str = "./model_output",
    timestamp: str = None,
) -> Tuple[str, str]:
    """
    Generate descriptive output and restart directory paths.

    Args:
        config: Model configuration
        theta_e_config: Theta-e profile configuration
        base_dir: Base output directory
        timestamp: Optional timestamp (YYYYMMDD_HHMMSS). If None, uses current time.

    Returns:
        (output_path, restart_dir): Full paths for output file and restart directory

    Example:
        >>> # With SB08 profile, no seasonal cycle, 51 grid points, 3600 days
        >>> generate_descriptive_path(config, theta_e_config)
        ('./model_output/SB08/run_20260111_134530_noseas_y0p0000_ny051_3600days_output.nc',
         './model_output/SB08')
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Directory: base_dir/theta_e_type/
    profile_dir = Path(base_dir) / theta_e_config.theta_e_type

    # Seasonal indicator (simplified)
    seasonal_str = "seas" if theta_e_config.y_0_seasonal_amp > 0 else "noseas"

    # Y0 value (always included, in km with sign indicator)
    y0_km = int(theta_e_config.y_0 / 1000)
    if y0_km >= 0:
        y0_str = f"y0p{y0_km:04d}"
    else:
        y0_str = f"y0n{abs(y0_km):04d}"

    # Resolution and duration strings
    resolution_str = f"ny{config.ny:03d}"
    duration_str = f"{config.total_integration_days}days"

    # Construct filename base (without file type suffix)
    filename_base = (
        f"run_{timestamp}_{seasonal_str}_{y0_str}_{resolution_str}_{duration_str}"
    )

    # Full paths
    output_path = str(profile_dir / f"{filename_base}_output.nc")
    restart_dir = str(profile_dir)

    return output_path, restart_dir


def generate_restart_filename(output_path: str, day: int) -> str:
    """
    Generate restart filename matching the output file's naming scheme.

    Args:
        output_path: Path to output file
        day: Day number for checkpoint

    Returns:
        Full path to restart file

    Example:
        >>> generate_restart_filename(
        ...     "./model_output/SB08/run_20260111_134530_seas_y0p0000_ny051_3600days_output.nc",
        ...     100
        ... )
        './model_output/SB08/run_20260111_134530_seas_y0p0000_ny051_3600days_restart_day0100.nc'
    """
    # Replace _output.nc with _restart_day{NNNN}.nc
    if output_path.endswith("_output.nc"):
        base = output_path[:-10]  # Remove "_output.nc"
        return f"{base}_restart_day{day:04d}.nc"
    else:
        # Fallback for custom paths
        directory = Path(output_path).parent
        return str(directory / f"restart_day{day:04d}.nc")
