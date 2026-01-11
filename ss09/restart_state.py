"""Restart state management for simulation checkpoints."""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import numpy as np
import xarray as xr
import logging
from datetime import datetime


@dataclass
class RestartState:
    """
    Complete state for restarting a simulation.

    Contains all information needed to exactly continue a simulation from a checkpoint,
    including instantaneous field values (NOT daily averages) and metadata.
    """

    # Time information
    current_time: float  # seconds from start
    current_step: int  # absolute timestep number
    current_day: int  # day index

    # Current INSTANTANEOUS state (timestep n)
    u: np.ndarray  # shape (ny,) - zonal wind
    v: np.ndarray  # shape (ny,) - meridional wind
    theta: np.ndarray  # shape (ny,) - potential temperature

    # Previous INSTANTANEOUS filtered state (timestep n-1) for leapfrog
    u_prev: np.ndarray  # shape (ny,)
    v_prev: np.ndarray  # shape (ny,)
    theta_prev: np.ndarray  # shape (ny,)

    # Grid
    y: np.ndarray  # shape (ny,) - meridional coordinate

    # Steady-state detector state (if enabled)
    steady_state_enabled: bool
    kinetic_energy_history: List[float]
    temp_variance_history: List[float]
    v_smoothness_history: List[float]
    v_grid_variance_history: List[float]
    is_converged: bool
    convergence_day: Optional[int]
    ke_converged: bool
    tvar_converged: bool
    smoothness_warning_issued: bool

    # Configuration snapshots (for validation on restart)
    config_snapshot: Dict[str, Any]  # SWConfig as dict
    theta_e_config_snapshot: Dict[str, Any]  # ThetaEConfig as dict

    def to_netcdf(self, filepath: str) -> None:
        """
        Write restart state to NetCDF file.

        Args:
            filepath: Path to output NetCDF file
        """
        ny = len(self.y)
        history_length = len(self.kinetic_energy_history)

        # Create xarray Dataset
        ds = xr.Dataset(
            # Scalar time metadata
            data_vars={
                "current_time": ([], self.current_time, {"units": "seconds", "long_name": "Current simulation time"}),
                "current_step": ([], self.current_step, {"long_name": "Current timestep number"}),
                "current_day": ([], self.current_day, {"long_name": "Current day index"}),
                # Current instantaneous state
                "u": (["y"], self.u, {"units": "m/s", "long_name": "Instantaneous zonal wind (timestep n)"}),
                "v": (["y"], self.v, {"units": "m/s", "long_name": "Instantaneous meridional wind (timestep n)"}),
                "theta": (["y"], self.theta, {"units": "K", "long_name": "Instantaneous potential temperature (timestep n)"}),
                # Previous instantaneous filtered state
                "u_prev": (["y"], self.u_prev, {"units": "m/s", "long_name": "Instantaneous zonal wind (timestep n-1, filtered)"}),
                "v_prev": (["y"], self.v_prev, {"units": "m/s", "long_name": "Instantaneous meridional wind (timestep n-1, filtered)"}),
                "theta_prev": (["y"], self.theta_prev, {"units": "K", "long_name": "Instantaneous potential temperature (timestep n-1, filtered)"}),
                # Steady-state detector history
                "kinetic_energy_history": (
                    ["history_length"],
                    np.array(self.kinetic_energy_history),
                    {"units": "m^2/s^2", "long_name": "Kinetic energy history for steady-state detection"},
                ),
                "temp_variance_history": (
                    ["history_length"],
                    np.array(self.temp_variance_history),
                    {"units": "K^2", "long_name": "Temperature variance history for steady-state detection"},
                ),
                "v_smoothness_history": (
                    ["history_length"],
                    np.array(self.v_smoothness_history),
                    {"long_name": "v-field neighbor correlation history"},
                ),
                "v_grid_variance_history": (
                    ["history_length"],
                    np.array(self.v_grid_variance_history),
                    {"units": "m^2/s^2", "long_name": "v-field grid-scale variance history"},
                ),
                # Detector state flags
                "steady_state_enabled": ([], int(self.steady_state_enabled), {"long_name": "Steady-state detection enabled"}),
                "is_converged": ([], int(self.is_converged), {"long_name": "Steady-state converged flag"}),
                "convergence_day": ([], self.convergence_day if self.convergence_day is not None else -1, {"long_name": "Day of convergence (-1 if not converged)"}),
                "ke_converged": ([], int(self.ke_converged), {"long_name": "Kinetic energy converged flag"}),
                "tvar_converged": ([], int(self.tvar_converged), {"long_name": "Temperature variance converged flag"}),
                "smoothness_warning_issued": ([], int(self.smoothness_warning_issued), {"long_name": "v-field smoothness warning issued flag"}),
            },
            coords={
                "y": (["y"], self.y, {"units": "m", "long_name": "Meridional distance"}),
                "history_length": np.arange(history_length),
            },
        )

        # Add global attributes
        ds.attrs["title"] = "Shallow Water Model Restart File"
        ds.attrs["creation_time"] = datetime.now().isoformat()
        ds.attrs["restart_type"] = "day_boundary"
        ds.attrs["description"] = "Contains instantaneous state for exact simulation continuation"

        # Add all config parameters as global attributes
        # Note: NetCDF doesn't support boolean type, convert to int
        for key, value in self.config_snapshot.items():
            if isinstance(value, bool):
                ds.attrs[f"config_{key}"] = int(value)
            elif isinstance(value, (int, float, str)):
                ds.attrs[f"config_{key}"] = value
            elif isinstance(value, np.ndarray):
                # Skip arrays (like y), already saved as variables
                continue
            else:
                ds.attrs[f"config_{key}"] = str(value)

        for key, value in self.theta_e_config_snapshot.items():
            if isinstance(value, bool):
                ds.attrs[f"theta_e_{key}"] = int(value)
            elif isinstance(value, (int, float, str)):
                ds.attrs[f"theta_e_{key}"] = value
            else:
                ds.attrs[f"theta_e_{key}"] = str(value)

        # Write to file
        ds.to_netcdf(filepath)
        logging.info(f"Wrote restart file: {filepath}")

    @classmethod
    def from_netcdf(cls, filepath: str) -> "RestartState":
        """
        Load restart state from NetCDF file.

        Args:
            filepath: Path to input NetCDF file

        Returns:
            RestartState object with all fields populated
        """
        ds = xr.open_dataset(filepath)

        # Extract scalar time metadata
        current_time = float(ds["current_time"].values)
        current_step = int(ds["current_step"].values)
        current_day = int(ds["current_day"].values)

        # Extract instantaneous state arrays
        u = ds["u"].values
        v = ds["v"].values
        theta = ds["theta"].values
        u_prev = ds["u_prev"].values
        v_prev = ds["v_prev"].values
        theta_prev = ds["theta_prev"].values
        y = ds["y"].values

        # Extract steady-state detector history
        kinetic_energy_history = ds["kinetic_energy_history"].values.tolist()
        temp_variance_history = ds["temp_variance_history"].values.tolist()
        v_smoothness_history = ds["v_smoothness_history"].values.tolist()
        v_grid_variance_history = ds["v_grid_variance_history"].values.tolist()

        # Extract detector flags
        steady_state_enabled = bool(ds["steady_state_enabled"].values)
        is_converged = bool(ds["is_converged"].values)
        convergence_day_val = int(ds["convergence_day"].values)
        convergence_day = convergence_day_val if convergence_day_val >= 0 else None
        ke_converged = bool(ds["ke_converged"].values)
        tvar_converged = bool(ds["tvar_converged"].values)
        smoothness_warning_issued = bool(ds["smoothness_warning_issued"].values)

        # Reconstruct config dictionaries from global attributes
        config_snapshot = {}
        theta_e_config_snapshot = {}

        for attr_name, attr_value in ds.attrs.items():
            if attr_name.startswith("config_"):
                key = attr_name.replace("config_", "")
                config_snapshot[key] = attr_value
            elif attr_name.startswith("theta_e_"):
                key = attr_name.replace("theta_e_", "")
                theta_e_config_snapshot[key] = attr_value

        ds.close()

        logging.info(f"Loaded restart file: {filepath}")

        return cls(
            current_time=current_time,
            current_step=current_step,
            current_day=current_day,
            u=u,
            v=v,
            theta=theta,
            u_prev=u_prev,
            v_prev=v_prev,
            theta_prev=theta_prev,
            y=y,
            steady_state_enabled=steady_state_enabled,
            kinetic_energy_history=kinetic_energy_history,
            temp_variance_history=temp_variance_history,
            v_smoothness_history=v_smoothness_history,
            v_grid_variance_history=v_grid_variance_history,
            is_converged=is_converged,
            convergence_day=convergence_day,
            ke_converged=ke_converged,
            tvar_converged=tvar_converged,
            smoothness_warning_issued=smoothness_warning_issued,
            config_snapshot=config_snapshot,
            theta_e_config_snapshot=theta_e_config_snapshot,
        )

    def validate_compatibility(self, config: Any, theta_e_config: Any) -> None:
        """
        Validate that restart state is compatible with current configuration.

        Raises errors for critical mismatches, warnings for non-critical ones.

        Args:
            config: Current SWConfig object
            theta_e_config: Current ThetaEConfig object

        Raises:
            ValueError: If critical parameters don't match
        """
        errors = []
        warnings = []

        # Critical parameters that must match
        critical_params = {
            "ny": ("Grid points", config.ny, self.config_snapshot.get("ny")),
            "dt": ("Timestep", config.dt, self.config_snapshot.get("dt")),
            "dy": ("Grid spacing", config.dy, self.config_snapshot.get("dy")),
            "domain_size": ("Domain size", config.domain_size, self.config_snapshot.get("domain_size")),
            "gravity": ("Gravity", config.gravity, self.config_snapshot.get("gravity")),
            "height": ("Height", config.height, self.config_snapshot.get("height")),
            "beta": ("Beta", config.beta, self.config_snapshot.get("beta")),
        }

        for param_name, (desc, current, restart) in critical_params.items():
            if restart is not None and not np.isclose(current, restart, rtol=1e-10):
                errors.append(f"{desc} mismatch: current={current}, restart={restart}")

        # Check theta_e config
        theta_e_critical = {
            "theta_e_type": theta_e_config.theta_e_type,
            "theta_00": theta_e_config.theta_00,
            "y_0": theta_e_config.y_0,
            "y_one": theta_e_config.y_one,
            "delta_y": theta_e_config.delta_y,
        }

        for param_name, current_val in theta_e_critical.items():
            restart_val = self.theta_e_config_snapshot.get(param_name)
            if restart_val is not None:
                if isinstance(current_val, (int, float)):
                    if not np.isclose(current_val, restart_val, rtol=1e-10):
                        errors.append(f"Theta_e {param_name} mismatch: current={current_val}, restart={restart_val}")
                elif current_val != restart_val:
                    errors.append(f"Theta_e {param_name} mismatch: current={current_val}, restart={restart_val}")

        # Non-critical parameters (just warn)
        if config.total_integration_days != self.config_snapshot.get("total_integration_days"):
            warnings.append(
                f"Total integration days differ: "
                f"current={config.total_integration_days}, restart={self.config_snapshot.get('total_integration_days')}. "
                f"This is normal when extending a simulation."
            )

        # Report errors
        if errors:
            error_msg = "Restart file incompatible with current configuration:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

        # Report warnings
        for warning in warnings:
            logging.warning(warning)

        logging.info("Restart file validated successfully - configuration compatible")
