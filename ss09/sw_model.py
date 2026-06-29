"""
The Sobel-Schneider single-layer shallow-water model.

Integrates u (zonal wind) and theta (potential temperature) on cell centers and
v (meridional wind) on cell faces using a staggered Arakawa C-grid (ss09.grid),
flux-form spatial operators (ss09.rhs), and a self-starting fixed-step RK4
method-of-lines time integrator (ss09.integrators).
"""

import os
import logging
import numpy as np
from typing import NamedTuple
from dataclasses import asdict
from .model_state import ModelState
from .theta_e import ThetaEProfile
from .sw_config import SWConfig
from .daily_results import DailyResults
from .steady_state import SteadyStateDetector
from .hadley_diagnostics import HadleyDiagnostics
from .restart_state import RestartState
from .output_path_utils import generate_restart_filename
from .integrators import RK4Integrator
from . import rhs as rhs_module

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
SECONDS_PER_DAY = 86400  # Number of seconds in a day
THETA_TO_TEMP = rhs_module.THETA_TO_TEMP  # Inverse of (p_s/p_t)^(R/c_p)


class TempVars(NamedTuple):
    """
    Temporary storage of model variables for daily averages.

    u, theta, theta_e are on cell centers (ny); v is on cell faces (ny+1).
    """
    u: np.ndarray
    v: np.ndarray
    theta: np.ndarray
    theta_e: np.ndarray
    time: np.ndarray


class SWModel:
    """
    The shallow water model on an equatorial beta plane (staggered C-grid, RK4).
    """
    def __init__(self, config: SWConfig, theta_e_profile: ThetaEProfile):
        self.config = config
        self.theta_e_profile = theta_e_profile
        ny = config.ny
        self.state = ModelState(
            t=0.0,
            u=np.zeros(ny),          # cell centers
            v=np.zeros(ny + 1),      # cell faces (v=0 at the two walls)
            theta=np.zeros(ny),      # cell centers
            y=config.y,              # cell-center coordinate
        )
        self.state = self.state._replace(theta=self.theta_e_profile(self.state))
        self.integrator = RK4Integrator()
        self.results = DailyResults(
            config.total_integration_days, ny,
            store_theta_e=self.has_seasonal_forcing()
        )
        self.temp_vars = TempVars(
            u=np.zeros((0, ny)),
            v=np.zeros((0, ny + 1)),
            theta=np.zeros((0, ny)),
            theta_e=np.zeros((0, ny)),
            time=np.zeros(0),
        )
        self.steady_state_detector = SteadyStateDetector(
            enabled=config.enable_steady_state,
            window_size=config.steady_state_window_size,
            threshold=config.steady_state_threshold,
            check_both_metrics=config.steady_state_check_both,
            smoothness_threshold=config.smoothness_threshold,
        )
        self.hadley_diagnostics = HadleyDiagnostics(
            ny=ny,
            total_days=config.total_integration_days
        )

    def has_seasonal_forcing(self) -> bool:
        """
        Check if the theta_e profile has seasonal forcing (time-varying y_0).

        Returns:
            True if seasonal cycle is active (y_0_seasonal_amp > 0), False otherwise
        """
        if hasattr(self.theta_e_profile, 'config'):
            config = self.theta_e_profile.config
            return getattr(config, 'y_0_seasonal_amp', 0.0) > 0
        return False

    def rhs(self, state: ModelState):
        """Full explicit tendency (du, dv, dtheta) for the current forcing."""
        return rhs_module.rhs(state, self.config, self.theta_e_profile)

    def init_temp_storage(self):
        """Initialize temporary storage for daily averages."""
        steps_per_day = int(SECONDS_PER_DAY / self.config.dt)
        ny = self.config.ny
        self.temp_vars = TempVars(
            u=np.zeros([steps_per_day, ny]),
            v=np.zeros([steps_per_day, ny + 1]),
            theta=np.zeros([steps_per_day, ny]),
            theta_e=np.zeros([steps_per_day, ny]),
            time=np.zeros(steps_per_day),
        )

    def store_temp_results(self, timestamp: float, j: int):
        """Store temporary results for daily averaging."""
        self.temp_vars.u[j - 1] = self.state.u
        self.temp_vars.v[j - 1] = self.state.v
        self.temp_vars.theta[j - 1] = self.state.theta
        self.temp_vars.theta_e[j - 1] = self.theta_e_profile(self.state)
        self.temp_vars.time[j - 1] = timestamp / SECONDS_PER_DAY

    def store_daily_avgs(self, day: int):
        """Store daily averages.

        v is averaged on faces, then interpolated to cell centers so that the
        output, Hadley diagnostics, and steady-state detector all see one
        (cell-center) y-grid.
        """
        theta_e_avg = (
            np.mean(self.temp_vars.theta_e, axis=0)
            if self.has_seasonal_forcing() else None
        )
        v_face_mean = np.mean(self.temp_vars.v, axis=0)
        v_center_mean = 0.5 * (v_face_mean[:-1] + v_face_mean[1:])
        self.results.store_day(
            day,
            np.mean(self.temp_vars.time),
            np.mean(self.temp_vars.u, axis=0),
            v_center_mean,
            np.mean(self.temp_vars.theta, axis=0),
            theta_e_avg,
        )

    def reset_temp_storage(self):
        """Reset temporary storage arrays."""
        self.temp_vars = TempVars(
            u=np.zeros_like(self.temp_vars.u),
            v=np.zeros_like(self.temp_vars.v),
            theta=np.zeros_like(self.temp_vars.theta),
            theta_e=np.zeros_like(self.temp_vars.theta_e),
            time=np.zeros_like(self.temp_vars.time),
        )

    def calc_ind_within_day(self, current_step: int) -> int:
        """Calculate the index within the day."""
        return (current_step + 1) % int(SECONDS_PER_DAY / self.config.dt)

    def run_sim(self):
        """Run the S-S model simulation."""
        total_time_steps = int(
            SECONDS_PER_DAY * self.config.total_integration_days / self.config.dt
        )

        # Determine starting day and step (for restart support)
        if getattr(self, "restart_day", None) is not None:
            day = self.restart_day
            starting_step = int(day * SECONDS_PER_DAY / self.config.dt)
            logging.info(f"Restarting from day {day}, step {starting_step}")
        else:
            day = 0
            starting_step = 0

        self.init_temp_storage()

        for i in range(starting_step, total_time_steps):
            # Self-starting RK4 advance; state.t becomes (i+1)*dt.
            self.state = self.integrator.step(self.state, self.config.dt, self.rhs)

            ind_within_day = self.calc_ind_within_day(i)
            self.store_temp_results(self.state.t, ind_within_day)
            if ind_within_day == 0:
                self.store_daily_avgs(day)

                # Record metrics for steady-state detection
                self.steady_state_detector.record_day(
                    day,
                    self.results.u[day],
                    self.results.v[day],
                    self.results.theta[day],
                    self.config.dy
                )

                # Record Hadley diagnostics
                self.hadley_diagnostics.record_day(
                    day,
                    self.results.u[day],
                    self.results.v[day],
                    self.config.y,
                    self.config.dy,
                    self.config.beta,
                )

                # Check convergence based on forcing type
                if self.has_seasonal_forcing() and self.config.seasonal_convergence_enabled:
                    # Year-to-year comparison for seasonal forcing (only if user enabled it)
                    seasonal_period = self.theta_e_profile.config.seasonal_period_days
                    if self.steady_state_detector.check_seasonal_convergence(
                        day,
                        seasonal_period,
                        window_size=self.config.seasonal_convergence_window,
                        threshold=self.config.seasonal_convergence_threshold
                    ):
                        logging.info(
                            f"Seasonal cycle converged at day {day} "
                            f"({day/seasonal_period:.1f} years). "
                            f"Current year matches previous year within threshold."
                        )
                        break
                elif not self.has_seasonal_forcing():
                    # Traditional steady-state check (only for non-seasonal runs)
                    if self.steady_state_detector.check_convergence(day):
                        logging.info(
                            f"Steady state reached at day {day}. "
                            f"KE converged: {self.steady_state_detector.ke_converged}, "
                            f"Tvar converged: {self.steady_state_detector.tvar_converged}"
                        )
                        break
                # If seasonal forcing but convergence disabled: no early stopping, run full integration

                # Save restart file if periodic checkpointing is enabled
                if self.config.save_restart_every > 0 and day % self.config.save_restart_every == 0 and day > 0:
                    self.save_restart_file(day)

                self.reset_temp_storage()
                day += 1
                logging.info(f"Day {day} finished.")

            if np.isnan(self.state.u).any():
                logging.warning("NaN detected in u, breaking the loop.")
                break

        # Warn if steady-state detection was enabled but convergence not reached
        if self.config.enable_steady_state and not self.steady_state_detector.is_converged:
            logging.warning(
                f"Steady-state convergence was enabled but simulation ended after {day} days "
                f"without reaching convergence. "
                f"KE converged: {self.steady_state_detector.ke_converged}, "
                f"Tvar converged: {self.steady_state_detector.tvar_converged}."
            )

        # Always save final restart file (unless save_restart_every is explicitly 0 to disable all restarts)
        # Note: if save_restart_every == 0, we still save a final restart file for manual continuation
        if day > 0:  # Only save if we actually ran some days
            self.save_restart_file(day)
            logging.info(f"Saved final restart file at day {day}")

    def save_restart_file(self, day: int) -> None:
        """
        Save the instantaneous staggered state to a NetCDF restart file.

        Saves the INSTANTANEOUS state (u, theta on centers; v on faces) at the
        current day boundary, NOT daily averages. RK4 is self-starting, so no
        n-1 level is needed (unlike the old leapfrog scheme).

        Args:
            day: Current day number for filename
        """
        os.makedirs(self.config.restart_output_dir, exist_ok=True)

        restart_state = RestartState(
            current_time=self.state.t,
            current_step=int(round(self.state.t / self.config.dt)),
            current_day=day,
            # INSTANTANEOUS current state - NOT daily averaged
            u=self.state.u.copy(),
            v=self.state.v.copy(),
            theta=self.state.theta.copy(),
            y=self.state.y.copy(),
            yf=self.config.yf.copy(),
            # Steady-state detector state
            steady_state_enabled=self.config.enable_steady_state,
            kinetic_energy_history=self.steady_state_detector.kinetic_energy_history.copy(),
            temp_variance_history=self.steady_state_detector.temp_variance_history.copy(),
            v_smoothness_history=self.steady_state_detector.v_smoothness_history.copy(),
            v_grid_variance_history=self.steady_state_detector.v_grid_variance_history.copy(),
            is_converged=self.steady_state_detector.is_converged,
            convergence_day=self.steady_state_detector.convergence_day,
            ke_converged=self.steady_state_detector.ke_converged,
            tvar_converged=self.steady_state_detector.tvar_converged,
            smoothness_warning_issued=self.steady_state_detector.smoothness_warning_issued,
            config_snapshot=asdict(self.config),
            theta_e_config_snapshot=asdict(self.theta_e_profile.config),
        )

        filepath = generate_restart_filename(self.config.output_path, day)
        restart_state.to_netcdf(filepath)

    def load_from_restart(self, restart_file: str) -> int:
        """
        Load state from restart file and restore model.

        Args:
            restart_file: Path to restart NetCDF file

        Returns:
            starting_day: The day number to resume from
        """
        restart_state = RestartState.from_netcdf(restart_file)
        restart_state.validate_compatibility(self.config, self.theta_e_profile.config)

        # Restore the instantaneous staggered state
        self.state = self.state._replace(
            t=restart_state.current_time,
            u=restart_state.u,
            v=restart_state.v,
            theta=restart_state.theta,
        )

        # Restore steady-state detector state if enabled
        if self.config.enable_steady_state and restart_state.steady_state_enabled:
            self.steady_state_detector.kinetic_energy_history = (
                restart_state.kinetic_energy_history
            )
            self.steady_state_detector.temp_variance_history = (
                restart_state.temp_variance_history
            )
            self.steady_state_detector.v_smoothness_history = (
                restart_state.v_smoothness_history
            )
            self.steady_state_detector.v_grid_variance_history = (
                restart_state.v_grid_variance_history
            )
            self.steady_state_detector.is_converged = restart_state.is_converged
            self.steady_state_detector.convergence_day = restart_state.convergence_day
            self.steady_state_detector.ke_converged = restart_state.ke_converged
            self.steady_state_detector.tvar_converged = restart_state.tvar_converged
            self.steady_state_detector.smoothness_warning_issued = (
                restart_state.smoothness_warning_issued
            )

        logging.info(f"Loaded restart state from day {restart_state.current_day}")
        return restart_state.current_day

    def save_results(self):
        """Save the simulation results to a NetCDF file."""
        ds = self.results.to_xarray(
            self.config, self.theta_e_profile, self.steady_state_detector,
            self.hadley_diagnostics
        )

        # Add coordinate attributes
        ds.y.attrs.update(
            {"units": "m", "long_name": "meridional distance from equator"}
        )

        # Add global attributes
        global_attrs = {
            "title": "Shallow Water Model Output",
            "creation_date": str(np.datetime64("now")),
            **{
                key: (
                    str(getattr(self.config, key))
                    if isinstance(getattr(self.config, key), bool)
                    else getattr(self.config, key)
                )
                for key in self.config.__dataclass_fields__
            },
            **self.steady_state_detector.get_convergence_info(),
            # Add theta_e configuration for reproducibility
            **{
                f"theta_e_{key}": (
                    str(getattr(self.theta_e_profile.config, key))
                    if isinstance(getattr(self.theta_e_profile.config, key), bool)
                    else getattr(self.theta_e_profile.config, key)
                )
                for key in self.theta_e_profile.config.__dataclass_fields__
            },
        }
        ds.attrs.update(global_attrs)

        # Save to file with explicit encoding to prevent time interpretation
        encoding = {"time": {"dtype": "float64", "_FillValue": None}}
        os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
        try:
            ds.to_netcdf(self.config.output_path, encoding=encoding)
            logging.info(f"Results successfully saved to {self.config.output_path}")
        except Exception as e:
            logging.error(f"Failed to save results: {str(e)}")
            raise
