"""
This file shows the python code for S-S model.
"""

import os
import logging
import numpy as np
from typing import Optional, Tuple, NamedTuple
from dataclasses import asdict
from .model_state import ModelState
from .theta_e import ThetaEProfile
from .sw_config import SWConfig
from .daily_results import DailyResults
from .steady_state import SteadyStateDetector
from .hadley_diagnostics import HadleyDiagnostics
from .restart_state import RestartState
from .output_path_utils import generate_restart_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
SECONDS_PER_DAY = 86400  # Number of seconds in a day
THETA_TO_TEMP = 1 / 1.6  # Inverse of (p_s/p_t)^(R/c_p)


def mc_limited_slope(u: np.ndarray, dy: float) -> np.ndarray:
    """Monotonized-central (MC) limited slope of u at each grid point.

    sigma_i = minmod3(2*dm_i, (dm_i + dp_i)/2, 2*dp_i), where dm/dp are the
    backward/forward one-sided slopes: zero wherever dm and dp differ in
    sign (local extrema) and at both endpoints, where the missing one-sided
    slope is padded with zero.
    """
    diff = (u[1:] - u[:-1]) / dy
    dm = np.zeros_like(u)
    dm[1:] = diff
    dp = np.zeros_like(u)
    dp[:-1] = diff
    centered = 0.5 * (dm + dp)
    mag = np.minimum(
        np.minimum(np.abs(2.0 * dm), np.abs(2.0 * dp)), np.abs(centered)
    )
    return np.where(dm * dp > 0, np.sign(dm) * mag, 0.0)


def muscl_mc_du_dy(u: np.ndarray, dy: float, y: np.ndarray) -> np.ndarray:
    """MUSCL upwind-biased du/dy with MC-limited slopes, for advection at
    the poleward velocity v_d*sgn(y).

    NH points use dm_i + (sigma_i - sigma_{i-1})/2, SH points
    dp_i - (sigma_{i+1} - sigma_i)/2: second-order in smooth regions (the
    limited-slope correction cancels the one-sided difference's leading
    (dy/2)*u'' truncation term), reverting toward first-order upwind where
    the limiter clips (extrema, discontinuities). An equator point takes the
    SH branch, which the sgn(0)=0 factor in the EMFD zeroes regardless.
    """
    diff = (u[1:] - u[:-1]) / dy
    dm = np.zeros_like(u)
    dm[1:] = diff
    dp = np.zeros_like(u)
    dp[:-1] = diff
    sigma = mc_limited_slope(u, dy)
    sigma_m = np.zeros_like(u)  # sigma_{i-1}
    sigma_m[1:] = sigma[:-1]
    sigma_p = np.zeros_like(u)  # sigma_{i+1}
    sigma_p[:-1] = sigma[1:]
    backward = dm + 0.5 * (sigma - sigma_m)
    forward = dp - 0.5 * (sigma_p - sigma)
    return np.where(y > 0, backward, forward)


def v_faces_to_centers(f: np.ndarray) -> np.ndarray:
    """Reconstruct v at the ny centers from the ny-1 face values.

    Center j (interior) is the two-point average of the faces on either side,
    0.5*(f[j-1] + f[j]); the wall centers carry v=0. This is the value u feels
    in the Coriolis and meridional-advection terms. Exactly antisymmetric when
    f is, because 0.5*((-b)+(-a)) == -(0.5*(a+b)) bit-for-bit.
    """
    vc = np.zeros(f.shape[0] + 1, dtype=f.dtype)
    vc[1:-1] = 0.5 * (f[:-1] + f[1:])
    return vc


def v_divergence_at_centers(f: np.ndarray, dy: float) -> np.ndarray:
    """Compact dv/dy at the ny centers from the ny-1 face values.

    Interior center j: (f[j] - f[j-1])/dy, a two-point difference blind to no
    grid-scale mode. Wall centers use the half-cell one-sided form consistent
    with v=0 at the wall (distance dy/2 from the outermost face). Exactly
    symmetric when f is antisymmetric (the divergence of an antisymmetric
    field), which parity of the vertical-advection terms depends on.
    """
    d = np.zeros(f.shape[0] + 1, dtype=f.dtype)
    d[1:-1] = (f[1:] - f[:-1]) / dy
    d[0] = 2.0 * f[0] / dy
    d[-1] = -2.0 * f[-1] / dy
    return d


def v_face_laplacian(f: np.ndarray, dy: float) -> np.ndarray:
    """Second y-derivative of v on the faces, for the k_v diffusion term.

    A 3-point Laplacian in the mirror-symmetric association
    (f_plus + f_minus) - 2 f_center, which is bit-exact under reflection (the
    naive (f_plus - 2 f_center) + f_minus form seeds a ~2e-18/step parity
    drift). The outermost faces borrow a ghost equal to -f[0] (resp. -f[-1]),
    the mirror image about the wall that enforces v=0 there, so the wall-face
    Laplacian is (f[1] - 3 f[0])/dy^2 rather than the phantom-zero
    (f[1] - 2 f[0])/dy^2.
    """
    fe = np.concatenate([[-f[0]], f, [-f[-1]]])
    return ((fe[2:] + fe[:-2]) - 2.0 * fe[1:-1]) / dy**2


def mc_face_values(w: np.ndarray, dy: float, c_f: np.ndarray) -> np.ndarray:
    """MUSCL face values of a center field w on the ny-1 interior faces,
    upwinded on the sign of the face transport velocity c_f.

    Face j sits between centers j and j+1. Where c_f > 0 the upwind cell is
    the left center j and the face value is its MC-limited linear
    reconstruction w[j] + (dy/2) sigma[j]; otherwise the right center j+1
    gives w[j+1] - (dy/2) sigma[j+1]. The limiter keeps every face value
    between the two adjacent cell values (non-oscillatory, so the transport
    cannot undershoot W < 0 at the sharp ITCZ front), and at a local
    extremum it reverts exactly to the upwind cell value. At c_f == 0 the
    branch choice is irrelevant: the advective flux c_f * w_face vanishes.
    """
    sigma = mc_limited_slope(w, dy)
    left = w[:-1] + 0.5 * dy * sigma[:-1]
    right = w[1:] - 0.5 * dy * sigma[1:]
    return np.where(c_f > 0, left, right)


def moisture_transport_tendency(
    w_adv: np.ndarray,
    w_diff: np.ndarray,
    v_f: np.ndarray,
    cwv_frac: float,
    d_w: float,
    dy: float,
) -> np.ndarray:
    """Finite-volume flux-form transport tendency of W on the ny centers.

    The flux on each interior face is F = c_f * W_f - D * dW/dy with
    c_f = -(2a-1) v_f the column transport velocity (opposite to the
    upper-layer v for a bottom-heavy column, a > 1/2), W_f the MUSCL-MC
    face value, and the diffusive gradient the compact face difference.
    The two W arguments let the integrator split time levels: w_adv is the
    central (n) field the advective flux sees, w_diff the lagged (n-1)
    field the diffusive flux sees. Wall fluxes are zero (zero total flux at
    the walls), and the wall centers use the half-cell divergence form
    (v_divergence_at_centers on the flux array), so the cell-weighted sum
    of the tendency telescopes to zero exactly: transport moves no total
    water, and total W changes only through E_0 - P.
    """
    c_f = -(2.0 * cwv_frac - 1.0) * v_f
    w_face = mc_face_values(w_adv, dy, c_f)
    flux = c_f * w_face - d_w * (w_diff[1:] - w_diff[:-1]) / dy
    return -v_divergence_at_centers(flux, dy)


def cwv_integral(w: np.ndarray, dy: float) -> float:
    """Discrete integral of a center field over the domain, with the FV
    cell widths (dy/2 half cells at the walls, dy interior): the measure
    under which the transport tendency conserves total water exactly."""
    return dy * (0.5 * w[0] + np.sum(w[1:-1]) + 0.5 * w[-1])


def precipitation(w: np.ndarray, w_crit: float, tau_c: float) -> np.ndarray:
    """P = (W - W_c)^+ / tau_c: quasi-equilibrium relaxation of CWV to the
    critical value on the convective timescale, zero at or below W_c. The
    only closure that keeps W bounded (a negative feedback for W > W_c)."""
    return np.maximum(w - w_crit, 0.0) / tau_c


class AuxiliaryVars(NamedTuple):
    """
    Values of u, v, and theta for the previous or future step.
    """
    u: np.ndarray
    v: np.ndarray
    theta: np.ndarray


class TempVars(NamedTuple):
    """
    Temporary storage of model variables for daily averages.
    w and precip are populated only when moisture is enabled (else None).
    """
    u: np.ndarray
    v: np.ndarray
    theta: np.ndarray
    theta_e: np.ndarray
    time: np.ndarray
    w: Optional[np.ndarray] = None
    precip: Optional[np.ndarray] = None


class SWModel:
    """
    The shallow water model on an equatorial beta plane.

    Instantiating ``SWModel(config, ...)`` returns a ``StaggeredSWModel`` when
    ``config.grid == "staggered"`` (the production default), so callers get the
    right dynamics from the config alone; the base class is the collocated
    (Zhang et al. 2025-lineage) layout and stays bit-for-bit unchanged.
    """
    def __new__(cls, config: SWConfig, theta_e_profile: ThetaEProfile, *args, **kwargs):
        if cls is SWModel and getattr(config, "grid", "collocated") == "staggered":
            return super().__new__(StaggeredSWModel)
        return super().__new__(cls)

    def __init__(self, config: SWConfig, theta_e_profile: ThetaEProfile):
        self.config = config
        self.theta_e_profile = theta_e_profile
        # u and theta live on the ny centers; v may live on a different grid
        # (config.nv = ny-1 interior faces for the staggered layout, ny for the
        # collocated layout).
        nv = config.nv
        self.state = ModelState(
            t=0.0,
            u=np.zeros(config.ny),
            v=np.zeros(nv),
            theta=np.zeros(config.ny),
            y=config.y,
        )
        self.state = self.state._replace(theta=self.theta_e_profile(self.state))
        # theta_E is time-invariant unless the forcing migrates seasonally;
        # cache it once so the RHS terms skip a per-step profile evaluation.
        self._theta_e_static = (
            None if self.has_seasonal_forcing()
            else self.theta_e_profile(self.state)
        )
        self.results = DailyResults(
            config.total_integration_days, config.ny,
            store_theta_e=self.has_seasonal_forcing(), nv=nv,
            store_moisture=config.enable_moisture,
        )
        self.temp_vars = TempVars(
            u=np.zeros((0, config.ny)),
            v=np.zeros((0, nv)),
            theta=np.zeros((0, config.ny)),
            theta_e=np.zeros((0, config.ny)),
            time=np.zeros(0),
        )
        self.vars_prev_step = AuxiliaryVars(
            u=np.zeros(config.ny), v=np.zeros(nv), theta=np.zeros(config.ny)
        )
        self.vars_next_step = AuxiliaryVars(
            u=np.zeros(config.ny), v=np.zeros(nv), theta=np.zeros(config.ny)
        )
        # Moist V1: prognostic column water vapor W on the ny centers, with
        # its own leapfrog history (w_prev is the filtered n-1 level, seeded
        # at run start or restored from a restart). None on a dry run.
        if config.enable_moisture:
            w0 = config.w_init if config.w_init is not None else config.w_crit
            self.w = np.full(config.ny, float(w0))
            self.w_prev = np.zeros(config.ny)
            self._moisture_negative_warned = False
        else:
            self.w = None
            self.w_prev = None
        self.steady_state_detector = SteadyStateDetector(
            enabled=config.enable_steady_state,
            window_size=config.steady_state_window_size,
            threshold=config.steady_state_threshold,
            check_both_metrics=config.steady_state_check_both,
            smoothness_threshold=config.smoothness_threshold,
        )
        self.hadley_diagnostics = HadleyDiagnostics(
            ny=config.ny,
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

    def current_theta_e(self) -> np.ndarray:
        """theta_E at the current state: the cached array when the forcing is
        stationary (non-seasonal), a fresh profile evaluation when seasonal."""
        if self._theta_e_static is not None:
            return self._theta_e_static
        return self.theta_e_profile(self.state)

    def init_prev_step_vars(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Initialize variables for the previous step."""
        u_prev = self.state.u - self.config.dt * self.du_dt()
        v_prev = self.state.v - self.config.dt * self.dv_dt()
        theta_prev = self.state.theta - self.config.dt * self.dtheta_dt()
        return u_prev, v_prev, theta_prev

    def init_temp_storage(self):
        """Initialize temporary storage for daily averages."""
        steps_per_day = int(SECONDS_PER_DAY / self.config.dt)
        moist = self.config.enable_moisture
        self.temp_vars = TempVars(
            u=np.zeros([steps_per_day, self.config.ny]),
            v=np.zeros([steps_per_day, self.config.nv]),
            theta=np.zeros([steps_per_day, self.config.ny]),
            theta_e=np.zeros([steps_per_day, self.config.ny]),
            time=np.zeros(steps_per_day),
            w=np.zeros([steps_per_day, self.config.ny]) if moist else None,
            precip=np.zeros([steps_per_day, self.config.ny]) if moist else None,
        )

    def du_dy_upwind(self) -> np.ndarray:
        """Calculate the upwind gradient of u with respect to y based on velocity v."""
        grad_u_upwind = np.zeros_like(self.state.u)
        # For positive velocity (backward difference)
        mask_pos = self.state.v > 0
        grad_u_upwind[1:][mask_pos[1:]] = (
            self.state.u[1:][mask_pos[1:]] - self.state.u[:-1][mask_pos[1:]]
        ) / self.config.dy
        # For negative velocity (forward difference)
        mask_neg = self.state.v < 0
        grad_u_upwind[:-1][mask_neg[:-1]] = (
            self.state.u[1:][mask_neg[:-1]] - self.state.u[:-1][mask_neg[:-1]]
        ) / self.config.dy
        return grad_u_upwind

    def edd_mom_flux_div_u(self) -> np.ndarray:
        """Calculate the eddy momentum flux divergence."""
        if self.config.emfd_heaviside_gate:
            # H(u)=1 if u>0, 0 if u<0, 0.5 at u=0
            gate = np.heaviside(self.state.u, 0.5)
        else:
            # Published Zhang et al. (2025) code omits the H(u) gate
            gate = 1.0
        if self.config.emfd_stencil == "upwind":
            # One-sided du/dy from the equatorward (upstream) side: the
            # effective advection velocity v_d*sgn(y) is poleward, so NH
            # points difference backward, SH points forward; sgn(0)=0 zeroes
            # the equator point regardless.
            u = self.state.u
            diff = (u[1:] - u[:-1]) / self.config.dy
            backward = np.zeros_like(u)
            backward[1:] = diff
            forward = np.zeros_like(u)
            forward[:-1] = diff
            du_dy = np.where(self.config.y > 0, backward, forward)
        elif self.config.emfd_stencil == "mc":
            du_dy = muscl_mc_du_dy(self.state.u, self.config.dy, self.config.y)
        else:
            du_dy = np.gradient(self.state.u, self.config.dy)
        return (
            self.config.v_d
            * gate
            * np.sign(self.config.y)
            * du_dy
        )

    def rayleigh_drag_u(self) -> np.ndarray:
        """Calculate the Rayleigh drag for u."""
        return self.state.u * self.config.epsilon_u

    def vert_advec_u(self) -> np.ndarray:
        """Calculate the vertical momentum advection."""
        return (
            self.state.u
            * self.dv_dy()
            * np.heaviside(self.current_theta_e() - self.state.theta, 0.5)
        )

    def coriolis_term(self, u_or_v: np.ndarray) -> np.ndarray:
        """Calculate the Coriolis term for the u or v equation."""
        return self.config.beta * self.config.y * u_or_v

    def merid_advec_u(self) -> np.ndarray:
        """Calculate the meridional advection term for u."""
        return self.state.v * self.du_dy_upwind()

    def du_dt(self) -> np.ndarray:
        """Calculate the time derivative of u."""
        vert_advec_u_term = (
            self.vert_advec_u() if self.config.include_vert_advec_u else 0
        )
        merid_advec_u_term = (
            self.merid_advec_u() if self.config.include_merid_advec_u else 0
        )
        return (
            self.coriolis_term(self.state.v)
            - merid_advec_u_term
            - vert_advec_u_term
            - self.rayleigh_drag_u()
            - self.edd_mom_flux_div_u()
        )

    def dv_dy(self) -> np.ndarray:
        """Calculate the gradient of v with respect to y."""
        return np.gradient(self.state.v, self.config.dy)

    def diffusion_v(self) -> np.ndarray:
        """Calculate the diffusion term for v."""
        return np.gradient(self.dv_dy(), self.config.dy) * self.config.k_v

    def dp_dy_term(self) -> np.ndarray:
        """Calculate the pressure gradient term."""
        dtemp_dy = np.gradient(self.state.theta * THETA_TO_TEMP, self.config.dy)
        return self.config.gravity * self.config.height * dtemp_dy / self.config.t_ref

    def dv_dt(self) -> np.ndarray:
        """Calculate the time derivative of v."""
        return (
            -self.coriolis_term(self.state.u) - self.dp_dy_term() + self.diffusion_v()
        ) / 2

    def newt_cool_term(self) -> np.ndarray:
        """Newtonian cooling term."""
        return (self.current_theta_e() - self.state.theta) / self.config.tau

    def vert_advec_theta(self) -> np.ndarray:
        """Calculate the vertical advection term for theta."""
        return (
            -self.config.delta
            * self.config.delta_z
            * self.dv_dy()
            / self.config.height
        )

    def eddy_heat_flux(self) -> np.ndarray:
        """Calculate the eddy heat flux divergence."""
        if self.config.coeff_eddy_heat_diff == 0.0:
            return np.zeros_like(self.state.theta)
        dtheta_dy = np.gradient(self.state.theta, self.config.dy)
        return self.config.coeff_eddy_heat_diff * np.gradient(dtheta_dy, self.config.dy)

    def dtheta_dt(self) -> np.ndarray:
        """Calculate the time derivative of theta."""
        return self.newt_cool_term() + self.vert_advec_theta() + self.eddy_heat_flux()

    def leapfrog_step(self, prev: np.ndarray, time_deriv_func) -> np.ndarray:
        """Perform a leapfrog step for a single variable."""
        return prev + 2 * self.config.dt * time_deriv_func()

    def asselin_filt(
        self, prev: np.ndarray, after: np.ndarray, now: np.ndarray
    ) -> np.ndarray:
        """Apply the Asselin filter to a single variable."""
        return now + self.config.asselin_filt_coef * (after + prev - 2 * now)

    def enforce_boundary_conditions(self):
        """Enforce boundary conditions."""
        self.state.u[0] = 0
        self.state.u[-1] = 0
        self.state.v[0] = 0
        self.state.v[-1] = 0

    def store_temp_results(self, timestamp: float, j: int):
        """Store temporary results for daily averaging."""
        self.temp_vars.u[j - 1] = self.state.u
        self.temp_vars.v[j - 1] = self.state.v
        self.temp_vars.theta[j - 1] = self.state.theta
        self.temp_vars.theta_e[j - 1] = self.current_theta_e()
        self.temp_vars.time[j - 1] = timestamp / SECONDS_PER_DAY
        if self.config.enable_moisture:
            # self.w is the just-stepped level, matching how the dry state is
            # stored; _last_precip is the sink actually applied in that step.
            self.temp_vars.w[j - 1] = self.w
            self.temp_vars.precip[j - 1] = self._last_precip

    def store_daily_avgs(self, day: int):
        """Store daily averages."""
        theta_e_avg = (
            np.mean(self.temp_vars.theta_e, axis=0)
            if self.has_seasonal_forcing() else None
        )
        w_avg = precip_avg = w_min = None
        if self.config.enable_moisture:
            w_avg = np.mean(self.temp_vars.w, axis=0)
            precip_avg = np.mean(self.temp_vars.precip, axis=0)
            # min over the day's instantaneous fields (a daily mean would
            # hide a transient negative-W undershoot)
            w_min = float(np.min(self.temp_vars.w))
        self.results.store_day(
            day,
            np.mean(self.temp_vars.time),
            np.mean(self.temp_vars.u, axis=0),
            np.mean(self.temp_vars.v, axis=0),
            np.mean(self.temp_vars.theta, axis=0),
            theta_e_avg,
            w=w_avg,
            precip=precip_avg,
            w_min=w_min,
        )

    def _daily_v_at_centers(self, v: np.ndarray) -> np.ndarray:
        """Daily-averaged v on the ny centers, for the diagnostics that ask
        what u feels. Identity on the collocated grid; the staggered subclass
        reconstructs centers from faces."""
        return v

    def _daily_v_faces(self, v: np.ndarray) -> Optional[np.ndarray]:
        """Daily-averaged v on its native grid, for the grid-scale smoothness
        monitor. None on the collocated grid (the monitor then uses the same
        v the kinetic-energy metric does); the staggered subclass returns the
        face field."""
        return None

    def _record_diagnostics(self, day: int):
        """Record steady-state and Hadley diagnostics for a completed day.

        The steady-state kinetic energy and the Hadley cell-latitude
        diagnostics consume center-reconstructed v (the field u feels), so
        their latitudes are correct on either grid; the v-smoothness monitor
        consumes the native-grid v (faces, on a staggered run) since its
        purpose is to detect grid-scale noise.
        """
        v_daily = self.results.v[day]
        v_centers = self._daily_v_at_centers(v_daily)
        self.steady_state_detector.record_day(
            day,
            self.results.u[day],
            v_centers,
            self.results.theta[day],
            self.config.dy,
            v_faces=self._daily_v_faces(v_daily),
        )
        self.hadley_diagnostics.record_day(
            day,
            self.results.u[day],
            v_centers,
            self.config.y,
            self.config.dy,
            self.config.beta,
        )

    def reset_temp_storage(self):
        """Reset temporary storage arrays."""
        moist = self.config.enable_moisture
        self.temp_vars = TempVars(
            u=np.zeros_like(self.temp_vars.u),
            v=np.zeros_like(self.temp_vars.v),
            theta=np.zeros_like(self.temp_vars.theta),
            theta_e=np.zeros_like(self.temp_vars.theta_e),
            time=np.zeros_like(self.temp_vars.time),
            w=np.zeros_like(self.temp_vars.w) if moist else None,
            precip=np.zeros_like(self.temp_vars.precip) if moist else None,
        )

    def calc_ind_within_day(self, current_step: int) -> int:
        """Calculate the index within the day."""
        return (current_step + 1) % int(SECONDS_PER_DAY / self.config.dt)

    def _setup_run(self) -> Tuple[int, int]:
        """Shared run initialization: restart bookkeeping, leapfrog seeding
        for fresh starts, and daily-buffer allocation. Returns the starting
        (day, step)."""
        # Determine starting day and step (for restart support)
        if getattr(self, "restart_day", None) is not None:
            day = self.restart_day
            starting_step = int(day * SECONDS_PER_DAY / self.config.dt)
            logging.info(f"Restarting from day {day}, step {starting_step}")
            # vars_prev_step (the filtered n-1 state) was restored from the
            # restart file by load_from_restart; reconstructing it here would
            # discard it and break exact continuation, so leave it untouched.
        else:
            day = 0
            starting_step = 0
            # Fresh start: seed the leapfrog scheme with a one-off backward step.
            self.vars_prev_step = AuxiliaryVars(*self.init_prev_step_vars())
            if self.config.enable_moisture:
                self._seed_moisture_prev()

        self.init_temp_storage()
        return day, starting_step

    def _process_day_end(self, day: int) -> Tuple[int, bool]:
        """Day-boundary bookkeeping shared by both backends: store the daily
        averages, record diagnostics, check convergence, save any periodic
        restart, and advance the day counter.

        Returns (new_day, stop). On stop the day counter is deliberately NOT
        advanced, so the final restart file is tagged with the just-completed
        day (unchanged from the pre-refactor loop)."""
        self.store_daily_avgs(day)
        self._record_diagnostics(day)

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
                return day, True
        elif not self.has_seasonal_forcing():
            # Traditional steady-state check (only for non-seasonal runs)
            if self.steady_state_detector.check_convergence(day):
                logging.info(
                    f"Steady state reached at day {day}. "
                    f"KE converged: {self.steady_state_detector.ke_converged}, "
                    f"Tvar converged: {self.steady_state_detector.tvar_converged}"
                )
                return day, True
        # If seasonal forcing but convergence disabled: no early stopping, run full integration

        # Save restart file if periodic checkpointing is enabled
        if self.config.save_restart_every > 0 and day % self.config.save_restart_every == 0 and day > 0:
            self.save_restart_file(day)

        day += 1
        logging.info(f"Day {day} finished.")
        return day, False

    def _finalize_run(self, day: int) -> None:
        """Shared run tail: unconverged warning and the final restart save."""
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

    def _run_sim_numba(self):
        raise RuntimeError(
            "backend='numba' is implemented only for the staggered grid; "
            "SWConfig validation should have rejected this configuration"
        )

    def _seed_moisture_prev(self):
        raise RuntimeError(
            "enable_moisture is implemented only for the staggered grid "
            "(the moisture fluxes live on the v faces); SWConfig validation "
            "should have rejected this configuration"
        )

    def _step_moisture(self):
        raise RuntimeError(
            "enable_moisture is implemented only for the staggered grid "
            "(the moisture fluxes live on the v faces); SWConfig validation "
            "should have rejected this configuration"
        )

    def run_sim(self):
        """Run the S-S model simulation."""
        if getattr(self.config, "backend", "numpy") == "numba":
            return self._run_sim_numba()

        total_time_steps = int(
            SECONDS_PER_DAY * self.config.total_integration_days / self.config.dt
        )

        day, starting_step = self._setup_run()

        for i in range(starting_step, total_time_steps):
            self.state = self.state._replace(t=i * self.config.dt)

            self.vars_next_step = self.vars_next_step._replace(
                u=self.leapfrog_step(self.vars_prev_step.u, self.du_dt),
                v=self.leapfrog_step(self.vars_prev_step.v, self.dv_dt),
                theta=self.leapfrog_step(self.vars_prev_step.theta, self.dtheta_dt),
            )

            self.vars_prev_step = self.vars_prev_step._replace(
                u=self.asselin_filt(
                    self.vars_prev_step.u, self.vars_next_step.u, self.state.u
                ),
                v=self.asselin_filt(
                    self.vars_prev_step.v, self.vars_next_step.v, self.state.v
                ),
                theta=self.asselin_filt(
                    self.vars_prev_step.theta,
                    self.vars_next_step.theta,
                    self.state.theta,
                ),
            )

            # W advances in its own leapfrog + Asselin cycle, here while
            # self.state still holds level n (its advective flux reads the
            # level-n face v). One-way coupled: no dry field is touched, so
            # the dry integration is bit-for-bit unchanged.
            if self.config.enable_moisture:
                self._step_moisture()

            self.state = self.state._replace(
                u=self.vars_next_step.u,
                v=self.vars_next_step.v,
                theta=self.vars_next_step.theta,
            )

            self.enforce_boundary_conditions()

            ind_within_day = self.calc_ind_within_day(i)
            self.store_temp_results(self.state.t, ind_within_day)
            if ind_within_day == 0:
                day, stop = self._process_day_end(day)
                if stop:
                    break
                self.reset_temp_storage()

            if np.isnan(self.state.u).any():
                logging.warning("NaN detected in u, breaking the loop.")
                break

        self._finalize_run(day)

    def save_restart_file(self, day: int) -> None:
        """
        Save restart state to NetCDF file.

        CRITICAL: Saves INSTANTANEOUS state variables, NOT daily averages.
        - self.state.u/v/theta: instantaneous values at current timestep (n)
        - self.vars_prev_step.u/v/theta: instantaneous filtered values at previous timestep (n-1)

        These are the live state variables used by the leapfrog integrator,
        completely independent from the daily averages stored in self.results.

        Args:
            day: Current day number for filename

        Filename: Matches output file naming scheme with _restart_day{day:04d}.nc suffix
        (e.g., run_20260111_134530_seas_y0p0000_ny051_3600days_restart_day0100.nc)
        """
        # Create restart directory if it doesn't exist
        os.makedirs(self.config.restart_output_dir, exist_ok=True)

        # Build RestartState with all necessary information
        restart_state = RestartState(
            current_time=self.state.t,
            current_step=int(self.state.t / self.config.dt),
            current_day=day,
            # INSTANTANEOUS current state (n) - NOT daily averaged
            u=self.state.u.copy(),
            v=self.state.v.copy(),
            theta=self.state.theta.copy(),
            # INSTANTANEOUS previous filtered state (n-1) - NOT daily averaged
            u_prev=self.vars_prev_step.u.copy(),
            v_prev=self.vars_prev_step.v.copy(),
            theta_prev=self.vars_prev_step.theta.copy(),
            y=self.state.y.copy(),
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
            grid=self.config.grid,
            y_v=self.config.y_v.copy(),
            # Moist state (format v3): instantaneous W at n and filtered n-1
            enable_moisture=self.config.enable_moisture,
            w=self.w.copy() if self.config.enable_moisture else None,
            w_prev=self.w_prev.copy() if self.config.enable_moisture else None,
        )

        # Generate filepath using descriptive naming scheme (matches output file)
        filepath = generate_restart_filename(self.config.output_path, day)
        restart_state.to_netcdf(filepath)

    @staticmethod
    def _migrate_v(v: np.ndarray, from_grid: str, to_grid: str) -> np.ndarray:
        """Interpolate v between the center and face grids for a one-time
        restart migration. Collocated->staggered averages adjacent centers onto
        the faces; staggered->collocated averages adjacent faces back to the
        centers (wall centers 0)."""
        if from_grid == "collocated" and to_grid == "staggered":
            return 0.5 * (v[:-1] + v[1:])
        if from_grid == "staggered" and to_grid == "collocated":
            return v_faces_to_centers(v)
        raise ValueError(
            f"unsupported v migration {from_grid!r} -> {to_grid!r}"
        )

    def load_from_restart(self, restart_file: str, migrate: bool = False) -> int:
        """
        Load state from restart file and restore model.

        Args:
            restart_file: Path to restart NetCDF file
            migrate: If True, permit a v-grid mismatch between the restart file
                and this run by interpolating v (and its n-1 state) between the
                center and face grids once. If False, a grid mismatch is a hard
                error.

        Returns:
            starting_day: The day number to resume from
        """
        # Load restart state
        restart_state = RestartState.from_netcdf(restart_file)

        # Validate compatibility with current configuration
        restart_state.validate_compatibility(
            self.config, self.theta_e_profile.config, allow_grid_migration=migrate
        )

        # Migrate v between grids once if requested (validate already enforced
        # that this only happens under an explicit migration).
        v = restart_state.v
        v_prev = restart_state.v_prev
        if restart_state.grid != self.config.grid:
            v = self._migrate_v(v, restart_state.grid, self.config.grid)
            v_prev = self._migrate_v(v_prev, restart_state.grid, self.config.grid)

        # Restore current state
        self.state = self.state._replace(
            t=restart_state.current_time,
            u=restart_state.u,
            v=v,
            theta=restart_state.theta,
        )

        # Restore previous step
        self.vars_prev_step = self.vars_prev_step._replace(
            u=restart_state.u_prev, v=v_prev, theta=restart_state.theta_prev
        )

        # Restore (or freshly initialize) the moisture state. validate_
        # compatibility already guaranteed the file/run moisture pairing is
        # legal, including that a dry-to-moist start has an explicit w_init.
        if self.config.enable_moisture:
            if restart_state.enable_moisture:
                self.w = restart_state.w
                self.w_prev = restart_state.w_prev
            else:
                self.w = np.full(self.config.ny, float(self.config.w_init))
                # Seed W's leapfrog history from the restored level-n state,
                # exactly as a fresh start would.
                self._seed_moisture_prev()
                logging.info(
                    "Restart file has no moisture state; W freshly "
                    f"initialized to the uniform w_init={self.config.w_init}"
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
                    # netCDF attrs cannot hold bools or None (w_init's
                    # "use w_crit" sentinel); store their str form.
                    if isinstance(getattr(self.config, key), bool)
                    or getattr(self.config, key) is None
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


class StaggeredSWModel(SWModel):
    """Arakawa C-grid layout: v on the ny-1 interior cell faces (y + dy/2),
    u and theta on the ny centers. SS09 section 2b's own layout, adopted as
    the production formulation 2026-07-11 because the centered dv/dy stencil
    of the collocated model is blind to the standing 2*dy v ripple the
    terminus forcing projects onto, and the compact face operators are not.

    Only the operators that couple the two grids are overridden; leapfrog,
    Asselin filtering, restart, and daily averaging carry through from the
    base class on the ny-1-long v arrays. There is no padding slot: every
    v array is exactly ny-1, and the shape mismatch with u/theta is the guard
    against grid confusion.
    """

    def __init__(self, config: SWConfig, theta_e_profile: ThetaEProfile):
        if config.grid != "staggered":
            raise ValueError(
                "StaggeredSWModel requires config.grid == 'staggered', got "
                f"{config.grid!r}"
            )
        super().__init__(config, theta_e_profile)

    def _v_at_centers(self) -> np.ndarray:
        """Instantaneous v reconstructed at the ny centers (what u feels)."""
        return v_faces_to_centers(self.state.v)

    def dv_dy(self) -> np.ndarray:
        """Compact divergence dv/dy at the ny centers from the face v."""
        return v_divergence_at_centers(self.state.v, self.config.dy)

    def diffusion_v(self) -> np.ndarray:
        """k_v diffusion of v on the faces (symmetric-association Laplacian
        with mirror wall ghosts)."""
        return v_face_laplacian(self.state.v, self.config.dy) * self.config.k_v

    def merid_advec_u(self) -> np.ndarray:
        """Meridional advection v*du/dy with v reconstructed at centers, first
        -order upwind biased by the sign of that center-v (mirrors the base
        model's du_dy_upwind, which reads the collocated v directly)."""
        vc = self._v_at_centers()
        u = self.state.u
        dy = self.config.dy
        grad = np.zeros_like(u)
        mask_pos = vc > 0
        grad[1:][mask_pos[1:]] = (
            u[1:][mask_pos[1:]] - u[:-1][mask_pos[1:]]
        ) / dy
        mask_neg = vc < 0
        grad[:-1][mask_neg[:-1]] = (
            u[1:][mask_neg[:-1]] - u[:-1][mask_neg[:-1]]
        ) / dy
        return vc * grad

    def du_dt(self) -> np.ndarray:
        """u tendency, with the Coriolis term fed the center-reconstructed v."""
        vert_advec_u_term = (
            self.vert_advec_u() if self.config.include_vert_advec_u else 0
        )
        merid_advec_u_term = (
            self.merid_advec_u() if self.config.include_merid_advec_u else 0
        )
        return (
            self.coriolis_term(self._v_at_centers())
            - merid_advec_u_term
            - vert_advec_u_term
            - self.rayleigh_drag_u()
            - self.edd_mom_flux_div_u()
        )

    def dv_dt(self) -> np.ndarray:
        """v tendency on the faces: Coriolis (-beta*y_v*u), pressure gradient
        (-dp/dy), and k_v diffusion, all evaluated on the ny-1 faces, halved
        per the SS09 formulation. u is averaged to faces and T differenced
        compactly across each face."""
        u = self.state.u
        dy = self.config.dy
        uf = 0.5 * (u[:-1] + u[1:])  # u averaged to the faces
        T = self.state.theta * THETA_TO_TEMP
        dt_dy = (T[1:] - T[:-1]) / dy  # compact dT/dy at the faces
        dp = self.config.gravity * self.config.height * dt_dy / self.config.t_ref
        coriolis = self.config.beta * self.config.y_v * uf
        return (-coriolis - dp + self.diffusion_v()) / 2

    def enforce_boundary_conditions(self):
        """Pin u=0 at the walls. v lives entirely on interior faces, so no
        face value sits on a wall to pin; the wall condition v=0 enters
        through the divergence half-cell form and the diffusion mirror ghost,
        not through pinning a face."""
        self.state.u[0] = 0
        self.state.u[-1] = 0

    def _daily_v_at_centers(self, v: np.ndarray) -> np.ndarray:
        """Daily-averaged face v reconstructed at the ny centers."""
        return v_faces_to_centers(v)

    def _daily_v_faces(self, v: np.ndarray) -> np.ndarray:
        """Daily-averaged v on its native face grid (for the smoothness
        monitor, whose job is grid-scale noise on that grid)."""
        return v

    def _moisture_rhs(
        self, w_adv: np.ndarray, w_lag: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """(dW/dt, applied precipitation) at the current state: flux-form
        transport through the face v, plus evaporation minus precipitation.
        The time-level split is the caller's: w_adv feeds the advective flux
        (central level n), w_lag the diffusive flux and the precipitation
        sink (lagged level n-1, an effectively forward step that keeps both
        dissipative terms off the Asselin stability budget). P is returned
        so the daily output can record the sink actually applied."""
        cfg = self.config
        p = precipitation(w_lag, cfg.w_crit, cfg.tau_c)
        transport = moisture_transport_tendency(
            w_adv, w_lag, self.state.v, cfg.cwv_frac, cfg.d_w, cfg.dy
        )
        return transport + cfg.evap - p, p

    def _seed_moisture_prev(self):
        """Seed W's leapfrog history with a one-off backward step (the
        moisture analog of init_prev_step_vars); both levels are the initial
        field, so the lagged terms see the same state as the central ones."""
        tendency, self._last_precip = self._moisture_rhs(self.w, self.w)
        self.w_prev = self.w - self.config.dt * tendency

    def _step_moisture(self):
        """Advance W one leapfrog step and Asselin-filter its history.
        Must run while self.state holds level n (the advective flux reads
        the level-n face v). Negative W is not clipped (clipping would break
        the mass budget); it is diagnosed with a one-shot warning."""
        tendency, self._last_precip = self._moisture_rhs(self.w, self.w_prev)
        w_next = self.w_prev + 2 * self.config.dt * tendency
        self.w_prev = self.asselin_filt(self.w_prev, w_next, self.w)
        self.w = w_next
        if not self._moisture_negative_warned and np.any(w_next < 0.0):
            logging.warning(
                "Negative W detected (min %.3g kg/m^2): the moisture "
                "transport has undershot; consider a smaller dt or d_w.",
                float(np.min(w_next)),
            )
            self._moisture_negative_warned = True

    def _run_sim_numba(self):
        """Day-granular driver around the fused numba kernel.

        Each iteration integrates one whole day in a single compiled call
        (the kernel fills the daily buffers and mutates the state and
        filtered-prev arrays in place), then runs the same day-end
        bookkeeping as the reference loop. Bitwise-identical to the numpy
        backend; see numba_backend.py. reset_temp_storage is skipped: the
        kernel overwrites every buffer row of a completed day, and partial
        (NaN) days are never consumed."""
        try:
            from . import numba_backend
        except ImportError as err:
            raise ImportError(
                "backend='numba' requires numba, which is not installed in "
                "this environment; install it with: "
                "conda install -c conda-forge numba"
            ) from err

        day, _ = self._setup_run()
        spd = int(SECONDS_PER_DAY / self.config.dt)
        total_days = self.config.total_integration_days
        seasonal = self.has_seasonal_forcing()
        if not seasonal:
            theta_e_day = self._theta_e_static.reshape(1, -1)

        while day < total_days:
            start_step = day * spd
            if seasonal:
                # Precompute the day's theta_E in vectorized numpy directly
                # into the daily buffer (exactly what the reference stores
                # per step); the kernel reads it row by row.
                ts = (start_step + np.arange(spd, dtype=np.int64)) * self.config.dt
                self.temp_vars.theta_e[:] = self.theta_e_profile.profile_at_times(
                    ts, self.config.y
                )
                theta_e_day = self.temp_vars.theta_e

            nan_step = numba_backend.run_day(
                **numba_backend.day_kernel_args(self, theta_e_day, start_step)
            )

            # Stamp t with the last executed step so restart files carry the
            # same current_time/current_step the reference loop would.
            last_k = spd - 1 if nan_step < 0 else nan_step
            self.state = self.state._replace(
                t=(start_step + last_k) * self.config.dt
            )

            if nan_step < 0 or nan_step == spd - 1:
                # Day completed (the reference stores the day before its NaN
                # check, so a NaN at the last step still stores the day).
                day, stop = self._process_day_end(day)
                if stop:
                    break
            if nan_step >= 0:
                logging.warning("NaN detected in u, breaking the loop.")
                break

        self._finalize_run(day)
