"""Right-hand-side assembly for the staggered-grid shallow-water model.

The spatial discretization lives in :mod:`ss09.grid` (flux-form C-grid
operators); this module wires those operators into the SS09 physics and exposes
the full explicit tendency for the method-of-lines time integration.

State layout (see :mod:`ss09.grid`):

  - ``u``, ``theta`` on the ``N`` cell centers (``config.y``)
  - ``v`` on the ``N+1`` cell faces (``config.yf``); the boundary faces are
    held at ``v = 0`` by giving them zero tendency.

The tendency is split into two groups so a future IMEX stepper can treat the
stiff-capable linear operators implicitly without changing any equation:

  - :func:`rhs_ex` -- nonstiff terms: Coriolis, flux-form advection, eddy-momentum
    flux divergence, pressure gradient, Newtonian relaxation, adiabatic heating,
    Rayleigh drag.
  - :func:`rhs_im` -- stiff-capable linear diffusion: ``k_v d^2 v`` and the
    optional ``kappa d^2 theta``. These are the terms that were only marginally
    stable under the old leapfrog scheme.

The production integrator (fixed-step RK4) evaluates ``rhs = rhs_ex + rhs_im``
fully explicitly; the split is structural, not yet used for an implicit solve.
Newtonian relaxation and the future precip sink would move into the implicit
group when IMEX is introduced.
"""

import numpy as np

from .model_state import ModelState
from . import grid as ops

THETA_TO_TEMP = 1 / 1.6  # inverse of (p_s/p_t)^(R/c_p)


def _heaviside(x: np.ndarray) -> np.ndarray:
    """Heaviside with H(0) = 0.5 (implicit smoothing at the boundary)."""
    return np.heaviside(x, 0.5)


# --------------------------------------------------------------------------
# Zonal momentum (u, on centers)
# --------------------------------------------------------------------------
def coriolis_u(v_face: np.ndarray, config) -> np.ndarray:
    """beta * y * v on centers (v averaged from faces)."""
    return config.beta * config.y * ops.avg_f2c(v_face)


def u_advection(u, v_face, theta, theta_e, config) -> np.ndarray:
    """Advective tendency of u, honoring the merid/vert toggles.

    Discretized exactly as the original SS09 code (which is the validated
    reference): the meridional term ``-v du/dy`` uses upwind differencing of
    ``du/dy`` biased by the sign of ``v`` (the numerical diffusion that keeps
    the advection stable), and the vertical-exchange term
    ``-H(theta_E - theta) u dv/dy`` is gated to the convecting region.

    A flux-form (``-d_y(vu)``) discretization was tried but is numerically
    unstable here: the model's advection is only conservative where convecting,
    and the non-conservative gated correction term required by the flux split
    (with centered ``dv/dy``) is undamped and drives an equatorial superrotation.
    """
    v_c = ops.avg_f2c(v_face)
    adv = np.zeros_like(u)
    if config.include_merid_advec_u:  # -v du/dy (upwind advective form)
        adv = adv - v_c * ops.ddy_upwind(u, v_c, config.dy)
    if config.include_vert_advec_u:  # -H(theta_E - theta) u dv/dy (gated to convection)
        adv = adv - _heaviside(theta_e - theta) * u * ops.div_f2c(v_face, config.dy)
    return adv


def emfd_u(u, config) -> np.ndarray:
    """Eddy-momentum flux divergence S = v_d H(u) sgn(y) du/dy on centers."""
    return (
        config.v_d
        * _heaviside(u)
        * np.sign(config.y)
        * ops.ddy_center(u, config.dy)
    )


def rayleigh_drag_u(u, config) -> np.ndarray:
    return config.epsilon_u * u


def momentum_diffusion_u(u, config) -> np.ndarray:
    """Explicit eddy viscosity on u: k_u d^2u/dy^2 (Neumann walls).

    Replaces the implicit damping the old leapfrog/Asselin scheme provided on u
    (its only u dissipation). Without it the eddy-momentum flux divergence, which
    is up-gradient near a westerly maximum, drives a slow equatorial
    superrotation. Default k_u is calibrated to reproduce the original climate.
    """
    if config.k_u == 0.0:
        return np.zeros_like(u)
    return config.k_u * ops.lap_center_neumann(u, config.dy)


# --------------------------------------------------------------------------
# Meridional momentum (v, on faces)
# --------------------------------------------------------------------------
def coriolis_v(u, config) -> np.ndarray:
    """beta * y * u on faces (u averaged from centers)."""
    return config.beta * config.yf * ops.avg_c2f(u)


def pressure_grad_v(theta, config) -> np.ndarray:
    """(g H / T0) dT/dy on faces, with T = theta * THETA_TO_TEMP."""
    dTdy = ops.grad_c2f(theta * THETA_TO_TEMP, config.dy)
    return config.gravity * config.height * dTdy / config.t_ref


# --------------------------------------------------------------------------
# Thermodynamics (theta, on centers)
# --------------------------------------------------------------------------
def newtonian_cooling(theta, theta_e, config) -> np.ndarray:
    return (theta_e - theta) / config.tau


def adiabatic_heating(v_face, config) -> np.ndarray:
    """-(delta delta_z / H) dv/dy on centers (adiabatic cooling from ascent)."""
    return -config.delta * config.delta_z * ops.div_f2c(v_face, config.dy) / config.height


# --------------------------------------------------------------------------
# Assembled tendencies
# --------------------------------------------------------------------------
def rhs_ex(state: ModelState, config, theta_e: np.ndarray):
    """Nonstiff explicit tendencies. ``theta_e`` is precomputed on centers."""
    u, v, theta = state.u, state.v, state.theta

    du = (
        coriolis_u(v, config)
        + u_advection(u, v, theta, theta_e, config)
        - rayleigh_drag_u(u, config)
        - emfd_u(u, config)
    )

    # v evolves only on interior faces; the 2 doubles the inertial term (the
    # tropopause and surface meridional velocities are equal and opposite).
    dv = np.zeros_like(v)
    dv[1:-1] = (
        -coriolis_v(u, config)[1:-1] - pressure_grad_v(theta, config)[1:-1]
    ) / 2.0

    dtheta = newtonian_cooling(theta, theta_e, config) + adiabatic_heating(v, config)
    return du, dv, dtheta


def rhs_im(state: ModelState, config):
    """Stiff-capable linear diffusion tendencies."""
    u, v, theta = state.u, state.v, state.theta

    du = momentum_diffusion_u(u, config)
    # k_v d^2v/dy^2, carrying the same factor of 1/2 as the rest of the v eqn.
    dv = ops.lap_face_dirichlet(v, config.dy) * config.k_v / 2.0
    if config.coeff_eddy_heat_diff != 0.0:
        dtheta = config.coeff_eddy_heat_diff * ops.lap_center_neumann(theta, config.dy)
    else:
        dtheta = np.zeros_like(theta)
    return du, dv, dtheta


def rhs(state: ModelState, config, theta_e_profile):
    """Full explicit tendency (rhs_ex + rhs_im) -> (du, dv, dtheta).

    Applies the u Dirichlet wall BC (u held at its initial 0 on the two boundary
    centers) by zeroing their tendency, so it is respected identically by the
    RK4 integrator and the scipy reference. v stays 0 on the wall faces because
    rhs_ex/rhs_im give those entries zero tendency.
    """
    theta_e = theta_e_profile(state)
    du_e, dv_e, dth_e = rhs_ex(state, config, theta_e)
    du_i, dv_i, dth_i = rhs_im(state, config)
    du = du_e + du_i
    du[0] = 0.0
    du[-1] = 0.0
    return du, dv_e + dv_i, dth_e + dth_i


# --------------------------------------------------------------------------
# Diagnostics and the flat scipy adapter
# --------------------------------------------------------------------------
def momentum_budget_terms(state: ModelState, config, theta_e_profile) -> dict:
    """Term-by-term zonal-momentum tendency on centers (for the harness)."""
    theta_e = theta_e_profile(state)
    u, v, theta = state.u, state.v, state.theta
    return {
        "coriolis (beta*y*v)": coriolis_u(v, config),
        "advection (-v du/dy ...)": u_advection(u, v, theta, theta_e, config),
        "rayleigh (-eps u)": -rayleigh_drag_u(u, config),
        "emfd (-S)": -emfd_u(u, config),
        "u diffusion (k_u d2u)": momentum_diffusion_u(u, config),
    }


def pack(state: ModelState) -> np.ndarray:
    """Flatten the staggered state to a single vector (u | v | theta)."""
    return np.concatenate([state.u, state.v, state.theta])


def unpack(vec: np.ndarray, t: float, config) -> ModelState:
    n = config.ny
    u = vec[:n]
    v = vec[n : 2 * n + 1]
    theta = vec[2 * n + 1 :]
    return ModelState(t=t, u=u, v=v, theta=theta, y=config.y)


def rhs_flat(t: float, vec: np.ndarray, config, theta_e_profile) -> np.ndarray:
    """``F(t, q)`` adapter for scipy.integrate.solve_ivp (reference integration)."""
    state = unpack(vec, t, config)
    du, dv, dtheta = rhs(state, config, theta_e_profile)
    return np.concatenate([du, dv, dtheta])
