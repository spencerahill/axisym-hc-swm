"""Numba backend: a fused day-runner kernel for the staggered-grid model.

run_day executes all steps of one model day (tendencies, leapfrog, Asselin,
boundary conditions, per-step storage into the daily buffers, per-step NaN
check) in a single @njit call; Python keeps everything day-granular (daily
averaging, diagnostics, convergence checks, restarts, NetCDF I/O). Every
expression transcribes the reference implementation in sw_model.py
operation-for-operation, so the integration is bitwise-identical to the
numpy backend; the parity tests in tests/test_numba_backend.py assert
max|delta| == 0.0, and any nonzero difference is a transcription bug here.

Two constructs necessarily differ from the reference text (numba 0.65.1
lacks them) but are value-identical, verified against the originals:
np.heaviside -> heaviside_gate (NaN input yields 0.5 instead of NaN, reached
only on already-NaN trajectories the driver is about to abort), and the
list-form np.concatenate ghost construction in v_face_laplacian -> explicit
ghost-array fill.

The kernel is written in "conservative array style" (vectorized expressions,
np.where branches, no in-place tricks beyond the reference's own) so it
transcribes nearly line-for-line to a future JAX port.
"""
import numpy as np
from numba import njit

from .sw_model import THETA_TO_TEMP

STENCIL_CODES = {"centered": 0, "upwind": 1, "mc": 2}


@njit(cache=True)
def heaviside_gate(x):
    """np.heaviside(x, 0.5), which numba lacks. NaN input yields 0.5 here
    (np.heaviside propagates NaN); reached only after the state is already
    NaN, one step before the NaN check aborts the run."""
    return np.where(x > 0.0, 1.0, np.where(x < 0.0, 0.0, 0.5))


@njit(cache=True)
def mc_limited_slope(u, dy):
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


@njit(cache=True)
def muscl_mc_du_dy(u, dy, y):
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


@njit(cache=True)
def v_faces_to_centers(f):
    vc = np.zeros(f.shape[0] + 1, dtype=f.dtype)
    vc[1:-1] = 0.5 * (f[:-1] + f[1:])
    return vc


@njit(cache=True)
def v_divergence_at_centers(f, dy):
    d = np.zeros(f.shape[0] + 1, dtype=f.dtype)
    d[1:-1] = (f[1:] - f[:-1]) / dy
    d[0] = 2.0 * f[0] / dy
    d[-1] = -2.0 * f[-1] / dy
    return d


@njit(cache=True)
def v_face_laplacian(f, dy):
    # explicit ghost fill in place of the reference's list-form
    # np.concatenate (numba-unsupported); same elements, and the same
    # (f_plus + f_minus) - 2 f_center association the parity invariant needs
    fe = np.empty(f.shape[0] + 2, dtype=f.dtype)
    fe[0] = -f[0]
    fe[1:-1] = f
    fe[-1] = -f[-1]
    return ((fe[2:] + fe[:-2]) - 2.0 * fe[1:-1]) / dy**2


@njit(cache=True)
def gradient_uniform(f, dy):
    """np.gradient(f, dy) for 1-D uniform spacing, default edge_order=1
    (0-ulp replica of the installed numpy 2.4.1, verified 2026-07-13)."""
    out = np.empty_like(f)
    out[1:-1] = (f[2:] - f[:-2]) / (2.0 * dy)
    out[0] = (f[1] - f[0]) / dy
    out[-1] = (f[-1] - f[-2]) / dy
    return out


@njit(cache=True)
def merid_advec_u_term(u, vc, dy):
    """v*du/dy with first-order upwinding by the sign of the center v.

    Rewrite of the reference's chained boolean-mask setitem (numba-
    unfriendly): vc>0 and vc<0 are disjoint, index 0 is reachable only via
    the vc<0 branch and index ny-1 only via vc>0, so the where-composition
    is value-identical (verified over 5000 states with planted zeros)."""
    diff = (u[1:] - u[:-1]) / dy
    grad = np.zeros_like(u)
    grad[1:] = np.where(vc[1:] > 0, diff, 0.0)
    grad[:-1] = np.where(vc[:-1] < 0, diff, grad[:-1])
    return vc * grad


@njit(cache=True)
def run_day(
    u, v, theta,
    u_prev, v_prev, theta_prev,
    theta_e_day,
    temp_u, temp_v, temp_theta, temp_time,
    start_step, dt,
    y, y_v, dy,
    beta, v_d, epsilon_u, k_v, gravity, height, t_ref,
    tau, delta, delta_z, theta_to_temp, asselin_coef, coeff_eddy_heat_diff,
    gate_on, include_merid, include_vert, stencil_code,
):
    """Integrate temp_u.shape[0] leapfrog steps (one model day), mutating the
    state (u, v, theta) and filtered prev arrays in place and filling the
    daily buffers row by row.

    theta_e_day holds theta_E per step (nsteps, ny) for seasonal forcing, or
    a single row (1, ny) for stationary forcing.

    Returns -1 if the day completed, else the within-day step index at which
    NaN was detected in u (checked after storing that step's row, mirroring
    the reference loop's ordering).

    The per-step sequence is load-bearing for bitwise parity: tendencies from
    the current state, leapfrog, Asselin consuming the PRE-boundary-condition
    leapfrog fields, state swap, then boundary conditions on the state arrays
    only; the filtered prev arrays keep their nonzero wall values.
    """
    nsteps = temp_u.shape[0]
    te_rows = theta_e_day.shape[0]
    for k in range(nsteps):
        t = (start_step + k) * dt
        row = k if te_rows > 1 else 0
        th_e = theta_e_day[row]

        # --- tendencies from the current state ---
        vc = v_faces_to_centers(v)
        dvdy = v_divergence_at_centers(v, dy)

        # du/dt: coriolis - merid advec - vert advec - Rayleigh drag - EMFD.
        # Disabled terms subtract a zeros array where the reference subtracts
        # scalar 0: x - 0.0 is exact for every float64 x including -0.0.
        if include_merid:
            merid = merid_advec_u_term(u, vc, dy)
        else:
            merid = np.zeros_like(u)
        if include_vert:
            vert = u * dvdy * heaviside_gate(th_e - theta)
        else:
            vert = np.zeros_like(u)
        if gate_on:
            gate = heaviside_gate(u)
        else:
            gate = np.ones_like(u)
        if stencil_code == 1:  # upwind: one-sided from the equatorward side
            diff = (u[1:] - u[:-1]) / dy
            backward = np.zeros_like(u)
            backward[1:] = diff
            forward = np.zeros_like(u)
            forward[:-1] = diff
            du_dy = np.where(y > 0, backward, forward)
        elif stencil_code == 2:  # mc
            du_dy = muscl_mc_du_dy(u, dy, y)
        else:  # centered
            du_dy = gradient_uniform(u, dy)
        emfd = v_d * gate * np.sign(y) * du_dy
        dudt = beta * y * vc - merid - vert - u * epsilon_u - emfd

        # dv/dt on the faces
        uf = 0.5 * (u[:-1] + u[1:])
        T = theta * theta_to_temp
        dt_dy = (T[1:] - T[:-1]) / dy
        dp = gravity * height * dt_dy / t_ref
        coriolis = beta * y_v * uf
        diffusion = v_face_laplacian(v, dy) * k_v
        dvdt = (-coriolis - dp + diffusion) / 2

        # dtheta/dt. The zeros eddy term is ADDED (not skipped) when the
        # coefficient is 0, as the reference does: x + 0.0 flips -0.0 to
        # +0.0, so skipping the add would not be bitwise-equivalent.
        newt = (th_e - theta) / tau
        vadv_th = -delta * delta_z * dvdy / height
        if coeff_eddy_heat_diff == 0.0:
            eddy = np.zeros_like(theta)
        else:
            dtheta_dy = gradient_uniform(theta, dy)
            eddy = coeff_eddy_heat_diff * gradient_uniform(dtheta_dy, dy)
        dthdt = newt + vadv_th + eddy

        # leapfrog (2 * dt formed in int, matching `2 * config.dt`)
        two_dt = 2 * dt
        u_next = u_prev + two_dt * dudt
        v_next = v_prev + two_dt * dvdt
        theta_next = theta_prev + two_dt * dthdt

        # Asselin filter, consuming the pre-BC leapfrog fields
        u_prev_new = u + asselin_coef * (u_next + u_prev - 2 * u)
        v_prev_new = v + asselin_coef * (v_next + v_prev - 2 * v)
        theta_prev_new = theta + asselin_coef * (theta_next + theta_prev - 2 * theta)
        u_prev[:] = u_prev_new
        v_prev[:] = v_prev_new
        theta_prev[:] = theta_prev_new

        # state <- next, then BCs on the state only
        u[:] = u_next
        v[:] = v_next
        theta[:] = theta_next
        u[0] = 0.0
        u[-1] = 0.0

        temp_u[k] = u
        temp_v[k] = v
        temp_theta[k] = theta
        temp_time[k] = t / 86400
        if np.isnan(u).any():
            return k
    return -1


def day_kernel_args(model, theta_e_day, start_step):
    """Assemble run_day's arguments from a StaggeredSWModel.

    Scalars are coerced to fixed types (float/int/bool) so every call hits
    one compiled signature regardless of how the config values were spelled
    (e.g. the int-valued k_v default); the coercions are value-exact."""
    config = model.config
    return dict(
        u=model.state.u,
        v=model.state.v,
        theta=model.state.theta,
        u_prev=model.vars_prev_step.u,
        v_prev=model.vars_prev_step.v,
        theta_prev=model.vars_prev_step.theta,
        theta_e_day=theta_e_day,
        temp_u=model.temp_vars.u,
        temp_v=model.temp_vars.v,
        temp_theta=model.temp_vars.theta,
        temp_time=model.temp_vars.time,
        start_step=int(start_step),
        dt=int(config.dt),
        y=config.y,
        y_v=config.y_v,
        dy=float(config.dy),
        beta=float(config.beta),
        v_d=float(config.v_d),
        epsilon_u=float(config.epsilon_u),
        k_v=float(config.k_v),
        gravity=float(config.gravity),
        height=float(config.height),
        t_ref=float(config.t_ref),
        tau=float(config.tau),
        delta=float(config.delta),
        delta_z=float(config.delta_z),
        theta_to_temp=float(THETA_TO_TEMP),
        asselin_coef=float(config.asselin_filt_coef),
        coeff_eddy_heat_diff=float(config.coeff_eddy_heat_diff),
        gate_on=bool(config.emfd_heaviside_gate),
        include_merid=bool(config.include_merid_advec_u),
        include_vert=bool(config.include_vert_advec_u),
        stencil_code=STENCIL_CODES[config.emfd_stencil],
    )
