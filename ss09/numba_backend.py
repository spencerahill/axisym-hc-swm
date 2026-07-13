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

    The body is fused explicit loops over grid points with scratch arrays
    allocated once per call: per-step array temporaries cost ~50 us/step at
    ny=801 (measured, ~70% of the array-style kernel's runtime), so the
    per-element operations of the operator functions above are inlined here
    in the SAME per-element order; those functions and their unit tests pin
    that order against the numpy reference. Notes that keep the loop forms
    bitwise-equal to the reference's array forms:
    - disabled du/dt terms subtract literal 0.0 (subtracting +0.0 never
      changes a float64, including -0.0);
    - the eddy heat term is ADDED even when its coefficient is 0, because
      adding +0.0 flips -0.0 to +0.0 exactly as the reference's zeros-array
      add does;
    - each one-sided difference (u[j+1]-u[j])/dy is precomputed once per
      step into `diff` and reused, matching the reference's shared array.
    """
    nsteps = temp_u.shape[0]
    te_rows = theta_e_day.shape[0]
    ny = u.shape[0]
    nv = v.shape[0]
    two_dt = 2 * dt

    # scratch, allocated once per day
    diff = np.empty(ny - 1)  # (u[j+1] - u[j]) / dy
    vc = np.empty(ny)  # v at centers
    dvdy = np.empty(ny)  # compact divergence at centers
    sigma = np.empty(ny)  # MC-limited slopes
    du_dy = np.empty(ny)  # EMFD stencil
    d1 = np.empty(ny)  # first gradient pass of the eddy heat term
    u_next = np.empty(ny)
    v_next = np.empty(nv)
    theta_next = np.empty(ny)

    for k in range(nsteps):
        t = (start_step + k) * dt
        row = k if te_rows > 1 else 0

        # --- grid-coupling scratches from the current state ---
        for j in range(ny - 1):
            diff[j] = (u[j + 1] - u[j]) / dy
        vc[0] = 0.0
        vc[ny - 1] = 0.0
        dvdy[0] = 2.0 * v[0] / dy
        dvdy[ny - 1] = -2.0 * v[nv - 1] / dy
        for j in range(1, ny - 1):
            vc[j] = 0.5 * (v[j - 1] + v[j])
            dvdy[j] = (v[j] - v[j - 1]) / dy

        # EMFD du/dy stencil
        if stencil_code == 1:  # upwind: one-sided from the equatorward side
            for j in range(ny):
                backward = diff[j - 1] if j >= 1 else 0.0
                forward = diff[j] if j <= ny - 2 else 0.0
                du_dy[j] = backward if y[j] > 0 else forward
        elif stencil_code == 2:  # mc
            for j in range(ny):
                dm = diff[j - 1] if j >= 1 else 0.0
                dp = diff[j] if j <= ny - 2 else 0.0
                centered = 0.5 * (dm + dp)
                mag = np.minimum(
                    np.minimum(np.abs(2.0 * dm), np.abs(2.0 * dp)), np.abs(centered)
                )
                sigma[j] = np.sign(dm) * mag if dm * dp > 0 else 0.0
            for j in range(ny):
                dm = diff[j - 1] if j >= 1 else 0.0
                dp = diff[j] if j <= ny - 2 else 0.0
                sigma_m = sigma[j - 1] if j >= 1 else 0.0
                sigma_p = sigma[j + 1] if j <= ny - 2 else 0.0
                backward = dm + 0.5 * (sigma[j] - sigma_m)
                forward = dp - 0.5 * (sigma_p - sigma[j])
                du_dy[j] = backward if y[j] > 0 else forward
        else:  # centered: np.gradient replica
            du_dy[0] = (u[1] - u[0]) / dy
            du_dy[ny - 1] = (u[ny - 1] - u[ny - 2]) / dy
            for j in range(1, ny - 1):
                du_dy[j] = (u[j + 1] - u[j - 1]) / (2.0 * dy)

        # eddy heat flux first pass: d1 = np.gradient(theta, dy)
        if coeff_eddy_heat_diff != 0.0:
            d1[0] = (theta[1] - theta[0]) / dy
            d1[ny - 1] = (theta[ny - 1] - theta[ny - 2]) / dy
            for j in range(1, ny - 1):
                d1[j] = (theta[j + 1] - theta[j - 1]) / (2.0 * dy)

        # --- u: tendency, leapfrog, Asselin (state committed later) ---
        for j in range(ny):
            if include_merid:
                if vc[j] > 0 and j >= 1:
                    grad = diff[j - 1]
                elif vc[j] < 0 and j <= ny - 2:
                    grad = diff[j]
                else:
                    grad = 0.0
                merid = vc[j] * grad
            else:
                merid = 0.0
            if include_vert:
                x = theta_e_day[row, j] - theta[j]
                gate_th = 1.0 if x > 0.0 else (0.0 if x < 0.0 else 0.5)
                vert = u[j] * dvdy[j] * gate_th
            else:
                vert = 0.0
            if gate_on:
                gate = 1.0 if u[j] > 0.0 else (0.0 if u[j] < 0.0 else 0.5)
            else:
                gate = 1.0
            emfd = v_d * gate * np.sign(y[j]) * du_dy[j]
            dudt = beta * y[j] * vc[j] - merid - vert - u[j] * epsilon_u - emfd
            un = u_prev[j] + two_dt * dudt
            u_prev[j] = u[j] + asselin_coef * (un + u_prev[j] - 2 * u[j])
            u_next[j] = un

        # --- v on the faces (reads the still-uncommitted u and v) ---
        for j in range(nv):
            uf = 0.5 * (u[j] + u[j + 1])
            dt_dy = (theta[j + 1] * theta_to_temp - theta[j] * theta_to_temp) / dy
            dp_term = gravity * height * dt_dy / t_ref
            coriolis = beta * y_v[j] * uf
            left = -v[0] if j == 0 else v[j - 1]
            right = -v[nv - 1] if j == nv - 1 else v[j + 1]
            lap = ((right + left) - 2.0 * v[j]) / dy**2
            dvdt = (-coriolis - dp_term + lap * k_v) / 2
            vn = v_prev[j] + two_dt * dvdt
            v_prev[j] = v[j] + asselin_coef * (vn + v_prev[j] - 2 * v[j])
            v_next[j] = vn

        # --- theta ---
        for j in range(ny):
            newt = (theta_e_day[row, j] - theta[j]) / tau
            vadv_th = -delta * delta_z * dvdy[j] / height
            if coeff_eddy_heat_diff == 0.0:
                eddy = 0.0
            else:
                if j == 0:
                    g2 = (d1[1] - d1[0]) / dy
                elif j == ny - 1:
                    g2 = (d1[ny - 1] - d1[ny - 2]) / dy
                else:
                    g2 = (d1[j + 1] - d1[j - 1]) / (2.0 * dy)
                eddy = coeff_eddy_heat_diff * g2
            dthdt = newt + vadv_th + eddy
            thn = theta_prev[j] + two_dt * dthdt
            theta_prev[j] = theta[j] + asselin_coef * (thn + theta_prev[j] - 2 * theta[j])
            theta_next[j] = thn

        # --- commit state, BCs on the state only, store, NaN check ---
        nan_found = False
        for j in range(ny):
            u[j] = u_next[j]
            theta[j] = theta_next[j]
        for j in range(nv):
            v[j] = v_next[j]
        u[0] = 0.0
        u[ny - 1] = 0.0
        for j in range(ny):
            temp_u[k, j] = u[j]
            temp_theta[k, j] = theta[j]
            if np.isnan(u[j]):
                nan_found = True
        for j in range(nv):
            temp_v[k, j] = v[j]
        temp_time[k] = t / 86400
        if nan_found:
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
