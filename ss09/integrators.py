"""Method-of-lines time integrators for the staggered shallow-water model.

The model state is advanced by an :class:`Integrator` whose ``step`` takes the
current :class:`~ss09.model_state.ModelState`, the time step ``dt``, and a
callable ``rhs(state) -> (du, dv, dtheta)``. Keeping the integrator separate
from the RHS makes the scheme swappable (a future IMEX stepper reuses the same
RHS split with no change to the physics).
"""


class Integrator:
    """Abstract time integrator (strategy interface)."""

    def step(self, state, dt, rhs):
        raise NotImplementedError


class RK4Integrator(Integrator):
    """Classic 4-stage, 4th-order explicit Runge-Kutta.

    Self-starting (no leapfrog ``n-1`` level, Asselin filter, or backward-Euler
    seed). Stage states carry the correct intermediate time so time-dependent
    forcing (the seasonal ``theta_e``) is sampled at ``t``, ``t+dt/2``,
    ``t+dt/2``, ``t+dt``. Because the RHS gives the boundary ``v`` faces zero
    tendency, they remain exactly 0 through every stage.
    """

    def step(self, state, dt, rhs):
        t0 = state.t
        u0, v0, th0 = state.u, state.v, state.theta

        def deriv(u, v, th, t):
            return rhs(state._replace(t=t, u=u, v=v, theta=th))

        k1u, k1v, k1t = deriv(u0, v0, th0, t0)
        k2u, k2v, k2t = deriv(
            u0 + 0.5 * dt * k1u, v0 + 0.5 * dt * k1v, th0 + 0.5 * dt * k1t, t0 + 0.5 * dt
        )
        k3u, k3v, k3t = deriv(
            u0 + 0.5 * dt * k2u, v0 + 0.5 * dt * k2v, th0 + 0.5 * dt * k2t, t0 + 0.5 * dt
        )
        k4u, k4v, k4t = deriv(
            u0 + dt * k3u, v0 + dt * k3v, th0 + dt * k3t, t0 + dt
        )

        u = u0 + dt / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
        v = v0 + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        th = th0 + dt / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t)
        return state._replace(t=t0 + dt, u=u, v=v, theta=th)
