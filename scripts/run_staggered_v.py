"""Staggered-v patch test: does putting v on cell faces beat compact-k_v?

Continues the equilibrated B1 gate-on+mc state (restart_day5475.nc) with
v moved in-process onto the C-grid faces y_i + dy/2 (u and theta stay on
the 801 centers), the arrangement SS09 section 2b uses for exactly this
noise-control purpose. The divergence dv/dy and the pressure gradient
dT/dy become compact 2-point stencils (no 2*dy blindness), and grid-scale
v structure acquires nondegenerate gravity-wave dispersion. No model code
is modified.

The standing interior v ripple is terminus-forced; the compact-Laplacian
k_v patch (run_compact_kv.py, 2026-07-10) bought only 1.1x (terminus
band) to 3.4x (deep tropics), so that is the floor to beat. The test
refutes the staggering lever if the reduction lands in the same 1-3x
class, confirms it if the envelope drops well below.

Storage: state.v[i] holds the face value at y_i + dy/2 for i < ny-1;
slot ny-1 is zero padding. Saved daily v is therefore on faces (shifted
dy/2 poleward of its labeled coordinate); the in-script analysis accounts
for this. End faces are pinned to zero (v=0 within half a cell of each
wall), a symmetric over-constraint confined to the dynamically dead wall
cells.

Usage:
    python scripts/run_staggered_v.py [--extend-days 200]
        [--restart .../b1_y0p0000_gateon_mc/restart_day5475.nc]
        [--outdir model_output/formulation_suite/mc_stencil/staggered_v]
"""
import argparse
import logging
import pathlib
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ss09.sw_config import SWConfig  # noqa: E402
from ss09.sw_model import SWModel, THETA_TO_TEMP  # noqa: E402
from ss09.theta_e import ThetaEConfig, Sin2Profile  # noqa: E402
from cmp_utils import sawtooth  # noqa: E402

Mm = 1e6
BANDS = [(0, 2), (2, 5), (5, 8)]
B1_DIR = pathlib.Path(
    "model_output/formulation_suite/mc_stencil/b1_y0p0000_gateon_mc")


class StaggeredVModel(SWModel):
    """SWModel with v on cell faces; u, theta unchanged on centers."""

    def _v_at_centers(self) -> np.ndarray:
        f = self.state.v
        vc = np.zeros_like(f)
        vc[1:-1] = 0.5 * (f[:-2] + f[1:-1])
        return vc

    def dv_dy(self) -> np.ndarray:
        """Compact divergence at centers from face v."""
        f = self.state.v
        d = np.zeros_like(f)
        d[1:-1] = (f[1:-1] - f[:-2]) / self.config.dy
        d[0] = 2.0 * f[0] / self.config.dy      # half-cell at south wall
        d[-1] = -2.0 * f[-2] / self.config.dy   # half-cell at north wall
        return d

    def merid_advec_u(self) -> np.ndarray:
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
        vert = self.vert_advec_u() if self.config.include_vert_advec_u else 0
        merid = (
            self.merid_advec_u() if self.config.include_merid_advec_u else 0
        )
        return (
            self.coriolis_term(self._v_at_centers())
            - merid
            - vert
            - self.rayleigh_drag_u()
            - self.edd_mom_flux_div_u()
        )

    def diffusion_v(self) -> np.ndarray:
        """Compact 3-point Laplacian on the faces, v=0 wall ghosts."""
        f = self.state.v
        dy2 = self.config.dy ** 2
        lap = np.zeros_like(f)
        lap[1:-2] = (f[2:-1] - 2.0 * f[1:-2] + f[:-3]) / dy2
        lap[0] = (f[1] - 2.0 * f[0]) / dy2
        lap[-2] = (f[-3] - 2.0 * f[-2]) / dy2
        return lap * self.config.k_v

    def dv_dt(self) -> np.ndarray:
        u = self.state.u
        y = self.config.y
        dy = self.config.dy
        uf = np.zeros_like(u)
        uf[:-1] = 0.5 * (u[:-1] + u[1:])
        yf = np.zeros_like(y)
        yf[:-1] = y[:-1] + 0.5 * dy
        T = self.state.theta * THETA_TO_TEMP
        dT = np.zeros_like(T)
        dT[:-1] = (T[1:] - T[:-1]) / dy
        dp = self.config.gravity * self.config.height * dT / self.config.t_ref
        out = (-self.config.beta * yf * uf - dp + self.diffusion_v()) / 2
        out[-1] = 0.0
        return out

    def enforce_boundary_conditions(self):
        self.state.u[0] = 0
        self.state.u[-1] = 0
        self.state.v[0] = 0    # southmost face
        self.state.v[-2] = 0   # northmost face (parity twin of v[0])
        self.state.v[-1] = 0   # padding slot

    def load_from_restart(self, restart_file: str) -> int:
        """Restore, then move v (and its n-1 state) from centers to faces."""
        day = super().load_from_restart(restart_file)
        for owner, setter in (
            (self.state.v, lambda f: setattr(
                self, "state", self.state._replace(v=f))),
            (self.vars_prev_step.v, lambda f: setattr(
                self, "vars_prev_step", self.vars_prev_step._replace(v=f))),
        ):
            f = np.zeros_like(owner)
            f[:-1] = 0.5 * (owner[:-1] + owner[1:])
            setter(f)
        return day


def analyze(outdir: pathlib.Path, extend_days: int):
    """Last-100-d staggered climatology vs the B1 (collocated) baseline."""
    ds = xr.open_dataset(outdir / "output.nc", decode_timedelta=False)
    nt = ds.sizes["time"]
    navg = min(100, nt)
    y = ds["y"].values
    u = ds["u"].values[nt - navg:].mean(axis=0)
    f = ds["v"].values[nt - navg:].mean(axis=0)
    T = ds["T"].values[nt - navg:].mean(axis=0)
    nan_days = int(np.isnan(ds["u"].values).any(axis=1).sum())
    ds.close()

    ref = xr.open_dataset(B1_DIR / "output.nc", decode_timedelta=False)
    ntr = ref.sizes["time"]
    ur = ref["u"].values[ntr - 1825:].mean(axis=0)
    vr = ref["v"].values[ntr - 1825:].mean(axis=0)
    Tr = ref["T"].values[ntr - 1825:].mean(axis=0)
    ref.close()

    dy = y[1] - y[0]
    yf = y[:-1] + 0.5 * dy
    fv = f[:-1]  # real faces, padding dropped

    print(f"\n=== staggered-v extension ({extend_days} d, last {navg} d "
          f"averaged); NaN days: {nan_days} ===")
    print("banded max sawtooth(v): staggered vs B1 collocated baseline")
    print(f"{'band Mm':>10} {'staggered':>11} {'B1':>9} {'reduction':>10}")
    for a, b in BANDS:
        m_f = (np.abs(yf) >= a * Mm) & (np.abs(yf) < b * Mm)
        m_c = (np.abs(y) >= a * Mm) & (np.abs(y) < b * Mm)
        s_f = np.nanmax(np.where(m_f, sawtooth(fv), np.nan))
        s_c = np.nanmax(np.where(m_c, sawtooth(vr), np.nan))
        print(f"[{a},{b}) {'':>3} {s_f:>11.6f} {s_c:>9.5f} {s_c / s_f:>9.1f}x")

    inner = np.abs(y) < 8 * Mm
    su = np.nanmax(np.where(inner, sawtooth(u), np.nan))
    sur = np.nanmax(np.where(inner, sawtooth(ur), np.nan))
    print(f"interior sawtooth(u): {su:.4f} vs B1 {sur:.4f}")

    ieq = int(np.argmin(np.abs(y)))
    notch = (y >= -10.5 * Mm) & (y <= -7 * Mm)
    print("\nclimate anchors (staggered vs B1):")
    print(f"  jet u_max      {u.max():.4f} vs {ur.max():.4f}")
    print(f"  v_absmax       {np.abs(fv).max():.4f} vs {np.abs(vr).max():.4f}"
          f"  (faces vs centers, dy/2 shift)")
    print(f"  T_eq           {T[ieq]:.4f} vs {Tr[ieq]:.4f}")
    print(f"  notch depth    {u[notch].min():.2f} vs {ur[notch].min():.2f}")
    print(f"  parity max|u(y)-u(-y)|   {np.max(np.abs(u - u[::-1])):.3g}")
    print(f"  parity max|vf+vf(mirror)| {np.max(np.abs(fv + fv[::-1])):.3g}")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extend-days", type=int, default=200)
    p.add_argument("--restart", default=str(B1_DIR / "restart_day5475.nc"))
    p.add_argument("--cold", action="store_true",
                   help="cold start (v=0 faces) instead of a warm restart")
    p.add_argument("--ndays", type=int, default=5475,
                   help="total integration days for a --cold run")
    p.add_argument("--vd", type=float, default=2.5, dest="v_d")
    p.add_argument("--restart-every", type=int, default=0,
                   dest="save_restart_every")
    p.add_argument("--skip-analysis", action="store_true",
                   help="skip the B1-baseline analyze() (e.g. at vd=0, "
                        "where that comparison is meaningless)")
    p.add_argument(
        "--outdir",
        default="model_output/formulation_suite/mc_stencil/staggered_v")
    args = p.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    total = args.ndays if args.cold else 5475 + args.extend_days
    config = SWConfig(
        total_integration_days=total,
        ny=801,
        dt=30,
        v_d=args.v_d,
        emfd_heaviside_gate=True,
        emfd_stencil="mc",
        save_restart_every=args.save_restart_every,
        output_path=str(outdir / "output.nc"),
        restart_output_dir=str(outdir),
    )
    model = StaggeredVModel(config, Sin2Profile(ThetaEConfig()))
    if not args.cold:
        model.restart_day = model.load_from_restart(args.restart)
    model.run_sim()
    model.save_results()
    if not args.skip_analysis:
        analyze(outdir, args.extend_days)


if __name__ == "__main__":
    main()
