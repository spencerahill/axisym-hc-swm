"""Bitwise parity of the numba backend with the numpy reference on the sphere.

The same contract as test_numba_backend.py (max|delta| == 0.0, machine-local):
the fused day kernel carries the spherical geometry arrays (cos(lat) at the
centers and faces, its square and reciprocal, f, the metric factor) in the
same positions as the numpy operators, so any nonzero difference is a
transcription bug in the kernel. Each integration test asserts its stored-day
count, so a configuration that blows up cannot pass on zero-filled buffers.

Grids: ny=41 (dy = 318 km) at dt=300 and ny=81 (dy = 159 km) at dt=120 are
finite through the runs below; ny=81 at dt=600 goes NaN in u on day 3
(measured 2026-11-08), which the NaN-stop test uses.
"""
import os

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("numba")

from ss09 import numba_backend
from ss09.sw_config import SWConfig
from ss09.sw_model import AuxiliaryVars, SWModel, TempVars
from ss09.theta_e import SphericalProfile, ThetaEConfig

pytestmark = pytest.mark.numba


def build(tmp_path, backend, ny=41, dt=300, ndays=3, theta_e_kwargs=None, **kw):
    outdir = tmp_path / backend
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = SWConfig(
        geometry="sphere", ny=ny, dt=dt, total_integration_days=ndays,
        backend=backend, output_path=str(outdir / "out.nc"),
        restart_output_dir=str(outdir), **kw,
    )
    te = dict(theta_e_type="sphere")
    te.update(theta_e_kwargs or {})
    return SWModel(cfg, SphericalProfile(ThetaEConfig(**te)))


def run_pair(tmp_path, **kw):
    m_np = build(tmp_path, "numpy", **kw)
    m_np.run_sim()
    m_nb = build(tmp_path, "numba", **kw)
    m_nb.run_sim()
    return m_np, m_nb


def assert_bitwise(a, b, name):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    if not np.array_equal(a, b, equal_nan=True):
        with np.errstate(invalid="ignore"):
            maxd = np.nanmax(np.abs(a - b))
        raise AssertionError(f"{name}: not bitwise identical, max|delta|={maxd!r}")


def stored_days(model):
    return int(np.count_nonzero(model.results.time != 0))


def assert_daily_parity(m_np, m_nb):
    for name in ("time", "u", "v", "theta"):
        assert_bitwise(getattr(m_np.results, name), getattr(m_nb.results, name),
                       f"daily {name}")


@pytest.mark.parametrize("form", ["angular_momentum", "vorticity", "literal"])
def test_single_step_parity_with_planted_zeros(tmp_path, form):
    """One kernel step against one reference loop-body step on the sphere,
    for each form of the eddy term, from a state planting the H(0) = 0.5
    branches (exact and signed zeros in u, theta == theta_E at points, zero
    faces), including the filtered prev state's unconstrained pole values."""
    m_ref = build(tmp_path, "numpy", ny=81, dt=120, ndays=1, emfd_form=form)
    m_nb = build(tmp_path, "numba", ny=81, dt=120, ndays=1, emfd_form=form)
    rng = np.random.default_rng(17)
    ny = m_ref.config.ny
    u0 = rng.normal(0.0, 20.0, ny)
    u0[[5, 20]] = 0.0
    u0[30] = -0.0
    v0 = rng.normal(0.0, 5.0, ny - 1)
    v0[3] = 0.0
    v0[17] = -0.0
    th0 = m_ref._theta_e_static + rng.normal(0.0, 2.0, ny)
    th0[[8, 40]] = m_ref._theta_e_static[[8, 40]]
    m_ref.state = m_ref.state._replace(u=u0.copy(), v=v0.copy(), theta=th0.copy())
    m_ref.vars_prev_step = AuxiliaryVars(*m_ref.init_prev_step_vars())
    m_nb.state = m_nb.state._replace(u=u0.copy(), v=v0.copy(), theta=th0.copy())
    m_nb.vars_prev_step = AuxiliaryVars(*(x.copy() for x in m_ref.vars_prev_step))

    m_ref.state = m_ref.state._replace(t=0)
    nu = m_ref.leapfrog_step(m_ref.vars_prev_step.u, m_ref.du_dt)
    nv = m_ref.leapfrog_step(m_ref.vars_prev_step.v, m_ref.dv_dt)
    nth = m_ref.leapfrog_step(m_ref.vars_prev_step.theta, m_ref.dtheta_dt)
    prev = AuxiliaryVars(
        m_ref.asselin_filt(m_ref.vars_prev_step.u, nu, m_ref.state.u),
        m_ref.asselin_filt(m_ref.vars_prev_step.v, nv, m_ref.state.v),
        m_ref.asselin_filt(m_ref.vars_prev_step.theta, nth, m_ref.state.theta),
    )
    m_ref.state = m_ref.state._replace(u=nu, v=nv, theta=nth)
    m_ref.enforce_boundary_conditions()

    m_nb.temp_vars = TempVars(
        u=np.zeros((1, ny)), v=np.zeros((1, ny - 1)), theta=np.zeros((1, ny)),
        theta_e=np.zeros((1, ny)), time=np.zeros(1),
    )
    nan_step = numba_backend.run_day(**numba_backend.day_kernel_args(
        m_nb, m_nb._theta_e_static.reshape(1, -1), start_step=0))
    assert nan_step == -1
    for name in ("u", "v", "theta"):
        assert_bitwise(getattr(m_ref.state, name), getattr(m_nb.state, name), name)
        assert_bitwise(getattr(prev, name), getattr(m_nb.vars_prev_step, name),
                       f"{name}_prev")


VARIANTS = [
    ("default", {}, {}),
    ("gate_off", dict(emfd_heaviside_gate=False), {}),
    ("upwind", dict(emfd_stencil="upwind"), {}),
    ("centered", dict(emfd_stencil="centered"), {}),
    ("no_merid", dict(include_merid_advec_u=False), {}),
    ("no_vert", dict(include_vert_advec_u=False), {}),
    ("eddy_heat", dict(coeff_eddy_heat_diff=5e3), {}),
    ("kinematic_gate", dict(vert_advec_gate="kinematic"), {}),
    ("off_equator", {}, dict(lat_0=8.0)),
    ("small_planet", dict(radius=3.0e6, rot_rate=1.5e-4), dict(radius=3.0e6)),
    ("vorticity_form", dict(emfd_form="vorticity"), {}),
    ("literal_form", dict(emfd_form="literal"), {}),
    ("vorticity_centered", dict(emfd_form="vorticity", emfd_stencil="centered"), {}),
    ("literal_upwind", dict(emfd_form="literal", emfd_stencil="upwind"), {}),
]


@pytest.mark.parametrize("name,kw,te_kw", VARIANTS, ids=[v[0] for v in VARIANTS])
def test_sphere_parity_variants(tmp_path, name, kw, te_kw):
    m_np, m_nb = run_pair(tmp_path, ndays=3, theta_e_kwargs=te_kw, **kw)
    assert stored_days(m_np) == stored_days(m_nb) == 3
    assert_daily_parity(m_np, m_nb)
    assert_bitwise(m_np.vars_prev_step.u, m_nb.vars_prev_step.u, "final u_prev")


def test_sphere_parity_finer_grid(tmp_path):
    m_np, m_nb = run_pair(tmp_path, ny=81, dt=120, ndays=2)
    assert stored_days(m_np) == stored_days(m_nb) == 2
    assert_daily_parity(m_np, m_nb)


def test_sphere_numba_holds_mirror_parity(tmp_path):
    m_nb = build(tmp_path, "numba", ny=81, dt=120, ndays=4)
    m_nb.run_sim()
    u, v = m_nb.state.u, m_nb.state.v
    assert stored_days(m_nb) == 4 and np.max(np.abs(v)) > 1.0
    assert np.max(np.abs(u - u[::-1])) == 0.0
    assert np.max(np.abs(v + v[::-1])) == 0.0


def test_metric_switch_changes_the_kernel_solution(tmp_path, monkeypatch):
    """Guard against a kernel that ignores the metric switch: with it off the
    solution differs, so the parity tests above exercise the metric term."""
    on = build(tmp_path / "on", "numba", ndays=2)
    on.run_sim()
    off = build(tmp_path / "off", "numba", ndays=2)
    real = numba_backend.day_kernel_args

    def no_metric(*args, **kwargs):
        d = real(*args, **kwargs)
        d["metric_on"] = False
        return d

    monkeypatch.setattr(numba_backend, "day_kernel_args", no_metric)
    off.run_sim()
    assert np.max(np.abs(on.state.u - off.state.u)) > 1e-3


def test_emfd_forms_give_different_kernel_solutions(tmp_path):
    """Guard against a kernel that ignores the form: each pair of forms gives
    a different solution, so the form variants above are not parity of three
    copies of one integration."""
    finals = {}
    for form in ("angular_momentum", "vorticity", "literal"):
        m = build(tmp_path / form, "numba", ndays=2, emfd_form=form)
        m.run_sim()
        assert stored_days(m) == 2
        finals[form] = m.state.u
    forms = list(finals)
    for i, a in enumerate(forms):
        for b in forms[i + 1:]:
            assert np.max(np.abs(finals[a] - finals[b])) > 1e-6, (a, b)


def test_sphere_nan_stop_equivalence(tmp_path):
    m_np, m_nb = run_pair(tmp_path, ny=81, dt=600, ndays=4)
    assert stored_days(m_np) == stored_days(m_nb) < 4
    assert np.isnan(m_np.state.u).any()
    assert_daily_parity(m_np, m_nb)
    assert_bitwise(m_np.state.u, m_nb.state.u, "final u (NaN pattern)")


def test_sphere_restart_numba_continuation_and_cross_backend(tmp_path):
    straight = build(tmp_path / "straight", "numba", ndays=4)
    straight.run_sim()
    first = build(tmp_path / "split", "numpy", ndays=2)
    first.run_sim()
    restart = str(tmp_path / "split" / "numpy" / "restart_day0002.nc")
    assert os.path.exists(restart)
    cont = build(tmp_path / "cont", "numba", ndays=4)
    cont.restart_day = cont.load_from_restart(restart)
    cont.run_sim()
    for day in (2, 3):
        for name in ("u", "v", "theta"):
            assert_bitwise(getattr(straight.results, name)[day],
                           getattr(cont.results, name)[day], f"{name} day {day}")
    assert_bitwise(straight.vars_prev_step.u, cont.vars_prev_step.u, "final u_prev")


def test_sphere_full_dataset_parity(tmp_path):
    kw = dict(ndays=4, enable_steady_state=True, steady_state_window_size=3,
              steady_state_threshold=0.0)
    m_np, m_nb = run_pair(tmp_path, **kw)
    m_np.save_results()
    m_nb.save_results()
    ds_np = xr.open_dataset(m_np.config.output_path, decode_times=False)
    ds_nb = xr.open_dataset(m_nb.config.output_path, decode_times=False)
    assert set(ds_np.data_vars) == set(ds_nb.data_vars)
    assert {"lat", "lat_face"} <= set(ds_np.coords)
    for var in ds_np.data_vars:
        assert_bitwise(ds_np[var].values, ds_nb[var].values, f"data_var {var}")
    for coord in ds_np.coords:
        assert_bitwise(ds_np[coord].values, ds_nb[coord].values, f"coord {coord}")
    ds_np.close()
    ds_nb.close()


def test_numba_kernel_invoked_on_the_sphere(tmp_path, monkeypatch):
    m = build(tmp_path, "numba", ndays=1)
    calls = []
    real = numba_backend.run_day

    def spy(**kwargs):
        calls.append(kwargs["metric_on"])
        return real(**kwargs)

    monkeypatch.setattr(numba_backend, "run_day", spy)
    m.run_sim()
    assert calls == [True]
