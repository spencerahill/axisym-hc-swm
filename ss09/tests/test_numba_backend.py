"""Tests for the numba backend: bitwise parity against the numpy reference.

Every parity assertion here is bitwise (max|delta| == 0.0), not tolerance:
the kernel transcribes the reference operation-for-operation, so any nonzero
difference is a transcription bug. Configs are pinned to smoke-verified
stable trajectories (ny=51/dt=3600 blows up by day 4-5 and cannot be used
for parity runs; ny=51/dt=1800 is finite through at least 6 days).
"""
import os
import subprocess

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("numba")

from ss09 import numba_backend
from ss09.model_state import ModelState
from ss09.sw_config import SWConfig
from ss09.sw_model import (
    AuxiliaryVars,
    SWModel,
    TempVars,
    mc_limited_slope,
    muscl_mc_du_dy,
    v_divergence_at_centers,
    v_face_laplacian,
    v_faces_to_centers,
)
from ss09.theta_e import SB08Profile, SS09Profile, Sin2Profile, ThetaEConfig

pytestmark = pytest.mark.numba

PROFILES = {"SS09": SS09Profile, "sin2": Sin2Profile, "SB08": SB08Profile}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model(tmp_path, backend, ny=51, dt=1800, ndays=5,
                theta_e_kwargs=None, **config_kwargs):
    outdir = tmp_path / backend
    outdir.mkdir(parents=True, exist_ok=True)
    config = SWConfig(
        total_integration_days=ndays,
        ny=ny,
        dt=dt,
        backend=backend,
        output_path=str(outdir / "out.nc"),
        restart_output_dir=str(outdir),
        **config_kwargs,
    )
    te_kwargs = dict(theta_e_type="sin2")
    te_kwargs.update(theta_e_kwargs or {})
    te_config = ThetaEConfig(**te_kwargs)
    profile = PROFILES[te_config.theta_e_type](te_config)
    return SWModel(config, profile)


def run_pair(tmp_path, **kwargs):
    m_np = build_model(tmp_path, "numpy", **kwargs)
    m_np.run_sim()
    m_nb = build_model(tmp_path, "numba", **kwargs)
    m_nb.run_sim()
    return m_np, m_nb


def assert_bitwise(a, b, name):
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    if not np.array_equal(a, b, equal_nan=True):
        with np.errstate(invalid="ignore"):
            maxd = np.nanmax(np.abs(a - b))
        raise AssertionError(f"{name}: not bitwise identical, max|delta|={maxd!r}")


def assert_daily_parity(m_np, m_nb, include_theta_e=False):
    assert_bitwise(m_np.results.time, m_nb.results.time, "daily time")
    assert_bitwise(m_np.results.u, m_nb.results.u, "daily u")
    assert_bitwise(m_np.results.v, m_nb.results.v, "daily v")
    assert_bitwise(m_np.results.theta, m_nb.results.theta, "daily theta")
    if include_theta_e:
        assert_bitwise(m_np.results.theta_e, m_nb.results.theta_e, "daily theta_e")


def stored_days(model):
    return int(np.count_nonzero(model.results.time != 0))


# ---------------------------------------------------------------------------
# Operator-level unit tests on adversarial inputs (exact zeros, signed zeros,
# walls, sign boundaries). Random fields alone never produce exact zeros, and
# the H(x=0)=0.5 branches are dynamically inert in normal runs, so these are
# the only tests that pin them.
# ---------------------------------------------------------------------------

def _adversarial(rng, n, scale):
    x = rng.normal(0.0, scale, n)
    idx = rng.choice(n, size=min(4, n), replace=False)
    x[idx[0]] = 0.0
    if n > 3:
        x[idx[1]] = -0.0
    return x


def test_kernel_gate_matches_heaviside():
    x = np.array([-5.0, -0.0, 0.0, 3.0, 1e-300, -1e-300, 2.5])
    assert_bitwise(np.heaviside(x, 0.5), numba_backend.heaviside_gate(x), "gate")


@pytest.mark.parametrize("ny", [20, 51])
def test_kernel_mc_stencil_matches_reference(ny):
    rng = np.random.default_rng(1)
    config = SWConfig(ny=ny, dt=1800, backend="numpy")
    for trial in range(50):
        u = _adversarial(rng, ny, 30.0)
        # plant a flat pair (zero one-sided slope) and a local extremum
        u[ny // 2] = u[ny // 2 - 1]
        assert_bitwise(
            mc_limited_slope(u, config.dy),
            numba_backend.mc_limited_slope(u, config.dy),
            f"mc_limited_slope trial {trial}",
        )
        assert_bitwise(
            muscl_mc_du_dy(u, config.dy, config.y),
            numba_backend.muscl_mc_du_dy(u, config.dy, config.y),
            f"muscl_mc_du_dy trial {trial}",
        )


def test_kernel_staggered_operators_match_reference():
    rng = np.random.default_rng(2)
    dy = 630040.0
    for trial in range(50):
        n_faces = int(rng.integers(4, 200))
        f = _adversarial(rng, n_faces, 5.0)
        assert_bitwise(
            v_faces_to_centers(f),
            numba_backend.v_faces_to_centers(f),
            f"faces_to_centers trial {trial}",
        )
        assert_bitwise(
            v_divergence_at_centers(f, dy),
            numba_backend.v_divergence_at_centers(f, dy),
            f"divergence trial {trial}",
        )
        assert_bitwise(
            v_face_laplacian(f, dy),
            numba_backend.v_face_laplacian(f, dy),
            f"laplacian trial {trial}",
        )


def test_kernel_gradient_matches_np_gradient():
    rng = np.random.default_rng(3)
    for trial in range(50):
        n = int(rng.integers(5, 400))
        f = rng.normal(320.0, 10.0, n)
        dy = float(rng.uniform(1e3, 1e6))
        assert_bitwise(
            np.gradient(f, dy),
            numba_backend.gradient_uniform(f, dy),
            f"gradient trial {trial}",
        )


def test_kernel_merid_advec_matches_reference(tmp_path):
    m = build_model(tmp_path, "numpy")
    rng = np.random.default_rng(4)
    ny = m.config.ny
    for trial in range(50):
        u = _adversarial(rng, ny, 30.0)
        v = _adversarial(rng, ny - 1, 5.0)
        m.state = m.state._replace(u=u, v=v)
        vc = v_faces_to_centers(v)
        assert_bitwise(
            m.merid_advec_u(),
            numba_backend.merid_advec_u_term(u, vc, m.config.dy),
            f"merid advec trial {trial}",
        )


def test_single_step_parity_with_planted_zeros(tmp_path):
    """One composed kernel step vs a manual reference loop-body step, on a
    state planting every H(x=0)=0.5 branch: exact/signed zeros in interior u
    (EMFD gate), theta == theta_e exactly at points (vert-advec gate), zero v
    faces (upwind branch selection). Compares the filtered prev state too,
    including its un-BC'd wall values: the Asselin filter must consume the
    pre-BC leapfrog fields and the BC must touch only the state arrays."""
    m_ref = build_model(tmp_path, "numpy", ndays=1)
    m_nb = build_model(tmp_path, "numba", ndays=1)
    rng = np.random.default_rng(7)
    ny = m_ref.config.ny

    u0 = rng.normal(0.0, 20.0, ny)
    u0[5] = 0.0
    u0[20] = 0.0
    u0[30] = -0.0
    v0 = rng.normal(0.0, 5.0, ny - 1)
    v0[3] = 0.0
    v0[17] = -0.0
    th0 = m_ref._theta_e_static + rng.normal(0.0, 2.0, ny)
    th0[8] = m_ref._theta_e_static[8]
    th0[40] = m_ref._theta_e_static[40]

    m_ref.state = m_ref.state._replace(u=u0.copy(), v=v0.copy(), theta=th0.copy())
    m_ref.vars_prev_step = AuxiliaryVars(*m_ref.init_prev_step_vars())
    m_nb.state = m_nb.state._replace(u=u0.copy(), v=v0.copy(), theta=th0.copy())
    m_nb.vars_prev_step = AuxiliaryVars(
        u=m_ref.vars_prev_step.u.copy(),
        v=m_ref.vars_prev_step.v.copy(),
        theta=m_ref.vars_prev_step.theta.copy(),
    )

    # reference: one manual execution of the run_sim loop body
    m_ref.state = m_ref.state._replace(t=0)
    next_u = m_ref.leapfrog_step(m_ref.vars_prev_step.u, m_ref.du_dt)
    next_v = m_ref.leapfrog_step(m_ref.vars_prev_step.v, m_ref.dv_dt)
    next_th = m_ref.leapfrog_step(m_ref.vars_prev_step.theta, m_ref.dtheta_dt)
    prev_u = m_ref.asselin_filt(m_ref.vars_prev_step.u, next_u, m_ref.state.u)
    prev_v = m_ref.asselin_filt(m_ref.vars_prev_step.v, next_v, m_ref.state.v)
    prev_th = m_ref.asselin_filt(m_ref.vars_prev_step.theta, next_th, m_ref.state.theta)
    m_ref.state = m_ref.state._replace(u=next_u, v=next_v, theta=next_th)
    m_ref.enforce_boundary_conditions()

    # kernel: a 1-step "day"
    m_nb.temp_vars = TempVars(
        u=np.zeros((1, ny)),
        v=np.zeros((1, ny - 1)),
        theta=np.zeros((1, ny)),
        theta_e=np.zeros((1, ny)),
        time=np.zeros(1),
    )
    theta_e_day = m_nb._theta_e_static.reshape(1, -1)
    nan_step = numba_backend.run_day(
        **numba_backend.day_kernel_args(m_nb, theta_e_day, start_step=0)
    )
    assert nan_step == -1

    assert_bitwise(m_ref.state.u, m_nb.state.u, "u after 1 step")
    assert_bitwise(m_ref.state.v, m_nb.state.v, "v after 1 step")
    assert_bitwise(m_ref.state.theta, m_nb.state.theta, "theta after 1 step")
    assert_bitwise(prev_u, m_nb.vars_prev_step.u, "u_prev incl un-BC'd walls")
    assert_bitwise(prev_v, m_nb.vars_prev_step.v, "v_prev")
    assert_bitwise(prev_th, m_nb.vars_prev_step.theta, "theta_prev")
    assert_bitwise(m_ref.state.u, m_nb.temp_vars.u[0], "stored u row")
    assert_bitwise(m_ref.state.v, m_nb.temp_vars.v[0], "stored v row")
    assert m_nb.temp_vars.time[0] == 0.0


# ---------------------------------------------------------------------------
# Integration parity: full runs, daily outputs bitwise. Each parity test
# asserts the expected stored-day count so a blowing-up config cannot pass
# vacuously on zero-filled buffers.
# ---------------------------------------------------------------------------

def test_parity_default_five_days(tmp_path):
    m_np, m_nb = run_pair(tmp_path, ndays=5)
    assert stored_days(m_np) == stored_days(m_nb) == 5
    assert_daily_parity(m_np, m_nb)


VARIANTS = [
    ("gate_off", dict(emfd_heaviside_gate=False), {}),
    ("gate_off_upwind", dict(emfd_heaviside_gate=False, emfd_stencil="upwind"), {}),
    ("upwind", dict(emfd_stencil="upwind"), {}),
    ("centered", dict(emfd_stencil="centered"), {}),
    ("no_merid", dict(include_merid_advec_u=False), {}),
    ("no_vert", dict(include_vert_advec_u=False), {}),
    ("eddy_heat", dict(coeff_eddy_heat_diff=5e3), {}),
    ("ss09_profile", {}, dict(theta_e_type="SS09")),
]


@pytest.mark.parametrize(
    "name,config_kwargs,theta_e_kwargs", VARIANTS, ids=[v[0] for v in VARIANTS]
)
def test_parity_variants(tmp_path, name, config_kwargs, theta_e_kwargs):
    m_np, m_nb = run_pair(
        tmp_path, ndays=3, theta_e_kwargs=theta_e_kwargs, **config_kwargs
    )
    assert stored_days(m_np) == stored_days(m_nb) == 3
    assert_daily_parity(m_np, m_nb)


def test_parity_asymmetric_y0(tmp_path):
    m_np, m_nb = run_pair(tmp_path, ndays=4, theta_e_kwargs=dict(y_0=1500e3))
    assert stored_days(m_np) == stored_days(m_nb) == 4
    assert_daily_parity(m_np, m_nb)


def test_parity_even_ny(tmp_path):
    # even ny has no gridpoint at y=0: pins hemisphere selection by where(y>0)
    m_np, m_nb = run_pair(tmp_path, ny=20, ndays=2)
    assert stored_days(m_np) == stored_days(m_nb) == 2
    assert_daily_parity(m_np, m_nb)


def test_parity_production_resolution(tmp_path):
    m_np, m_nb = run_pair(tmp_path, ny=801, dt=30, ndays=2)
    assert stored_days(m_np) == stored_days(m_nb) == 2
    assert_daily_parity(m_np, m_nb)


@pytest.mark.parametrize("cycle", ["sin", "square", "tanh"])
def test_parity_seasonal(tmp_path, cycle):
    m_np, m_nb = run_pair(
        tmp_path,
        ndays=4,
        theta_e_kwargs=dict(
            theta_e_type="SB08",
            y_0_seasonal_amp=700e3,
            seasonal_cycle_type=cycle,
        ),
    )
    assert stored_days(m_np) == stored_days(m_nb) == 4
    assert m_np.results.store_theta_e and m_nb.results.store_theta_e
    assert_daily_parity(m_np, m_nb, include_theta_e=True)


@pytest.mark.parametrize("cycle", ["sin", "square", "tanh"])
def test_sb08_profile_at_times_matches_per_step(cycle):
    te_config = ThetaEConfig(
        theta_e_type="SB08",
        y_0_seasonal_amp=700e3,
        seasonal_cycle_type=cycle,
        seasonal_period_days=360.0,
        seasonal_phase_days=17.0,
    )
    profile = SB08Profile(te_config)
    config = SWConfig(ny=51, dt=1800, total_integration_days=1)
    y = config.y
    ts = (4321 + np.arange(48)) * 1800
    block = profile.profile_at_times(ts, y)
    assert block.shape == (48, 51)
    zeros = np.zeros_like(y)
    for k, t in enumerate(ts):
        ref = profile(ModelState(t=int(t), u=zeros, v=zeros, theta=zeros, y=y))
        assert_bitwise(ref, block[k], f"theta_e block row {k}")


def test_sb08_profile_at_times_zero_amplitude():
    """The amp=0 branch of profile_at_times (unused by the production driver,
    which only calls the helper for seasonal runs) still matches per-step
    evaluation."""
    te_config = ThetaEConfig(theta_e_type="SB08", y_0=300e3)
    profile = SB08Profile(te_config)
    config = SWConfig(ny=51, dt=1800, total_integration_days=1)
    y = config.y
    ts = np.arange(4) * 1800
    block = profile.profile_at_times(ts, y)
    zeros = np.zeros_like(y)
    for k, t in enumerate(ts):
        ref = profile(ModelState(t=int(t), u=zeros, v=zeros, theta=zeros, y=y))
        assert_bitwise(ref, block[k], f"static theta_e block row {k}")


def test_numba_backend_actually_invoked(tmp_path, monkeypatch):
    """Guard against a silent fallback to the reference loop, which would make
    every parity test pass vacuously."""
    m = build_model(tmp_path, "numba", ndays=1)
    calls = []
    real = numba_backend.run_day

    def spy(**kwargs):
        calls.append(1)
        return real(**kwargs)

    monkeypatch.setattr(numba_backend, "run_day", spy)
    m.run_sim()
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Restart semantics
# ---------------------------------------------------------------------------

def _final_restart(model_dir, day):
    path = os.path.join(model_dir, f"restart_day{day:04d}.nc")
    assert os.path.exists(path), f"missing {path}"
    return path


def test_restart_numba_continuation_bitwise(tmp_path):
    straight = build_model(tmp_path / "straight", "numba", ndays=4)
    straight.run_sim()
    first = build_model(tmp_path / "split", "numba", ndays=2)
    first.run_sim()
    restart_file = _final_restart(str(tmp_path / "split" / "numba"), 2)
    cont = build_model(tmp_path / "split", "numba", ndays=4)
    cont.restart_day = cont.load_from_restart(restart_file)
    cont.run_sim()
    for day in (2, 3):
        assert_bitwise(straight.results.u[day], cont.results.u[day], f"u day {day}")
        assert_bitwise(straight.results.v[day], cont.results.v[day], f"v day {day}")
        assert_bitwise(
            straight.results.theta[day], cont.results.theta[day], f"theta day {day}"
        )
    assert_bitwise(straight.state.u, cont.state.u, "final u")
    assert_bitwise(straight.state.v, cont.state.v, "final v")
    assert_bitwise(straight.state.theta, cont.state.theta, "final theta")
    assert_bitwise(straight.vars_prev_step.u, cont.vars_prev_step.u, "final u_prev")


def test_restart_cross_backend_bitwise(tmp_path):
    base = build_model(tmp_path / "base", "numpy", ndays=2)
    base.run_sim()
    restart_file = _final_restart(str(tmp_path / "base" / "numpy"), 2)
    continuations = {}
    for backend in ("numpy", "numba"):
        cont = build_model(tmp_path / f"cont_{backend}", backend, ndays=4)
        cont.restart_day = cont.load_from_restart(restart_file)
        cont.run_sim()
        continuations[backend] = cont
    c_np, c_nb = continuations["numpy"], continuations["numba"]
    for day in (2, 3):
        assert_bitwise(c_np.results.u[day], c_nb.results.u[day], f"u day {day}")
        assert_bitwise(c_np.results.v[day], c_nb.results.v[day], f"v day {day}")
    assert_bitwise(c_np.state.u, c_nb.state.u, "final u")
    assert_bitwise(c_np.vars_prev_step.u, c_nb.vars_prev_step.u, "final u_prev")
    assert_bitwise(c_np.vars_prev_step.v, c_nb.vars_prev_step.v, "final v_prev")


def test_restart_seasonal_numba_continuation(tmp_path):
    """A restart continuation must reconstruct the absolute seasonal phase:
    a helper that re-zeroes time after restart diverges here."""
    seasonal = dict(
        theta_e_type="SB08", y_0_seasonal_amp=700e3, seasonal_cycle_type="sin"
    )
    straight = build_model(
        tmp_path / "straight", "numba", ndays=4, theta_e_kwargs=seasonal
    )
    straight.run_sim()
    first = build_model(tmp_path / "split", "numba", ndays=2, theta_e_kwargs=seasonal)
    first.run_sim()
    restart_file = _final_restart(str(tmp_path / "split" / "numba"), 2)
    cont = build_model(tmp_path / "split", "numba", ndays=4, theta_e_kwargs=seasonal)
    cont.restart_day = cont.load_from_restart(restart_file)
    cont.run_sim()
    for day in (2, 3):
        assert_bitwise(straight.results.u[day], cont.results.u[day], f"u day {day}")
        assert_bitwise(
            straight.results.theta_e[day],
            cont.results.theta_e[day],
            f"theta_e day {day}",
        )
    # cross-backend anchor for the same seasonal continuation
    ref = build_model(tmp_path / "ref", "numpy", ndays=4, theta_e_kwargs=seasonal)
    ref.run_sim()
    for day in (2, 3):
        assert_bitwise(ref.results.u[day], cont.results.u[day], f"u vs numpy day {day}")


def test_restart_files_field_parity(tmp_path):
    """Restart files are the only observable of the filtered prev state's wall
    values (they are dynamically inert in the daily outputs): compare every
    field of the periodic and final restart files across backends."""
    for backend in ("numpy", "numba"):
        m = build_model(tmp_path, backend, ndays=4, save_restart_every=2)
        m.run_sim()
    fields = (
        "u", "v", "theta", "u_prev", "v_prev", "theta_prev",
        "current_time", "current_step", "current_day",
    )
    for day in (2, 4):
        ds_np = xr.open_dataset(
            str(tmp_path / "numpy" / f"restart_day{day:04d}.nc"), decode_times=False
        )
        ds_nb = xr.open_dataset(
            str(tmp_path / "numba" / f"restart_day{day:04d}.nc"), decode_times=False
        )
        for field in fields:
            assert_bitwise(
                ds_np[field].values, ds_nb[field].values, f"{field} day {day}"
            )
        ds_np.close()
        ds_nb.close()


# ---------------------------------------------------------------------------
# Early-stop paths: steady-state break, seasonal-convergence break, NaN break
# ---------------------------------------------------------------------------

def test_steady_state_break_parity(tmp_path):
    kwargs = dict(
        ndays=8,
        enable_steady_state=True,
        steady_state_window_size=2,
        steady_state_threshold=1e6,
    )
    m_np, m_nb = run_pair(tmp_path, **kwargs)
    for m in (m_np, m_nb):
        assert m.steady_state_detector.is_converged
        assert m.steady_state_detector.convergence_day == 1
        assert stored_days(m) == 2
    assert_daily_parity(m_np, m_nb)
    assert_bitwise(
        np.asarray(m_np.steady_state_detector.kinetic_energy_history),
        np.asarray(m_nb.steady_state_detector.kinetic_energy_history),
        "KE history",
    )
    assert_bitwise(
        np.asarray(m_np.steady_state_detector.temp_variance_history),
        np.asarray(m_nb.steady_state_detector.temp_variance_history),
        "Tvar history",
    )
    # break taken with day un-incremented: final restart tagged day0001
    _final_restart(str(tmp_path / "numpy"), 1)
    _final_restart(str(tmp_path / "numba"), 1)


def test_seasonal_convergence_break_parity(tmp_path):
    kwargs = dict(
        ndays=8,
        # the detector records history only when enable_steady_state is on,
        # so seasonal convergence checking requires both flags (as the CLI's
        # --stop-at-steady-state --seas-conv combination provides)
        enable_steady_state=True,
        steady_state_window_size=2,
        seasonal_convergence_enabled=True,
        seasonal_convergence_window=1,
        seasonal_convergence_threshold=1e6,
        theta_e_kwargs=dict(
            theta_e_type="SB08",
            y_0_seasonal_amp=50e3,
            seasonal_period_days=2.0,
        ),
    )
    m_np, m_nb = run_pair(tmp_path, **kwargs)
    assert stored_days(m_np) == stored_days(m_nb)
    assert stored_days(m_np) < 8, "seasonal-convergence break was never taken"
    assert_daily_parity(m_np, m_nb, include_theta_e=True)


def test_nan_stop_equivalence(tmp_path):
    """ny=101/dt=1800 NaNs in u mid-day-3 (smoke-verified, u first: v and theta
    still finite at the break): both backends must stop at the same step with
    the same stored days and identical final state including the NaN pattern."""
    m_np, m_nb = run_pair(tmp_path, ny=101, dt=1800, ndays=4)
    assert stored_days(m_np) == stored_days(m_nb) == 2
    assert_daily_parity(m_np, m_nb)
    assert_bitwise(m_np.state.u, m_nb.state.u, "final u (NaN pattern)")
    assert_bitwise(m_np.state.v, m_nb.state.v, "final v")
    assert_bitwise(m_np.state.theta, m_nb.state.theta, "final theta")
    _final_restart(str(tmp_path / "numpy"), 2)
    _final_restart(str(tmp_path / "numba"), 2)


# ---------------------------------------------------------------------------
# Full-dataset and end-to-end CLI parity
# ---------------------------------------------------------------------------

def test_full_dataset_parity(tmp_path):
    """save_results() output must be identical across backends in every data
    variable, coordinate, and attribute (except backend itself and
    path/timestamp attrs): catches a driver that skips diagnostics or
    mis-stamps time."""
    kwargs = dict(
        ndays=5,
        enable_steady_state=True,
        steady_state_window_size=3,
        steady_state_threshold=0.0,
    )
    m_np, m_nb = run_pair(tmp_path, **kwargs)
    m_np.save_results()
    m_nb.save_results()
    ds_np = xr.open_dataset(m_np.config.output_path, decode_times=False)
    ds_nb = xr.open_dataset(m_nb.config.output_path, decode_times=False)
    assert set(ds_np.data_vars) == set(ds_nb.data_vars)
    assert set(ds_np.coords) == set(ds_nb.coords)
    for var in ds_np.data_vars:
        assert_bitwise(ds_np[var].values, ds_nb[var].values, f"data_var {var}")
    for coord in ds_np.coords:
        assert_bitwise(ds_np[coord].values, ds_nb[coord].values, f"coord {coord}")
    exclude = {"backend", "creation_date", "output_path", "restart_output_dir"}
    keys_np = set(ds_np.attrs) - exclude
    keys_nb = set(ds_nb.attrs) - exclude
    assert keys_np == keys_nb
    for key in keys_np:
        va = np.asarray(ds_np.attrs[key])
        vb = np.asarray(ds_nb.attrs[key])
        assert np.array_equal(va, vb), f"attr {key}: {va!r} vs {vb!r}"
    assert ds_np.attrs["backend"] == "numpy"
    assert ds_nb.attrs["backend"] == "numba"
    ds_np.close()
    ds_nb.close()


@pytest.mark.regression
def test_cli_numba_reproduces_staggered_baseline(tmp_path):
    """End-to-end through the CLI: the numba backend at the production default
    formulation reproduces the staggered regression baseline bit-for-bit."""
    out = str(tmp_path / "numba_baseline.nc")
    subprocess.run(
        [
            "run-sw-model", "--ndays", "5", "--ny", "801", "--dt", "30",
            "--backend", "numba", "--output-path", out,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    baseline = xr.open_dataset("ss09/tests/baseline/output_staggered.nc")
    test_ds = xr.open_dataset(out)
    for var in ("u", "v", "T"):
        max_diff = np.abs(baseline[var].values - test_ds[var].values).max()
        assert max_diff == 0.0, f"{var} differs from baseline by {max_diff}"
    baseline.close()
    test_ds.close()


def test_cli_rejects_collocated_numba():
    proc = subprocess.run(
        ["run-sw-model", "--backend", "numba", "--grid", "collocated", "--ndays", "1"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "staggered" in (proc.stderr + proc.stdout)
