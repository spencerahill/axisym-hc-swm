"""Tests for the moist V1 passive column-water-vapor extension.

V1 (guides/moist_axisymmetric_model_spec.pdf Eq. Q1) adds a prognostic
column water vapor W(y, t) riding passively on the dry circulation:

    dW/dt + d/dy[-(2a-1) v W - D dW/dy] = E_0 - P,   P = (W - W_c)^+ / tau_c

W lives on the ny centers; the transport is finite-volume in flux form with
fluxes on the ny-1 interior faces where the staggered v already lives. Two
hard invariants define success: (1) with enable_moisture on, the dry fields
u, v, theta are bit-for-bit identical to the same run with it off (W is
one-way coupled); (2) with it off, the dry code path is untouched (guarded
by the existing regression baselines).
"""

from typing import Any

import numpy as np
import pytest

from ss09.cli import parse_arguments, setup_sw_config, setup_theta_e_config
from ss09.sw_config import SWConfig
from ss09.sw_model import (
    cwv_integral,
    mc_face_values,
    moisture_transport_tendency,
)


def _even_centers(ny, seed):
    """A center array w (length ny) with w[i] == w[ny-1-i] exactly."""
    rng = np.random.default_rng(seed)
    half = rng.uniform(30.0, 60.0, ny // 2)
    if ny % 2 == 0:
        return np.concatenate([half, half[::-1]])
    return np.concatenate([half, [45.0], half[::-1]])


def _odd_faces(nf, seed):
    """A face array f (length nf) with f[j] == -f[nf-1-j] exactly."""
    rng = np.random.default_rng(seed)
    half = rng.normal(0.0, 5.0, nf // 2)
    if nf % 2 == 0:
        return np.concatenate([half, -half[::-1]])
    return np.concatenate([half, [0.0], -half[::-1]])


# --- Config validation -----------------------------------------------------

def _moist_config(**overrides):
    """A valid moist configuration on the production (staggered) grid."""
    kwargs: dict[str, Any] = dict(
        total_integration_days=2,
        ny=51,
        dt=1800,
        enable_moisture=True,
    )
    kwargs.update(overrides)
    return SWConfig(**kwargs)


def test_config_enable_moisture_accepted_on_defaults():
    config = _moist_config()
    assert config.enable_moisture is True
    # placeholder defaults from the V1 plan (ERA5-calibrated values pending)
    assert config.cwv_frac == 0.85
    assert config.d_w == 1.0e6
    assert config.w_crit == 50.0
    assert config.tau_c == 14400.0
    assert config.evap == 4.6e-5
    assert config.w_init is None  # None means "use w_crit"


@pytest.mark.parametrize(
    "param, value",
    [
        ("cwv_frac", 0.9),
        ("d_w", 2.0e6),
        ("w_crit", 40.0),
        ("tau_c", 7200.0),
        ("evap", 5.0e-5),
        ("w_init", 45.0),
    ],
)
def test_config_moist_param_without_enable_moisture_raises(param, value):
    """Repo style: passing a moist parameter without enable_moisture is a
    hard error, never a silent no-op."""
    with pytest.raises(ValueError, match="enable_moisture"):
        SWConfig(total_integration_days=2, ny=51, dt=1800, **{param: value})


def test_config_enable_moisture_rejects_numba_backend():
    """The numba kernel does not mirror the moisture step yet (deferred until
    the V1 physics is frozen)."""
    with pytest.raises(ValueError, match="numba"):
        _moist_config(backend="numba")


def test_config_enable_moisture_rejects_collocated_grid():
    """The collocated layout is the frozen Zhang et al. (2025) reproduction
    path; moisture assumes face-v fluxes."""
    with pytest.raises(ValueError, match="collocated"):
        _moist_config(grid="collocated")


# --- CLI -------------------------------------------------------------------

def test_cli_enable_moisture_flag(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["run-sw-model", "--enable-moisture", "--dt", "1800"],
    )
    args = parse_arguments()
    config = setup_sw_config(args, setup_theta_e_config(args))
    assert config.enable_moisture is True
    assert config.w_crit == 50.0


def test_cli_moist_values_reach_config(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run-sw-model", "--enable-moisture", "--dt", "1800",
            "--cwv-frac", "0.9", "--dw", "2e6", "--w-crit", "45",
            "--tau-c", "7200", "--evap", "5e-5", "--w-init", "40",
        ],
    )
    args = parse_arguments()
    config = setup_sw_config(args, setup_theta_e_config(args))
    assert config.cwv_frac == 0.9
    assert config.d_w == 2.0e6
    assert config.w_crit == 45.0
    assert config.tau_c == 7200.0
    assert config.evap == 5.0e-5
    assert config.w_init == 40.0


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--cwv-frac", "0.9"),
        ("--dw", "2e6"),
        ("--w-crit", "45"),
        ("--tau-c", "7200"),
        ("--evap", "5e-5"),
        ("--w-init", "40"),
    ],
)
def test_cli_moist_flag_without_enable_moisture_exits(monkeypatch, flag, value):
    monkeypatch.setattr(
        "sys.argv", ["run-sw-model", "--dt", "1800", flag, value]
    )
    with pytest.raises(SystemExit) as exc_info:
        parse_arguments()
    # our explicit error, not argparse's unknown-argument exit (code 2)
    assert "--enable-moisture" in str(exc_info.value.code)


def test_cli_dry_default_has_moisture_off(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run-sw-model", "--dt", "1800"])
    args = parse_arguments()
    config = setup_sw_config(args, setup_theta_e_config(args))
    assert config.enable_moisture is False


# --- mc_face_values (MUSCL-MC face reconstruction) -------------------------

def test_mc_face_values_constant_field_exact():
    """A constant field reconstructs to the constant on every face,
    whichever way each face upwinds."""
    w = np.full(9, 47.25)
    c_f = np.array([1.0, -1.0, 0.5, 0.0, -0.5, 2.0, -2.0, 1.0])
    np.testing.assert_array_equal(mc_face_values(w, 1.0, c_f), np.full(8, 47.25))


def test_mc_face_values_linear_field_exact_interior():
    """On a linear field the MC slope is the exact slope, so interior faces
    reconstruct the exact midpoint value from either side. The outermost
    cells have a zero-padded one-sided slope (limiter clips to 0), so the
    boundary face served by an endpoint cell reverts to the cell value."""
    w = np.array([0.0, 2.0, 4.0, 6.0, 8.0])  # dy = 1 -> exact arithmetic
    # upwind from the left everywhere: face j reconstructs from cell j
    left = mc_face_values(w, 1.0, np.full(4, 1.0))
    np.testing.assert_array_equal(left[1:], [3.0, 5.0, 7.0])
    assert left[0] == w[0]  # endpoint cell: limited slope = 0
    # upwind from the right everywhere: face j reconstructs from cell j+1
    right = mc_face_values(w, 1.0, np.full(4, -1.0))
    np.testing.assert_array_equal(right[:-1], [1.0, 3.0, 5.0])
    assert right[-1] == w[-1]


def test_mc_face_values_extremum_reverts_to_cell_value():
    """At a local extremum the limited slope is zero, so the face value is
    exactly the upwind cell value (no overshoot at the sharp ITCZ front)."""
    w = np.array([1.0, 2.0, 7.0, 2.0, 1.0])  # max in cell 2
    up_left = mc_face_values(w, 1.0, np.ones(4))
    assert up_left[2] == 7.0  # face 2 upwinds from the extremum cell 2
    up_right = mc_face_values(w, 1.0, -np.ones(4))
    assert up_right[1] == 7.0  # face 1 upwinds from cell 2


def test_mc_face_values_bounded_by_adjacent_cells():
    """Every face value lies between the two adjacent cell values (the MC
    limiter's non-oscillatory guarantee), for random rough fields and mixed
    upwind directions."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        w = rng.uniform(0.0, 60.0, 41)
        c_f = rng.normal(0.0, 2.0, 40)
        w_face = mc_face_values(w, 39377.5, c_f)
        lo = np.minimum(w[:-1], w[1:])
        hi = np.maximum(w[:-1], w[1:])
        assert np.all(w_face >= lo - 1e-12)
        assert np.all(w_face <= hi + 1e-12)


# --- moisture_transport_tendency -------------------------------------------

def test_moisture_transport_tendency_conserves_mass():
    """Interior fluxes telescope and the wall fluxes are zero, so the
    cell-weighted sum of the transport tendency vanishes to roundoff:
    transport alone moves no total water."""
    rng = np.random.default_rng(11)
    dy = 39377.5
    for ny in (21, 51, 52):
        w_adv = rng.uniform(20.0, 60.0, ny)
        w_diff = rng.uniform(20.0, 60.0, ny)
        v_f = rng.normal(0.0, 3.0, ny - 1)
        tend = moisture_transport_tendency(
            w_adv, w_diff, v_f, cwv_frac=0.85, d_w=1.0e6, dy=dy
        )
        total = cwv_integral(tend, dy)
        # scale: the largest single flux crossing a face
        scale = np.max(np.abs(v_f)) * np.max(np.abs(w_adv))
        assert abs(total) < 1e-12 * scale


def test_moisture_transport_tendency_uniform_w_zero_v_is_zero():
    """No gradients and no flow: transport contributes exactly nothing."""
    ny = 31
    w = np.full(ny, 50.0)
    v_f = np.zeros(ny - 1)
    tend = moisture_transport_tendency(
        w, w, v_f, cwv_frac=0.85, d_w=1.0e6, dy=1000.0
    )
    np.testing.assert_array_equal(np.abs(tend), np.zeros(ny))


def test_moisture_transport_tendency_parity_bitexact():
    """Even W and odd face-v give an exactly even tendency, bit-for-bit:
    the discrete parity that lets a symmetric moist run hold
    max|W(y) - W(-y)| == 0.0 through the full integration."""
    for ny in (20, 21, 51, 52):
        w_adv = _even_centers(ny, seed=ny)
        w_diff = _even_centers(ny, seed=1000 + ny)
        v_f = _odd_faces(ny - 1, seed=2000 + ny)
        tend = moisture_transport_tendency(
            w_adv, w_diff, v_f, cwv_frac=0.85, d_w=1.0e6, dy=39377.5
        )
        assert np.max(np.abs(tend - tend[::-1])) == 0.0


def test_moisture_transport_tendency_pure_diffusion_of_parabola():
    """With v_f = 0 and a parabolic W, the interior tendency is the exact
    constant D * d2W/dy2 (the compact face-difference divergence is exact on
    quadratics away from the zero-flux walls)."""
    ny, dy, d_w = 41, 100.0, 2.0e5
    y = np.arange(ny) * dy
    w = 1e-6 * (y - y[ny // 2]) ** 2
    tend = moisture_transport_tendency(
        w, w, np.zeros(ny - 1), cwv_frac=0.85, d_w=d_w, dy=dy
    )
    np.testing.assert_allclose(
        tend[1:-1], np.full(ny - 2, d_w * 2.0e-6), rtol=1e-10
    )


# --- cwv_integral ----------------------------------------------------------

def test_cwv_integral_trapezoid_weights():
    """The discrete integral uses the FV cell widths: dy/2 at the wall
    centers (half cells), dy in the interior."""
    w = np.array([2.0, 3.0, 5.0, 7.0])
    assert cwv_integral(w, 10.0) == 10.0 * (1.0 + 3.0 + 5.0 + 3.5)
