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
