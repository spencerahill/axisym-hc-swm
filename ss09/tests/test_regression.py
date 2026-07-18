import pytest
import os
import xarray as xr
import numpy as np
import subprocess
import tempfile


@pytest.fixture
def baseline_path():
    return "ss09/tests/baseline/output.nc"


@pytest.fixture
def baseline_staggered_path():
    return "ss09/tests/baseline/output_staggered.nc"


@pytest.fixture
def baseline_moist_path():
    return "ss09/tests/baseline/output_moist.nc"


@pytest.fixture
def test_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield os.path.join(temp_dir, "test_output.nc")


def run_model(
    output_path, grid: "str | None" = "collocated", legacy=False, moist=False
):
    args = [
        "run-sw-model",
        "--ndays",
        "5",
        "--ny",
        "801",
        "--dt",
        "30",
        "--output-path",
        output_path,
    ]
    # grid=None omits --grid entirely, exercising the default (staggered).
    if grid is not None:
        args.extend(["--grid", grid])
    # legacy=True pins the pre-2026-07-12 gate/stencil defaults (gate off,
    # centered) that the collocated Zhang25-repro baseline was generated with,
    # now that the production defaults are gate-on + mc.
    if legacy:
        args.extend(["--no-emfd-heaviside-gate", "--emfd-stencil", "centered"])
    if moist:
        args.append("--enable-moisture")
    try:
        subprocess.run(args, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Command failed with error:")
        print(e.stdout)
        print(e.stderr)
        raise


def compare_outputs(baseline_path: str, test_path: str) -> bool:
    """
    Compare baseline and test outputs for physics regression.

    Checks:
    1. State variables (u, v, T) - the model solution
    2. Forcing field (theta_e) - the boundary condition
    3. Physics parameters - ensures model configuration is identical

    Ignores diagnostic/monitoring variables added in newer versions.
    """
    baseline_ds = xr.open_dataset(baseline_path)
    test_ds = xr.open_dataset(test_path)

    # State variables to check (the model solution)
    state_vars = ['u', 'v', 'T']

    # Forcing field (boundary condition for temperature)
    forcing_vars = ['theta_e']

    # Physics parameters (SW model configuration)
    # Note: Attribute names may be uppercase or lowercase depending on version
    physics_params = {
        'beta': 2e-11,
        'delta': 4000.0,
        'delta_z': 60,
        'epsilon_u': 1e-08,
        'k_v': 778600,
        'v_d': 2.5,
    }

    # Check state variables
    for var in state_vars:
        if var not in test_ds:
            print(f"Missing state variable in test output: {var}")
            return False
        if not np.allclose(baseline_ds[var], test_ds[var], atol=1e-6):
            print(f"Difference found in state variable: {var}")
            max_diff = np.abs(baseline_ds[var].values - test_ds[var].values).max()
            print(f"  Maximum difference: {max_diff}")
            return False

    # Check forcing fields (these define the BCs)
    for var in forcing_vars:
        if var not in test_ds:
            print(f"Missing forcing variable in test output: {var}")
            return False
        if not np.allclose(baseline_ds[var], test_ds[var], atol=1e-6):
            print(f"Difference found in forcing variable: {var}")
            max_diff = np.abs(baseline_ds[var].values - test_ds[var].values).max()
            print(f"  Maximum difference: {max_diff}")
            return False

    # Check physics parameters (model configuration)
    for param, expected_value in physics_params.items():
        # Try both lowercase and uppercase
        test_value = None
        if param in test_ds.attrs:
            test_value = test_ds.attrs[param]
        elif param.upper() in test_ds.attrs:
            test_value = test_ds.attrs[param.upper()]

        if test_value is None:
            print(f"Missing physics parameter in test output: {param}")
            return False

        if not np.isclose(expected_value, test_value, atol=1e-10):
            print(f"Difference in physics parameter {param}: expected={expected_value}, test={test_value}")
            return False

    return True


@pytest.mark.regression
def test_regression(baseline_path, test_path):
    """The collocated (legacy) path reproduces the stored baseline. The
    baseline was generated before the staggered-v refactor, so this guards
    that the collocated path is unchanged by it (--grid collocated is the
    Zhang et al. 2025-lineage reproduction path, gate off + centered)."""
    run_model(test_path, grid="collocated", legacy=True)
    assert compare_outputs(
        baseline_path, test_path
    ), "Regression test failed: Outputs differ."


@pytest.mark.regression
def test_regression_staggered(baseline_staggered_path, test_path):
    """The default path reproduces the staggered baseline. Run with no flags,
    so it guards the full production default (staggered grid + gate-on + mc
    stencil): if any of the three reverted, the solution would differ (grid via
    v's coordinate length, gate/stencil via the flank/notch structure)."""
    run_model(test_path, grid=None)
    assert compare_outputs(
        baseline_staggered_path, test_path
    ), "Staggered regression test failed: Outputs differ."


@pytest.mark.regression
def test_regression_moist(baseline_moist_path, test_path):
    """The production default path with --enable-moisture reproduces the
    moist baseline: guards the W/P numerics (transport, precipitation,
    lagged stepping) and, jointly with the dry-invariance test, that the
    dry solution under moisture stays on the staggered baseline."""
    run_model(test_path, grid=None, moist=True)
    assert compare_outputs(
        baseline_moist_path, test_path
    ), "Moist regression test failed: Outputs differ."
    # and the moisture fields themselves
    baseline_ds = xr.open_dataset(baseline_moist_path)
    test_ds = xr.open_dataset(test_path)
    for var in ("W", "P", "W_mean", "W_min"):
        assert var in test_ds, f"Missing moisture variable: {var}"
        assert np.allclose(
            baseline_ds[var], test_ds[var], atol=1e-6
        ), f"Difference found in moisture variable: {var}"


@pytest.mark.regression
def test_collocated_path_bit_identical(baseline_path, test_path):
    """The collocated path reproduces the baseline state (u, v, T) bit-for-bit,
    not merely within tolerance: the staggered-v refactor must not perturb a
    single floating-point bit of the legacy solution."""
    run_model(test_path, grid="collocated", legacy=True)
    baseline_ds = xr.open_dataset(baseline_path)
    test_ds = xr.open_dataset(test_path)
    for var in ("u", "v", "T"):
        max_diff = np.abs(
            baseline_ds[var].values - test_ds[var].values
        ).max()
        assert max_diff == 0.0, (
            f"{var} differs from baseline by {max_diff} (expected bit-exact)"
        )
