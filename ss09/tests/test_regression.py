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
def test_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield os.path.join(temp_dir, "test_output.nc")


def run_model(output_path):
    try:
        subprocess.run(
            [
                "run-sw-model",
                "--total_integration_days",
                "5",
                "--ny",
                "801",
                "--dt",
                "30",
                "--output_path",
                output_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("Command failed with error:")
        print(e.stdout)
        print(e.stderr)
        raise


def compare_outputs(baseline_path: str, test_path: str) -> bool:
    """Compare the baseline and test outputs."""
    baseline_ds = xr.open_dataset(baseline_path)
    test_ds = xr.open_dataset(test_path)

    # Compare all variables
    for var in baseline_ds.data_vars:
        if not np.allclose(baseline_ds[var], test_ds[var], atol=1e-6):
            print(f"Difference found in variable: {var}")
            return False

    return True


@pytest.mark.regression
def test_regression(baseline_path, test_path):
    """Test the model output against the baseline."""
    run_model(test_path)
    assert compare_outputs(
        baseline_path, test_path
    ), "Regression test failed: Outputs differ."
