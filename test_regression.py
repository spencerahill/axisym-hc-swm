import os
import xarray as xr
import numpy as np
import subprocess


def run_model(output_path: str, total_integration_days: int = 5):
    """Run the model and save the output to the specified path."""
    subprocess.run(
        [
            "python",
            "SS_Model.py",
            "--total_integration_days",
            str(total_integration_days),
        ],
        check=True,
    )
    os.rename("./model_output/output.nc", output_path)


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


def test_regression():
    """Test the model output against the baseline."""
    baseline_path = "baseline/output.nc"
    test_path = "test_output.nc"

    # Run the model to generate test output
    run_model(test_path)

    # Compare the outputs
    assert compare_outputs(
        baseline_path, test_path
    ), "Regression test failed: Outputs differ."

    # Clean up test output
    os.remove(test_path)


if __name__ == "__main__":
    test_regression()
