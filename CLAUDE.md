# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python implementation of the Sobel-Schneider single-layer shallow water model for simulating Hadley circulation with parameterized eddy momentum forcing, based on Sobel and Schneider (2009, 2013). The model integrates momentum and thermodynamic equations on an equatorial beta plane using a leapfrog time-stepping scheme with Asselin filtering.

## Commands

### Installation
```bash
pip install -e .
```

### Running the Model
```bash
# Run with default parameters
run-sw-model

# Run with custom parameters
run-sw-model --total_integration_days 250 --ny 51 --dt 3600 --theta_e_type sin2 --output_path ./model_output/output.nc
```

### Testing
```bash
# Run all tests
pytest ss09/tests/

# Run a specific test file
pytest ss09/tests/test_sw_model.py

# Run a specific test function
pytest ss09/tests/test_sw_model.py::test_function_name

# Run regression tests (compare against baseline)
pytest ss09/tests/test_regression.py
```

## Architecture

### Core Model Flow

The model execution follows this sequence:
1. **CLI (`cli.py`)** parses command-line arguments and creates configuration objects
2. **Configuration objects** are instantiated:
   - `SWConfig`: Physical parameters, numerical settings, domain setup
   - `ThetaEConfig`: Parameters for equilibrium potential temperature profile
3. **ThetaEProfile** subclass is selected based on `theta_e_type`:
   - `SS09Profile`: Original (y/y₁)² profile
   - `Sin2Profile`: sin²(πy/2y₁) profile (default)
   - `SB08Profile`: Schneider & Bordoni (2008) formula
4. **SWModel** is instantiated with config and theta_e profile
5. **run_sim()** executes the leapfrog integration loop
6. **save_results()** writes daily-averaged output to NetCDF

### Key Components

**SWModel (`sw_model.py`)**: Main simulation class
- Integrates u (zonal wind), v (meridional wind), theta (potential temperature)
- Uses leapfrog time-stepping with Asselin filtering for temporal stability
- Enforces boundary conditions (u=0, v=0 at domain edges)
- Stores daily averages during integration

**ModelState (`model_state.py`)**: NamedTuple containing instantaneous state variables (t, u, v, theta, y)

**SWConfig (`sw_config.py`)**: Dataclass with model configuration
- Automatically computes `dy` (grid spacing) and `y` (grid points) from `domain_size` and `ny` in `__post_init__`
- Key numerical parameters: `dt`, `asselin_filt_coef`, `include_vert_advec_u`

**ThetaEProfile (`theta_e.py`)**: Abstract base class with three implementations
- Defines equilibrium potential temperature profile
- Called during model integration for Newtonian cooling

**DailyResults (`daily_results.py`)**: Accumulates and exports daily-averaged output
- Converts to xarray Dataset with proper metadata
- Filters out unused timesteps (where time == 0)

### Physics and Numerics

The model solves prognostic equations for:
- **u (zonal wind)**: Coriolis, meridional advection, vertical advection (optional), Rayleigh drag, eddy momentum flux divergence
- **v (meridional wind)**: Coriolis, pressure gradient, vertical eddy diffusion
- **theta (potential temperature)**: Newtonian cooling to theta_e, vertical advection, eddy heat flux (optional)

Numerical scheme:
- Leapfrog time-stepping for temporal integration
- Asselin filter to suppress computational mode
- Upwind differencing for meridional advection of u
- Centered differences for spatial derivatives elsewhere

### Testing Strategy

Tests are organized by functionality:
- `test_sw_model.py`: Unit tests for physics terms and numerical methods
- `test_theta_e_profile.py`: Tests for theta_e profile implementations
- `test_theta_e_config.py`: Configuration validation
- `test_cli.py`: Command-line interface tests
- `test_regression.py`: End-to-end regression tests against baseline output in `ss09/tests/baseline/`

Regression tests run short simulations (5 days) and compare against stored baseline using `xarray.testing.assert_allclose`.

## Steady-State Detection

The model supports optional early termination when it reaches statistical steady state, which can significantly reduce computational cost for long simulations.

### Usage

Enable via CLI:
```bash
# Basic usage - stop when both KE and Tvar converge
run-sw-model --enable-steady-state

# Custom window and threshold
run-sw-model --enable-steady-state --steady-state-window 15 --steady-state-threshold 0.0005

# Require only one metric to converge instead of both
run-sw-model --enable-steady-state --steady-state-either

# Run for up to 500 days but stop early if steady state reached
run-sw-model --total_integration_days 500 --enable-steady-state
```

### Parameters

- `--enable-steady-state`: Turn on detection (default: disabled)
- `--steady-state-window N`: Use N-day window for convergence check (default: 10)
- `--steady-state-threshold X`: Relative change threshold (default: 0.001 = 0.1%)
- `--steady-state-either`: Require only one metric to converge instead of both
- `--smoothness-threshold X`: Neighbor correlation threshold for v field smoothness (default: 0.5)

### Convergence Metrics

By default, both metrics must satisfy convergence criteria:
- **Kinetic Energy**: Domain-averaged KE = mean(u² + v²)
- **Temperature Variance**: Spatial std(theta)

Convergence criterion: `(max(last_N_days) - min(last_N_days)) / mean(last_N_days) < threshold`

### Output

When enabled, the NetCDF output includes:
- `steady_state_kinetic_energy`: Time series of KE during simulation
- `steady_state_temp_variance`: Time series of temperature variance
- `v_neighbor_correlation`: Time series of v field smoothness (if steady-state enabled)
- `v_grid_variance`: Time series of grid-scale variance in v (if steady-state enabled)
- Global attributes indicating if/when convergence occurred and smoothness statistics

### v Field Smoothness Monitoring

When steady-state detection is enabled, the model automatically monitors the smoothness of the meridional wind (v) field to detect grid-scale oscillations from the leapfrog time-stepping scheme.

**Smoothness Metric**: Correlation between neighboring grid points. Values close to 1.0 indicate smooth fields, while values below 0.5 indicate grid-scale (2Δy) computational mode oscillations.

**Behavior**:
- If neighbor correlation falls below the threshold (default: 0.5), a warning is logged suggesting to increase `k_v` (vertical eddy viscosity)
- Smoothness history is saved to NetCDF output for post-analysis
- Warning is issued only once per simulation to avoid log spam
- Does not stop the simulation, only warns

**Configuration**:
```bash
# Adjust smoothness threshold (default: 0.5)
run-sw-model --enable-steady-state --smoothness-threshold 0.7
```

**Physical Interpretation**: Grid-scale oscillations typically occur when friction (especially `k_v`) is too weak to damp the 2Δy computational mode inherent to leapfrog schemes. Increasing `k_v` suppresses these oscillations. See the `analyze_v_smoothness.py` script for detailed smoothness analysis.

## Important Implementation Notes

- The factor of 2 in `dv_dt()` at line 163 is intentional per the physics formulation
- `THETA_TO_TEMP = 1/1.6` converts potential temperature to temperature assuming (p_s/p_t)^(R/c_p) = 1.6
- Vertical advection of zonal momentum can be toggled via `include_vert_advec_u` (enabled by default)
- Eddy heat diffusion is disabled by default (`coeff_eddy_heat_diff=0.0`); values <1e4 have minimal effect
- NaN detection breaks the integration loop early with a warning
- Output NetCDF files include all configuration parameters as global attributes
- Steady-state detection is disabled by default and has zero performance overhead when off
