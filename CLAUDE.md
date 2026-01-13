# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python implementation of the Sobel-Schneider single-layer shallow water model for simulating Hadley circulation with parameterized eddy momentum forcing, based on Sobel and Schneider (2009, 2013). The model integrates momentum and thermodynamic equations on an equatorial beta plane using a leapfrog time-stepping scheme with Asselin filtering.

## Git Workflow

**IMPORTANT**: Always create a git commit after every substantive change to the codebase. This ensures:
- Changes are tracked incrementally with clear history
- Easy rollback if needed
- Better collaboration and code review

When committing:
1. Review `git status` and `git diff` to understand what changed
2. Check recent commit messages (`git log --oneline`) for style consistency
3. Write descriptive commit messages focusing on the "why" not just the "what"
4. Include the Co-Authored-By tag: `Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>`

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

### Restart/Checkpoint Functionality

The model supports saving and loading simulation state for extending runs or recovery from interruptions.

**Saving restart files:**
```bash
# Save restart file only at end of simulation
run-sw-model --total_integration_days 100

# Save periodic restart files every 20 days (plus final file)
run-sw-model --total_integration_days 100 --save-restart-every 20

# Specify restart file directory
run-sw-model --total_integration_days 100 --save-restart-every 20 --restart-output-dir ./checkpoints
```

**Restart files** are named `restart_day{NNNN}.nc` (e.g., `restart_day0050.nc`) and contain:
- Instantaneous state variables (u, v, theta) at timesteps n and n-1 (NOT daily averages)
- Steady-state detector history (if enabled)
- All configuration parameters for validation

**Continuing from a restart:**
```bash
# Continue from day 50 to day 100
run-sw-model --restart-from ./model_output/restart_day0050.nc --total_integration_days 100

# Continue with different output path
run-sw-model --restart-from ./model_output/restart_day0050.nc \
             --total_integration_days 150 \
             --output_path ./extended_run.nc
```

**Important notes:**
- Restart files contain instantaneous snapshots at day boundaries, not daily-averaged output
- Configuration parameters (ny, dt, domain_size, etc.) must match between restart file and new run
- Steady-state detector history is preserved across restarts for continuous convergence monitoring
- Output files contain only the days simulated in that run (filtered automatically)

## Output File Organization

By default, the model automatically generates descriptive output paths based on run configuration, making it easy to organize and identify different model runs.

### Directory Structure

```
./model_output/
  {theta_e_type}/              # Profile type (SS09, sin2, or SB08)
    run_{timestamp}_{params}_output.nc
    run_{timestamp}_{params}_restart_day{NNNN}.nc
```

Files are organized with:
- **One directory level** grouping by fundamental physics choice (theta_e_type)
- **Descriptive filenames** encoding run-specific parameters and timestamp

### Filename Components

**Format:** `run_{timestamp}_{seasonal}_{y0}_{resolution}_{duration}_{file_type}.nc`

**Components:**
- **timestamp**: YYYYMMDD_HHMMSS (ensures uniqueness, chronological sorting)
- **seasonal**: `seas` (seasonal cycle enabled) or `noseas` (no seasonal cycle)
- **y0**: Mean ITCZ position in km with sign indicator (e.g., `y0p0000`, `y0p0700`, `y0n0500`)
- **resolution**: Grid points `ny{NNN}` (e.g., `ny051`, `ny101`)
- **duration**: Total integration days `{N}days` (e.g., `3600days`, `250days`)
- **file_type**: `output` (main results) or `restart_day{NNNN}` (checkpoint files)

### Examples

```bash
# Seasonal SB08 run: 700 km ITCZ migration, 51 grid points, 3600 days
./model_output/SB08/run_20260111_134530_seas_y0p0000_ny051_3600days_output.nc
./model_output/SB08/run_20260111_134530_seas_y0p0000_ny051_3600days_restart_day0100.nc

# Non-seasonal sin2 run: centered ITCZ, 101 grid points, 250 days
./model_output/sin2/run_20260112_093000_noseas_y0p0000_ny101_250days_output.nc
./model_output/sin2/run_20260112_093000_noseas_y0p0000_ny101_250days_restart_day0250.nc

# Shifted ITCZ: 1500 km north, SS09 profile
./model_output/SS09/run_20260113_101500_noseas_y0p1500_ny051_1000days_output.nc
```

### Custom Paths

You can override the automatic naming with explicit paths:

```bash
# Use custom output path
run-sw-model --output_path ./my_custom_path/run_001.nc

# Use custom restart directory
run-sw-model --restart-output-dir ./my_checkpoints
```

When custom paths are provided, descriptive naming is disabled for those specific paths.

### Benefits

- **Self-documenting**: Key parameters visible in filesystem
- **No overwrites**: Timestamp ensures unique filenames
- **Easy browsing**: Profile types organized in directories
- **Chronological**: Files sort by timestamp within each directory
- **Restart matching**: Restart files clearly associated with output files

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

### Nonlinear Advection Terms

Control which nonlinear advection terms are included in the u momentum equation:

```bash
# Disable meridional advection of u (v*du/dy) for linear dynamics
run-sw-model --no-merid-advec-u

# Disable vertical advection of u (u*dv/dy)
run-sw-model --no-vert-advec-u

# Fully linear u equation (no advection terms)
run-sw-model --no-merid-advec-u --no-vert-advec-u
```

**Physical interpretation**:
- **Nonlinear case** (default, both enabled): Full advection, realistic dynamics with nonlinear interactions
- **Linear case** (advection disabled): Linearized around rest state, enables comparison with analytical solutions and isolates linear wave dynamics

**Common use cases**:
- Linear vs nonlinear comparison for understanding mechanisms
- Verifying linear theory predictions
- Isolating effects of advection on circulation strength

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
