#!/usr/bin/env python
"""
Analyze smoothness of meridional wind (v) field with different friction parameters.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import signal

def compute_grid_scale_variance(v, dy):
    """
    Compute variance of second differences (proxy for grid-scale noise).

    Second difference approximates d²v/dy² at grid scale.
    Large values indicate rapid oscillations.
    """
    # Second difference: v[i+1] - 2*v[i] + v[i-1]
    second_diff = np.diff(v, n=2)
    # Normalize by grid spacing squared
    normalized = second_diff / (dy**2)
    return np.var(normalized)

def compute_first_derivative_variance(v, dy):
    """
    Compute variance of first derivative dv/dy.
    Large variance indicates rapid spatial changes.
    """
    dv_dy = np.gradient(v, dy)
    return np.var(dv_dy)

def compute_power_spectrum(v, dy):
    """
    Compute power spectral density to see energy at different wavelengths.

    Returns:
        wavenumbers: spatial frequencies (cycles per meter)
        power: power spectral density
    """
    # Detrend (remove mean)
    v_detrend = v - np.mean(v)

    # FFT
    fft = np.fft.rfft(v_detrend)
    power = np.abs(fft)**2 / len(v)

    # Wavenumbers (spatial frequencies)
    freqs = np.fft.rfftfreq(len(v), d=dy)

    # Convert to wavelengths (m)
    wavelengths = 1.0 / (freqs + 1e-10)  # avoid division by zero

    return freqs, wavelengths, power

def compute_smoothness_ratio(v, dy, threshold_wavelength=1e6):
    """
    Ratio of power in short wavelengths (< threshold) to long wavelengths (> threshold).

    High ratio indicates more grid-scale noise relative to large-scale features.
    """
    freqs, wavelengths, power = compute_power_spectrum(v, dy)

    # Identify short vs long wavelengths
    short_wave_mask = wavelengths < threshold_wavelength
    long_wave_mask = wavelengths >= threshold_wavelength

    # Sum power in each range
    short_power = np.sum(power[short_wave_mask])
    long_power = np.sum(power[long_wave_mask])

    ratio = short_power / (long_power + 1e-10)
    return ratio, short_power, long_power

def analyze_case(filename, label):
    """Analyze a single case."""
    ds = xr.open_dataset(filename)

    # Time-average over last 50 days
    v = ds.v.isel(time=slice(-50, None)).mean(dim='time').values
    # v lives on the staggered cell faces (y_edge); fall back to y for
    # legacy collocated output files.
    y = ds.y_edge.values if 'y_edge' in ds.coords else ds.y.values
    dy = y[1] - y[0]

    # Compute metrics
    results = {
        'label': label,
        'epsilon_u': ds.attrs['epsilon_u'],
        'k_v': ds.attrs['k_v'],
        'grid_var': compute_grid_scale_variance(v, dy),
        'deriv_var': compute_first_derivative_variance(v, dy),
        'v_max': np.max(np.abs(v)),
        'v_std': np.std(v),
    }

    # Smoothness ratio
    ratio, short_pow, long_pow = compute_smoothness_ratio(v, dy, threshold_wavelength=1e6)
    results['smoothness_ratio'] = ratio
    results['short_wavelength_power'] = short_pow
    results['long_wavelength_power'] = long_pow

    # Power spectrum
    freqs, wavelengths, power = compute_power_spectrum(v, dy)
    results['freqs'] = freqs
    results['wavelengths'] = wavelengths
    results['power'] = power
    results['v'] = v
    results['y'] = y

    return results

def main():
    # Run different friction cases
    cases = []

    print("Running friction sensitivity tests...\n")

    # Case 1: Default friction
    print("Case 1: Default friction (epsilon_u=1e-8, k_v=778600)")
    import subprocess
    subprocess.run([
        'run-sw-model',
        '--ndays', '100',
        '--vd', '0.0',
        '--output-path', './model_output/smooth_test_default.nc'
    ], capture_output=True)
    cases.append(('Default\n(εᵤ=1e-8, kᵥ=7.8e5)', './model_output/smooth_test_default.nc'))

    # Case 2: No Rayleigh drag, keep k_v
    print("Case 2: No Rayleigh drag (epsilon_u=0, k_v=778600)")
    subprocess.run([
        'run-sw-model',
        '--ndays', '100',
        '--vd', '0.0',
        '--eps-u', '0.0',
        '--output-path', './model_output/smooth_test_no_eps.nc'
    ], capture_output=True)
    cases.append(('No Rayleigh\n(εᵤ=0, kᵥ=7.8e5)', './model_output/smooth_test_no_eps.nc'))

    # Case 3: Keep Rayleigh drag, no k_v
    print("Case 3: No eddy viscosity (epsilon_u=1e-8, k_v=0)")
    subprocess.run([
        'run-sw-model',
        '--ndays', '100',
        '--vd', '0.0',
        '--kv', '0.0',
        '--output-path', './model_output/smooth_test_no_kv.nc'
    ], capture_output=True)
    cases.append(('No Eddy Visc\n(εᵤ=1e-8, kᵥ=0)', './model_output/smooth_test_no_kv.nc'))

    # Case 4: Both off
    print("Case 4: Both off (epsilon_u=0, k_v=0)")
    subprocess.run([
        'run-sw-model',
        '--ndays', '100',
        '--vd', '0.0',
        '--eps-u', '0.0',
        '--kv', '0.0',
        '--output-path', './model_output/smooth_test_both_off.nc'
    ], capture_output=True)
    cases.append(('Both Off\n(εᵤ=0, kᵥ=0)', './model_output/smooth_test_both_off.nc'))

    # Case 5: Smaller k_v
    print("Case 5: Reduced k_v (epsilon_u=1e-8, k_v=10000)")
    subprocess.run([
        'run-sw-model',
        '--ndays', '100',
        '--vd', '0.0',
        '--kv', '10000',
        '--output-path', './model_output/smooth_test_small_kv.nc'
    ], capture_output=True)
    cases.append(('Small kᵥ\n(εᵤ=1e-8, kᵥ=1e4)', './model_output/smooth_test_small_kv.nc'))

    print("\nAnalyzing results...\n")

    # Analyze all cases
    results_list = []
    for label, filename in cases:
        results = analyze_case(filename, label)
        results_list.append(results)

    # Print summary table
    print("=" * 90)
    print("SMOOTHNESS METRICS SUMMARY")
    print("=" * 90)
    print(f"{'Case':<20} {'εᵤ':<10} {'kᵥ':<12} {'Grid Var':<12} {'Deriv Var':<12} {'Smooth Ratio':<12}")
    print("-" * 90)

    for r in results_list:
        print(f"{r['label']:<20} {r['epsilon_u']:<10.1e} {r['k_v']:<12.1e} "
              f"{r['grid_var']:<12.2e} {r['deriv_var']:<12.2e} {r['smoothness_ratio']:<12.2f}")

    print("=" * 90)
    print("\nMetric definitions:")
    print("  Grid Var:     Variance of d²v/dy² (higher = more grid-scale oscillations)")
    print("  Deriv Var:    Variance of dv/dy (higher = more spatial variability)")
    print("  Smooth Ratio: Power(<1000km) / Power(>1000km) (higher = noisier)")
    print()

    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))

    # 1. Meridional profiles
    ax1 = plt.subplot(3, 2, 1)
    for r in results_list:
        y_deg = r['y'] / 111000  # Convert to degrees
        ax1.plot(y_deg, r['v'], label=r['label'], linewidth=2, alpha=0.8)
    ax1.set_xlabel('Latitude (degrees)')
    ax1.set_ylabel('v (m/s)')
    ax1.set_title('Meridional Wind Profiles (time-averaged)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.axhline(0, color='k', linewidth=0.5)

    # 2. Zoomed region showing grid-scale structure
    ax2 = plt.subplot(3, 2, 2)
    for r in results_list:
        y_deg = r['y'] / 111000
        # Zoom to equatorial region
        mask = np.abs(y_deg) < 20
        ax2.plot(y_deg[mask], r['v'][mask], 'o-', label=r['label'], markersize=3, alpha=0.7)
    ax2.set_xlabel('Latitude (degrees)')
    ax2.set_ylabel('v (m/s)')
    ax2.set_title('Zoomed: Equatorial Region (±20°)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # 3. Power spectra
    ax3 = plt.subplot(3, 2, 3)
    for r in results_list:
        # Only plot non-zero wavenumbers
        mask = r['wavelengths'] < 1e8  # Avoid very long wavelengths
        ax3.loglog(r['wavelengths'][mask]/1000, r['power'][mask],
                   label=r['label'], linewidth=2, alpha=0.8)
    ax3.axvline(1000, color='k', linestyle='--', alpha=0.5, label='1000 km threshold')
    ax3.set_xlabel('Wavelength (km)')
    ax3.set_ylabel('Power Spectral Density')
    ax3.set_title('Power Spectrum of v')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    ax3.set_xlim(100, 2e4)

    # 4. First derivative
    ax4 = plt.subplot(3, 2, 4)
    for r in results_list:
        y_deg = r['y'] / 111000
        dv_dy = np.gradient(r['v'], r['y'][1] - r['y'][0])
        ax4.plot(y_deg, dv_dy * 1e6, label=r['label'], linewidth=2, alpha=0.8)
    ax4.set_xlabel('Latitude (degrees)')
    ax4.set_ylabel('dv/dy (10⁻⁶ s⁻¹)')
    ax4.set_title('First Derivative: dv/dy')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    ax4.axhline(0, color='k', linewidth=0.5)

    # 5. Second derivative (what k_v acts on)
    ax5 = plt.subplot(3, 2, 5)
    for r in results_list:
        y_deg = r['y'] / 111000
        dy = r['y'][1] - r['y'][0]
        dv_dy = np.gradient(r['v'], dy)
        d2v_dy2 = np.gradient(dv_dy, dy)
        ax5.plot(y_deg, d2v_dy2 * 1e12, label=r['label'], linewidth=2, alpha=0.8)
    ax5.set_xlabel('Latitude (degrees)')
    ax5.set_ylabel('d²v/dy² (10⁻¹² m⁻¹s⁻¹)')
    ax5.set_title('Second Derivative: d²v/dy² (diffusion acts on this)')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8)
    ax5.axhline(0, color='k', linewidth=0.5)

    # 6. Bar chart of metrics
    ax6 = plt.subplot(3, 2, 6)
    labels = [r['label'].replace('\n', ' ') for r in results_list]
    grid_vars = [r['grid_var'] for r in results_list]
    x = np.arange(len(labels))
    width = 0.35

    bars = ax6.bar(x, grid_vars, width)
    ax6.set_ylabel('Grid-Scale Variance (d²v/dy²)²')
    ax6.set_title('Grid-Scale Noise Metric')
    ax6.set_xticks(x)
    ax6.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3, axis='y')

    # Highlight which has lowest variance
    min_idx = np.argmin(grid_vars)
    bars[min_idx].set_color('green')
    bars[min_idx].set_alpha(0.7)

    plt.tight_layout()
    plt.savefig('v_field_smoothness_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: v_field_smoothness_analysis.png")

    # Additional diagnostic: correlation between neighboring points
    print("\n" + "=" * 90)
    print("NEIGHBOR CORRELATION (anticorrelation indicates 2Δy oscillations)")
    print("=" * 90)
    for r in results_list:
        # Correlation between v[i] and v[i+1]
        corr = np.corrcoef(r['v'][:-1], r['v'][1:])[0, 1]
        print(f"{r['label']:<20} Correlation: {corr:6.3f}  "
              f"{'(SMOOTH)' if corr > 0.9 else '(NOISY)' if corr < 0.5 else ''}")

    print("\n" + "=" * 90)
    print("INTERPRETATION:")
    print("  - Correlation near 1.0 = smooth field")
    print("  - Correlation < 0.5 = significant grid-scale oscillations")
    print("=" * 90)

if __name__ == '__main__':
    main()
