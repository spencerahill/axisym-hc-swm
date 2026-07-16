# Known issues

Latent defects documented for future cleanup. Each is found and verified
empirically; an entry stays here until a session fixes it with a decision
plus tests, since every fix changes user-visible behavior.

- **Custom `--output-path` with a nonexistent parent directory crashes the
  end-of-run restart write**
  ([issue #4](https://github.com/spencerahill/axisym-hc-swm/issues/4)):
  nothing creates the parent directory, so after the full integration the
  final restart's netCDF create fails with the library's misleading
  `PermissionError` (its standard complaint for a missing parent), and the
  daily output, written after the restart, is lost with it. Found
  2026-07-16 launching the fixed-Ro suite Tier-0 runs; workaround is
  pre-creating the directory. Candidate fix: `makedirs(exist_ok=True)`
  for the output and restart parents at startup, so a doomed path fails
  fast instead of after the run.

## Fixed

- **`--seas-conv` could never trigger without `--stop-at-steady-state`**
  ([issue #2](https://github.com/spencerahill/axisym-hc-swm/issues/2)):
  the steady-state detector records the daily history the seasonal
  year-to-year check reads, and it records only when enabled, so
  `--seas-conv` alone silently completed the full run length. Fixed
  2026-07-14: `SWConfig` now raises when `seasonal_convergence_enabled`
  is set without `enable_steady_state`, and the CLI refuses `--seas-conv`
  without `--stop-at-steady-state`.
- **Reference-loop step accounting inconsistent for dt not dividing 86400**
  ([issue #3](https://github.com/spencerahill/axisym-hc-swm/issues/3)):
  the numpy loop's total-step count, day-boundary bookkeeping, and restart
  resume disagreed for such dt; only the numba backend rejected them. Fixed
  2026-07-14: the divisibility validation in `SWConfig` now applies to all
  backends (the integer-dt requirement stays numba-specific, since it
  guards that kernel's integer-seconds bookkeeping).
