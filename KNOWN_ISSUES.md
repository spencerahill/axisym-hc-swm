# Known issues

Latent defects documented for future cleanup. Each was found and verified
empirically but deliberately left unfixed to keep the discovering session's
scope contained; fixing any of them changes user-visible behavior, so each
needs a small decision plus tests.

## 1. `--seas-conv` can never trigger without `--stop-at-steady-state`

**Behavior.** `--seas-conv` (`seasonal_convergence_enabled`) alone never
stops a run: the simulation silently completes its full length with no
warning.

**Mechanism.** `SteadyStateDetector.record_day` returns immediately when
`not self.enabled` (`ss09/steady_state.py:181-182`), and `enabled` comes
only from `config.enable_steady_state`. `check_seasonal_convergence`
requires `len(kinetic_energy_history) >= min_days_needed`
(`ss09/steady_state.py:277-279`); with the detector disabled the history
stays empty forever, so the check returns False on every day.

**Found** 2026-07-13 while building the numba parity suite: a
seasonal-convergence-break test configured with only
`seasonal_convergence_enabled=True` never broke early on either backend
until `enable_steady_state=True` was added
(`ss09/tests/test_numba_backend.py::test_seasonal_convergence_break_parity`
carries the explanatory comment).

**Workaround.** Pass both flags: `run-sw-model --stop-at-steady-state
--seas-conv ...` (compatible; `--stop-at-steady-state` conflicts only with
`--ndays`).

**Candidate fixes.** (a) Record detector history whenever
`seasonal_convergence_enabled` is set, decoupling recording from the
steady-state stop flag; (b) validate in `SWConfig.__post_init__` that
`seasonal_convergence_enabled` requires `enable_steady_state`, with a clear
error (smallest change); (c) have the CLI's `--seas-conv` imply detector
recording. Each needs tests for the newly reachable or forbidden
combinations.

## 2. Reference-loop step accounting is inconsistent for dt not dividing 86400

**Behavior.** For a `dt` that does not divide 86400, the numpy loop's
total-step count, day-boundary bookkeeping, and restart resume disagree
with each other. The numba backend rejects such `dt` at config time; the
numpy path accepts it silently.

**Mechanism** (all in `ss09/sw_model.py`). The run executes
`total_time_steps = int(86400 * ndays / dt)` steps, but daily storage
cycles on `spd = int(86400 / dt)` steps per "day", and
`ndays * spd != int(86400 * ndays / dt)` for non-divisor dt. Measured
(2026-07-13 review probe, dt=7000/ndays=5): 61 total steps vs 60
day-cycled steps; 48 vs 45 at dt=9000. The trailing sub-day steps evolve
the state into the final restart file but belong to no stored day, and
each stored "day" spans `spd*dt != 86400` seconds. Restart resume computes
`starting_step = int(day * 86400 / dt)`, which is not a multiple of `spd`:
at dt=7000, day 3 resumes at step 37 while the original run's day-3
boundary was step 36, so a continuation duplicates a step relative to the
uninterrupted run.

**Scope.** No known workflow uses a non-divisor dt (production values
3600/1800/30/15 all divide 86400); latent correctness, no active harm.

**Candidate fix.** Extend the numba backend's divisibility validation
(`ss09/sw_config.py`, the `86400 % dt` check) to all backends, plus an
error-path test, after confirming no use case needs sub-day-incommensurate
dt.
