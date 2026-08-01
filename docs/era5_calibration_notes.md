# ERA5 calibration: working notes for the report

Status 2026-08-01. Branch `feature/era5-a-calibration` (5 commits, UNMERGED).
These notes hold everything needed to write `docs/era5_calibration_report.tex`
in a fresh session. Terse register: written for future-me, not for Spencer.

## What the report has to be

A self-contained LaTeX PDF status report per the project convention
(`[[self-contained-status-reports]]`): non-chronological, figures embedded,
equations and parameter values and a run/data inventory in appendices. Not a
log. It must lay out the d/H argument explicitly, since Spencer twice said the
chat version was unclear.

It should also carry the follow-up calculations listed under "Open
calculations" below.

## Scripts and data

All under `scripts/`, all run with plain `python` from `scripts/`:

- `era5_a_calibration.py` -> `model_output/era5_a_calibration.nc` + table
- `era5_a_calibration_figures.py` -> `_map`, `_profiles`, `_vertical`, `_transport` PNGs
- `era5_stability_calibration.py` -> table + `_regression.png`
- `era5_vertical_structure.py` -> table + PNG
- `era5_moisture_budget.py` -> table + PNG

ERA5 monthly zonal means at `~/Dropbox/data/ecmwf/era5/monthly/` (symlinked in
the repo as `era5-data/`, created by Spencer 2026-08-01). On disk and used:
`hus`, `cwv`, `ps`, `va`, `ta`, `zg`, `evap`, `pr` (the only lat-lon file,
4 GB, streamed month by month; no dask in this env).

Empty directories, i.e. the highest-value downloads: `col_wv_flux_north`,
`col_mse_flux_north_st_eddy`, `col_dse_flux_north_st_eddy`,
`div_col_wv_flux`, `tend_col_wv`. Also wanted: the ERA5 mass-consistent
product (Trenberth-adjusted column fluxes and tendencies).

## THE CENTRAL RESULT (found last, supersedes earlier framing)

**The model mixes two incompatible readings of what `v` means, and that single
inconsistency is the whole negative-GMS problem.**

- **Half-column reading.** `v` is the branch velocity of two branches each
  holding half the column mass. Forced by the moisture equation: the form
  `-(2a-1) v W` requires the upper-branch fraction to be `1-a`, i.e. the two
  branches exhaust the column. Also forced by the factor of 2 in the `v`
  equation.
- **Delta-layer reading.** `v` is the velocity of a layer of depth `d` at the
  top and another at the bottom, with `v = 0` between. Forced by the
  thermodynamic equation's `(d Delta_z / H) dv/dy`, which comes from
  `w = d dv/dy` advecting `Delta_z/H`.

Measured (ERA5 2000-2019, |lat|<=20, mass-adjusted, W = 39.6 kg/m^2):

| reading | Shat (J/m^2) | 2a-1 | Hhat (J/m^2) |
|---|---|---|---|
| half-column | 1.978e8 | 1.392 | +5.986e7 |
| delta-layer, dp = 150 hPa | 5.93e7 | 0.418 | +1.80e7 |
| delta-layer, dp = 200 hPa | 7.91e7 | 0.557 | +2.40e7 |
| delta-layer, dp = 250 hPa | 9.89e7 | 0.696 | +2.99e7 |
| delta-layer, dp = 300 hPa | 1.187e8 | 0.835 | +3.59e7 |

Everything scales linearly with `dp`, since `v_delta / v_half = dp / (p_s/2)`.

**The model's own `Shat = C d Delta_z / H = 7.746e7` corresponds to
`dp = 196 hPa`**, which is essentially Spencer's eyeball estimate of ~200 hPa
from the observed `[v]` profile. So the model's Shat is RIGHT, not 2.55x too
small; my 2026-07-31 claim was wrong because it normalised by the half-column
`v_model`.

At that same `dp = 196 hPa`: **`2a-1 = 0.545` (so `a = 0.773`), and
`Hhat = +2.34e7 J/m^2`, positive.**

The negative GMS came from mixing readings:

| combination | Hhat |
|---|---|
| model Shat + half-column flux-matched 2a-1 = 1.392 | -6.05e7 |
| model Shat + half-column mass-partition 2a-1 = 0.900 | -1.17e7 |
| model Shat + delta-layer 2a-1 = 0.545 (CONSISTENT) | **+2.34e7** |

**Consequence for V2**: with a consistent `a ~ 0.77`, `W* = Shat/(L_v(2a-1))
= 56.9 kg/m^2`, above the quiescent column `W_c + tau_c E_0 = 50.66`. The
default `W_c = 50` is then POSITIVE-GMS and the banded/aggregated regime of
2026-07-31 should not appear. This needs a run to confirm.

**Retract**: the 2026-07-31 recommendation of `a = 0.95` and the 2026-08-01
claim that Shat is 2.55x too small. Both used the half-column normalisation
against a delta-layer Shat.

## The a calibration (mass partition), unchanged and still valid

ERA5 1979-2023, monthly zonal-mean `hus`, below-ground masked by zonal-mean `ps`.
`a = W_lower / W_total` at an interface `p_d`.

Half-mass interface `p_s/2` (derived: equal and opposite branch VELOCITIES
force equal branch masses):

- a = 0.9477 (|lat|<=10), 0.9501 (<=20), 0.9506 (<=30), 0.9501 global
- across latitude, annual mean within +/-30: 0.946 to 0.960
- tropical-mean seasonal cycle: 0.9496 (Mar) to 0.9508 (Oct), range 0.0012
- at fixed latitude the swing is 20x bigger: 0.939-0.964 at 15N (max Feb),
  0.945-0.972 at 15S (max Jul). Max in local winter, min in local summer;
  hemispheres cancel in the tropical mean. Mechanism [plausible]: winter
  subtropical subsidence plus trade inversion traps water low; summer
  ITCZ/monsoon convection moistens the free troposphere.
- interannual: 0.9466-0.9529 over 45 annual means, trend -0.0011/decade.
  Mechanism [speculative]: Clausius-Clapeyron moistens the colder free
  troposphere fractionally faster.

Interface ladder (|lat|<=20): 400 hPa -> 0.982, 500 -> 0.950, 600 -> 0.883,
700 -> 0.772. Interface sensitivity dominates all geophysical variation.

Robustness: piecewise-constant vs trapezoidal integration agree to 0.0006 in
`a`; removing the below-ground mask entirely moves `a` by 0.0007; the column
integral runs 2.0% below ERA5 tcwv in the tropics (bias concentrated at
30-40N, Tibet/Sahara, under 1.3% in the deep tropics). Spencer 2026-08-01:
not concerned about the zonal-asymmetry error at this magnitude.

Identity verified to 2.8e-14: the model's `-(2a-1)vW` IS the two-layer bulk
flux at `p_s/2`.

Level-of-nondivergence interface: `Psi` is nearly flat through the dead layer,
so `|Psi|` stays within 5% of its extremum over 549-705 hPa (|lat|<=15) and
`a` read there spans 0.87 to 0.72. Reported as a band in the figures.

## Effective vs literal bulk (Spencer asked; define carefully in the report)

- **Effective**: least-squares slope of the observed flux on the model's `v`.
  Answers "how much does the atmosphere actually move per unit branch
  velocity". Absorbs the within-branch correlation between `v` and the
  transported quantity.
- **Literal bulk**: `M_branch x (mean_upper - mean_lower)` at the interface.
  Answers "what would a two-layer model with genuinely uniform branch
  velocities carry".

DSE: 1.978e8 (effective) vs 2.283e8 (literal), 13% apart, opposite sign of
correction from moisture because `s` rises smoothly with height while the
upper branch's transport sits below its own branch mean.
Moisture: 1.401 (effective) vs 0.900 (literal), 56% apart, because `q` is
concentrated at the very bottom where `|v|` is largest.

Spencer 2026-07-31 leans to the effective/"effective fraction" reading and is
comfortable with it exceeding 1 in the half-column normalisation.

## Spencer's notes of 2026-08-01, to be addressed in the report

1. **"Degenerate" was a pedantic dodge; CONCEDE.** His condition, read as
   intended (the level separating the poleward branch from the equatorward
   branch), is exactly the level of nondivergence, which is what
   `dynamical_interface` computes. The technical point I made (that
   `Psi(p_d) = -(Psi(p_s)-Psi(p_d))` reduces to `Psi(p_s)=0` and so holds at
   every level) is true but unhelpful: only at the sign-change level do the
   two integrals equal the FULL poleward and equatorward transports; at other
   levels each mixes both signs. State it his way and drop the framing.
   Time-mean mass conservation giving `int v dp = 0` is exactly right.

2. **Equal pressure depths, not equal geometric depths.** He is right and this
   is the fix that makes the delta-layer picture self-consistent: equal `dp`
   gives equal mass, which is what equal-and-opposite velocities require. His
   ~200 hPa eyeball from `era5_vertical_structure.png` panel a matches the
   196 hPa the model's own Shat implies. Measured mismatch under the current
   equal-geometric-depth reading: a 4 km layer spans 153 hPa below the model
   tropopause and 369 hPa above the ground, mass ratio 2.4.

3. **Lay out the d/H argument in the PDF.** It confused him twice in chat.
   The chain to write out slowly: the theta equation forces
   `F_DSE = C (d Delta_z/H) v`, so `Shat = C d Delta_z/H` is not an extra
   assumption; `d Delta_z/H = 15 K` is the theta change across ONE layer of
   depth d; whether 15 K is right depends entirely on what `v` means, which
   is the central result above. The "(H-d) Delta_z/H = 45 K" alternative I
   floated matches the literal-bulk half-column Shat to 2% (2.324e8 vs
   2.283e8) but is the WRONG fix, because the model's `v` is a layer velocity
   and not a half-column velocity. Present the two-readings table instead.

4. **beta: use the Coriolis mapping.** `beta = 2e-11` equals Earth's
   `2 Omega cos(phi)/a` at **phi = 29.1 deg**. Equatorial value 2.289e-11.
   His suggestion, worth computing properly in the report: a ~15 deg value is
   2.211e-11, and the cos-weighted average over 0-30 deg is **2.190e-11**
   (`<beta> = 2 Omega int cos^2 / (a int cos)`), 9.5% above the model's 2e-11.
   Mapping `y = 3000 km`: 24.3 deg (Coriolis, USE THIS), 27.0 deg (arc length
   on Earth's radius), 23.6 deg (arc length on the 7292 km radius the model's
   beta implies).

5. **RCE, not RE, and it is unobservable.** He is fully convinced the target
   should be RCE, and says he does not know how he convinced himself
   otherwise when the spec was written. He is also right that boundary-layer
   `theta_e` in reanalysis already contains the circulation's influence, so it
   is a contaminated proxy: true lat-by-lat RCE has time-mean `v = w = 0` by
   definition and is not observable. **Given that ambiguity, treat the RCE
   contrast as a tuning parameter within reasonable limits**, with ERA5
   setting the limits rather than the value.

## Measured numbers to carry into the report

### Stability (ERA5 2000-2019, regressed on v_model, half-column normalisation)

| band | Shat_eff | Shat_bulk | Hhat_eff | W | Hhat/Shat |
|---|---|---|---|---|---|
| \|lat\|<=10 | 1.909e8 | 2.287e8 | 4.926e7 | 44.5 | 0.258 |
| \|lat\|<=15 | 1.946e8 | 2.283e8 | 5.459e7 | 42.2 | 0.281 |
| \|lat\|<=20 | 1.978e8 | 2.283e8 | 5.986e7 | 39.6 | 0.303 |
| \|lat\|<=30 | 1.997e8 | 2.301e8 | 6.348e7 | 35.1 | 0.318 |

**Correction to make explicitly**: the LOCAL `Hhat = F_MSE/v_model` is not
uniformly positive. It goes negative within ~8 deg of the equator (about
-50 MJ/m^2 just south of it) and rises to +140 MJ/m^2 by 25N. The band
aggregate is dominated by large-|v| subtropical points. Caveat: `v_model`
passes through zero near the equator so the local ratio is ill-conditioned
there; do not quote -50 as firm. This is the observed counterpart of the
"very slightly negative GMS" case Spencer said might still interest him.

### Budget closure

- Barotropic mass adjustment on `[v]`: changes Shat by **-33.8%** (2.989e8 ->
  1.978e8), peak |Psi_ext| by -10.2%, peak |v_model| by -10.3%. Shat is the
  MOST sensitive because `s` has a ~330 kJ/kg mean, so a small spurious
  barotropic `[v]` fabricates a large DSE flux. Treat 34% as the floor on the
  Shat uncertainty until the mass-consistent product is used.
- Without adjustment, |Psi(p_s)| averages 232 and peaks at 780 kg/m/s.
- Global mean E-P = -0.0129 mm/day (0.4% of global P). Integrating it pole to
  pole puts -20.0 kg/m/s at 85N where the flux must vanish; removing the
  global mean first gives +1.7 and moves the tropical D only from 7.20e5 to
  7.24e5. Moisture route robust, energy route not.

### Eddy diffusivity D (budget residual, stationary + transient together)

Regression of `-F_eddy` on `dW/dy`, ERA5 2000-2009:

| band | D (m^2/s) |
|---|---|
| 0-10 | 3.474e5 |
| 0-15 | 5.444e5 |
| 0-20 | 7.629e5 |
| 10-20 | 9.404e5 |
| 15-25 | 1.833e6 |
| 20-30 | 2.437e6 |
| 25-35 | 2.794e6 |
| 0-35 | 1.376e6 |

Reconciles with the postdoc's lat-lon tropical 1-2e5: D rises an order of
magnitude from deep tropics to subtropics, my earlier 7e5 aggregate was
subtropics-dominated, and the residual gap is plausibly the stationary eddies
the zonal-mean route lumps in. Model default `d_w = 1e6` is a subtropical
value applied to the deep tropics.

### E_0

ERA5 mean evaporation: 4.44e-5 (|lat|<=10), **4.66e-5** (<=20), 4.52e-5
(<=30) kg/m^2/s. Model 4.6e-5. Matches to 1.3%, no change needed.

### Temperature structure

Free-tropospheric theta (200-850 hPa mass weighted), ERA5:
theta(0) = 325.6 K, i.e. model-diagnostic `T = 0.625 theta` = 203.5 K.
Equator-minus-value contrasts: 0.02 K at 10 deg, 0.99 at 18, 4.87 at 27,
6.76 at 30. On the Coriolis mapping: 0.56 K at 15.9 deg, 3.35 at 24.3,
8.95 at 33.3.

ERA5 200 hPa zonal-mean u: max 31.1 m/s at 30.2 S.

Boundary-layer theta_e (surface to 100 hPa above), equator-minus-value:
2.11 K at 10 deg, 7.75 at 16, 12.40 at 20, 16.59 at 24, 23.93 at 30.
So a column-by-column RCE free troposphere would be ~5x steeper than observed
at 24 deg. CAVEAT PER SPENCER: this proxy is itself contaminated by the
circulation; it bounds rather than determines the RCE contrast.

### Model equilibria vs ERA5 (existing runs in `model_output/moist_v2/`)

Contrasts at the Coriolis-mapped latitudes, last-10-day means:

| run | dTh@16 | dTh@24 | dTh@33 | T_eq | jet | max\|v\| |
|---|---|---|---|---|---|---|
| ERA5 | 0.56 | 3.35 | 8.95 | 203.5 | 31.1 | 0.49 |
| V1 (dy=50, no LH) | 0.18 | 1.13 | 4.16 | 200.3 | 28.2 | 0.365 |
| V2_0 (dyr=75, no LH) | 0.26 | 1.63 | 5.99 | 197.1 | 41.6 | 0.562 |
| V2 Wc=35 | 0.62 | 3.65 | 12.22 | 245.0 | 55.1 | 1.42 |
| V2 Wc=40 | 0.92 | 5.07 | 15.96 | 246.1 | 60.8 | 2.27 |
| V2 Wc=50 | 1.71 | 7.62 | 21.62 | 248.1 | 73.8 | 5.87 |
| V2 dyr=55 | 1.64 | 7.19 | 19.63 | 249.6 | 61.2 | - |
| V2 dyr=95 | 1.78 | 8.01 | 23.38 | 246.3 | 88.6 | - |

Key readings: V1 matches the jet, overturning and temperature LEVEL but is 3x
too flat in gradient. Every V2 config overshoots the jet 1.8-2.4x. Latent
heating STEEPENS the model's meridional gradient (V2_0 1.63 -> V2 3.65-7.62 at
24 deg), the opposite of the spec's stated "convection flattens it back"
rationale. NOTE: `max|v|` comparison is itself normalisation-dependent and
must be redone under the resolved reading.

### The a=0.95 -> GMS arithmetic (now superseded but keep for the narrative)

`W* = Shat/(L_v(2a-1))`: 44.26 at a=0.85, 34.42 at a=0.9501, both with the
model Shat. This is what made the GMS look worse, and it was the
normalisation mismatch.

## Open calculations for the report session

1. **Literal delta-slab mass partition.** Under equal-pressure-depth slabs of
   ~200 hPa, compute `(W_lower_slab - W_upper_slab)/W` and compare with the
   flux-matched 0.545. Determines the recommended `a`.
2. **Redo the a(lat, month) climatology at the ~200 hPa slab interface**, not
   just the half-mass one, since that is now the model-consistent geometry.
3. **Confirm the resolved reading fixes V2**: run `W_c = 50` with
   `--cwv-frac 0.77` and check the banded state does not appear.
4. **beta**: cos-weighted average over the Hadley range, and what changes if
   `beta = 2.19e-11`.
5. **Redo the model-vs-ERA5 `max|v|` comparison** under the resolved
   normalisation (the current table mixes conventions).
6. **Delta_z and H** directly from ERA5 (theta at tropopause minus surface;
   lapse-rate tropopause height), to check whether 60 K and 16 km hold up.
7. **tau** from clear-sky radiative cooling (`rlntcs`, `rsntcs`, `olr`).
8. **(W_c, tau_c)** from regressing P on W; note monthly zonal means give only
   a smoothed pickup curve, so this bounds rather than fits.
9. Optional: `v_d` from the zonal-momentum-budget residual.

## Systematic parameter table (for an appendix)

Validatable and DONE: `Shat`, `Hhat`, `a`/`2a-1`, `E_0`, `Delta_theta` (proxy),
`D`.
Validatable, not yet done: `Delta_z`, `H`, `theta_00`, `tau`, `y_1`, `y_0`,
`(W_c, tau_c)`, `v_d`.
Over-determined: `d` (two slab parameters, three observables).
Not ERA5-validatable: `Delta_theta^rad` (needs offline radiative equilibrium;
and per note 5 the target should be RCE anyway), `beta` (a choice of reference
latitude), `epsilon` (regularisation), `kappa_v` (explicitly numerical).

**Organising principle to state in the report**: calibrate the COMBINATIONS the
equations use, not the individual symbols. Calibrating `a` alone pushed the GMS
negative; the symbols only mean something once the normalisation of `v` is
fixed.

## Figures available

- `era5_a_calibration_map.png` (month on x, latitude on y per
  [[lat-month-plot-orientation]])
- `era5_a_calibration_profiles.png`, `_vertical.png`, `_transport.png`
- `era5_stability_regression.png`
- `era5_vertical_structure.png`
- `era5_moisture_budget.png`

All need regenerating once the resolved normalisation lands, since several
carry the half-column framing.
