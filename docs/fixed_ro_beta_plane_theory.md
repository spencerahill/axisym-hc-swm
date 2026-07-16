# Fixed-Ro equal-area theory on the equatorial beta plane, for this SWM

Beta-plane translations of the closed forms in the fixed-Ro manuscript
(`papers/fixed-ro-eq-area_ms-draft-v1.pdf`, "ms" below), derived against this
model's exact equations. Every formula carries a basis tag; "pending run"
tags get upgraded as the suite's runs confirm them. Section 8 lists the
falsifying checks and their status.

## 1. Setup and notation

Steady state, hemispherically symmetric forcing (y0 = 0), northern
hemisphere (y > 0). The SWM's steady equations (SCIENCE.md section 2), with
T = theta/1.6 and the v equation's factor of 2 multiplying only the
time-tendency (so it drops out of every steady balance):

- zonal momentum: v (beta y - u_y) = H(theta_E - theta) (v_y) u
  + epsilon_u u + v_d H(u) sgn(y) u_y
- meridional momentum: beta y u = -(g H / T0) T_y + k_v v_yy
- thermodynamic: (delta Delta_z / H) v_y = (theta_E - theta) / tau

Forcing (SS09 parabolic profile, the theory-matched choice): theta_E =
theta_00 - Delta_y (y / y1)^2 for |y| < y1. In temperature units T_E =
T_E(0) - b y^2 with curvature b = Delta_T / y1^2 and Delta_T = Delta_y / 1.6.
T0 = t_ref = 300 K.

The small-angle limit of the ms is exact here: the beta plane has no cos(lat)
factors, so no truncation enters any of the following.

## 2. The one-line mapping

Every ms result carries over with the substitutions

    Ro_th -> R_beta = 4 g H Delta_T / (T0 beta^2 y1^4)
    Delta_h theta_0 -> Delta_T,    a phi -> y,    phi_Ro -> Y_Ro / y1.

`[derived]` R_beta comes from the sphere definition Ro_th = g H Delta_h /
(Omega a)^2 with Omega = beta a / 2 and Delta_h = Delta_T a^2 / (T0 y1^2)
(matching the forcing curvature at the equator); the a's cancel, as they
must on a beta plane. At this repo's defaults (g = 9.81, H = 16 km,
Delta_y = 50 K so Delta_T = 31.25 K, T0 = 300 K, beta = 2e-11, y1 =
9439 km): **R_beta = 0.0206**.

## 3. Wind and temperature `[derived: pending run confirmation]`

Uniform Ro = u_y / (beta y) integrates to u_Ro(y) = Ro beta y^2 / 2 (ascent
at y = 0). Gradient balance (neglecting k_v) gives dT/dy = -(T0 / g H) Ro
beta^2 y^3 / 2, so

    T_Ro(y) = T_Ro(0) - c y^4,    c = Ro T0 beta^2 / (8 g H).

## 4. Equal-area edge and equatorial temperature `[derived: pending run confirmation]`

With D(y) = T_Ro(y) - T_E(y), imposing D(Y) = 0 and the zero net energy
integral over [0, Y] (measure dy; the Newtonian relaxation has uniform tau):
D(Y) = 0 and int D = 0 give (2/3) b Y^2 = (4/5) c Y^4, i.e. Y^2 = (5/6) b/c,
which in the Section-2 variables is

    Y_Ro = y1 (5 R_beta / (3 Ro))^(1/2)              [ms Eq. 13 analog]
    T_E(0) - T_Ro(0) = (5/18) R_beta Delta_T / Ro    [ms Eq. 15 analog]

Defaults, Ro = 1: **Y_Ro = 1.75e6 m** and equatorial depression **0.18 K**
(in T; multiply by 1.6 for theta units). At Ro = 0.15 the depression is
1.2 K (T) / 1.9 K (theta).

Consistency note: the analogous derivation for the sin2 profile needs the
numerical equal-area solver (puffins `equal_area_bouss` with Ro-scaled
`del_h_over_ro`, verified in puffins PR #46). Near the equator sin2 behaves
like a parabola with curvature enhanced by (pi/2)^2 = 2.47, since
sin^2(pi y / (2 y1)) = (pi y / (2 y1))^2 + O(y^4); Section 7's postdiction
check uses exactly this factor.

## 5. Meridional wind, heat flux, momentum flux `[derived: pending run confirmation]`

The steady thermodynamic balance integrates to the SWM's v profile directly
(theta_E - theta = 1.6 (T_E - T) = -1.6 D):

    v(y) = -(1.6 H / (delta Delta_z tau)) int_0^y D dy'
         = (1.6 H / (delta Delta_z tau)) (5/18) (R_beta Delta_T / Ro)
           Y_Ro [x - 2 x^3 + x^5],    x = y / Y_Ro.

The bracket is exactly the ms Eq. (16) shape; it peaks at x = 5^(-1/2) =
0.447 with value 0.286, and v_max scales as Ro^(-3/2) (ms Eq. 20 analog).
Defaults, Ro = 1: **v_max = 3.0e-3 m/s** at y = 0.78e6 m.

Column fluxes, using the two-layer reading (upper slab of depth delta with
velocity +v and potential temperature theta_top, return slab of depth delta
with -v and theta_bot, theta_top - theta_bot = Delta_z; return-flow u is
negligible):

    heat flux:      int v theta dz = delta Delta_z v(y)      [ms Eq. 16]
    momentum flux:  int v u dz   = delta v(y) u_Ro(y)        [ms Eq. 17]
    vertical velocity: w = delta v_y, w_max = delta v'(0) ~ Ro^(-1)  [ms sec. 3f]

The ms's counterintuitive momentum-flux result carries over: expanding
delta v u_Ro, the leading term in y is Ro-independent (the Ro^(-1)
strengthening of v cancels the Ro weakening of u in the product) and
d(flux)/dRo < 0 strictly inside the cell, exactly as verified for the
sphere forms in puffins `test_eq_area.py`.

## 6. Diagnosed quantities: conventions for the suite

- **Local Ro(y):** u_y / (beta y), the model's existing diagnostic (NaN
  within |y| < dy of the equator).
- **Cell-mean Ro, two definitions, both reported** (paralleling Hill et al.
  2025): (i) the area mean of local Ro over [y of Ro-max, Y_edge]; (ii) the
  best-fit Ro regressing u(y) against beta y^2 / 2 over the same interval.
  The near-ascent region is excluded because Ro -> 0 at the equator whenever
  v_d > 0 (Zhang et al. 2025, u ~ y^3).
- **Cell edge, three definitions, all reported:** the v-threshold edge
  (|v| below 10% of its extremum, existing diagnostic), the jet latitude
  (existing spline diagnostic), and the theta-merge latitude (where
  theta rejoins theta_E within a tolerance). The theory says they coincide;
  their splitting is itself a result.
- **Momentum-budget shares:** each steady u-equation term evaluated from
  output; the drag share epsilon_u u and vertical-advection share quantify
  the accuracy of the EMFD-Coriolis balance underlying Zhang et al. (2025)
  Eq. 24, Ro = v / (v + v_d).

## 7. Emergent cell-mean Ro closure `[derived: pending run confirmation]`

Combining Zhang's Eq. 24 at the cell-mean level with Section 5's v
magnitude (cell mean of the bracket over [0, 1] is 1/6):

    Rbar = vbar / (vbar + v_d),
    vbar(Rbar) = (1.6 H / (delta Delta_z tau)) (5/18)
                 (R_beta Delta_T / Rbar) Y_Ro(Rbar) / 6,

a transcendental equation for Rbar(v_d) with no simulation input. This is
the suite's analysis G; it extends Zhang et al. (2025) section 4 (their
max-Ro scalings v_d^(-2/5), v_d^(-1/4)) to the cell-mean variable the ms
uses.

Postdiction check `[computed]`: Section 5's v with the sin2 profile's
larger near-equator curvature ((pi/2)^2 = 2.47 in b, entering v through
b^2 Y / c as a factor ~9.6) predicts v_max ~ 0.03 m/s for the published
defaults, matching Zhang et al. (2025) Figs. 2-3 magnitudes within ~1.5x.

## 8. Falsifying checks and status

1. **Edge, equatorial depression, v profile vs diagnosed Ro (this memo's
   most-falsifying single check):** SS09-parabolic forcing, v_d = 0.
   Framing correction from the first run: with the default vertical
   momentum advection on, v_d = 0 is NOT the Ro = 1 limit (Zhang et al.
   2025: u ~ beta y^2 / 3 near the ascent, and drag lowers it further);
   the run's diagnosed cell-mean Ro is what the theory must be evaluated
   at. The Ro -> 1 anchor requires --no-vert-advec-u. Status: see run
   record below.
2. **Edge collapse:** Y_edge (Ro / R_beta)^(1/2) / y1 = (5/3)^(1/2) flat
   across the v_d, Delta_y, beta, and radius ladders. Status: pending
   Tier 1-2.
3. **Radius-ladder null:** dimensional Y_edge invariant under (beta -> beta/k,
   y1 -> k y1, L -> k L). Status: pending Tier 2.
4. **v-profile shape and Ro^(-3/2) magnitude scaling** (Section 5). Status:
   pending Tier 1.
5. **Momentum flux increases as Ro decreases** (Section 5). Status: pending
   Tier 1.
6. **Emergent Rbar(v_d)** (Section 7). Status: pending Tier 1 + analysis G.

## Run record

(Each confirming or refuting run gets an entry: config, output path, measured
vs predicted.)

### 2026-07-16, check 1a: v_d = 0, default physics (vertical advection ON)

Config: SS09 parabolic forcing, v_d = 0, ny = 801, dt = 30, numba backend,
steady-state stop (window 30 d, threshold 1e-4), converged day 959. Output:
`model_output/fixed_ro_suite/tier0_amc_edge/amc_ss09_ny801_dt30_vd0.nc`.
Steadiness caveat `[computed]`: the jet was still drifting +0.4% per 60 d at
stop (drag-timescale tail), so numbers carry roughly a 2% allowance.

Diagnosed: cell-mean Ro 0.469 (area mean) / 0.470 (u-regression); local
Ro(y) is NOT uniform: ~0.68 at the ascent (the vertical-advection 2/3 of
Zhang et al. 2025, drag-reduced), declining through ~0.6 mid-cell to ~0.2
approaching the jet, i.e. the ms's linear-Ro motivation in miniature.

Theory at diagnosed Ro = 0.470 vs run `[computed]`:

| quantity | predicted | measured | rel. gap |
|---|---|---|---|
| equatorial depression T_E(0)-T(0) | 0.381 K | 0.379 K | 0.5% |
| v_max | 9.28e-3 m/s | 9.39e-3 m/s | 1.2% |
| y of v_max (0.447 Y_Ro) | 1.14e6 m | 1.18e6 m | 3.5% |
| cell edge Y_Ro (vs jet latitude) | 2.55e6 m | 2.36e6 m | -7.4% |

Sections 4-5 upgraded to `[confirmed at the 1-8% level, single run]` for
these observables at this Ro. Additional findings: the u plateau outside
the cell is the parabolic forcing's RCE gradient wind, u_RCE = 2 g H
Delta_T / (T0 beta y1^2) = 18.35 m/s, constant out to y1 (verified to 3
digits); the v-threshold edge diagnostic is unusable at small v_d because
the drag circulation v = epsilon_u u_RCE / (beta y) (~2e-3 m/s) exceeds
10% of v_max out to ~9.3e6 m; the theta-merge diagnostic must search
poleward of the jet, not of the v max, to avoid the equal-area lobe
crossing.

### 2026-07-16, check 1b: v_d = 0, vertical advection OFF

Config: as 1a plus --no-vert-advec-u; converged day 971. Output:
`.../tier0_amc_edge/amc_ss09_ny801_dt30_vd0_novert.nc`.

Removing vertical advection left the cell-mean Ro essentially unchanged
(fit 0.473 vs 0.470; area mean 0.536 vs 0.469), because **the Rayleigh
drag, at its default epsilon_u = 1e-8, is the leading-order non-AMC agent
for this forcing** `[computed]`: the steady zonal momentum balance over
the cell interior is v (beta y - u_y) = epsilon_u u to within ~5-10%
(median ratio 1.05 over 0.3e6 < y < 2.2e6 m). The parabolic forcing's
weak circulation (v_max ~ 9e-3 m/s) makes (1 - Ro) ~ epsilon_u u /
(v beta y) order one. Local Ro(y) declines monotonically from ~1 at the
ascent to ~0.2 at the jet.

Theory at diagnosed Ro = 0.473 vs run `[computed]`: depression 0.378
predicted vs 0.358 K measured (5.3%); v_max 9.18e-3 vs 8.81e-3 m/s
(4.2%); edge 2.54e6 vs jet 2.36e6 m (-7.1%).

Consequences for the suite: (i) the Ro -> 1 anchor needs a reduced
epsilon_u, so the Tier-3 drag ladder is load-bearing rather than optional;
(ii) at the sin2 forcing the circulation is ~3x stronger, so the drag
share is ~3x smaller but still non-negligible at v_d = 0 (consistent with
Zhang et al. 2025 Fig. 4's max Ro ~ 0.9 there); (iii) the steady-state
detector (window 30 d, 1e-4) fires while the jet still drifts ~0.4% per
60 d on the drag timescale, so budget-grade runs need a stricter threshold
or a fixed post-detection extension.
