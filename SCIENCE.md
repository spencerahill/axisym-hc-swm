# SCIENCE.md

Scientific background for the Sobel-Schneider single-layer Hadley circulation model. This document provides physics context for Claude Code; see `CLAUDE.md` for coding workflow and repository mechanics.

## 1. Physical Setup

This model represents the **zonal-mean upper troposphere** as a single thin layer near the tropopause on an **equatorial β-plane**. It simulates the **Hadley circulation**—Earth's tropical overturning circulation characterized by:
- Rising motion near the equator (ITCZ)
- Poleward flow aloft
- Subsidence in the subtropics (~30°N/S)
- Equatorward return flow at the surface (not explicitly modeled)

The model captures the interaction between thermally-driven mean meridional circulation and **parameterized eddy momentum fluxes** from extratropical baroclinic eddies. This interaction determines whether the circulation is in an angular momentum-conserving (AMC) regime or an eddy-dominated regime.

## 2. Governing Equations

The model solves three prognostic equations for zonal wind $u$, meridional wind $v$, and potential temperature $\theta$.

### 2.1 Zonal Momentum Equation

$$\frac{\partial u}{\partial t} - v(\beta y - \frac{\partial u}{\partial y}) = -\mathcal{H}(\theta_E - \theta) \cdot (\frac{\partial v}{\partial y}) \cdot u - \mathcal{F} - \mathcal{S}$$

| Term | Expression | Physical Meaning |
|------|------------|------------------|
| Tendency | $\partial u / \partial t$ | Rate of change of zonal wind |
| Coriolis + advection | $v(\beta y - \partial u / \partial y)$ | Acceleration from planetary vorticity and relative vorticity advection |
| Vertical advection | $-\mathcal{H}(\theta_E - \theta)(\partial_y v) u$ | Momentum exchange with lower layer; only active where atmosphere is cooler than equilibrium ($\theta_E > \theta$). Assumes ascending air carries **zero zonal momentum**. |
| Rayleigh drag | $\mathcal{F} = \epsilon_u u$ | Linear damping; ensures numerical stability, no direct atmospheric analog |
| EMFD | $\mathcal{S}$ | Eddy momentum flux divergence (see §3.1) |

> **Deviation from SS09**: SS09's published equation (their Eq. 2.1) gates the vertical advection term with $\mathcal{H}(\partial_y v)$, not $\mathcal{H}(\theta_E - \theta)$. This implementation changes only the Heaviside *argument* (the on/off gate), keeping the kinematic $\partial_y v$ as the multiplicative magnitude. The two gates coincide in steady state; the thermodynamic gate suppresses a grid-scale numerical artifact. See §3.3 for the full rationale.

### 2.2 Meridional Momentum Equation

$$2 \frac{\partial v}{\partial t} + \beta y u = -\frac{gH}{T_0} \frac{\partial T}{\partial y} + k_v \frac{\partial^2 v}{\partial y^2}$$

**Note**: The original SS09 paper included a meridional advection term $2v \partial v / \partial y$, but this was removed in the 2013 corrigendum (see §2.4 below).

**Why the factor of 2?** The equation is integrated vertically over the entire troposphere. The model assumes:
1. Meridional velocities at the tropopause and surface are **equal and opposite** ($v_{top} = -v_{bottom}$)
2. The surface zonal velocity is negligible compared to the upper-tropospheric value

This vertical structure doubles the inertial term relative to a single-layer formulation. See Xian & Miller (2008) and the appendix in Zhang et al. (2025) for the derivation.

| Term | Physical Meaning |
|------|------------------|
| $\beta y u$ | Coriolis force from zonal wind |
| $-(gH/T_0) \partial T / \partial y$ | Pressure gradient force (thermal wind balance) |
| $k_v \partial^2 v / \partial y^2$ | Eddy viscosity; damps grid-scale numerical noise |

### 2.3 Thermodynamic Equation

$$\frac{\partial \theta}{\partial t} + \frac{\delta \Delta_z}{H} \frac{\partial v}{\partial y} = \frac{\theta_E - \theta}{\tau} + \kappa_\theta \frac{\partial^2 \theta}{\partial y^2}$$

| Term | Physical Meaning |
|------|------------------|
| $(\delta \Delta_z / H) \partial v / \partial y$ | Adiabatic cooling/warming from vertical motion; divergence implies ascent |
| $(\theta_E - \theta) / \tau$ | Newtonian cooling toward radiative-convective equilibrium (RCE) profile |
| $\kappa_\theta \partial^2 \theta / \partial y^2$ | Eddy heat diffusion; optional extension (default $\kappa_\theta = 0$) |

**Note**: The eddy heat diffusion term is an optional extension not in the original SS09 equations. It is disabled by default (`coeff_eddy_heat_diff=0.0`) and values below ~10⁴ m²/s have minimal effect.

### 2.4 Sobel & Schneider (2013) Corrigendum

The 2013 correction to the original SS09 paper fixed two errors:

1. **Removed meridional advection in v-equation**: The term $2v \partial v / \partial y$ was erroneous and has been dropped. The corrected meridional momentum equation is shown in §2.2 above.

2. **Temperature vs. potential temperature bug**: In the original numerical implementation (not the written equations), potential temperature $\theta$ was inadvertently used in place of temperature $T$ in the pressure-gradient term of the meridional momentum equation. Since $\theta = 1.6\,T$, this made the effective pressure gradient and the computed circulations too strong. It quantitatively (but not qualitatively) affected SS09's figures numbered 3 and higher; the corrected (weaker) circulations in the 2013 figures are the right validation targets.

This implementation follows the corrected (2013) equations.

## 3. Key Parameterizations

### 3.1 Eddy Momentum Flux Divergence (EMFD)

$$\mathcal{S} = v_d \cdot \mathcal{H}(u) \cdot \text{sgn}(y) \cdot \frac{\partial u}{\partial y}$$

**Physical interpretation**: Extratropical baroclinic eddies (Rossby waves) propagate equatorward from midlatitudes. They have westerly or zero phase speeds and reach their **critical latitudes** where the mean flow matches their phase speed. There they dissipate or reflect, depositing momentum.

- $v_d$ ≈ 2.5 m/s: drag velocity controlling EMFD magnitude (tuned to match GCM results)
- $\mathcal{H}(u)$: Heaviside function ($\mathcal{H}(u) = 1$ for $u > 0$, $0$ for $u < 0$, and $0.5$ at $u = 0$ for implicit smoothing at the boundary); eddies only act where $u > 0$ (westerlies)
- $\text{sgn}(y)$: ensures correct sign in each hemisphere
- $\partial u / \partial y$: EMFD scales with wind shear

The total EMFD integrated over a Hadley cell is proportional to the subtropical jet strength, which by thermal wind balance scales with the meridional temperature gradient.

**The EMFD is anti-diffusive near a wind maximum.** Because $\mathcal{S} \propto \text{sgn}(y)\,\partial_y u$, the eddy momentum flux is *up-gradient* where $\partial_y u$ reverses (at a zonal-wind extremum), sharpening the jet there rather than smoothing it. The poleward jet flanks, where $u$ turns over and the $\mathcal{H}(u)$ gate switches, are therefore the numerically delicate region of the model.

**Gate provenance (deviation between the Zhang et al. 2025 paper and its code)**: The $\mathcal{H}(u)$ gate appears in SS09's Eq. (2.5) and in Zhang et al. (2025) Eq. (5), but the published source code behind the Zhang et al. (2025) figures ([github.com/zpcllyj/SobelSchneiderModel](https://github.com/zpcllyj/SobelSchneiderModel)) omits it (the gated line is commented out). At the paper's equinoctial forcing ($y_0 = 0$) the omission is physically inconsequential because $u > 0$ throughout the Hadley cell, but numerically the discontinuous gate switching at flank $u=0$ crossings excites a grid-scale mode on this collocated grid: in matched runs at ny=801/dt=30s/v_d=2.5, the gateless model reproduces the published code to machine roundoff (max$|\Delta u| = 8\times10^{-12}$ m/s over 30 days) while the gated model grows a spurious grid-scale flank jet saturating at 420 m/s near $|y| \approx 13$ Mm by day ~500, with the Hadley cell interior left superficially normal (verified 2026-07-09). The gate is configurable via `emfd_heaviside_gate`. It defaulted to disabled from 2026-07-09 to 2026-07-11 to match the published code and figures, and since 2026-07-12 defaults to enabled as part of the production formulation. The 420 m/s runaway above is specific to the gate on the *collocated* grid with the *centered* stencil; the production mc stencil (§3.1, next) and staggered grid (§3.4) make the gate-on integration stable and bounded, which is what lets the gate be the default. For the gateless reproduction of the published figures, pass `--no-emfd-heaviside-gate --emfd-stencil centered --grid collocated`.

### 3.2 Rayleigh Drag

$$\mathcal{F} = \epsilon_u u$$

A weak linear drag ($\epsilon_u \sim 10^{-8}$ s⁻¹, ~1000-day timescale) ensures the model has a stable steady state. It has no clear physical analog in the real atmosphere but is numerically necessary.

### 3.3 Vertical Momentum Advection

The term $-\mathcal{H}(\theta_E - \theta)(\partial_y v) u$ represents exchange between the upper layer and lower troposphere. Two limiting assumptions are possible:
1. **Ascending air carries zero zonal momentum** (default): creates the term shown
2. **Ascending air carries the upper-layer zonal momentum**: eliminates this term entirely

The model uses assumption (1). Sensitivity to this choice is modest compared to EMFD effects.

**Heaviside gate (deviation from SS09)**: SS09's published zonal momentum equation (their Eq. 2.1) gates this term with $\mathcal{H}(\partial_y v)$, so both the gate and the magnitude use the kinematic divergence. This implementation instead gates with $\mathcal{H}(\theta_E - \theta)$, a thermodynamic criterion: vertical momentum exchange is active where the atmosphere is cooler than radiative-convective equilibrium ($\theta_E > \theta$), which indicates a convective (ascending) tendency; when $\theta \geq \theta_E$ the term is inactive. Only the gate is changed. The multiplicative magnitude remains the kinematic $\partial_y v$.

**Why the two gates coincide in steady state**: The steady thermodynamic balance (§2.3, dropping the tendency and diffusion) is $\frac{\delta \Delta_z}{H}\partial_y v = \frac{\theta_E - \theta}{\tau}$, so $\partial_y v \propto (\theta_E - \theta)$ with the same sign and the two Heaviside arguments switch at the same latitudes. Away from equilibrium they differ, and that difference is the reason for the change: $\partial_y v$ is a spatial derivative of the prognostic $v$ field, so it amplifies the 2Δy computational mode of the leapfrog scheme; $\mathcal{H}(\partial_y v)$ then flips on and off between adjacent grid points and sustains grid-scale wiggles. SS09 saw these fluctuations too (their §2b) and controlled them with a staggered grid plus time averaging. $\mathcal{H}(\theta_E - \theta)$ is a smooth function of the smooth $\theta$ field, so the gate switches once at the convective margin and the artifact is suppressed. This gate change originated with Pengcheng Zhang in January 2024.

**Selectable gate (`--vert-advec-gate`)**: the thermodynamic gate above is the default for dry and V1 runs, and SS09's kinematic $\mathcal{H}(\partial_y v)$ is available and is the default under `--enable-latent-heating`. Two things changed since the original substitution. The staggered production grid removed the $2\Delta y$ null space that sustained the gridpoint oscillations (§3.4), and dry A/B twins run to slow-drift-gated steady state (2026-07-18) confirm the kinematic gate is stable there: no reintroduced grid-scale noise, exact mirror parity, climate anchors within 2%, and 99% sign agreement between the two gate arguments, with the only persistent difference a 433 km poleward shift of the westerly terminus notch. And the coincidence argument above fails in V2 (§3.6): with explicit latent heating the steady balance ties $\partial_y v$ to $(\theta_E-\theta)/\tau + \mathcal{L}P$, so where the column precipitates it can be warmer than its relaxation target, $\theta > \theta_E$, and the thermodynamic gate then switches off exactly where convection is strongest.

**Relation to the precursor literature**: Several earlier one- and two-layer models replace *both* the gate and the magnitude with the thermodynamic heating $Q \propto (\theta_E - \theta)$, that is, the convective mass flux, in place of the kinematic divergence: Esler et al. (2000), Shell & Held (2004), and Adam & Paldor (2009). SS09 and Xian & Miller (2008) keep the kinematic $\partial_y v$ magnitude. This implementation is a hybrid: the thermodynamic gate of the former group, the kinematic magnitude of the latter. Replacing the magnitude as well was considered and not adopted, to stay closer to SS09's Eq. 2.1 and to keep the vertical velocity consistent with the adiabatic term of the thermodynamic equation (§2.3), which also uses $\partial_y v$.

### 3.4 Staggered v-Grid (Arakawa C-grid)

The default grid staggers $v$ onto the $ny-1$ interior cell faces $y_i + \Delta y/2$, with $u$ and $\theta$ on the $ny$ centers, the arrangement SS09 use in their §2b. It exists to control the standing grid-scale $v$ noise that the collocated grid cannot.

**Why the collocated grid leaves a ripple.** On the collocated grid every field sits at the centers and $\partial_y v$ is the centered difference $(v_{i+1} - v_{i-1})/2\Delta y$, whose stencil skips $v_i$ and is therefore blind to the $2\Delta y$ checkerboard component of $v$. With the anti-diffusive EMFD forcing a sharp westerly-terminus front (§3.1), the terminus projects onto that checkerboard, and because the divergence and pressure-gradient operators cannot feel it, the grid-scale $v$ is a dynamical null space: it neither radiates nor is damped by them. Adding $k_v$ diffusion damps only *within* that null space, which is why increasing $k_v$ reduced the ripple at most ~3x.

**Why staggering removes it.** On the C-grid the divergence at a center is the compact two-point difference of the two adjacent faces, $(v_{i} - v_{i-1})/\Delta y$, and the meridional pressure gradient at a face is the compact difference of the two adjacent temperatures. Both stencils now see the $2\Delta y$ component, so grid-scale $v$ acquires the gravity-wave dispersion of the resolved dynamics rather than sitting in a null space. At ny=801 the standing interior ripple drops 24-91x by band onto the gateless noise floor, while the climate anchors (jet strength and position, equatorial temperature) move less than 0.02% and the notch physics is unchanged.

**Discretization and parity.** $v$ at a wall center is zero (the physical boundary); the divergence there uses the half-cell one-sided form consistent with that. The $k_v$ diffusion of $v$ is a three-point Laplacian on the faces with a mirror wall ghost (the image value $-v_0$ across the wall, so $v=0$ is enforced there), written in the symmetric association $(v_+ + v_-) - 2 v_c$. That association and a face coordinate defined as the exact midpoint $\tfrac12(y_i + y_{i+1})$ make every operator preserve mirror symmetry to the last floating-point bit: from a symmetric initial state the integration holds $\max|u(y)-u(-y)| = 0$ exactly, the same invariant the collocated model has. The EMFD, drag, Newtonian cooling, and Coriolis coupling ($v$ averaged to centers for the $u$ equation, $u$ averaged to faces for the $v$ equation) are unchanged in form.

The collocated layout remains available (`--grid collocated`) as the bit-exact reproduction path for the Zhang et al. (2025) lineage.

### 3.5 Passive Column Moisture (Moist V1)

With `--enable-moisture` the model carries a prognostic column water vapor $W(y,t)$ on the dry circulation (Version 1 of the moist extension; full derivation and parameter provenance in `guides/moist_axisymmetric_model_spec.pdf`):

$$\partial_t W + \partial_y\big[-(2a-1)\,v\,W - D\,\partial_y W\big] = E_0 - P, \qquad P = \frac{(W-W_c)^+}{\tau_c}$$

The mean transport velocity is $-(2a-1)v$: the lower branch of the overturning moves opposite to the upper-layer $v$, and with CWV fraction $a$ in the lower layer the column-integrated moisture flux reduces to this single coefficient. $-D\,\partial_y W$ is the downgradient eddy moisture flux (the ERA5-calibrated object of the moist program), $E_0$ uniform evaporation, and $P$ a quasi-equilibrium relaxation of $W$ to the critical value $W_c$ on the convective timescale $\tau_c$, the standard precipitating-shallow-water closure and the negative feedback that keeps $W$ bounded. In V1 the coupling is one-way: $W$ never enters the momentum or thermodynamic equations, so the dry solution is bit-for-bit that of a dry run (a tested invariant). The diagnostic eddy MSE flux is $-L_v D\,\partial_y W$, computed in analysis scripts.

Discretely, $W$ lives on the cell centers and its transport is finite-volume in flux form through the $ny-1$ interior faces where the staggered $v$ lives: face values of $W$ are MUSCL reconstructions with monotonized-central limited slopes upwinded on the transport velocity's sign (non-oscillatory at the ITCZ front the precipitation kink creates), wall fluxes are zero (half-cell divergence at the wall centers), and total water is conserved by transport to roundoff. Advection is stepped at the central leapfrog level; $D$ and $P$ are stepped at the lagged $n-1$ level, an effectively forward step that keeps both dissipative terms off the Asselin filter's stability budget. The transport operators preserve mirror parity exactly at any resolution, so on the odd-$n_y$ grid used at every production resolution (where the mesh is itself bit-antisymmetric) a symmetric run holds $\max|W(y)-W(-y)| = 0$ bit-for-bit, the same mirror symmetry the dry integration has. At even $n_y$ the linspace mesh is not bit-antisymmetric, so both dry and moist fields inherit a $\sim\!10^{-13}$ asymmetry.

### 3.6 Latent Heating Feedback (Moist V2)

`--enable-latent-heating` lets precipitation heat the column, so moisture feeds back on the circulation (Version 2 of the moist extension). The moisture equation of §3.5 is unchanged; the thermodynamic equation becomes

$$\partial_t\theta + \frac{d\,\Delta_z}{H}\,\partial_y v = \frac{\theta_E - \theta}{\tau} + \mathcal{L}P,$$

with $\theta_E$ the same relaxation profile V1 uses.

**Why the target does not change (Spencer, 2026-11-07).** The spec argues that V1's single relaxation bundles every diabatic term, convective heating included, so that adding $\mathcal{L}P$ releases the same latent heat twice unless the relaxation is retargeted to a steeper, purely radiative profile ($\Delta_\theta^{\rm rad} = 75$ to 100 K against V1's 50 K). With uniform evaporation that argument does not reach the gradient. Switch off every transport term and each column relaxes independently to $P = E_0$, $W = W_c + \tau_c E_0$ and $\theta = \theta_E + \tau\mathcal{L}E_0$, so the latent heating that would be counted twice raises the no-circulation temperature by the same $\tau\mathcal{L}E_0 = 71$ K at every latitude and adds no gradient for a steeper target to remove. The default therefore keeps $\theta_E$. The retargeted form remains available: `--delta-y-rad` rebuilds the relaxation profile, in the same forcing-profile family, at the stated contrast $\Delta_\theta^{\rm rad}$. A 75 K target was the default from 2026-07-30 to 2026-11-07; in the 2026-08-09 parameter sweep, on its positive-stability reference, the 75 K target gave a subtropical jet of 53.6 m/s, against 35.8 m/s at 50 K and 31.0 m/s in ERA5. A latitude-dependent $E_0$ would give the no-circulation latent heating a gradient, and with it a physical basis for a steeper target; the model has no such option.

**Why $\mathcal{L} = L_v/C$.** The column heat capacity $C = c_p\Pi\Delta p/g \approx 5.16\times10^6$ J m⁻² K⁻¹ converts a $\theta$ tendency to a column-energy tendency, and the same $C$ sets the gross dry stability $\hat{S} = C\,d\Delta_z/H$. Requiring the heating $C\mathcal{L}P$ to equal the latent release $L_vP$ gives $\mathcal{L} = L_v/C \approx 0.484$ K per kg m⁻² s⁻¹, so 10 mm/day of precipitation heats the column about 4.8 K/day. The consequence is the column MSE budget

$$\partial_t\langle h\rangle + \partial_y\big[v\hat{H} - L_vD\,\partial_y W\big] = \frac{C}{\tau}(\theta_E-\theta) + L_vE_0,$$

with $\langle h\rangle = C\theta + L_vW$ and gross moist stability $\hat{H} = \hat{S} - L_v(2a-1)W$: precipitation is internal and cancels, leaving the relaxation and the surface latent flux as the only sources. The discrete scheme keeps the cancellation exact, because the latent term reads the same lagged $P$ array the moisture sink uses and $\theta$ and $W$ share a leapfrog with the same $\Delta t$ and Asselin coefficient.

**Mean-state consequence.** Away from the circulation $v = 0$ and $P = E_0$, so the steady balance gives $\theta = \theta_E + \tau\mathcal{L}E_0$, a warming of 71 K at the default parameters. This is the drizzle-everywhere state of §3.5 acting on temperature: the whole domain precipitates at the evaporation rate, so the whole domain is heated at $\mathcal{L}E_0 \approx 1.9$ K/day and relaxes to a warmer equilibrium. Every $\theta$ term is invariant under $\theta \to \theta + c$ with $\theta_E \to \theta_E + c$, so the offset changes no dynamics, and lowering `--theta00` by 71 K gives realistic absolute temperatures (verified to $2\times10^{-13}$ relative in $u$). In the deep tropics the offset differs, since both $P$ and the adiabatic export $\frac{d\Delta_z}{H}\partial_y v$ are larger there; that difference is the feedback V2 exists to study. In the 2026-08-09 sweep, latent heating at the unchanged 50 K target doubled the equator-to-24-degree $\theta$ contrast relative to V1 (2.38 against 1.13 K).

**The regime parameter.** The sign of $\hat{H}$ for an undisturbed column is set by $r = W_q/W^*$, with $W_q = W_c + \tau_c E_0$ and $W^* = \hat{S}/(L_v(2a-1))$ the column water vapor at which $\hat{H}$ vanishes. Below $r \approx 0.94$ the circulation exports moist static energy from its ascending branch and V2 gives one broad Hadley cell; above it, ascent imports moist static energy, reinforces itself, and the rain aggregates into narrow bands separated by dry gaps (measured at the 75 K target; see the Moist V2 section of `CLAUDE.md`). The defaults ($a = 0.79$, $W_c = 40$ kg m$^{-2}$, 2026-11-07) give $r = 0.76$.

**Two readings of $v$ in the moist equations.** The moisture transport $-(2a-1)vW$ and the factor of 2 in the $v$ equation treat $v$ as the velocity of two branches, each containing half the column mass. The thermodynamic term $\frac{d\,\Delta_z}{H}\partial_y v$ treats it as the velocity of a slab of depth $d$ at the top and another at the bottom, with $v = 0$ between them. The two velocities are related by $v_{\rm slab}/v_{\rm half} = (p_s/2)/\delta p$, with $\delta p$ the slab's pressure depth, and every coefficient defined as a transport per unit $v$ differs by that factor between the readings: 2.55 at the observed tropical $p_s = 997$ hPa and a slab depth near 195 hPa. The 2026-08-01 ERA5 calibration (in the private `trop-eddies-swm` project) showed that $W^*$ and $\hat{H}/\hat{S}$ take the same value under either reading, so they are the quantities to compare with observations and no model result depends on the choice. The calibration left open whether to restate $d$ as a pressure depth, so the model's stated assumptions match what it computes.

The vertical-advection gate switches to SS09's kinematic form for this version, for the reason given at the end of §3.3.

### 3.7 Spherical Geometry

`--geometry sphere` (`SWConfig(geometry="sphere")`) replaces the equatorial beta plane with the sphere, from pole to pole. The plan behind it, with the derivation of each term, is `docs/spherical-swm-plan/spherical-swm-plan.pdf`. Every beta-plane run is bit-for-bit what it was before the option existed: the geometry enters the operators only through arrays (`ss09/geometry.py`) that are exactly 1.0 on the beta plane, where $f$ is $\beta y$.

**Coordinate and grid.** The meridional coordinate is $y = a\sin\varphi$, for planetary radius $a$ and latitude $\varphi$, on a grid uniform in $y$ from $y = -a$ to $y = +a$, so cells of equal width in $y$ have equal area. The meridional distance between neighboring centers is $\Delta y/\cos\varphi$. Writing $c = \cos\varphi = (1 - y^2/a^2)^{1/2}$, the relative vorticity is $\zeta = -\partial_y(cu)$, the divergence is $\partial_y(cv)$, a derivative along the meridian is $c\,\partial_y$, and $u\tan\varphi/a = yu/(a^2c)$. The Coriolis parameter is exactly $f = 2\Omega\sin\varphi = (2\Omega/a)\,y$. The defaults are Earth's mean radius, 6371 km, and the WGS84 rotation rate, $7.292115\times10^{-5}$ s$^{-1}$, one turn per sidereal day. `ny` must be odd, so that a center sits on the equator, and `--ny 325` gives the production grid spacing, 39.3 km at the equator.

**Equations**, as implemented in `StaggeredSWModel`:

$$\partial_t u - v\big[f - \partial_y(cu)\big] = -\mathcal{G}\,u\,\partial_y(cv) - \epsilon_u u - \mathcal{S}$$

$$2\,\partial_t v + \Big(f + \frac{u\tan\varphi}{a}\Big)u = -\frac{gH}{T_0}\,c\,\partial_y T + k_v\,c\,\partial_y\big[\partial_y(cv)\big]$$

$$\partial_t\theta + \frac{\delta\Delta_z}{H}\,\partial_y(cv) = \frac{\theta_E - \theta}{\tau} + \kappa_\theta\,\partial_y\big(c^2\,\partial_y\theta\big)$$

with $\mathcal{G}$ either gate of §3.3, unchanged. Setting $c = 1$, dropping the metric term $u^2\tan\varphi/a$ and writing $f = \beta y$ recovers §2. Most of these forms are fixed by physics, given SS09's two-layer assumptions: the Coriolis and advection terms are the meridional flux of absolute vorticity, the vertical advection assumes that air rising into the upper layer has the lower layer's angular momentum, which is zero relative angular momentum, the metric and pressure-gradient terms follow from Xian and Miller's (2008) two-layer derivation, and the adiabatic and eddy-heat terms are the spherical divergence and the diffusion of a scalar. Two forms are choices, both Spencer's of 2026-11-07:

- **The eddy momentum flux divergence** is $\mathcal{S} = v_d\,\mathcal{H}(u)\,\mathrm{sgn}(y)\,c^{-1}\partial_y(c^2u)$, the spherical divergence of the eddy angular-momentum flux $\overline{u'v'} = v_d\,\mathcal{H}(u)\,\mathrm{sgn}(y)\,u$ whose beta-plane divergence is SS09's term. Of the three spherical readings of SS09's empirical closure it is the only one that conserves angular momentum: $c\,\mathcal{S}$ is an exact derivative in $y$, so the eddy torque over the westerlies reduces to boundary values. For a uniform-Rossby-number wind it is $1 - \tan^2\varphi/2$ times the reading that keeps the closure $\mathrm{Ro} = v/(v + v_d)$ exact, 17% smaller at 30 degrees. The other two readings are options, selected by `--emfd-form` (sphere only, added 2026-11-08): `vorticity`, $v_d\,\mathcal{H}(u)\,\mathrm{sgn}(y)\,\partial_y(cu)$, the reading that keeps $\mathrm{Ro} = v/(v + v_d)$ exact; and `literal`, $v_d\,\mathcal{H}(u)\,\mathrm{sgn}(y)\,c\,\partial_y u$, SS09's term with the derivative along the meridian. The default, `angular_momentum`, is the form above. The stencil acts on $c^2u$, $cu$ and $u$ respectively.
- **The smoothing of $v$** is the meridional component of the vector Laplacian, $k_v[\nabla^2 v - v/(a^2\cos^2\varphi)]$, which for an axisymmetric field is the gradient of the divergence. The smoothing's discrete form is the face gradient of the discrete divergence, and summed over the faces $v$ times it is minus the equal-area sum of the squared divergence, so it never raises $\sum v^2$.

**Forcing.** The spherical profile (`--theta-e-type sphere`, §4.4) is the only one allowed on the sphere, and `--lat-0` sets its heating maximum in degrees; `--y0`, `--y1`, `--beta` and `--domain-size` are refused there.

**Discretization.** The scheme is the beta plane's: the staggered grid, leapfrog stepping with the Asselin filter, first-order upwind advection (now of $cu$), the MC stencil of the eddy term (now applied to $c^2u$ in the default form), and both Heaviside gates. The outermost centers are the poles, where $u = 0$: that is the regularity condition, since $\zeta$ contains $u\tan\varphi/a$, and it is the wall condition the model already imposes. $v$ stays on the interior faces, and since $c = 0$ at the poles no flux crosses them. The reciprocal $1/c$ is set to zero at the two pole centers, whose $u$ tendency the condition $u = 0$ discards. The grid is built as $a\,(k/m)$ for the integers $k = -m,\dots,m$, which is exactly antisymmetric, where `np.linspace(-a, a, ny)` is not at these sizes; with it, a hemispherically symmetric forcing keeps $u$ and $\theta$ exactly symmetric and $v$ exactly antisymmetric, bit for bit, as on the beta plane.

**Accuracy, measured 2026-11-08** on regular axisymmetric test fields (`ss09/tests/test_sphere_model.py`). Within 60 degrees of the equator every operator converges at its design order: second order, and first for the upwind advection. Near the poles, the operators acting on fields that are odd in colatitude (the face-to-center average of $v$, the eddy term's derivative of $c^2u$) and the half-cell divergence at the poles lose accuracy. Over the whole domain the divergence converges at first order, the smoothing and the face-to-center average of $v$ at half order, and the eddy term's error in the outermost interior cells stays near 0.9% of its maximum without shrinking as the grid is refined. The angular-momentum-conserving wind is steady up to the upwind truncation, and the steady $v$ equation is the full-angle gradient-wind balance to second order.

**The polar westerly jet (measured 2026-11-08, unresolved).** With the eddy term in the default form, the spherical steady state has westerlies increasing poleward to a jet near 75 degrees (53 m/s at $\Delta_y = 50$ K), which ends one grid cell poleward in an easterly notch (-42 m/s) with $\theta$ 20 K below $\theta_E$. The beta-plane twin with the same forcing in latitude has its westerly maximum at 52 degrees (24 m/s). The three forms differ by one term: since $\partial_y c = -\tan\varphi/a$, each is $v_d\,\mathcal{H}(u)\,\mathrm{sgn}(y)\,(c\,\partial_y u - n\,u\tan\varphi/a)$, with $n = 2$, 1 and 0 for the angular-momentum, vorticity and literal forms. In the $u$ equation the second part is a westerly acceleration at the rate $n\,v_d|\tan\varphi|/a$ in both hemispheres, which at the default $v_d = 2.5$ m/s is 112 to 216 times the drag rate $\epsilon_u$ between 55 and 70 degrees for $n = 2$; there it is the largest part of the eddy term, and the Coriolis force on an equatorward flow balances the eddy term. Halving the grid spacing (325 to 649 points) moves the default form's jet from 75.73 to 73.10 degrees, and within 40 degrees of the equator the sphere's coarse-to-fine changes of $u$, $v$ and $\theta$ are then 3.27, 4.73 and 5.98 times the twin's. The vorticity form has a plateau of westerlies to 78 degrees whose maximum and notch also move with resolution, and its ratios are 3.19, 3.83 and 3.59. The literal form ($n = 0$) has its westerly maximum at 55.18 degrees on both grids, and its ratios are 0.16, 0.24 and 0.17. With $v_d = 0$ the sphere's ratios are 1.03, 0.94 and 1.00. Whether to keep the angular-momentum-conserving form, adopt the literal form, or confine the closure away from the poles is Spencer's open decision; the Spherical Geometry section of `CLAUDE.md` gives the numbers.

**Not yet supported on the sphere:** moisture (moist V1 and V2), seasonal forcing and the collocated grid, each refused with an error. The spherical moisture flux, $\partial_y\{c[-(2a-1)vW - D\,c\,\partial_y W]\}$, is implemented and tested for exact water conservation and second-order convergence, but no moist run on the sphere has been validated. The numba backend supports the sphere, bitwise identical to the numpy reference.

## 4. Equilibrium Temperature Profiles ($\theta_E$)

### 4.1 SS09 Profile (Parabolic)

$$\theta_E = \begin{cases} \theta_{00} - \Delta_y (y/y_1)^2 & |y| < y_1 \\ \theta_{00} - \Delta_y & |y| \geq y_1 \end{cases}$$

Simple parabolic profile from the original SS09 paper. Has a discontinuous derivative at $y = y_1$.

### 4.2 Sin² Profile

$$\theta_E = \theta_{00} - \Delta_y \sin^2\left(\frac{\pi y}{2 y_1}\right), \quad |y| < y_1$$

Smoother than SS09; avoids discontinuities at the Hadley cell edge. Default in this implementation.

### 4.3 SB08 Profile (Seasonal Forcing)

$$\theta_E = \theta_{00} - \Delta_y \left[ \sin^2\left(\frac{\pi y}{2 y_1}\right) - 2 \sin\left(\frac{\pi y_0}{2 y_1}\right) \sin\left(\frac{\pi y}{2 y_1}\right) \right]$$

Adds an off-equator term via $y_0$, the latitude of maximum RCE temperature: with the minus sign on the cross term as implemented here, the $\theta_E$ maximum sits at $y = +y_0$. (SS09's printed Eq. (3.2) has a plus sign, which places the maximum at $y = -y_0$ and contradicts their own solstitial figures, whose winter jet lies opposite the stated $y_0$; the minus sign here makes positive $y_0$ put the ITCZ in the northern hemisphere.) This enables:
- **Equinoctial conditions** ($y_0 = 0$): symmetric two-cell Hadley circulation
- **Solstitial/monsoon conditions** ($y_0 \neq 0$): asymmetric single dominant cell

Seasonal cycles are imposed by varying $y_0$ sinusoidally in time.

### 4.4 Spherical Profile (geometry="sphere")

$$\theta_E = \theta_{00} - \Delta_y\left(\sin^2\varphi - 2\sin\varphi_0\sin\varphi\right) = \theta_{00} - \Delta_y\,\frac{y^2 - 2y_0y}{a^2}, \qquad y_0 = a\sin\varphi_0$$

The profile of Schneider and Bordoni's (2008) idealized GCM, with the heating maximum at latitude $\varphi_0$ (`--lat-0`) and $\Delta_y$ the equator-to-pole drop for $\varphi_0 = 0$. The three beta-plane profiles are renderings of it: the sin² and SB08 profiles replace $\sin\varphi$ with $\sin(\pi y/2y_1)$, so that $y_1$ stands in for the pole, and SS09's parabola is its small-angle form. In the spherical model's coordinate it is SS09's parabola with $y_1 = a$, exact on the sphere and with no plateau, since $y$ ends at $\pm a$; $y_1$ and the beta-plane offset $y_0$ have no other spherical counterpart and are refused.

## 5. Dynamical Regimes

The **local Rossby number** quantifies the balance between relative and planetary vorticity:

$$\text{Ro} = \frac{-\zeta}{\beta y} = \frac{\partial u / \partial y}{\beta y}$$

### 5.1 Angular Momentum Conserving (AMC) Regime

**Ro ≈ 1** throughout the Hadley cell.

- Zonal wind follows: $u \approx \frac{1}{2} \beta y^2$
- Temperature: $T(y) \sim C_0 - C_4 y^4$
- Meridional circulation determined by thermal forcing alone
- **Nonlinear dynamics**: advection terms dominate momentum budget

### 5.2 Eddy-Dominated Regime

**Ro → 0** (especially near equator).

- Zonal wind follows: $u \sim y^3$ near the equator (cubic, not quadratic!)
- Temperature: $T(y) \sim C_0 - C_5 y^5$
- EMFD balances Coriolis acceleration: $\beta y v \approx v_d \partial u / \partial y$
- **Linear dynamics**: eddy drag dominates over nonlinear advection

## 6. Expected Model Behavior

*The following assumes Earth-like, default parameter values.*

### Steady-State Structure
- **Subtropical jets**: Westerly maxima at Hadley cell edges (~30° equivalent latitude)
- **Equatorial easterlies**: Arise only when ascent is off-equator ($y_0 \neq 0$); with symmetric forcing ($y_0 = 0$), equatorial winds are near zero
- **Temperature**: Flatter than RCE profile within the Hadley cell due to advection

### Parameter Sensitivities
| Change | Effect |
|--------|--------|
| Increase $v_d$ (stronger eddies) | Weaker zonal winds, stronger $v$, wider Hadley cell, lower Ro |
| Increase $y_0$ (off-equator forcing) | Stronger winter jet, weaker summer jet, transition toward AMC |
| Increase $\Delta_y$ (larger equator-pole gradient) | Stronger circulation overall |

## 7. Key Parameters

| Symbol | Default Value | Physical Meaning |
|--------|---------------|------------------|
| $\tau$ | 37 days | Thermal relaxation timescale |
| $H$ | 16 km | Tropopause height |
| $\delta$ | 4 km | Upper-layer thickness |
| $T_0$ | 300 K | Reference surface temperature |
| $\Delta_z$ | 60 K | Vertical potential temperature difference (static stability) |
| $\Delta_y$ | 50 K | Equator-to-pole RCE temperature difference |
| $\theta_{00}$ | 330 K | Background mean potential temperature |
| $\beta$ | 2×10⁻¹¹ m⁻¹s⁻¹ | Meridional gradient of Coriolis parameter |
| $v_d$ | 2.5 m/s | EMFD coefficient |
| $k_v$ | 778,600 m²/s | Eddy viscosity on $v$; 100× SS09's published 7786 m²/s, following the Zhang et al. (2025) code. Effective diffusivity is $k_v/2$ from the $v$ equation's factor of 2 |
| $\epsilon_u$ | 10⁻⁸ s⁻¹ | Rayleigh drag coefficient |
| $y_1$ | 9439 km | Half-width of RCE profile (~85° latitude equivalent) |

### Unit Conversions
- $\theta$ to $T$: $T = \theta (p_t / p_s)^{R/c_p}$ with $(p_s/p_t)^{R/c_p} \approx 1.6$
- Model uses meters for $y$; 1° latitude ≈ 111 km

## 8. References

**Primary**:
- Sobel, A. H. & Schneider, T. (2009). Single-layer axisymmetric model for a Hadley circulation with parameterized eddy momentum forcing. *J. Adv. Model. Earth Syst.*, 1, 10. doi:10.3894/JAMES.2009.1.10
- Sobel, A. H. & Schneider, T. (2013). Correction to "Single-layer axisymmetric model for a Hadley circulation with parameterized eddy momentum forcing". *J. Adv. Model. Earth Syst.*, 5, 654-657. doi:10.1002/jame.20030

**Extensions**:
- Zhang, P., Lutsko, N. J., Hill, S. A., & Xie, S.-P. (2025). Hadley Cell Dynamics in an Axisymmetric Single-Layer Model: Effects of Parameterized Eddies and Equatorial Heating. *J. Atmos. Sci.*, 82, 2757-2770. doi:10.1175/JAS-D-25-0076.1

**Background**:
- Held, I. M. & Hou, A. Y. (1980). Nonlinear axially symmetric circulations in a nearly inviscid atmosphere. *J. Atmos. Sci.*, 37, 515-533.
- Schneider, T. & Bordoni, S. (2008). Eddy-mediated regime transitions in the seasonal cycle of a Hadley circulation. *J. Atmos. Sci.*, 65, 915-934.
- Xian, P. & Miller, R. L. (2008). Abrupt seasonal migration of the ITCZ into the summer hemisphere. *J. Atmos. Sci.*, 65, 1878-1895.

**Vertical momentum exchange parameterizations** (referenced in §3.3):
- Esler, J. G., Polvani, L. M. & Plumb, R. A. (2000). The effect of a Hadley circulation on the propagation and reflection of planetary waves in a simple one-layer model. *J. Atmos. Sci.*, 57, 1536-1556. doi:10.1175/1520-0469(2000)057<1536:TEOAHC>2.0.CO;2
- Shell, K. M. & Held, I. M. (2004). Abrupt transition to strong superrotation in an axisymmetric model of the upper troposphere. *J. Atmos. Sci.*, 61, 2928-2935. doi:10.1175/JAS-3312.1
- Adam, O. & Paldor, N. (2009). Global circulation in an axially symmetric shallow water model forced by equinoctial differential heating. *J. Atmos. Sci.*, 66, 1418-1433. doi:10.1175/2008JAS2685.1
