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

$$\frac{\partial u}{\partial t} - v(\beta y - \frac{\partial u}{\partial y}) = -\mathcal{H}(\theta_E - \theta) \cdot (\frac{\partial v}{\partial y}) \cdot u - \mathcal{F} - \mathcal{S} + k_u \frac{\partial^2 u}{\partial y^2}$$

| Term | Expression | Physical Meaning |
|------|------------|------------------|
| Tendency | $\partial u / \partial t$ | Rate of change of zonal wind |
| Coriolis + advection | $v(\beta y - \partial u / \partial y)$ | Acceleration from planetary vorticity and relative vorticity advection |
| Vertical advection | $-\mathcal{H}(\theta_E - \theta)(\partial_y v) u$ | Momentum exchange with lower layer; only active where atmosphere is cooler than equilibrium ($\theta_E > \theta$). Assumes ascending air carries **zero zonal momentum**. |
| Rayleigh drag | $\mathcal{F} = \epsilon_u u$ | Linear damping; ensures numerical stability, no direct atmospheric analog |
| EMFD | $\mathcal{S}$ | Eddy momentum flux divergence (see §3.1) |
| Momentum diffusion | $k_u \partial^2 u / \partial y^2$ | Eddy viscosity on $u$ (analog of $k_v$ on $v$); damps the EMFD-driven equatorial superrotation. See §3.4 and the numerics note in §8. |

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

2. **Temperature vs. potential temperature bug**: In the original numerical implementation (not the written equations), the RHS of the meridional momentum equation incorrectly used temperature $T$ where it should have used potential temperature $\theta$. This quantitatively (but not qualitatively) affected several figures in the original paper.

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

**Equatorial superrotation.** Because $\mathcal{S} \propto \text{sgn}(y)\,\partial_y u$, the EMFD is an *up-gradient* (anti-diffusive) momentum flux near a westerly maximum: once a small westerly forms at the equator ($u>0$ so $\mathcal{H}(u)=1$), the EMFD reinforces it. Left unchecked this drives a slow equatorial superrotation. The original leapfrog/Asselin scheme suppressed it through implicit numerical damping; the steady-state version of this growth is the long-documented "spurious momentum source." The current scheme controls it with the explicit momentum diffusion $k_u$ (§3.4).

### 3.2 Rayleigh Drag

$$\mathcal{F} = \epsilon_u u$$

A weak linear drag ($\epsilon_u \sim 10^{-8}$ s⁻¹, ~1000-day timescale) ensures the model has a stable steady state. It has no clear physical analog in the real atmosphere but is numerically necessary.

### 3.3 Vertical Momentum Advection

The term $-\mathcal{H}(\theta_E - \theta)(\partial_y v) u$ represents exchange between the upper layer and lower troposphere. Two limiting assumptions are possible:
1. **Ascending air carries zero zonal momentum** (default): creates the term shown
2. **Ascending air carries the upper-layer zonal momentum**: eliminates this term entirely

The model uses assumption (1). Sensitivity to this choice is modest compared to EMFD effects.

**Activation condition**: The Heaviside function $\mathcal{H}(\theta_E - \theta)$ uses a thermodynamic criterion rather than a kinematic one. Vertical momentum exchange is active where the atmosphere is cooler than radiative-convective equilibrium ($\theta_E > \theta$), which indicates a convective tendency. When the atmosphere is at or above equilibrium ($\theta \geq \theta_E$), no convective mixing occurs and the term is inactive.

### 3.4 Momentum Diffusion

$$k_u \frac{\partial^2 u}{\partial y^2}$$

An eddy viscosity on $u$, the direct analog of $k_v$ on $v$. It supplies the only scale-selective dissipation acting on the zonal wind and so sets the steady-state momentum balance against the up-gradient EMFD (§3.1). The original model had no explicit $\partial^2 u$ term; its zonal-wind dissipation came entirely from the implicit damping of the leapfrog/Asselin time scheme. The current self-starting RK4 scheme has no such implicit damping, so $k_u$ makes that dissipation explicit and tunable. The default $k_u = 10^5$ m²/s is calibrated to reproduce the original steady climate (symmetric subtropical jet ≈ 28 m/s; off-equatorial $y_0 = 1000$ km winter jet ≈ 41 m/s). Setting $k_u = 0$ recovers the undamped equations, in which the EMFD superrotation grows.

## 4. Equilibrium Temperature Profiles ($\theta_E$)

### 4.1 SS09 Profile (Parabolic)

$$\theta_E = \begin{cases} \theta_{00} - \Delta_y (y/y_1)^2 & |y| < y_1 \\ \theta_{00} - \Delta_y & |y| \geq y_1 \end{cases}$$

Simple parabolic profile from the original SS09 paper. Has a discontinuous derivative at $y = y_1$.

### 4.2 Sin² Profile

$$\theta_E = \theta_{00} - \Delta_y \sin^2\left(\frac{\pi y}{2 y_1}\right), \quad |y| < y_1$$

Smoother than SS09; avoids discontinuities at the Hadley cell edge. Default in this implementation.

### 4.3 SB08 Profile (Seasonal Forcing)

$$\theta_E = \theta_{00} - \Delta_y \left[ \sin^2\left(\frac{\pi y}{2 y_1}\right) + 2 \sin\left(\frac{\pi y_0}{2 y_1}\right) \sin\left(\frac{\pi y}{2 y_1}\right) \right]$$

Adds an off-equator term via $y_0$, the latitude of maximum RCE temperature. This enables:
- **Equinoctial conditions** ($y_0 = 0$): symmetric two-cell Hadley circulation
- **Solstitial/monsoon conditions** ($y_0 \neq 0$): asymmetric single dominant cell

Seasonal cycles are imposed by varying $y_0$ sinusoidally in time.

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
| $k_v$ | 7786 m²/s | Eddy viscosity on $v$ |
| $k_u$ | 10⁵ m²/s | Eddy viscosity on $u$ (momentum diffusion, §3.4) |
| $\epsilon_u$ | 10⁻⁸ s⁻¹ | Rayleigh drag coefficient |
| $y_1$ | 9439 km | Half-width of RCE profile (~85° latitude equivalent) |

### Unit Conversions
- $\theta$ to $T$: $T = \theta (p_t / p_s)^{R/c_p}$ with $(p_s/p_t)^{R/c_p} \approx 1.6$
- Model uses meters for $y$; 1° latitude ≈ 111 km

## 8. Numerical Methods

The model integrates on a one-dimensional **staggered Arakawa C-grid** with a self-starting **fixed-step RK4** time integrator.

- **Grid**: $N$ cell centers carry $u$ and $\theta$; $N+1$ cell faces carry $v$, with $v = 0$ on the two wall faces. $N$ even places a face at the equator (so an odd $v$ vanishes there exactly) and no center on it. Default $N = 50$, $dy = $ domain $/ N$.
- **Spatial operators**: single adjacent differences (centers→faces for $\partial_y\theta$; faces→centers for $\partial_y v$), so the $v$-$\theta$ gravity-wave couple has no $2\,dy$ null space. The meridional advection $-v\,\partial_y u$ uses upwind differencing for stability; $u$ at the walls is held at zero (Dirichlet).
- **Time stepping**: RK4 needs no Asselin filter and no leapfrog $n{-}1$ level, so a restart stores a single instantaneous state.

This replaces the earlier collocated grid with leapfrog/Asselin time stepping, which supported an undamped $2\,dy$ computational mode and went unstable off the equator (constant $y_0 > 0$ from rest reached NaN within a few steps at $dt = 3600$ s). The staggered/RK4 scheme runs that regime stably at $dt = 3600$ s. The Asselin filter had also been the only damping on $u$; the explicit $k_u$ (§3.4) now provides it.

## 9. References

**Primary**:
- Sobel, A. H. & Schneider, T. (2009). Single-layer axisymmetric model for a Hadley circulation with parameterized eddy momentum forcing. *J. Adv. Model. Earth Syst.*, 1, 10. doi:10.3894/JAMES.2009.1.10
- Sobel, A. H. & Schneider, T. (2013). Correction to "Single-layer axisymmetric model for a Hadley circulation with parameterized eddy momentum forcing". *J. Adv. Model. Earth Syst.*, 5, 654-657. doi:10.1002/jame.20030

**Extensions**:
- Zhang, P., Lutsko, N. J., Hill, S. A., & Xie, S.-P. (2025). Hadley Cell Dynamics in an Axisymmetric Single-Layer Model: Effects of Parameterized Eddies and Equatorial Heating. *J. Atmos. Sci.*, 82, 2757-2770. doi:10.1175/JAS-D-25-0076.1

**Background**:
- Held, I. M. & Hou, A. Y. (1980). Nonlinear axially symmetric circulations in a nearly inviscid atmosphere. *J. Atmos. Sci.*, 37, 515-533.
- Schneider, T. & Bordoni, S. (2008). Eddy-mediated regime transitions in the seasonal cycle of a Hadley circulation. *J. Atmos. Sci.*, 65, 915-934.
- Xian, P. & Miller, R. L. (2008). Abrupt seasonal migration of the ITCZ into the summer hemisphere. *J. Atmos. Sci.*, 65, 1878-1895.
