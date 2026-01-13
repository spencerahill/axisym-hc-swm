# Effective Seasonal Response with Thermal Inertia

## Introduction

This document derives the **effective seasonal cycle** that the atmospheric circulation responds to when the equilibrium potential temperature profile $\theta_e$ varies seasonally, accounting for thermal inertia from Newtonian cooling.

### Physical Setup

The potential temperature equation in the shallow water model includes Newtonian relaxation to an equilibrium profile:

$$
\frac{\partial \theta}{\partial t} + \text{advection terms} = -\frac{\theta - \theta_e}{\tau}
$$

where:
- $\theta(y,t)$ is the actual potential temperature
- $\theta_e(y,t)$ is the equilibrium potential temperature profile
- $\tau$ is the Newtonian cooling timescale (thermal damping time)

For the Schneider & Bordoni (2008) profile with seasonal forcing, the ITCZ location varies as:

$$
y_0(t) = y_{00} + A \sin\left(\frac{2\pi t}{T_{\text{seasonal}}}\right)
$$

where:
- $y_{00}$ is the time-mean ITCZ latitude
- $A$ is the seasonal migration amplitude
- $T_{\text{seasonal}}$ is the seasonal period (e.g., 360 days)

## Competing Timescales

Two timescales determine the system response:

1. **Forcing timescale**: $T_{\text{seasonal}}$ — how fast the forcing varies
2. **Thermal inertia timescale**: $\tau$ — how fast the atmosphere forgets its past state

Define the dimensionless parameter:

$$
\varepsilon = \frac{\tau}{T_{\text{seasonal}}}
$$

This ratio controls whether the circulation can "keep up" with the seasonal forcing.

## Three Response Regimes

### 1. Fast Response Limit: $\varepsilon \ll 1$ (or $\tau \ll T_{\text{seasonal}}$)

**Physical meaning**: Thermal relaxation is rapid compared to seasonal variations. The atmosphere adjusts quickly and "forgets" previous states.

**Response**: $\theta \approx \theta_e(t)$ nearly instantaneously. The circulation follows the forcing with negligible lag.

**Effective forcing**:
$$
y_{0,\text{eff}}(t) \approx y_0(t)
$$
No damping or phase lag.

---

### 2. Slow Response Limit: $\varepsilon \gg 1$ (or $\tau \gg T_{\text{seasonal}}$)

**Physical meaning**: Thermal inertia is huge. The system cannot adjust before the forcing reverses direction.

**Response**: The atmosphere time-averages over the seasonal cycle, filtering out rapid oscillations.

**Effective forcing**:
$$
y_{0,\text{eff}} \approx y_{00}
$$
Seasonal amplitude is strongly damped; only the time-mean remains.

---

### 3. Intermediate Regime: $\varepsilon \sim 1$

**Physical meaning**: Partial thermal inertia. The system responds but with both damping and time lag.

## General Analytical Solution

Consider the linearized response near equilibrium. If the forcing is:

$$
\theta_e(t) = \bar{\theta}_e + \Delta\theta_e \sin(\omega t)
$$

where $\omega = 2\pi / T_{\text{seasonal}}$ is the seasonal frequency, the steady periodic solution is:

$$
\theta(t) = \bar{\theta} + \Delta\theta \sin(\omega t - \varphi)
$$

Substituting into the Newtonian cooling equation (neglecting advection):

$$
\frac{\partial \theta}{\partial t} = -\frac{\theta - \theta_e}{\tau}
$$

yields:

$$
\omega \Delta\theta \cos(\omega t - \varphi) = -\frac{\Delta\theta \sin(\omega t - \varphi) - \Delta\theta_e \sin(\omega t)}{\tau}
$$

Solving for the amplitude ratio and phase lag:

**Amplitude ratio (damping factor)**:
$$
\frac{\Delta\theta}{\Delta\theta_e} = \frac{1}{\sqrt{1 + \omega^2 \tau^2}}
$$

**Phase lag**:
$$
\varphi = \arctan(\omega \tau)
$$

## Physical Interpretation

### Damping

The response amplitude is reduced by the factor $1/\sqrt{1 + \omega^2 \tau^2}$:

- When $\omega\tau \ll 1$ (fast response): $\Delta\theta / \Delta\theta_e \approx 1$ — **full amplitude**, no damping
- When $\omega\tau \gg 1$ (slow response): $\Delta\theta / \Delta\theta_e \approx 1/(\omega\tau) \propto 1/\tau$ — **strong damping**

### Phase Lag

The response lags the forcing by $\varphi = \arctan(\omega\tau)$:

- When $\omega\tau \ll 1$: $\varphi \approx \omega\tau \approx 0$ — **negligible lag**
- When $\omega\tau \gg 1$: $\varphi \to \pi/2$ — **quarter-cycle lag** (90° phase shift)

## Effective Seasonal Cycle

The **effective** ITCZ position that the circulation actually responds to is:

$$
y_{0,\text{eff}}(t) = y_{00} + \frac{A}{\sqrt{1 + \omega^2 \tau^2}} \sin\left(\omega t - \arctan(\omega\tau)\right)
$$

**Key features**:

1. **Reduced amplitude**:
   $$
   A_{\text{eff}} = \frac{A}{\sqrt{1 + \omega^2 \tau^2}} < A
   $$

2. **Phase lag**: The response lags the forcing by $\varphi = \arctan(\omega\tau)$

3. **Limiting behavior**:
   - As $\tau \to 0$: $A_{\text{eff}} \to A$ and $\varphi \to 0$ (perfect following)
   - As $\tau \to \infty$: $A_{\text{eff}} \to 0$ and $\varphi \to \pi/2$ (no response)

## Numerical Example

For typical parameters in shallow water Hadley cell models:

- $T_{\text{seasonal}} = 360$ days $= 3.11 \times 10^7$ s
- $\tau = 20$ days $= 1.73 \times 10^6$ s
- $\omega\tau = 2\pi (20/360) \approx 0.35$

Then:

**Damping factor**:
$$
\frac{1}{\sqrt{1 + 0.35^2}} \approx 0.94
$$
The response amplitude is **94% of the forcing amplitude** (6% reduction).

**Phase lag**:
$$
\varphi = \arctan(0.35) \approx 19° \approx 19 \text{ days}
$$
The circulation lags the forcing by approximately **19 days**.

## Conclusion

For realistic Newtonian cooling timescales ($\tau \sim 10$–$30$ days) and seasonal periods ($T_{\text{seasonal}} \sim 360$ days), thermal inertia causes:

1. **Modest amplitude reduction** (~5–10%)
2. **Phase lag** of ~10–20 days

The circulation does not respond to the imposed $y_0(t)$ directly, but rather to a damped and lagged effective seasonal cycle $y_{0,\text{eff}}(t)$.

---

**Symbol Summary**:
- $\theta$: potential temperature
- $\theta_e$: equilibrium potential temperature
- $\tau$: Newtonian cooling timescale
- $y_0(t)$: ITCZ latitude (imposed forcing)
- $y_{0,\text{eff}}(t)$: effective ITCZ latitude (what circulation responds to)
- $A$: seasonal migration amplitude
- $T_{\text{seasonal}}$: seasonal period
- $\omega = 2\pi/T_{\text{seasonal}}$: seasonal frequency
- $\varepsilon = \tau/T_{\text{seasonal}}$: dimensionless timescale ratio
- $\varphi$: phase lag between response and forcing
