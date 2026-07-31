"""Column thermodynamic constants for the moist extension.

These pin the conversion between a potential-temperature tendency and a
column-energy tendency, which is what makes the moist static energy budget
close: the same column heat capacity C sets the gross dry stability
Shat = C d Delta_z / H (the mean DSE flux is Shat v) and the latent-heating
coefficient Lambda = L_v / C. Consistency requires C Lambda = L_v exactly,
so that the latent release Lambda P added to the theta equation and the
sink P removed from the moisture equation cancel term-by-term in the column
MSE budget (guides/moist_axisymmetric_model_spec.pdf Eqs. Cdef, Lconv, mse2).

The Exner factor is SS13's Pi = (p_t/p_s)^(R/c_p) = 1/1.6, the same number
the dynamics use as sw_model.THETA_TO_TEMP (asserted by a test). It fixes the
tropopause pressure at p_t = p_s Pi^(c_p/R) = 193 hPa and hence the layer
thickness Delta_p = 807 hPa. Note that H = 16 km is a dynamical length scale
elsewhere in the model and is NOT hydrostatically tied to p_t here, so no
consistency between the two is required (spec item 2).
"""

C_P = 1004.0  # J kg^-1 K^-1, dry-air specific heat at constant pressure
R_OVER_CP = 2.0 / 7.0
EXNER = 1.0 / 1.6  # Pi = (p_t/p_s)^(R/c_p), SS13; == sw_model.THETA_TO_TEMP
P_SURF = 1.0e5  # Pa
P_TOP = P_SURF * EXNER ** (1.0 / R_OVER_CP)  # Pa, ~193 hPa
DELTA_P = P_SURF - P_TOP  # Pa, ~807 hPa
L_V = 2.5e6  # J kg^-1, latent heat of vaporization


def column_heat_capacity(gravity: float) -> float:
    """C = c_p Pi Delta_p / g, the column heat capacity converting a theta
    tendency to a column-energy tendency (~5.2e6 J m^-2 K^-1)."""
    return C_P * EXNER * DELTA_P / gravity


def latent_heating_coeff(gravity: float) -> float:
    """Lambda = L_v / C, the precipitation-to-theta-tendency conversion
    (~0.48 K per kg m^-2 s^-1, i.e. 10 mm/day of precipitation heats the
    column ~4.8 K/day)."""
    return L_V / column_heat_capacity(gravity)


def gross_dry_stability(gravity: float, delta: float, delta_z: float,
                        height: float) -> float:
    """Shat = C d Delta_z / H (~7.7e7 J m^-2): the mean DSE flux is Shat v,
    and the gross moist stability is Hhat = Shat - L_v (2a-1) W."""
    return column_heat_capacity(gravity) * delta * delta_z / height
