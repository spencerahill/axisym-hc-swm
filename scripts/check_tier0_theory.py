# /// script
# requires-python = ">=3.10"
# dependencies = ["sympy"]
# ///
"""Symbolic verification of the Tier-0 fixed-Ro beta-plane memo.

Each check() call asserts one [derived] claim from
docs/fixed_ro_beta_plane_theory.tex. sympy proves the algebra exactly
(rational arithmetic, no floating point); a FAIL means the derivation,
or its transcription here, is wrong.

Run either way:

    uv run scripts/check_tier0_theory.py      # isolated env from the
                                              # inline metadata above
    python scripts/check_tier0_theory.py      # any env with sympy
"""

import sympy as sp

n_fail = 0


def check(name: str, claim: bool) -> None:
    global n_fail
    if not claim:
        n_fail += 1
    print(("PASS" if claim else "FAIL") + "  " + name)


# ----------------------------------------------------------------- symbols
# positive=True matters: it licenses simplifications like sqrt(Ro**2) -> Ro
# and lets solve() discard unphysical roots.
y, Y, x = sp.symbols("y Y x", positive=True)
Ro, g, H, T0, DT, beta, y1 = sp.symbols(
    "Ro g H T_0 Delta_T beta y_1", positive=True)

# Memo sec. 1-3: forcing curvature b, response curvature c, and R_beta.
b = DT / y1**2                              # T_E  = T_E(0)  - b y^2
c = Ro * T0 * beta**2 / (8 * g * H)         # T_Ro = T_Ro(0) - c y^4
Rb = 4 * g * H * DT / (T0 * beta**2 * y1**4)

# ---------------------------------------- sec. 3: T_Ro from gradient balance
# beta*y*u = -(gH/T0) dT/dy with u = Ro*beta*y^2/2 should give c as above.
u_Ro = Ro * beta * y**2 / 2
dT_dy = -(T0 / (g * H)) * beta * y * u_Ro
check("T_Ro curvature: gradient balance of u_Ro gives c = Ro T0 beta^2/(8gH)",
      sp.simplify(sp.integrate(-dT_dy, (y, 0, y)) - c * y**4) == 0)

# ------------------------------------- sec. 4: equal-area edge and depression
# Unknowns: the edge Y and the equatorial depression d0 = T_E(0) - T_Ro(0).
# Impose D(Y) = 0 and the equal-area condition int_0^Y D dy = 0, then let
# sympy solve the system instead of doing the algebra by hand.
d0 = sp.Symbol("d0", positive=True)
D = -d0 + b * y**2 - c * y**4               # D(y) = T_Ro(y) - T_E(y)
sols = sp.solve([sp.Eq(D.subs(y, Y), 0),
                 sp.Eq(sp.integrate(D, (y, 0, Y)), 0)],
                [Y, d0], dict=True)
sol = sols[0]

check("edge: Y^2 = (5/6) b/c",
      sp.simplify(sol[Y] ** 2 - sp.Rational(5, 6) * b / c) == 0)
check("edge: Y_Ro = y1 sqrt(5 Rb/(3 Ro))          [ms Eq. 13 analog]",
      sp.simplify(sol[Y] - y1 * sp.sqrt(5 * Rb / (3 * Ro))) == 0)
check("depression: d0 = (5/18) Rb DT / Ro          [ms Eq. 15 analog]",
      sp.simplify(sol[d0] - sp.Rational(5, 18) * Rb * DT / Ro) == 0)

# --------------------------------------------------- sec. 5: the v profile
# Claim: -int_0^y D dy' = d0 * Y * (x - 2x^3 + x^5) with x = y/Y, so v(y)
# is the positive prefactor 1.6 H/(delta Dz tau) times this.
Ysol, d0sol = sol[Y], sol[d0]
integral = -sp.integrate(D.subs(d0, d0sol), (y, 0, y))
bracket = x - 2 * x**3 + x**5
check("v shape: -int_0^y D = d0 Y (x - 2x^3 + x^5)  [ms Eq. 16 shape]",
      sp.simplify(integral - (d0sol * Ysol * bracket).subs(x, y / Ysol)) == 0)

xpk = [r for r in sp.solve(bracket.diff(x), x) if r < 1][0]
check("v peak at x = 1/sqrt(5)",
      sp.simplify(xpk - 1 / sp.sqrt(5)) == 0)
check("v peak bracket value = 16/(25 sqrt 5) = 0.28622",
      sp.simplify(bracket.subs(x, xpk) - sp.Rational(16, 25) / sp.sqrt(5)) == 0)
check("cell-mean of bracket over [0,1] = 1/6        [sec. 7 closure input]",
      sp.integrate(bracket, (x, 0, 1)) == sp.Rational(1, 6))

# v_max = prefactor * d0 * Y * 0.286; check the Ro scaling via the log
# derivative d(log vmax)/d(log Ro), which sympy evaluates exactly.
vmax = d0sol * Ysol * bracket.subs(x, xpk)
check("v_max scales as Ro^(-3/2)                    [ms Eq. 20 analog]",
      sp.simplify(sp.log(vmax).diff(Ro) * Ro + sp.Rational(3, 2)) == 0)

# ------------------------------------- sec. 5: momentum flux counterintuitive
# flux(y) = delta * v(y) * u_Ro(y); drop the positive Ro-free prefactor.
flux = (d0sol * Ysol * bracket).subs(x, y / Ysol) * u_Ro
lead = flux.series(y, 0, 4).removeO()
check("momentum flux: leading O(y^3) term is Ro-independent",
      sp.simplify(lead.diff(Ro)) == 0)

# d(flux)/dRo at FIXED y (differentiate first, substitute y = x*Y after).
dflux = sp.simplify(flux.diff(Ro).subs(y, x * Ysol))
ratio = sp.simplify(dflux / (x**2 - 1))
check("momentum flux: d(flux)/dRo < 0 strictly inside the cell (x < 1)",
      bool(ratio.is_positive))

# ------------------------------------------- sec. 8: RCE plateau outside cell
# Gradient balance with T_E itself: u_RCE = 2 g H DT/(T0 beta y1^2), y-free.
u_rce = sp.solve(sp.Eq(beta * y * sp.Symbol("u"),
                       -(g * H / T0) * sp.diff(-(b) * y**2, y)),
                 sp.Symbol("u"))[0]
check("RCE wind: u_RCE = 2 g H DT/(T0 beta y1^2), independent of y",
      sp.simplify(u_rce - 2 * g * H * DT / (T0 * beta * y1**2)) == 0
      and u_rce.diff(y) == 0)

# ---------------------------------------------------- numeric spot checks
# Substitute the repo defaults and compare against the memo's quoted numbers.
defaults = {g: 9.81, H: 16e3, DT: 31.25, T0: 300,
            beta: 2e-11, y1: 9.439e6}
Rb_num = float(Rb.subs(defaults))
print(f"\nR_beta at defaults          = {Rb_num:.4f}   (memo: 0.0206)")
Yro_num = float(sol[Y].subs(defaults).subs(Ro, 0.470))
print(f"Y_Ro at diagnosed Ro=0.470  = {Yro_num:.3e} m  (memo run 1a: 2.55e6)")
d0_num = float(sol[d0].subs(defaults).subs(Ro, 0.470))
print(f"depression at Ro=0.470      = {d0_num:.3f} K  (memo run 1a: 0.381)")
urce_num = float(u_rce.subs(defaults))
print(f"u_RCE at defaults           = {urce_num:.2f} m/s  (memo: 18.35)")

print(f"\n{n_fail} failures" if n_fail else "\nall symbolic checks passed")
