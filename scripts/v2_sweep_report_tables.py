"""Every number the moist V2 sweep report quotes, emitted by one script.

A figure quoted in a write-up has to appear in the committed entry point's own
output, or nobody can regenerate it. So this script prints the report's
statistics AND writes them to numbers.json, and writes the LaTeX table
fragments the report \\input{}s. Nothing in the report is transcribed by hand.

The statistics were listed before the first run rather than discovered while
drafting: a range over a set, a count of how many members clear a threshold,
a sign or direction tally, and the paired difference between twins.

Usage:
    python scripts/v2_sweep_report_tables.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/shill/py/axisym-hc-swm")

from scripts.v2_sweep_diagnostics import (  # noqa: E402
    Y_ONE, lat_to_y, load_run, load_seasonal, scalars, seasonal_scalars,
)
from scripts.v2_sweep_figures import load_all  # noqa: E402
from scripts.v2_sweep_manifest import (  # noqa: E402
    DT_6, DT_15, LAMBDA_0, SWEEP_DIR, manifest, r_param, w_star,
)

OUT = os.path.join(SWEEP_DIR, "tables")
# The LaTeX fragments are also written here, beside the document that reads
# them, because model_output/ is a symlink to a scratch filesystem and a
# document should not depend on it to build.
DOCS_TABLES = "/home/shill/py/axisym-hc-swm/docs/tables"

# ERA5 anchors, from docs/era5_calibration_report.pdf (Tables 8 and 9 of the
# .tex: tab:fluxcompare and tab:score) and its stability section. Quoted here
# so the plausibility comparison has a fixed reference, with the source named.
ERA5 = dict(
    jet=31.0,                    # 200 hPa zonal-mean maximum, m/s
    v_slab=(1.07, 1.31),         # slab velocity at dp = 195 and 238 hPa, m/s
    dtheta16=0.56, dtheta24=3.21, dtheta33=8.75,   # K
    dse24=50.7, lvq24=-22.9, mse24=27.9, eddy24=57.7, tot24=85.6,  # MW/m
    w10=44.5, w20=39.6, w30=35.1,   # kg/m^2, band means
    hhat_over_shat=(0.258, 0.318),  # |phi| <= 10 to 30 deg
    w_star=(44.0, 57.0),            # kg/m^2, range over time-mean conventions
)


def fmt(x, n=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{n}f}"


def by_name(rows):
    return {r["name"]: r for r in rows}


def rng(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return (min(v), max(v)) if v else (np.nan, np.nan)


_ESCAPES = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$",
            "&": r"\&", "#": r"\#", "_": r"\_", "%": r"\%",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _esc(text: str) -> str:
    """Escape a plain string for LaTeX text mode.

    The manifest notes are written for a terminal and contain things like
    'D=0 m^2/s' and 'tau_c=1800 s'; both the caret and the underscore are
    LaTeX-special, and an unescaped caret raises 'Missing $ inserted' rather
    than rendering.
    """
    return "".join(_ESCAPES.get(c, c) for c in text)


class _tex:
    """Write one LaTeX fragment to both the sweep tree and docs/tables.

    The document \\input{}s from docs/tables so that it builds without the
    model_output symlink, which points at a scratch filesystem.
    """

    def __init__(self, name):
        self.paths = [os.path.join(OUT, name), os.path.join(DOCS_TABLES, name)]

    def __enter__(self):
        os.makedirs(OUT, exist_ok=True)
        os.makedirs(DOCS_TABLES, exist_ok=True)
        self.handles = [open(p, "w") for p in self.paths]
        return self

    def write(self, text):
        for h in self.handles:
            h.write(text)

    def __exit__(self, *exc):
        for h in self.handles:
            h.close()
        return False


# ---------------------------------------------------------------------------
def write_inventory():
    """The run table, from the manifest alone. Independent of whether any run
    has finished, so it is written first."""
    with _tex("inventory.tex") as fh:
        fh.write("{\\footnotesize\n\\begin{longtable}{llrrp{7.0cm}}\n")
        fh.write("\\caption{Every run in the sweep.}\\label{tab:inventory}\\\\\n")
        fh.write("\\toprule\nrun & block & $n_y$ & $\\Delta t$ (s) & what it "
                 "varies\\\\\n\\midrule\n\\endfirsthead\n\\toprule\nrun & block"
                 " & $n_y$ & $\\Delta t$ (s) & what it varies\\\\\n\\midrule\n"
                 "\\endhead\n\\bottomrule\n\\endfoot\n")
        for e in manifest():
            fh.write(f"\\texttt{{{_esc(e['name'])}}} & {e['block']} & "
                     f"{e['args']['ny']} & {e['args']['dt']} & "
                     f"{_esc(e['note'])}\\\\\n")
        fh.write("\\end{longtable}\n}\n")
    print(f"Wrote {OUT}/inventory.tex")


def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=SWEEP_DIR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--complete-frac", type=float, default=0.98)
    a = ap.parse_args()
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    write_inventory()
    eq, seas, meta, missing = load_all(a.tree, 1000, a.complete_frac)
    rows = [scalars(eq[n], n, meta[n]["block"], meta[n]["note"]) for n in eq]
    srows = [seasonal_scalars(seas[n], n, meta[n]["block"], meta[n]["note"])
             for n in seas]
    R = by_name(rows)
    S = by_name(srows)
    N = {}   # the numbers dictionary the report quotes
    lines = []   # console echo

    def note(key, val, text):
        N[key] = val
        lines.append(text)

    n_manifest = len(manifest())
    note("n_runs_total", n_manifest, f"manifest runs: {n_manifest}")
    note("n_eq", len(rows), f"equilibrium runs analysed: {len(rows)}")
    note("n_seas", len(srows), f"seasonal runs analysed: {len(srows)}")
    note("n_missing", len(missing),
         f"runs missing or diverged: {len(missing)}"
         + (" (" + " ".join(missing) + ")" if missing else ""))
    note("missing_names", missing, "")
    note("n_dt15", len(DT_15), f"configs needing dt=15 s: {len(DT_15)}")
    note("n_dt6", len(DT_6), f"configs needing dt=6 s: {len(DT_6)}")

    # -- equilibrium quality -------------------------------------------------
    for key, label, digits in (("p_mean_over_e0", "domain-mean P / E_0", 6),
                               ("resid_rel", "relative MSE budget residual", 3),
                               ("drift_u", "relative u drift, last two 1000-d windows", 4),
                               ("drift_w", "relative W drift", 5)):
        lo, hi = rng([r[key] for r in rows])
        note(f"{key}_min", lo, "")
        note(f"{key}_max", hi, f"{label}: {lo:.3g} to {hi:.3g} over {len(rows)} runs")
    if rows:
        worst_drift = max(rows, key=lambda r: r["drift_u"])
        note("worst_drift_name", worst_drift["name"], "")
        note("worst_drift_u", worst_drift["drift_u"],
             f"largest u drift: {worst_drift['name']} at "
             f"{worst_drift['drift_u']:.3g}")
    # -- steady versus never-settling ---------------------------------------
    steady = [r for r in rows if r["steady"]]
    unsteady = [r for r in rows if not r["steady"]]
    note("n_steady", len(steady), f"steady solutions: {len(steady)}")
    note("n_unsteady", len(unsteady),
         f"solutions that never settle: {len(unsteady)}"
         + (" (" + " ".join(r["name"] for r in unsteady) + ")" if unsteady else ""))
    note("unsteady_names", [r["name"] for r in unsteady], "")
    if steady:
        lo, hi = rng([r["resid_rel"] for r in steady])
        note("resid_steady_max", hi,
             f"MSE budget residual over the {len(steady)} steady runs: "
             f"{lo:.2g} to {hi:.2g} of the source")
        lo, hi = rng([r["drift_u"] for r in steady])
        note("drift_steady_max", hi,
             f"u drift over the steady runs: {lo:.2g} to {hi:.2g}")
    if unsteady:
        lo, hi = rng([r["r"] for r in unsteady])
        note("r_unsteady_min", lo, "")
        note("r_unsteady_max", hi,
             f"the never-settling runs span r = {lo:.3f} to {hi:.3f}")
        lo, hi = rng([r["jet_range"] for r in unsteady])
        note("unsteady_jet_range_min", lo, "")
        note("unsteady_jet_range_max", hi,
             f"their jet magnitude swings by {lo:.1f} to {hi:.1f} m/s over the "
             f"trailing 2000 days")
        lo, hi = rng([r["pmax_range"] for r in unsteady])
        note("unsteady_pmax_range_max", hi,
             f"their peak rain rate swings by {lo:.0f} to {hi:.0f} mm/day")
        # the largest r that IS steady, to bracket the second boundary
        rs = [r["r"] for r in steady if r["r"] > 1.0]
        if rs:
            note("r_max_steady_negative_gms", max(rs),
                 f"the largest r that still settles is {max(rs):.3f}")
    lines.append("steadiness by run (r, jet range over the trailing window):")
    for r in sorted(rows, key=lambda r: r["r"]):
        if not r["steady"]:
            lines.append(f"    {r['name']:28s} r={r['r']:.3f}  jet swings "
                         f"{r['jet_range']:6.2f} m/s  peak P swings "
                         f"{r['pmax_range']:7.1f} mm/day  budget residual "
                         f"{r['resid_rel']:.1e}")

    n_neg_w = sum(1 for r in rows if r["w_min"] < 0)
    note("n_negative_w", n_neg_w,
         f"runs with negative W in the time mean: {n_neg_w}")
    n_neg_w_run = sum(1 for r in rows if np.isfinite(r["w_min_run"])
                      and r["w_min_run"] < 0)
    note("n_negative_w_run", n_neg_w_run,
         f"runs whose instantaneous W ever went negative: {n_neg_w_run}")

    # -- the regime transition ----------------------------------------------
    # Where does the domain stop raining everywhere? Measured on the a = 0.85
    # W_c transect, which is the finest sampling of r available.
    transect = sorted([r for r in rows if abs(r["a"] - 0.85) < 1e-9
                       and r["block"] in ("A", "B")], key=lambda r: r["r"])
    note("transect_r", [r["r"] for r in transect], "")
    note("transect_wet", [r["wet_frac"] for r in transect], "")
    note("transect_pmax", [r["p_max"] for r in transect], "")
    note("transect_names", [r["name"] for r in transect], "")
    lines.append("a = 0.85 transect (r, wet fraction, peak P mm/day, bands):")
    for r in transect:
        lines.append(f"    W_c={r['w_crit']:5.1f}  r={r['r']:.3f}  "
                     f"wet={r['wet_frac']:.3f}  Pmax={r['p_max']:8.2f}  "
                     f"bands={r['n_bands']}  Hmin={r['hhat_min']:7.2f}  "
                     f"jet={r['jet']:5.1f}  |v|={r['v_max']:5.2f}")
    # last r that still rains essentially everywhere, first that does not
    wet_thresh = 0.98
    below = [r for r in transect if r["wet_frac"] >= wet_thresh]
    above = [r for r in transect if r["wet_frac"] < wet_thresh]
    if below and above:
        note("r_last_wet", below[-1]["r"], "")
        note("r_first_dry", above[0]["r"], "")
        note("wc_last_wet", below[-1]["w_crit"], "")
        note("wc_first_dry", above[0]["w_crit"], "")
        lines.append(f"  transition brackets r = {below[-1]['r']:.3f} "
                     f"(W_c={below[-1]['w_crit']:.0f}) to {above[0]['r']:.3f} "
                     f"(W_c={above[0]['w_crit']:.0f})")

    # -- matched-r collapse test --------------------------------------------
    # This is the falsification test: four values of a, W_c chosen to put each
    # at the same r. If r is the control parameter these should agree.
    collapse = {}
    for rr in (0.85, 1.05):
        grp = [R[f"C_r{int(rr*100):03d}_a{int(a*100):03d}"]
               for a in (0.75, 0.79, 0.83, 0.87)
               if f"C_r{int(rr*100):03d}_a{int(a*100):03d}" in R]
        if len(grp) < 2:
            continue
        entry = {}
        for key in ("wet_frac", "p_max", "jet", "v_max", "t_eq", "hhat_min",
                    "itcz_spread", "eddy_share", "dtheta24"):
            v = np.array([g[key] for g in grp], dtype=float)
            entry[key] = dict(
                mean=float(np.mean(v)), lo=float(np.min(v)), hi=float(np.max(v)),
                spread_rel=float((np.max(v) - np.min(v))
                                 / max(abs(np.mean(v)), 1e-30)))
        entry["a_values"] = [g["a"] for g in grp]
        entry["wc_values"] = [g["w_crit"] for g in grp]
        collapse[f"r{rr}"] = entry
        lines.append(f"matched r = {rr}, a in {entry['a_values']} "
                     f"(W_c {[round(w,2) for w in entry['wc_values']]}):")
        for key in ("wet_frac", "p_max", "jet", "v_max", "hhat_min"):
            e = entry[key]
            lines.append(f"    {key:12s} {e['lo']:9.3f} to {e['hi']:9.3f}  "
                         f"spread {100*e['spread_rel']:6.2f}% of the mean")
    note("collapse", collapse, "")

    # For contrast: the spread of the SAME metrics across the whole sweep, so
    # "the matched-r sets agree" has a scale to be judged against.
    allsel = [r for r in rows if r["block"] in ("A", "B", "C", "E", "G", "I")]
    span = {}
    for key in ("wet_frac", "p_max", "jet", "v_max", "hhat_min"):
        v = np.array([r[key] for r in allsel], dtype=float)
        span[key] = dict(lo=float(np.min(v)), hi=float(np.max(v)),
                         spread_rel=float((np.max(v) - np.min(v))
                                          / max(abs(np.mean(v)), 1e-30)))
    note("gms_block_span", span, "")
    lines.append("for scale, the same metrics across all GMS-varying blocks "
                 f"({len(allsel)} runs):")
    for key, e in span.items():
        lines.append(f"    {key:12s} {e['lo']:9.3f} to {e['hi']:9.3f}  "
                     f"spread {100*e['spread_rel']:7.1f}% of the mean")

    # -- plausibility scorecard against ERA5 --------------------------------
    # Row labels are the three numbers that identify the run, in their own
    # columns rather than crammed into a text label: a text label wide enough to
    # carry them pushed both generated tables 265 pt past the margin.
    score_names = [("A_a075_wc30", "0.75 & 30 & 0.49"),
                   ("A_a079_wc40", "0.79 & 40 & 0.76"),
                   ("A_a085_wc40", "0.85 & 40 & 0.92"),
                   ("B_a085_wc44", "0.85 & 44 & 1.01"),
                   ("A_a085_wc50", "0.85 & 50 & 1.14"),
                   ("N_P_v1", r"\multicolumn{3}{l}{V1 twin, $a$=0.79 $W_c$=40}")]
    score_rows = []
    for nm, lab in score_names:
        if nm not in R:
            continue
        r = R[nm]
        score_rows.append((lab, r))
    with _tex("score.tex") as fh:
        fh.write("{\\small\n\\begin{tabular}{cccccccc}\n\\toprule\n")
        fh.write("$a$ & $W_c$ & $r$ & jet & peak $|v|$ & "
                 "$\\Delta\\theta(16^\\circ)$ & $(24^\\circ)$ & $(33^\\circ)$\\\\\n")
        fh.write(" & & & (m\\,s$^{-1}$) & (m\\,s$^{-1}$) & (K) & (K) & "
                 "(K)\\\\\n\\midrule\n")
        fh.write(f"\\multicolumn{{3}}{{l}}{{ERA5}} & {ERA5['jet']:.1f} & "
                 f"{ERA5['v_slab'][0]:.2f}--{ERA5['v_slab'][1]:.2f} & "
                 f"{ERA5['dtheta16']:.2f} & {ERA5['dtheta24']:.2f} & "
                 f"{ERA5['dtheta33']:.2f}\\\\\n\\midrule\n")
        for lab, r in score_rows:
            fh.write(f"{lab} & {r['jet']:.1f} & {r['v_max']:.2f} & "
                     f"{r['dtheta16']:.2f} & {r['dtheta24']:.2f} & "
                     f"{r['dtheta33']:.2f}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n}\n")
    note("score_rows", [(lab, {k: r[k] for k in
                              ("jet", "v_max", "dtheta16", "dtheta24",
                               "dtheta33", "w_trop20", "r")})
                        for lab, r in score_rows], "")
    lines.append("plausibility scorecard written to tables/score.tex")
    for lab, r in score_rows:
        lines.append(f"    {lab[:34]:34s} jet {r['jet']:5.1f}  |v| "
                     f"{r['v_max']:5.2f}  dth24 {r['dtheta24']:5.2f}  "
                     f"W20 {r['w_trop20']:5.1f}")
    # jet overshoot factor, the quantity the ERA5 report flagged
    ov = [(r["name"], r["jet"] / ERA5["jet"]) for r in rows
          if r["block"] in ("A", "B") and r["latent"]]
    lo, hi = rng([x[1] for x in ov])
    note("jet_overshoot_min", lo, "")
    note("jet_overshoot_max", hi,
         f"jet / observed 31.0 m/s over the GMS plane: {lo:.2f} to {hi:.2f}")

    # -- flux table at 24 deg ------------------------------------------------
    with _tex("fluxes.tex") as fh:
        fh.write("{\\small\n\\begin{tabular}{cccccc}\n\\toprule\n")
        fh.write("$a$ & $W_c$ & $r$ & mean $v\\hat H$ & eddy $-L_vD\\py W$ & "
                 "total\\\\\n & & & (MW\\,m$^{-1}$) & (MW\\,m$^{-1}$) & "
                 "(MW\\,m$^{-1}$)\\\\\n\\midrule\n")
        fh.write(f"\\multicolumn{{3}}{{l}}{{ERA5}} & {ERA5['mse24']:.1f} & "
                 f"{ERA5['eddy24']:.1f} & {ERA5['tot24']:.1f}\\\\\n\\midrule\n")
        for lab, r in score_rows:
            fh.write(f"{lab} & {r['f_mean_ref']:.2f} & {r['f_eddy_ref']:.2f} & "
                     f"{r['f_tot_ref']:.2f}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n}\n")
    note("flux_rows", [(lab, {k: r[k] for k in
                             ("f_mean_ref", "f_eddy_ref", "f_tot_ref",
                              "eddy_share")}) for lab, r in score_rows], "")

    # -- ladders: the direction and size of each parameter's effect ---------
    ladders = [
        ("delta_y_rad", "D", r"$\Delta_\theta^{\rm rad}$ (K)"),
        ("d_w", "F", r"$D$ (m$^2$\,s$^{-1}$)"),
        ("tau_c", "E", r"$\tau_c$ (s)"),
        ("evap", "G", r"$E_0$ (kg\,m$^{-2}$\,s$^{-1}$)"),
        ("v_d", "H", r"$v_d$ (m\,s$^{-1}$)"),
        ("delta_z", "I", r"$\Delta_z$ (K)"),
        ("lam", "J", r"$\Lambda$"),
    ]
    lad = {}
    for xkey, block, xlab in ladders:
        for ref, a_ref, wc_ref in (("P", 0.79, 40.0), ("N", 0.85, 50.0)):
            sub = sorted([r for r in rows
                          if r["block"] in (block, "A")
                          and abs(r["a"] - a_ref) < 1e-9
                          and abs(r["w_crit"] - wc_ref) < 1e-9
                          and (r["block"] == block
                               or (r["block"] == "A" and r["latent"]))],
                         key=lambda r: (r[xkey] if r[xkey] is not None else 0))
            if len(sub) < 3:
                continue
            key = f"{xkey}_{ref}"
            lad[key] = dict(
                x=[r[xkey] for r in sub], names=[r["name"] for r in sub],
                **{m: [r[m] for r in sub] for m in
                   ("jet", "v_max", "p_max", "wet_frac", "hhat_min", "w_eq",
                    "t_eq", "eddy_share", "dtheta24", "r")})
            lines.append(f"ladder {xkey} on {ref}: x = "
                         + ", ".join(f"{v:.4g}" for v in lad[key]["x"]))
            for m in ("jet", "v_max", "p_max", "wet_frac"):
                vv = lad[key][m]
                lines.append(f"    {m:10s} " + "  ".join(f"{v:8.3f}" for v in vv))
    note("ladders", lad, "")

    # -- initial-condition test ---------------------------------------------
    ic = []
    for wc in (40.0, 42.0, 44.0, 46.0):
        base = R.get(f"A_a085_wc{int(wc)}") or R.get(f"B_a085_wc{int(wc)}")
        if base is None:
            continue
        entry = dict(w_crit=wc, r=base["r"],
                     base=dict(w_init=wc, wet_frac=base["wet_frac"],
                               p_max=base["p_max"], jet=base["jet"],
                               v_max=base["v_max"], w_eq=base["w_eq"]))
        entry["bumped"] = []
        for bump in (15.0, 30.0):
            nm = f"K_wc{int(wc):02d}_wi{int(wc+bump):02d}"
            if nm in R:
                b = R[nm]
                entry["bumped"].append(dict(
                    w_init=wc + bump, wet_frac=b["wet_frac"],
                    p_max=b["p_max"], jet=b["jet"], v_max=b["v_max"],
                    w_eq=b["w_eq"],
                    rel_diff_jet=abs(b["jet"] - base["jet"])
                    / max(abs(base["jet"]), 1e-12),
                    rel_diff_pmax=abs(b["p_max"] - base["p_max"])
                    / max(abs(base["p_max"]), 1e-12)))
        ic.append(entry)
    note("initial_condition", ic, "")
    worst_ic = 0.0
    for e in ic:
        for b in e["bumped"]:
            worst_ic = max(worst_ic, b["rel_diff_jet"], b["rel_diff_pmax"])
        lines.append(f"W_c={e['w_crit']:.0f} (r={e['r']:.3f}): base jet "
                     f"{e['base']['jet']:.2f} Pmax {e['base']['p_max']:.2f}; "
                     + "; ".join(f"W0={b['w_init']:.0f} jet {b['jet']:.2f} "
                                 f"Pmax {b['p_max']:.2f}" for b in e["bumped"]))
    note("worst_ic_rel_diff", worst_ic,
         f"largest relative difference from raising initial W: {worst_ic:.3g}")

    # -- off-equatorial -----------------------------------------------------
    offeq = {}
    for ref, a_ref in (("P", 0.79), ("N", 0.85)):
        for prof in ("sin2", "sb08"):
            # The sin^2 series takes its y_0 = 0 member from block A (the
            # reference IS a sin^2 y_0 = 0 run); the SB08 series has its own
            # y_0 = 0 member and must not borrow the sin^2 one, which is a
            # different profile family.
            names = list(
                ([f"A_a{int(a_ref*100):03d}_wc{40 if ref == 'P' else 50}"]
                 if prof == "sin2" else [])
                + [f"L_{ref}_{prof}_y{v:04d}" for v in
                   ((700, 1400, 2100) if prof == "sin2"
                    else (0, 700, 1400, 2100, 2800))])
            sub = sorted([R[n] for n in names if n in R],
                         key=lambda r: r["y_0"])
            if len(sub) < 3:
                continue
            y0 = np.array([r["y_0"] for r in sub])
            itcz = np.array([r["itcz"] for r in sub])
            ok = np.isfinite(itcz)
            slope = (float(np.polyfit(y0[ok], itcz[ok], 1)[0])
                     if ok.sum() > 1 else np.nan)
            offeq[f"{ref}_{prof}"] = dict(
                y0=[float(v) for v in y0], itcz=[float(v) for v in itcz],
                efe=[r["efe"] for r in sub], u_eq=[r["u_eq"] for r in sub],
                jetN=[r["jet"] for r in sub], jetS=[r["jet_s"] for r in sub],
                v_max=[r["v_max"] for r in sub],
                p_max=[r["p_max"] for r in sub],
                wet_frac=[r["wet_frac"] for r in sub],
                names=[r["name"] for r in sub], itcz_per_y0=slope)
            lines.append(
                f"off-equatorial {ref} {prof}: y0 (Mm) "
                + ", ".join(f"{float(v)/1e6:.2f}" for v in y0)
                + " -> rain centroid "
                + ", ".join(f"{float(v)/1e6:.3f}" if np.isfinite(v) else "nan"
                            for v in itcz)
                + f"; d(centroid)/d(y0) = {slope:.3f}")
            lines.append(f"     equatorial u {[round(r['u_eq'],2) for r in sub]}"
                         f"  summer jet {[round(r['jet'],1) for r in sub]}"
                         f"  winter jet {[round(r['jet_s'],1) for r in sub]}")
    note("offeq", offeq, "")

    # -- seasonal -----------------------------------------------------------
    if srows:
        with _tex("seasonal.tex") as fh:
            fh.write("{\\footnotesize\n\\begin{tabular}{lcccccccc}\n\\toprule\n")
            fh.write("run & amp. & period & $\\hat H$ & centroid amp. & gain & "
                     "lag & summer jet & winter jet\\\\\n")
            fh.write(" & (km) & (d) & & (km) & & (d) & (m\\,s$^{-1}$) & "
                     "(m\\,s$^{-1}$)\\\\\n\\midrule\n")
            for r in sorted(srows, key=lambda r: (r["a"], r["period"],
                                                  r["amp_km"])):
                sign = "$+$" if r["r"] < 1 else "$-$"
                fh.write(f"\\texttt{{{r['name'].replace('_', chr(92)+'_')}}} & "
                         f"{r['amp_km']:.0f} & {r['period']:.0f} & {sign} & "
                         f"{r['amp_itcz_km']:.0f} & {r['gain']:.3f} & "
                         f"{r['lag_days']:.1f} & {r['jetN_max']:.1f} & "
                         f"{r['jetS_max']:.1f}\\\\\n")
            fh.write("\\bottomrule\n\\end{tabular}\n}\n")
        note("seasonal_rows", srows, "")
        lines.append("seasonal composites:")
        for r in sorted(srows, key=lambda r: (r["a"], r["period"], r["amp_km"])):
            lines.append(f"    {r['name']:26s} amp {r['amp_km']:5.0f} km  "
                         f"per {r['period']:5.0f} d  centroid amp "
                         f"{r['amp_itcz_km']:6.0f} km  gain {r['gain']:6.3f}  "
                         f"lag {r['lag_days']:6.1f} d  jetN {r['jetN_max']:5.1f}"
                         f"  jetS {r['jetS_max']:5.1f}  repeat "
                         f"{r['repeat_itcz_km']:6.1f} km")
        lo, hi = rng([r["gain"] for r in srows])
        note("gain_min", lo, "")
        note("gain_max", hi, f"centroid-to-forcing amplitude ratio: "
                             f"{lo:.3f} to {hi:.3f}")
        lo, hi = rng([r["lag_days"] for r in srows])
        note("lag_min", lo, "")
        note("lag_max", hi, f"lag: {lo:.1f} to {hi:.1f} days")
        lo, hi = rng([r["repeat_itcz_km"] for r in srows])
        note("repeat_itcz_max", hi,
             f"cycle-to-cycle repeatability of the rain centroid: "
             f"at worst {hi:.1f} km")
        lo, hi = rng([r["p_mean_over_e0"] for r in srows])
        note("seas_water_min", lo, "")
        note("seas_water_max", hi,
             f"seasonal domain-mean P / E_0: {lo:.6f} to {hi:.6f}")

    # -- seasonal against perpetual -----------------------------------------
    svp = []
    for sn, pn, lab in (("M_P_amp0700_p360", "L_P_sb08_y0700", "700 km, P"),
                        ("M_P_amp1400_p360", "L_P_sb08_y1400", "1400 km, P"),
                        ("M_P_amp2100_p360", "L_P_sb08_y2100", "2100 km, P"),
                        ("M_N_amp1400_p360", "L_N_sb08_y1400", "1400 km, N")):
        if sn not in seas or pn not in R:
            continue
        s = seas[sn]
        k = int(np.argmax(s["y0_t"]))
        p_seas = s["p"][k]
        y = s["y"]
        itcz_seas = float(np.sum(np.maximum(p_seas, 0) * y)
                          / max(np.sum(np.maximum(p_seas, 0)), 1e-30))
        entry = dict(label=lab, seasonal=sn, perpetual=pn,
                     y0_peak=float(s["y0_t"][k]),
                     itcz_seasonal=itcz_seas, itcz_perpetual=R[pn]["itcz"],
                     jetN_seasonal=float(s["u"][k][y > 0].max()),
                     jetN_perpetual=R[pn]["jet"],
                     pmax_seasonal=float(p_seas.max()) * 86400,
                     pmax_perpetual=R[pn]["p_max"])
        entry["itcz_gap_km"] = (entry["itcz_seasonal"]
                                - entry["itcz_perpetual"]) / 1e3
        entry["jet_gap"] = entry["jetN_seasonal"] - entry["jetN_perpetual"]
        svp.append(entry)
        lines.append(f"seasonal vs perpetual, {lab}: ITCZ "
                     f"{entry['itcz_seasonal']/1e6:.3f} vs "
                     f"{entry['itcz_perpetual']/1e6:.3f} Mm "
                     f"({entry['itcz_gap_km']:+.0f} km), jet "
                     f"{entry['jetN_seasonal']:.1f} vs "
                     f"{entry['jetN_perpetual']:.1f} "
                     f"({entry['jet_gap']:+.1f} m/s), peak P "
                     f"{entry['pmax_seasonal']:.2f} vs "
                     f"{entry['pmax_perpetual']:.2f} mm/day")
    note("seasonal_vs_perpetual", svp, "")

    # -- numerics twins: paired max|delta| on a common grid -----------------
    twins = []
    for base, other, lab in (("A_a079_wc40", "O_P_dt15", "P: dt 30 to 15"),
                             ("A_a079_wc40", "O_P_ny1601", "P: ny 801 to 1601"),
                             ("A_a079_wc40", "O_P_ny401", "P: ny 801 to 401"),
                             ("A_a085_wc50", "O_N_dt15", "N: dt 30 to 15"),
                             ("A_a085_wc50", "O_N_ny1601", "N: ny 801 to 1601"),
                             ("A_a085_wc50", "O_N_ny401", "N: ny 801 to 401"),
                             ("B_a085_wc44", "O_crit_ny1601",
                              "near-critical: ny 801 to 1601"),
                             ("O_P_sb08_y0700_ny0801_dt6",
                              "O_P_sb08_y0700_ny1601_dt6",
                              "off-equatorial: ny 801 to 1601 at dt 6, 2500 d")):
        if base not in eq or other not in eq:
            continue
        A, B = eq[base], eq[other]
        entry = dict(label=lab, base=base, other=other)
        for f in ("u", "temp", "w", "p"):
            # interpolate the finer/coarser field onto the base grid; a paired
            # per-point difference, not a difference of two summaries
            vb = np.interp(A["y"], B["y"], B[f])
            d = np.abs(A[f] - vb)
            scale = max(float(np.max(np.abs(A[f]))), 1e-30)
            entry[f"max_abs_{f}"] = float(np.max(d))
            entry[f"max_rel_{f}"] = float(np.max(d) / scale)
            entry[f"argmax_{f}_Mm"] = float(A["y"][int(np.argmax(d))] / 1e6)
        for m in ("jet", "v_max", "p_max", "wet_frac", "hhat_min", "t_eq"):
            a_ = scalars(A)[m]
            b_ = scalars(B)[m]
            entry[f"d_{m}"] = float(b_ - a_)
            entry[f"drel_{m}"] = float(abs(b_ - a_) / max(abs(a_), 1e-30))
        twins.append(entry)
        lines.append(f"twin {lab}: max|du| {entry['max_abs_u']:.4g} m/s "
                     f"({100*entry['max_rel_u']:.2f}% of peak u, at "
                     f"{entry['argmax_u_Mm']:+.2f} Mm); max|dP| "
                     f"{entry['max_abs_p']*86400:.4g} mm/day; jet moves "
                     f"{entry['d_jet']:+.3f} m/s "
                     f"({100*entry['drel_jet']:.2f}%), peak P "
                     f"{100*entry['drel_p_max']:.2f}%")
    note("twins", twins, "")
    if twins:
        lo, hi = rng([t["drel_jet"] for t in twins])
        note("twin_jet_rel_max", hi,
             f"jet changes at most {100*hi:.2f}% between resolution/dt twins "
             f"({len(twins)} pairs)")
        lo, hi = rng([t["max_rel_u"] for t in twins])
        note("twin_u_rel_max", hi,
             f"whole-profile max|du|/max|u| at most {100*hi:.2f}%")

    # -- V1 to V2 attribution ------------------------------------------------
    attrib = []
    for ref, a_ref, wc in (("P", 0.79, 40.0), ("N", 0.85, 50.0)):
        chain = [(f"N_{ref}_v1", "V1: no latent heating, no retarget"),
                 (f"J_{ref}_lam000", r"V2$_0$: retarget only, $\Lambda=0$"),
                 (f"D_{ref}_dyr050", r"$\Lambda$ on, $\Delta^{\rm rad}=50$ K"),
                 (f"A_a{int(a_ref*100):03d}_wc{int(wc)}",
                  r"V2: $\Delta^{\rm rad}=75$ K")]
        got = [(R[n], lab) for n, lab in chain if n in R]
        if len(got) < 3:
            continue
        attrib.append(dict(ref=ref, steps=[
            dict(label=lab, name=r["name"], jet=r["jet"], v_max=r["v_max"],
                 t_eq=r["t_eq"], dtheta24=r["dtheta24"], p_max=r["p_max"],
                 wet_frac=r["wet_frac"], w_eq=r["w_eq"],
                 hhat_min=r["hhat_min"], theta_offset=r["theta_offset"])
            for r, lab in got]))
        lines.append(f"V1 to V2 chain, {ref} reference:")
        for r, lab in got:
            lines.append(f"    {lab[:42]:42s} jet {r['jet']:6.2f}  |v| "
                         f"{r['v_max']:6.3f}  T_eq {r['t_eq']:7.2f}  dth24 "
                         f"{r['dtheta24']:6.2f}  Pmax {r['p_max']:7.2f}  "
                         f"wet {r['wet_frac']:.3f}")
    note("attribution", attrib, "")

    # -- the whole table, so nothing is invisible ---------------------------
    cols = ["name", "block", "r", "a", "w_crit", "tau_c", "evap", "d_w", "v_d",
            "delta_z", "delta_y_rad", "lam", "y_0", "theta_e_type", "ny", "dt",
            "days", "jet", "jet_lat", "jet_s", "u_eq", "v_max", "t_eq",
            "dtheta16", "dtheta24", "dtheta33", "w_eq", "w_max", "w_trop20",
            "p_max", "p_mean_over_e0", "p_wet_mean_over_e0", "wet_frac",
            "dry_frac", "n_bands", "itcz", "itcz_spread", "efe", "asc_edge",
            "north_edge", "ro_cell", "hhat_min", "hhat_eq", "hhat_neg_frac",
            "w_gap", "hhat_gap", "gap_hhat_pos_frac", "f_mean_ref",
            "f_eddy_ref", "f_tot_ref", "eddy_share", "resid_rel", "drift_u",
            "drift_w", "w_min", "w_min_run", "steady", "jet_range", "jet_sd",
            "pmax_range", "wmean_range", "itcz_excess", "itcz_peak",
            "ascent_axis", "p_wet_mean_over_e0", "var_window"]
    csv_path = os.path.join(OUT, "..", "all_runs.csv")
    with open(csv_path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in sorted(rows, key=lambda r: (r["block"], r["name"])):
            fh.write(",".join(
                (f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]))
                for c in cols) + "\n")
    print(f"\nWrote the full {len(rows)}-run table to {csv_path}")
    show = ["r", "steady", "jet", "v_max", "t_eq", "w_eq", "p_max", "wet_frac",
            "n_bands", "hhat_min", "w_gap", "hhat_gap", "f_mean_ref",
            "f_eddy_ref", "resid_rel", "drift_u"]
    print("\n" + f"{'run':28s}" + "".join(f"{c:>10}" for c in show))
    for r in sorted(rows, key=lambda r: (r["block"], r["r"], r["name"])):
        print(f"{r['name']:28s}" + "".join(
            (f"{r[c]:>10.4g}" if isinstance(r[c], float) else f"{str(r[c]):>10}")
            for c in show))

    with open(os.path.join(OUT, "..", "numbers.json"), "w") as fh:
        json.dump(N, fh, indent=1, default=float)
    print("\n".join(l for l in lines if l))
    print(f"\nWrote {OUT}/score.tex, fluxes.tex, inventory.tex"
          + (", seasonal.tex" if srows else "")
          + f" and {os.path.join(OUT, '..', 'numbers.json')}")


if __name__ == "__main__":
    main()
