# ERA5 calibration: working notes

**Superseded 2026-08-01 by `docs/era5_calibration_report.pdf`** (source
`era5_calibration_report.tex`, rebuild with
`latexmk -pdf -output-directory=docs docs/era5_calibration_report.tex` from the
repo root). Everything these notes held is in the report, verified and with its
figures regenerated. Read the report, not this file.

What changed between the notes and the report, so an old memory of the notes
does not mislead:

- **The figures are no longer suspect.** All seven were regenerated under the
  resolved reading of `v`. The "DO NOT REUSE" table that used to be here is
  gone because the condition it warned about is fixed.
- **The recommended `a` moved from 0.77 to ~0.79, with a range 0.77-0.84.**
  The notes quoted only the transport-weighted monthly regression. Three other
  defensible time means give 0.79 to 0.84, and the spread straddles the regime
  boundary. Report section 7.
- **The model's `Shat` is 18% low, not right.** The notes said the model's
  `Shat` corresponded to `dp = 196` hPa and was therefore correct. That is
  circular: 196 hPa was derived from `Shat` itself. The independent estimate is
  the branch depth the observed `[v]` shape implies, 238 hPa, against which the
  model is 18% low. Report section 5.
- **Open calculations 1, 2, 4 and 5 are done** and in the report. Calculation 3
  (confirm the resolved reading fixes V2) is done and confirmed: run
  `model_output/moist_v2/wc50_a077/`, report section 9. Calculations 6-9 remain
  open, listed in report section 12.
- **`Wstar` and `Hhat/Shat` are the normalisation-invariant quantities.** This
  framing is new in the report and is the cleanest statement of the whole
  result; the notes never identified it.

Scripts are inventoried in the report's Appendix C. The new one is
`scripts/era5_normalization.py`, which owns the two-readings resolution, plus
`scripts/era5_calibration_v2_check.py` for the model test.
