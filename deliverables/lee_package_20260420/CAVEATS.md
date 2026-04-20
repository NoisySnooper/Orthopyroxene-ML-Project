# Caveats and reconstruction provenance

Items an independent reviewer should know before acting on the numbers
in this package.

## 1. Reconstructed conformal calibration (`results/nb07_conformal_qhat.json`)

The conformal calibration file (qhat_T=92.58 C, qhat_P=10.0 kbar at alpha=0.10)
was rebuilt from archive after the post-consolidation notebook layout dropped
it. Values originate from the pre-v10 calibration run (n_calibration=43). The
semantic validity of these qhat values under the **current** calibration set
has NOT been independently verified. Downstream: nb08 LEPR comparison uses
these qhat values for conformal half-widths on natural-sample predictions.

## 2. Reconstructed per-family winners (`results/nb03_per_family_winners.json`)

Rebuilt via `scripts/data_prep/build_per_family_winners.py` selecting
argmin(mean RMSE) -> argmin(std) -> alphabetical from
`opx_multiseed_summary.csv`. No cross-validation against a pre-consolidation
archive file. Selections:

- Forest family: RF/pwlr (opx_only T, opx_only P, opx_liq T) + RF/alr (opx_liq P)
- Boosted family: LightGBM/alr, XGB/pwlr, GB/raw, GB/raw

## 3. Ship-if-better threshold tightness

The ship-if-better rule uses `overall_delta > 1e-6 AND
max_regime_degradation <= 1e-6`. Effectively "never degrade any regime
by any float amount". Defensible but strict; a looser threshold (e.g.
0.5% relative degradation) would likely let more corrections ship.
Reported RMSE values are unaffected; the rule only gates shipping.

## 4. Phase 2 figure scope

The TabPFN integration originally planned 6 new figures. Shipped 2
(fig44 scoreboard, fig45 opx-only P headline) and relied on pre-existing
figures (fig28, fig30-35) for the remaining panels. No numeric data
impact; cosmetic scope only.

## 5. Papermill execution is visual-only

The 48-cell notebook runs error-free but contains no numeric assertions
(no `assert abs(rmse - 10.35) < 0.1` style guards). Verification is by
eye against the underlying CSVs. Every cell DOES raise
FileNotFoundError if its source CSV is missing, so structural
failures are caught; value-level regressions would require adding
explicit asserts.

## 6. CSV column name retention

Columns such as `v10_pre_rmse` and `v10_post_rmse` remain in the CSVs
as machine contracts; the notebook performs display-time aliasing to
"tuned pre" / "tuned post" for the reader. Renaming the CSV columns
themselves would break every downstream script.
