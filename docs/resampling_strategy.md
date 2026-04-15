# P-T Grid Tempered Resampling Strategy

**Status:** proposed for v9 Phase 2 (NB03 rebuild)
**Author:** NQTa
**Date:** 2026-04-15

---

## Background

The opx-liq training data (n=600 from ExPetDB) has a highly imbalanced P-T distribution:

- P quintile breaks: [2.1, 10.0, 12.0, 20.0] kbar
- ~40% of samples sit between 0-10 kbar
- ~20% of samples span 20-60 kbar (the upper tail)
- T coverage is likewise concentrated at crustal conditions, with mantle-interior temperatures underrepresented

Tree-based regressors trained on imbalanced data tend to underpredict the underrepresented regions and regress toward the training-data mean. This regression-to-the-mean phenomenon is documented for ML thermobarometry by Ágreda-López et al. (2024) and is the mechanism driving the ArcPL T underprediction observed in v7/v8 runs.

---

## Approach: tempered resampling

We resample the training data toward a tempered target distribution that sits halfway between the current empirical distribution and a uniform distribution over occupied P-T cells.

### Procedure

1. Define a P-T grid: 5 P bins × 5 T bins, with **range-based** bin edges (equal-width bins spanning `[min, max]` of the training-set P and T respectively). Range-based edges mean dense cells stay dense and sparse cells stay sparse, so the resampling actually has something to do. Quintile edges would force each bin to hold ~20% of samples by construction, making the empirical distribution already near-uniform over bins and defeating the purpose.
2. Count samples per cell: `c_i` for each `(p_bin, t_bin)` cell that is occupied (`c_i > 0`).
3. Compute the uniform target: `u_i = N_total / n_occupied_cells`. Unoccupied cells stay at zero; we do not invent synthetic regions of P-T space that have no experimental support.
4. Set the tempered target: `t_i = round((c_i + u_i) / 2)`.
5. For each occupied cell:
   - If `t_i > c_i`: bootstrap with replacement to add `t_i - c_i` copies
   - If `t_i < c_i`: subsample without replacement down to `t_i`
   - If `t_i == c_i`: leave as-is
6. Concatenate resampled cells to produce the resampled training dataframe.
7. Log a diagnostic table: cell, current count, target count, action taken.

### Expected output

n=550-700 after resampling (tempering moves counts toward uniform, so dense cells shrink and sparse cells grow). Exact number depends on occupied-cell count and rounding.

---

## Why tempered, not uniform, not ArcPL-targeted

**Uniform:** gives every occupied cell equal weight. Maximizes generalization to rare regions but hurts performance on common P-T regions (which is where most downstream use cases sit). The v7/v8 models already perform well on common regions; giving up that performance for marginal gains on the tail is a poor trade.

**ArcPL-targeted:** weighting training toward the ArcPL test distribution would quietly bias the model toward the held-out benchmark. This is methodologically indefensible in peer review ("you trained to the test"). Rejected.

**Tempered:** the middle path. Every occupied cell gets at least uniform weight; common cells retain most of their natural bias. Defensible as a principled compromise between generalization and in-distribution accuracy.

---

## Scope of application

**Training time only.** Resampling is applied to the outer training fold during NB03's final model fit.

**NOT applied to:**
- CV folds during Optuna hyperparameter search (resampling test splits would bias HP selection)
- Outer test set (would bias in-domain evaluation)
- ArcPL external validation set (would bias external reporting)

This is the same pattern used for SMOTE in the imbalanced classification literature: resample training, evaluate on the original distribution.

---

## Interaction with RF/ERT internal bootstrap

`RandomForestRegressor` and `ExtraTreesRegressor` default to `bootstrap=True`, which resamples the training set internally for each tree. When the outer training set has already been bootstrap-augmented by our tempered resampling step, stacking RF's internal bootstrap on top creates a double-bootstrap: trees see a bootstrap of a bootstrap, which shrinks effective per-tree sample diversity and inflates the variance of the ensemble.

**Rule:** when fitting RF or ERT on a resampled training set (Phase 3.9 canonical `_resampled` models), set `bootstrap=False`. This disables the inner resampling and lets the outer tempered resampling be the sole source of sample-level randomization.

For the non-resampled canonical models (Phase 3.4), keep `bootstrap=True` as usual.

XGBoost and GradientBoosting do not use sample-level bootstrap by default (they use `subsample` which is a per-tree row fraction, not replacement sampling), so no adjustment is needed for them.

---

## Expected impact

Medium probability of 2-5 °C improvement on T RMSE (head-to-head n=166 subset and full-ArcPL n=197 coverage).
Modest improvement plausible on high-P pressure prediction.
Minor risk of slightly worse RMSE on the most common P-T region.

RMSE may go up or down after resampling. Both outcomes are valid findings. If RMSE gets worse, we document the experiment as a negative-result ablation in the manuscript and keep the non-resampled canonical models as the primary recommendation.

---

## Implementation location

`src/resampling.py` (new module, to be written in Phase 2).

Functions:
- `compute_pt_grid_bins(df, n_p_bins=5, n_t_bins=5)` → returns P and T bin edges
- `assign_pt_cells(df, p_bins, t_bins)` → returns (p_cell, t_cell) tuples per row
- `tempered_resample(df, target_col_p='P_kbar', target_col_t='T_C', n_p_bins=5, n_t_bins=5, seed=42)` → returns resampled df + logs diagnostics

Called from NB03 Phase 3.9 (new cell) after Optuna final training, to produce a parallel set of `_resampled` canonical models alongside the baseline canonical set.

---

## References

- Ágreda-López et al. (2024) *Computers & Geosciences* — ML thermobarometry regression-to-mean diagnosis
- Branco, Torgo, Ribeiro (2016) *ACM Computing Surveys* — imbalanced regression survey; tempered resampling is consistent with their "relevance-based" approach
- Chawla et al. (2002) — SMOTE; the training-only application pattern originates here
