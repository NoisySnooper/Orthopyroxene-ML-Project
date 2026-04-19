# P-T Grid Tempered Resampling Strategy

**Status:** proposed for v9 Phase 2 (NB03 rebuild)
**Author:** NQTa
**Date:** 2026-04-15
**Last updated:** 2026-04-16 with v9 empirical outcome block

---

## Background

The opx-liq training data (n=600 from ExPetDB) has a highly imbalanced P-T distribution:

- P quintile breaks: [2.1, 10.0, 12.0, 20.0] kbar
- ~40% of samples sit between 0-10 kbar
- ~20% of samples span 20-60 kbar (the upper tail)
- T coverage is likewise concentrated at crustal conditions, with mantle-interior temperatures underrepresented

Tree-based regressors trained on imbalanced data tend to underpredict the underrepresented regions and regress toward the training-data mean. This regression-to-the-mean phenomenon is documented for ML thermobarometry by Agreda-Lopez et al. (2024) and is the mechanism driving the ArcPL T underprediction observed in v7/v8 runs.

---

## Approach: tempered resampling

We resample the training data toward a tempered target distribution that sits halfway between the current empirical distribution and a uniform distribution over occupied P-T cells.

### Procedure

1. Define a P-T grid: 5 P bins x 5 T bins, with **range-based** bin edges (equal-width bins spanning `[min, max]` of the training-set P and T respectively). Range-based edges mean dense cells stay dense and sparse cells stay sparse, so the resampling actually has something to do. Quintile edges would force each bin to hold ~20% of samples by construction, making the empirical distribution already near-uniform over bins and defeating the purpose.
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

Medium probability of 2-5 C improvement on T RMSE (head-to-head n=166 subset and full-ArcPL n=197 coverage).
Modest improvement plausible on high-P pressure prediction.
Minor risk of slightly worse RMSE on the most common P-T region.

RMSE may go up or down after resampling. Both outcomes are valid findings. If RMSE gets worse, we document the experiment as a negative-result ablation in the manuscript and keep the non-resampled canonical models as the primary recommendation.

---

## Implementation location

`src/resampling.py` (new module, to be written in Phase 2).

Functions:
- `compute_pt_grid_bins(df, n_p_bins=5, n_t_bins=5)` -> returns P and T bin edges
- `assign_pt_cells(df, p_bins, t_bins)` -> returns (p_cell, t_cell) tuples per row
- `tempered_resample(df, target_col_p='P_kbar', target_col_t='T_C', n_p_bins=5, n_t_bins=5, seed=42)` -> returns resampled df + logs diagnostics

Called from NB03 Phase 3.9 (new cell) after Optuna final training, to produce a parallel set of `_resampled` canonical models alongside the baseline canonical set.

---

## References

- Agreda-Lopez et al. (2024) *Computers & Geosciences* — ML thermobarometry regression-to-mean diagnosis
- Branco, Torgo, Ribeiro (2016) *ACM Computing Surveys* — imbalanced regression survey; tempered resampling is consistent with their "relevance-based" approach
- Chawla et al. (2002) — SMOTE; the training-only application pattern originates here

---

## v9 empirical outcome (added 2026-04-16)

### What happened

Tempered resampling was implemented per specification. `bootstrap=False` correctly set on RF and ERT resampled variants. 8 resampled canonical joblibs produced (4 forest-family + 4 boosted-family).

From `results/nb03_resampling_impact.csv` and the ArcPL paired benchmark:

| Config | Test RMSE change | ArcPL RMSE change |
|---|---|---|
| Forest T_C opx_liq | +3.2 C (worse) | +5.8 C (worse) |
| Boosted T_C opx_liq | +1.3 C (worse) | +2.8 C (worse) |
| Forest P_kbar opx_liq | +1.6 kbar (worse, 41% relative) | +0.64 kbar (worse) |
| Boosted P_kbar opx_liq | +0.2 kbar (worse) | +0.17 kbar (worse) |
| Forest T_C opx_only | +5.1 C (worse) | +14.1 C (worse) |
| Boosted T_C opx_only | +0.8 C (basically tie) | +13.1 C (worse) |
| Forest P_kbar opx_only | +0.6 kbar (worse) | +0.21 kbar (worse) |
| Boosted P_kbar opx_only | +0.3 kbar (worse) | +0.94 kbar (worse) |

**8 of 8 resampled configs worse on ArcPL, 6 of 8 worse on test.** The hypothesis that resampling would reduce ArcPL T underprediction was falsified.

### Why it hurt

Inspection of the resampling diagnostics reveals the mechanism:

1. The P-T grid has 25 cells (5x5) of which only ~12-15 are occupied given the ExPetDB distribution.
2. Tempering toward uniform shrinks the largest populated cells (crustal P 2-10 kbar, T 1000-1200 C) from ~120 samples to ~60, and grows the tail cells from ~5-15 samples to ~35 via bootstrap.
3. The bootstrap oversampling on sparse cells means the RF/XGB see repeated copies of the same 5-10 samples, which causes overfitting to specific outlier experiments in the tail.
4. Combined: common-region performance drops because training density dropped; tail-region performance does not improve because the bootstrap duplicates are not new information.

### v10 shipped decision

- Resampling is **NOT canonical** per test T03 in `v10_nb03_test_protocol.md`.
- Resampled joblibs move to `models/ablation/resampled/` as part of Phase A cleanup.
- `nb03_per_family_winners.json` gains an `ablation_tested` metadata block listing resampling as rejected.
- Resampling code in `src/resampling.py` remains importable for reproducibility but is not called by the canonical NB03 path.

### Lessons for v11 framework paper

If cpx training data is less imbalanced than opx (ExPetDB likely has more low-P cpx experiments), resampling may have different effects. Re-test per-track in the cpx pipeline rather than assuming the opx outcome carries over.

---

## v10 scope update (2026-04-16)

Resampling was ablated for opx in v9 (8/8 worse on ArcPL, 6/8 worse on test).
v10 re-tests per pipeline because the opx outcome does not mechanically carry
over to cpx, twopx, or universal. Each pipeline has its own training-data P-T
distribution and its own grid occupancy pattern.

### Per-pipeline re-test via test T03

Per `docs/nb03_test_protocol.md`, test T03 ("resampling vs none") runs
independently for each of opx_only, opx_liq, cpx_only, cpx_liq, twopx, and
universal. For each pipeline:

1. Compute P-T grid occupancy and report to `results/v10_resampling_occupancy_{pipeline}.csv`
2. Fit all 8 base models with and without tempered resampling
3. Evaluate on pipeline-specific test set AND ArcPL (where applicable)
4. Ship condition: resampled version must beat non-resampled on BOTH test AND ArcPL with 95% bootstrap CI not crossing zero, for at least 5 of 8 base models
5. Record outcome in `results/v10_nb03_test_log.csv`

### Predicted outcomes

- **opx_only, opx_liq:** likely to fail again (v9 result reproduces). Confirmation adds statistical weight; ablated definitively across 8 models rather than 4.
- **cpx_only, cpx_liq:** unclear. Cpx is more abundant in ExPetDB (~1500-2000 expected vs 600 opx-liq); the P-T distribution may be broader and more even, which would make tempering less aggressive and less damaging to dense cells. Could pass for cpx.
- **twopx:** small training set (~500-800) with likely strong P-T clustering at mantle conditions. Resampling may over-duplicate rare cells. Likely fails.
- **universal:** masking architecture amplifies resampling effects because a resampled twopx pair is counted once per phase-combination variant. Likely fails.

### Interaction with 8-model roster

v9 tested resampling on 4 models (RF, ERT, XGB, GB). v10 adds CatBoost,
LightGBM, ElasticNet, and MLP. ElasticNet and MLP are particularly sensitive
to duplicated samples through their gradient-based fits; resampling could
cause mode collapse. Test T03 reports per-model pass/fail; acceptance gate
is majority of models (>=5/8).

### Code path

`src/resampling.py` stays as-is. No changes needed. v10 adds a driver in
`notebooks/nb03_{pipeline}_baseline_models.ipynb` that invokes
`tempered_resample` once per pipeline per test T03 invocation.

### Cross-references

- Per-pipeline test definition: `docs/nb03_test_protocol.md` test T03
- 8-model roster: `docs/master_plan.md` Section 3
- Per-pipeline test log schema: `docs/master_plan.md` Section 7
