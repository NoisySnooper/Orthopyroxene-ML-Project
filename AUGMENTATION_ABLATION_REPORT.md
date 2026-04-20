# Augmentation ablation (nb04b) report

**Date:** 2026-04-19
**Branch:** main
**Safety tag:** `pre-nb04b-aug-20260419` -> 134b025
**Interpretation label:** not_supported_aug_degrades

## Scope

4 opx combinations (opx_liq T_C/P_kbar, opx_only T_C/P_kbar) x 3 feature sets (raw/alr/pwlr) x 8 Optuna-tuned models (RF, ERT, XGB, GB, CatBoost, LightGBM, ElasticNet, MLP) x 20 seeds (42-61) = 1920 augmented test-RMSE fits, plus 80 augmented OOF runs (4 winning cells x 20 seeds) for Form A / Form B / ship-if-better.

## Headline table

| track    | target   | non_aug_model   | non_aug_feature_set   |   non_aug_mean_rmse |   non_aug_std | non_aug_winner   | aug_model   | aug_feature_set   |   aug_mean_rmse |   aug_std | aug_ship_a_rate   | aug_ship_b_rate   |   aug_winner_A |   aug_winner_B |   aug_winner_none |   aug_form_a_post_rmse_mean |   aug_form_b_post_rmse_mean |
|:---------|:---------|:----------------|:----------------------|--------------------:|--------------:|:-----------------|:------------|:------------------|----------------:|----------:|:------------------|:------------------|---------------:|---------------:|------------------:|----------------------------:|----------------------------:|
| opx_liq  | T_C      | ElasticNet      | raw                   |              77.062 |         0.000 | excluded         | ElasticNet  | raw               |          77.190 |     0.107 | 0/20              | 0/20              |              0 |              0 |                20 |                      77.694 |                      76.524 |
| opx_liq  | P_kbar   | MLP             | raw                   |               4.404 |         0.460 | excluded         | XGB         | raw               |           4.762 |     0.075 | 17/20             | 0/20              |             17 |              0 |                 3 |                       3.064 |                       5.988 |
| opx_only | T_C      | LightGBM        | alr                   |             146.627 |         0.588 | excluded         | XGB         | raw               |         153.781 |     0.782 | 20/20             | 4/20              |             20 |              0 |                 0 |                     146.684 |                     190.413 |
| opx_only | P_kbar   | RF              | pwlr                  |              10.347 |         0.043 | A                | ERT         | pwlr              |          11.122 |     0.108 | 20/20             | 0/20              |             20 |              0 |                 0 |                       5.951 |                      12.319 |

## Interpretation

The augmentation hypothesis is not supported. Form B fails to ship on 0/4 opx combinations under the Agreda-Lopez 15x Gaussian augmentation protocol, and augmentation additionally degrades aggregate RMSE on every combination. For small experimental petrology datasets with publication-level clustering, 15x Gaussian composition noise is counterproductive. The differentiator between our negative Form B result and Agreda-Lopez positive one must be mineral-specific (orthopyroxene vs clinopyroxene residual structure) or protocol-specific beyond augmentation alone.

## Validation checks

| # | Check | Status | Detail |
|---|-------|--------|--------|
| 1 | pytest all pass | OK | 69 passed in 3.52s |
| 2 | test_augmentation 7 unit tests pass | OK | 7 passed in 1.13s |
| 3 | notebook executed via papermill | OK | looked for C:\Users\NQTa\Documents\MLCourse\Final Project\notebooks\executed\nb04b_aug_test_executed.ipynb |
| 4 | augmentation_ablation_opx_multiseed_results.csv exists | OK | 148532 bytes |
| 5 | augmentation_ablation_opx_bias_correction.csv exists | OK | 22769 bytes |
| 6 | augmentation_ablation_opx_regime_rmse.csv exists | OK | 170380 bytes |
| 7 | augmentation_ablation_opx_headline.csv exists | OK | 945 bytes |
| 8 | fig_aug01_ship_verdict_comparison: PDF + PNG | OK | pdf=True png=True |
| 9 | fig_aug02_aggregate_rmse_delta: PDF + PNG | OK | pdf=True png=True |
| 10 | fig_aug03_residual_structure_per_regime: PDF + PNG | OK | pdf=True png=True |
| 11 | fig_aug04_form_b_breakpoint_stability: PDF + PNG | OK | pdf=True png=True |
| 12 | CANONICAL_FIGURES registers fig_aug01-04 | OK | all 4 registered |
| 13 | results CSV row count == 1920 (4 x 3 x 8 x 20) | OK | got 1920 |
| 14 | no NaN RMSEs in results | OK | n_nan=0 |
| 15 | ship verdict is A/B/none for all 80 rows | OK | rows=80 bad_winner=0 |

Overall: ALL CHECKS PASSED

## Manuscript paragraphs

### Section 3.9 Augmentation sensitivity (Methods)

We test whether the negative Form B ship-if-better result on opx depends on the presence or absence of the 15x Gaussian composition-noise augmentation protocol used by Agreda-Lopez et al. (2024) on their cpx training set. For each of the four opx (track, target) combinations, at each of 20 seeds 42-61, we augmented the training set with 15 noisy copies per original sample (4 combinations x 3 feature sets x 8 models x 20 seeds = 1920 cell-refits). Per-sample noise was multiplicative Gaussian with 3% relative standard deviation, applied independently per copy to every oxide feature, with non-negative clipping. Citation groups were preserved across copies; citation-grouped 10-fold CV splits were computed on the original data so augmented rows always inherit their parent's fold membership and never leak into a held-out fold. Test sets were not augmented. Optuna hyperparameters were frozen at the non-augmented values (`results/optuna_best_params_opx.json`) so this ablation isolates augmentation as the only independent variable against our pre-registered v10 pipeline. Form A and Form B fits and the ship-if-better decision used the same definitions as in Section 3.8 (per-regime OLS for Form A, piecewise bounds on predicted-value quantiles for Form B, ship if overall RMSE improves beyond numerical tolerance and no regime degrades by more than tolerance). Implementation in `src/ablations/augmentation.py`, runnable via `scripts/ablations/run_augmentation_ablation_opx.py` and `scripts/ablations/run_augmentation_oof_bias.py`. Reproducible at tag `pre-nb04b-aug-20260419`.

### Section 5.3 Augmentation sensitivity (Discussion, autofilled)

The augmentation hypothesis is not supported. Form B fails to ship on 0/4 opx combinations under the Agreda-Lopez 15x Gaussian augmentation protocol, and augmentation additionally degrades aggregate RMSE on every combination. For small experimental petrology datasets with publication-level clustering, 15x Gaussian composition noise is counterproductive. The differentiator between our negative Form B result and Agreda-Lopez positive one must be mineral-specific (orthopyroxene vs clinopyroxene residual structure) or protocol-specific beyond augmentation alone.

Per-combination headline (best cell under each protocol):

- **opx_liq/T_C**: non-aug ElasticNet/raw RMSE 77.06+/-0.00 (canon winner: excluded); aug ElasticNet/raw RMSE 77.19+/-0.11; Form A ships 0/20 seeds, Form B ships 0/20 seeds.
- **opx_liq/P_kbar**: non-aug MLP/raw RMSE 4.40+/-0.46 (canon winner: excluded); aug XGB/raw RMSE 4.76+/-0.08; Form A ships 17/20 seeds, Form B ships 0/20 seeds.
- **opx_only/T_C**: non-aug LightGBM/alr RMSE 146.63+/-0.59 (canon winner: excluded); aug XGB/raw RMSE 153.78+/-0.78; Form A ships 20/20 seeds, Form B ships 4/20 seeds.
- **opx_only/P_kbar**: non-aug RF/pwlr RMSE 10.35+/-0.04 (canon winner: A); aug ERT/pwlr RMSE 11.12+/-0.11; Form A ships 20/20 seeds, Form B ships 0/20 seeds.

Across all 80 augmented (combo, seed) cells: Form A wins 57, Form B wins 0, neither ships 23.
See fig_aug01 for ship-verdict comparison, fig_aug02 for aggregate RMSE delta, fig_aug03 for residual structure per regime on opx-only P_kbar, and fig_aug04 for Form B breakpoint stability.

## Known residuals / deferrals

- fig_aug03 visualizes augmented OOF residuals only (not a non-aug vs aug side-by-side). A full non-aug OOF refit on the 4 winner cells was out of scope for the 12-hour budget; the aggregate RMSE comparison in fig_aug02 already captures the net residual magnitude change.
- The Section 5.3 draft language in `manuscripts/opx_2026/text/aug_discussion_5_3.md` is autofilled from this run. Hand-edit for voice before merging into the main manuscript draft.

## Pre-registered commitments reaffirmed

- Canonical artifacts (`results/opx_multiseed_*.csv`, `results/bias_correction_*.csv`, `results/preregistered_scorecard_*.csv`) untouched.
- Optuna hyperparameters frozen at non-aug values; no re-tuning.
- Citation-grouped CV preserved under augmentation via the augmented-aware OOF helper `oof_predict_augmented`.
