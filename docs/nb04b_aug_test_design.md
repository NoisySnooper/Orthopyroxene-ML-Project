# nb04b augmentation ablation design

**Added:** 2026-04-19
**Notebook:** `notebooks/nb04b_aug_test.ipynb`
**Module:** `src/ablations/augmentation.py`
**Drivers:**

- `scripts/ablations/run_augmentation_ablation_opx.py` - Section 4 big compute (1920 fits)
- `scripts/ablations/run_augmentation_oof_bias.py` - Sections 5-7 OOF + bias correction on the 4 winning cells

## Why this notebook exists

The Section 5.3 discussion draft of the opx manuscript attributes the 0/8
Form B ship rate on opx pipelines to the absence of the 15x Gaussian-
noise augmentation protocol that Agreda-Lopez et al. (2024) used on cpx.
That attribution is currently unsupported by direct evidence. nb04b runs
the obvious counterfactual: refit the same 8 Optuna-tuned families on
the same 4 opx combinations with 15x Gaussian-noise augmentation and
test whether Form B now ships.

A positive result (Form B ships on >=1 opx combination under
augmentation) supports the Section 5.3 hypothesis. A negative result
(Form B still fails to ship on 0/4) is informative too - it rules out
augmentation as the explanatory variable and shifts attribution to
mineral-specific or protocol-specific factors.

## What stays fixed

- **Optuna hyperparameters.** Frozen at the values in
  `results/optuna_best_params_opx.json`, reused verbatim. No re-tuning.
  This isolates augmentation as the single independent variable.
- **Test set.** Un-augmented. Test augmentation would be circular since
  it inflates training data variance into the error estimate.
- **Citation grouping.** Preserved across augmented copies. 10-fold
  citation-grouped CV splits are computed on the original data; augmented
  rows inherit their parent's fold membership and never leak into a
  held-out fold.
- **Canonical artefacts.** `results/opx_multiseed_*.csv`,
  `results/bias_correction_*.csv`, and `results/preregistered_scorecard_*.csv`
  are never touched. nb04b writes only to
  `results/augmentation_ablation_opx_*.csv`.

## Augmentation protocol

Multiplicative Gaussian noise per the Agreda-Lopez (2024) specification:

```
X_aug[i, j, k] = X[i, j] * (1 + epsilon[i, j, k])
epsilon ~ N(0, 0.03^2)   # 3% relative std
y_aug[i, j, k] = y[i]    # target unchanged
```

with `n_copies=15` (15 noisy copies per original, stacked on top of the
originals for a 16x total). `clip_nonneg=True` guarantees oxide
concentrations stay non-negative. Independent noise per copy, seeded by
the split seed so the protocol is reproducible.

## Scope

- **4 (track, target) combinations**: `opx_liq`/T_C, `opx_liq`/P_kbar,
  `opx_only`/T_C, `opx_only`/P_kbar.
- **3 feature sets**: `raw`, `alr`, `pwlr`.
- **8 Optuna-tuned models**: RF, ERT, XGB, GB, CatBoost, LightGBM,
  ElasticNet, MLP. TabPFN is excluded (TabPFN's in-context architecture
  has no "refit with augmented training data" semantics).
- **20 seeds**: 42-61, matching the canonical multiseed protocol.
- **Total**: 1920 augmented test-RMSE fits + 80 augmented OOF runs for
  bias correction.

## Deliverables

### Code

- `src/ablations/augmentation.py` - augmentation module (`augment_gaussian`,
  `make_augmented_cv_splits`, `oof_predict_augmented`, `noise_profile`).
- `tests/test_augmentation.py` - 7 unit tests.
- `scripts/ablations/run_augmentation_ablation_opx.py` - Section 4 driver.
- `scripts/ablations/run_augmentation_oof_bias.py` - Sections 5-7 driver.
- `notebooks/nb04b_aug_test.ipynb` - the notebook itself.

### Results CSVs

- `augmentation_ablation_opx_multiseed_results.csv` - 1920 per-cell rows
  (pipeline, track, target, feature_set, model, seed, test_rmse).
- `augmentation_ablation_opx_multiseed_summary.csv` - aggregated (mean,
  std, min, max, count) per (track, target, feature_set, model).
- `augmentation_ablation_opx_bias_correction.csv` - 80 rows per (track,
  target, seed) with ship verdicts and Form A/B parameters.
- `augmentation_ablation_opx_regime_rmse.csv` - per-regime pre/post RMSE.
- `augmentation_ablation_opx_headline.csv` - 4-row headline comparison
  table ready for the manuscript.
- `augmentation_ablation_opx_oof.pkl` - pickled OOF arrays for figure
  generation.

### Figures (CANONICAL_FIGURES num 40-43)

- `fig_aug01_ship_verdict_comparison` - Form A / Form B / none counts,
  non-augmented vs augmented, across 4 opx combinations.
- `fig_aug02_aggregate_rmse_delta` - aggregate test RMSE per combination,
  non-aug best vs aug best, with Putirka reference where available.
- `fig_aug03_residual_structure_per_regime` - OOF residual violins per P
  regime, non-aug (top) vs aug (bottom), for opx_only P_kbar.
- `fig_aug04_form_b_breakpoint_stability` - Form B (alpha_L, alpha_R)
  vs seed for opx_only P_kbar under aug.

Each figure has PDF, PNG, and `.txt` caption sidecar.

### Manuscript paragraphs

- Section 3.9 Methods paragraph describing the augmentation protocol.
- Section 5.3 Discussion paragraph reporting findings (branch-selected
  by actual result).

## Relationship to the main manuscript

nb04b is an **optional ablation**, not on the critical path to Section 4
results. The paper's pre-registered verdict (Form A ships opx-liq T_C;
Form B ships nothing on opx) stands regardless of nb04b's outcome.
What nb04b provides is honest attribution for *why* Form B fails on opx.

If Form B ships under augmentation, Section 5.3 becomes "Form B is
training-distribution-gated; our non-augmented protocol sits on the
wrong side of the gate, explaining the disagreement with Agreda-Lopez."
If Form B still fails under augmentation, Section 5.3 becomes "Form B
failure on opx is not an augmentation artefact; the differentiator is
mineral-specific or protocol-specific beyond augmentation."

Either finding is publishable. Hiding the test result would not be.

## Wall-clock estimate

Timing probe (8 RF cells, raw opx_liq T_C): 6.2s/cell. Projected total:
1920 x 6.2s = ~3.3 h for Section 4 + ~30 min for Sections 5-7 OOF/bias =
**~4 h total**. Well under the 12-hour compute budget.
