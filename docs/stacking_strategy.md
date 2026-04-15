# Ridge Regression Stacking Strategy

**Status:** proposed for v9 Phase 2 (NB03 rebuild)
**Author:** NQTa
**Date:** 2026-04-15

---

## Background

This pipeline trains 4 base models per (target, track, feature_set) combination: Random Forest, Extra Trees, XGBoost, Gradient Boosting. The v7/v8 canonical model selection picks one winner per family (`forest_family`, `boosted_family`) per (target, track), yielding 8 canonical models total (4 per family × 2 families).

Stacking adds a meta-model that learns optimal weights for combining all 4 base predictions per (target, track). When base models make partly uncorrelated errors, stacked predictions cancel noise and improve on any single model. When base models are near-identical, stacking collapses to the best single weight and adds no value; this would be an honest, informative result.

---

## Approach: Ridge regression with cross-validated alpha

**Meta-model:** `sklearn.linear_model.RidgeCV` with 5-fold CV over `alpha = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`.

**Input to meta-model:** out-of-fold (OOF) predictions from the 4 base models. Each OOF prediction was produced by a base model trained on a fold that excluded the predicted sample, so the meta-model sees honest predictions, not in-sample fits.

**Output:** a single blended prediction per (target, track).

### OOF generation procedure

For each base model:
1. Use 5-fold `GroupKFold` by Citation (same as the existing hyperparameter CV).
2. For each fold: fit on train indices, predict on val indices, store predictions at val positions in a length-N array.
3. Result: one N-length OOF vector per base model.

Stack 4 OOF vectors column-wise → `(N, 4)` matrix.

### Per-base-model feature_set

Each of the 4 base models uses its own winning feature_set per Phase 3.5 canonical selection (e.g., RF-T-opx_liq winner = alr, XGB-T-opx_liq winner = raw). The stack therefore blends predictions from heterogeneous feature representations. This matches deployment behavior: each canonical model in production uses its own feature_set, so the meta-model sees the same input distribution at training and prediction time.

### Ridge fit

Fit `RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=5)` on the `(N, 4)` matrix with the ground-truth target as y.

Retain: the fitted meta-model object (has `coef_` for the 4 base weights, `alpha_` for the selected regularization strength, `intercept_` for the bias term).

### Prediction-time pipeline

For new data X:
1. Run all 4 base models on X → 4 prediction vectors
2. Stack column-wise → `(n_new, 4)` matrix
3. Pass through fitted meta-model → 1 blended prediction vector

---

## Why Ridge, not linear, not RF

**Unregularized linear regression:** when base predictions are highly correlated (they often are across forest/boosted models on the same input), the Gram matrix is near-singular and unregularized coefficients can blow up or swing wildly. One trial removed a tree could flip the weights.

**RF or gradient-boosted meta-model:** can capture nonlinear blending patterns but (a) with only 4 base predictors, nonlinearity has little room to help, (b) adds substantial variance without validation signal, (c) tanks interpretability (no weight per base model to report).

**Ridge:** regularized linear blending. CV-selected alpha picks the right amount of shrinkage for the correlation structure of the OOF matrix. Interpretable: the 4 coefficients tell you which base models contribute most. Standard choice in stacking literature.

---

## Stacking is a new canonical family

Per the existing family naming convention (`forest_family`, `boosted_family`), we add:

```
stacked_family: meta-model output, no candidate ranking, single member per (target, track)
```

Canonical model count grows from 8 (4 winners × 2 = T+P) to 12 (8 base + 4 stacked).

**Placement in pipeline:**
Phase 3.10 (new NB03 cell), after Phase 3.4 (Optuna final training) and Phase 3.5 (winner selection). Base model winners remain valid canonical models in their own right; stacking adds one more canonical model per (target, track), it does not replace the base winners.

---

## Sanity checks and failure modes

The Phase 2.6 implementation should log and flag:

1. **Alpha at endpoint:** if `alpha_ == 100.0` (max) or `alpha_ == 0.001` (min), the search range is too narrow. Widen and refit.
2. **Single-model collapse:** if any base weight has `|coef_| > 0.9` and the others are near zero, stacking is effectively picking one model. Report this as an honest finding, not a bug.
3. **Negative weights:** Ridge does not constrain weights to be non-negative. A small negative weight (say -0.1) is fine and statistically informative. A large negative weight (< -0.5) may indicate severe multicollinearity or overfit; investigate before trusting it.
4. **Blank OOF column:** if one base model failed during OOF and the column has NaNs, drop that base model from the stack and note it in the manifest.

---

## Expected impact

Medium probability of 1-4 °C improvement on T RMSE vs the best single base model.
Marginal improvement plausible on P.
May not generalize across distribution shift to ArcPL; primarily benefits in-distribution test performance.

Plausible outcomes include:
- Stacked wins both targets → stacked becomes primary recommendation
- Stacked wins one, base wins the other → recommend per-target
- Stacked matches but does not beat → document as ablation showing diminishing returns
- Stacked underperforms best base → single-model collapse result; still a publishable finding

---

## Implementation location

`src/stacking.py` (new module, to be written in Phase 2).

Functions:
- `generate_oof_predictions(model_ctor, X, y, groups, cv_splitter, seed=42)` → returns length-N OOF vector
- `fit_ridge_meta_model(oof_matrix, y_train, alphas=None)` → returns fitted RidgeCV
- `stacking_predict(meta_model, base_predictions_dict)` → returns blended predictions

Integration in NB03 Phase 3.10 generates OOF vectors for all 4 base models per (target, track), fits 4 RidgeCV meta-models, and saves them as `meta_ridge_{target}_{track}_stacked.joblib` in `models/`.

`src/data.py` gets a `load_stacked_model(target, track)` helper returning a callable that runs the full (4 base → stack) pipeline.

---

## References

- Wolpert (1992) *Neural Networks* — "Stacked generalization"; the original proposal
- Caruana et al. (2004) ICML — "Ensemble selection from libraries of models"; practical tips for meta-model selection
- Hoerl & Kennard (1970) — Ridge regression; the meta-model family
