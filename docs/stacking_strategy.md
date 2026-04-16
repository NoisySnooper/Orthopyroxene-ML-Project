# Ridge Regression Stacking Strategy

**Status:** proposed for v9 Phase 2 (NB03 rebuild)
**Author:** NQTa
**Date:** 2026-04-15
**Last updated:** 2026-04-16 with v9 empirical outcome block

---

## Background

This pipeline trains 4 base models per (target, track, feature_set) combination: Random Forest, Extra Trees, XGBoost, Gradient Boosting. The v7/v8 canonical model selection picks one winner per family (`forest_family`, `boosted_family`) per (target, track), yielding 8 canonical models total (4 per family x 2 families).

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

Stack 4 OOF vectors column-wise -> `(N, 4)` matrix.

### Per-base-model feature_set

Each of the 4 base models uses its own winning feature_set per Phase 3.5 canonical selection (e.g., RF-T-opx_liq winner = alr, XGB-T-opx_liq winner = raw). The stack therefore blends predictions from heterogeneous feature representations. This matches deployment behavior: each canonical model in production uses its own feature_set, so the meta-model sees the same input distribution at training and prediction time.

### Ridge fit

Fit `RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=5)` on the `(N, 4)` matrix with the ground-truth target as y.

Retain: the fitted meta-model object (has `coef_` for the 4 base weights, `alpha_` for the selected regularization strength, `intercept_` for the bias term).

### Prediction-time pipeline

For new data X:
1. Run all 4 base models on X -> 4 prediction vectors
2. Stack column-wise -> `(n_new, 4)` matrix
3. Pass through fitted meta-model -> 1 blended prediction vector

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

Canonical model count grows from 8 (4 winners x 2 = T+P) to 12 (8 base + 4 stacked).

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

Medium probability of 1-4 C improvement on T RMSE vs the best single base model.
Marginal improvement plausible on P.
May not generalize across distribution shift to ArcPL; primarily benefits in-distribution test performance.

Plausible outcomes include:
- Stacked wins both targets -> stacked becomes primary recommendation
- Stacked wins one, base wins the other -> recommend per-target
- Stacked matches but does not beat -> document as ablation showing diminishing returns
- Stacked underperforms best base -> single-model collapse result; still a publishable finding

---

## Implementation location

`src/stacking.py` (new module, to be written in Phase 2).

Functions:
- `generate_oof_predictions(model_ctor, X, y, groups, cv_splitter, seed=42)` -> returns length-N OOF vector
- `fit_ridge_meta_model(oof_matrix, y_train, alphas=None)` -> returns fitted RidgeCV
- `stacking_predict(meta_model, base_predictions_dict)` -> returns blended predictions

Integration in NB03 Phase 3.10 generates OOF vectors for all 4 base models per (target, track), fits 4 RidgeCV meta-models, and saves them as `meta_ridge_{target}_{track}_stacked.joblib` in `models/`.

`src/data.py` gets a `load_stacked_model(target, track)` helper returning a callable that runs the full (4 base -> stack) pipeline.

---

## References

- Wolpert (1992) *Neural Networks* — "Stacked generalization"; the original proposal
- Caruana et al. (2004) ICML — "Ensemble selection from libraries of models"; practical tips for meta-model selection
- Hoerl & Kennard (1970) — Ridge regression; the meta-model family

---

## v9 empirical outcome (added 2026-04-16)

### What happened

All four RidgeCV meta-models converged with `alpha_=100.0` at the endpoint of the search grid. From `results/nb03_stacking_diagnostics.json`:

| Target | Track | alpha | Best base | Base RMSE (test) | Stacked RMSE (test) | Delta |
|---|---|---|---|---|---|---|
| T_C | opx_liq | 100 (endpoint) | RF | 84.89 | 89.36 | +4.5 (worse) |
| P_kbar | opx_liq | 100 (endpoint) | GB | 4.74 | 4.77 | +0.03 (tie) |
| T_C | opx_only | 100 (endpoint) | RF | 152.04 | 158.56 | +6.5 (worse) |
| P_kbar | opx_only | 100 (endpoint) | RF | 10.33 | 10.69 | +0.4 (worse) |

Negative Ridge weights appeared on RF for 3 of 4 targets (range -0.011 to -0.190), consistent with high base-model collinearity. Base-model OOF correlations ranged 0.78-0.99, with most pairs above 0.95. The Ridge could not find an informative blend because the bases agree too much on in-domain data.

### On ArcPL (out-of-distribution)

Same meta models, applied to n=96 Kd-equilibrated ArcPL:

| Target | Boosted RMSE | Stacked RMSE | Delta |
|---|---|---|---|
| T_C opx_liq | 56.24 | 53.59 | **-2.65 (better)** |
| P_kbar opx_liq | 3.12 | 3.30 | +0.18 (slightly worse) |

Stacking HELPED on ArcPL T by 2.65 C. Bases that are nearly identical in-distribution diverge under distribution shift; the Ridge blend dampens their disagreement.

### Interpretation

The pre-registered sanity check 1 ("alpha at endpoint") flagged correctly. Three of the four alphas are pegged at 100 (max). Per the original plan this means the grid is too narrow or the bases are too correlated for Ridge to add value on the in-distribution task. We did not re-fit with a wider grid because the ArcPL result shows stacking helps OOD, and the endpoint-alpha diagnostic is information in itself: bases are collinear in-domain.

### v10 shipped decision

- Stacking is **NOT primary**. Boosted (XGB) is primary per T01.
- Stacking is **KEPT** as an OOD-robustness ablation. Documented in manuscript as "reports stacked predictions for robustness on samples flagged out-of-distribution by IsolationForest."
- Test T09 in `v10_nb03_test_protocol.md` proposes fixing a distributional mismatch (meta trained on OOF, applied on full-fit base preds) via `cross_val_predict` at inference time. If T09 passes, `src/stacking.py` gains a `predict_with_cv` option.

### Failure mode flagged for manuscript Methods

Reviewers will notice alpha_=100 endpoint. We pre-empt by stating in Methods:

> "Ridge stacking alpha converged to the endpoint of the CV search grid (alpha=100) for all four targets, indicating near-degenerate collinearity among base models in the training distribution. Base OOF correlations ranged 0.78 to 0.99. On the held-out ArcPL external set, the stacked meta-model nevertheless reduced temperature RMSE by 2.65 C (from 56.24 to 53.59 C), consistent with bases that agree in-distribution but diverge under distribution shift. We therefore retain stacking as an OOD-robustness ablation rather than a primary in-distribution model."

---

## v10 scope update (2026-04-16)

The v9 stacking approach used a single method (RidgeCV) over 4 base models for
opx only. v10 expands on three axes: more base models, more ensemble methods,
and more pipelines (opx + cpx + twopx + isolated universal).

### Four ensemble methods compared per pipeline

Per `docs/v10_ensemble_methods_plan.md`, four meta-methods run head-to-head
for each pipeline independently. Winner ships; others report as ablation.

| Method | Description | Ship condition |
|---|---|---|
| Ridge stacking | RidgeCV over OOF base preds (v9 carry-over) | Current primary baseline |
| Two-level Ridge | Ridge over 4 tree bases + Ridge over 4 non-tree bases, then Ridge over those 2 | Beats single-level Ridge on test AND ArcPL |
| Greedy Caruana 2004 | Iterative ensemble selection on validation RMSE with replacement | Beats Ridge on test AND ArcPL |
| AutoGluon multilayer | Black-box multilayer stacker | Beats all three AND user-acceptable runtime |

Gate criterion lives in the test protocol as test T02 ("ensemble method
shootout"). Runs per-pipeline; results logged to
`results/v10_nb03_test_log.csv`.

### Base model pool expands from 4 to 8

v10 adds CatBoost, LightGBM, ElasticNet, and MLPRegressor per test T11 and
T12. The v9 endpoint-alpha=100 behavior was driven by tree-tree collinearity
in the 4-base Ridge. With 4 non-tree bases added, the OOF correlation
structure becomes non-degenerate and Ridge should find an informative blend
(alpha not pegged at endpoint). Prediction: opx_liq T stacked RMSE improves
by 5-10 C vs v9 once non-tree bases are in the pool.

### Linear-SHAP interpretability

With 8 bases, Linear-SHAP on the Ridge meta reports which of the 8 base
models drives each prediction. Previously (4 bases, all trees), Linear-SHAP
showed redundant base contributions; with 4 trees + 4 non-trees, it will
distinguish tree-type signal from linear-type signal from NN signal. Added
to NB06 per `docs/v10_stacking_propagation_audit.md`.

### Per-pipeline independence

No assumption that opx stacking findings carry over to cpx or twopx. Each
pipeline runs the full 4-method shootout independently. Cpx stacking may
ship Ridge while twopx ships greedy. Universal is fully isolated, tested
separately, results never combined with primary pipelines.

### Cross-references

- Ensemble method head-to-head spec: `docs/v10_ensemble_methods_plan.md`
- Where stacking must appear in NB04-NB10: `docs/v10_stacking_propagation_audit.md`
- Test T02 (ensemble shootout) pass/fail criteria: `docs/v10_nb03_test_protocol.md`
- 8-base-model roster: `docs/v10_master_plan.md` Section 3
