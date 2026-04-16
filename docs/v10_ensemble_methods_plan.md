# v10 ensemble methods plan

**Status:** canonical spec for Phase 3.10 of each NB03 variant
**Author:** NQTa
**Date:** 2026-04-16
**Companion to:** `v10_master_plan.md`, `v10_nb03_test_protocol.md`

Four ensemble methods tested per pipeline. Winner ships. Implements test T02 from `v10_nb03_test_protocol.md`.

---

## 1. The four methods

### 1.1 Method A: Ridge stacking (v9 continuation)

Fit `RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=5)` on OOF predictions from 8 base models.

**Pros:** simple, interpretable weights, fast, documented in `docs/stacking_strategy.md`.

**Cons:** collapses when base predictions are highly correlated (v9 result: alpha at endpoint, worse than best base on test).

**v9 outcome (opx):** lost on test, won on ArcPL. Ship as OOD-only ablation.

### 1.2 Method B: Two-level stacking

```
Level 0: 8 base models → 8 OOF vectors
Level 1: group by model family
  - Tree group: [RF, ERT, XGB, GB, CatBoost, LightGBM] → fit Ridge A
  - Non-tree group: [ElasticNet, MLP] → fit Ridge B
Level 2: stack [Ridge A output, Ridge B output] → fit Ridge final
```

**Pros:** forces diversity via family-level aggregation; less prone to collapse.

**Cons:** more complex, more overfit risk, 3 meta models to track.

**Expected:** may beat Method A when tree vs non-tree have complementary errors.

### 1.3 Method C: Greedy ensemble selection (Caruana 2004)

Iteratively add base models to ensemble with replacement, minimizing validation RMSE.

Algorithm:
```
1. E = [] (empty ensemble)
2. For k in 1..K_max:
     For each base model m:
       Compute RMSE of average(E + [m]) on validation set
     Add best m to E (with replacement)
     Stop if no improvement for 3 iterations
3. Return E
```

`K_max = 50`. Validation = OOF.

**Pros:** non-linear ensembling, handles collinearity well, often wins on tabular benchmarks.

**Cons:** can overfit validation set, interpretability only via which-models-got-picked.

### 1.4 Method D: AutoGluon-Tabular (automated)

AutoGluon's `TabularPredictor` with multi-layer stacking enabled. Trains its own base models AND its own stacking layer.

Use case: if AutoGluon beats our hand-tuned pipeline on a given target, that's an embarrassing but honest finding that we report.

**Pros:** state-of-the-art automated tabular; can include models we don't hand-specify.

**Cons:** black-box, may not agree with our base-model selection, installation issues on Windows per master plan risk 10.

**Fallback if AutoGluon unavailable:** skip and note in methods. Methods A, B, C still tested.

---

## 2. Per-pipeline evaluation

For each pipeline P in {opx, cpx, twopx, universal}:

```
1. Fit 8 base models per standard NB03 protocol
2. Compute 8 OOF vectors (from GroupKFold CV during training)
3. Fit Method A, B, C, D on these OOF vectors
4. Compute test RMSE for each method
5. Compute external-benchmark RMSE for each method (ArcPL for opx/cpx/twopx, held-out for universal)
6. Bootstrap 95% CI on RMSE per method
7. Select winning method per target × track
```

---

## 3. Selection criterion

Winner is declared per (target, track):

1. Lower test RMSE than best single base, OR
2. Lower external RMSE than best single base with test RMSE not worse than best base + 1 C (T) / 0.1 kbar (P)

If multiple methods tie within 0.5 C (T) or 0.05 kbar (P), prefer the simpler method (A > B > C > D).

If no method meets criterion: ship "no ensemble" — canonical model = best single base. Document null result.

---

## 4. Output artifacts

Per pipeline × target × track:

- `models/canonical/{pipeline}/meta_{method}_{target}_{track}.joblib` for each tested method (4 joblibs even if only one ships)
- `results/{pipeline}/nb03_ensemble_winner_{target}_{track}.json`:
  ```json
  {
    "winner_method": "greedy",
    "winner_test_rmse": 77.3,
    "winner_external_rmse": 55.2,
    "method_A_test_rmse": 89.4,
    "method_B_test_rmse": 82.1,
    "method_C_test_rmse": 77.3,
    "method_D_test_rmse": 78.6,
    "bootstrap_cis": {...},
    "selection_rationale": "..."
  }
  ```
- `results/{pipeline}/nb03_ensemble_diagnostics.json` — across-target summary

---

## 5. Implementation

`src/stacking.py` extended (not replaced):

```python
# Existing
def fit_ridge_meta_model(oof_matrix, y_train, alphas=None, cv=5): ...
def stacking_predict(meta_model, base_predictions, base_order): ...

# New for Method B
def fit_two_level_stacking(oof_tree, oof_nontree, y_train): ...

# New for Method C
def fit_greedy_ensemble(oof_matrix, y_train, max_K=50, patience=3): ...

# New for Method D
def fit_autogluon_ensemble(X_train, y_train, groups=None, time_limit=3600): ...

# New cross-method evaluator
def evaluate_ensemble_methods(
    X_train, y_train, oof_dict, y_test_pred_dict, X_external, y_external,
    methods=('ridge', 'two_level', 'greedy', 'autogluon'),
    target='T_C', track='opx_liq', pipeline='opx'
): 
    """Returns dict with method -> {test_rmse, external_rmse, ci_lo, ci_hi}."""
    ...
```

---

## 6. T02 test wiring

From `v10_nb03_test_protocol.md`:

> T02 — Ensemble method shootout: at least one of four ensemble methods beats best single base on both test and external.

The test:

1. Run `evaluate_ensemble_methods(...)` per (target, track) in pipeline
2. Compute winner per criterion in Section 3
3. Log to `v10_nb03_test_log.csv`: `T02, {pipeline}, ..., passed: Y if winner exists else N`
4. Write winner to `nb03_ensemble_winner_{target}_{track}.json`

---

## 7. Stacking propagation

Whatever method wins per pipeline × target × track becomes "the stacked canonical model" for that combination. Every downstream NB (NB04, NB05, NB06, NB07, NB08, NB10) loads it via `load_stacked_model(pipeline, target, track)`.

See `v10_stacking_propagation_audit.md` for explicit NB-by-NB usage requirements.

---

## 8. Figures

Per pipeline:

- `fig_nb03_{pipeline}_ensemble_comparison.png` — 4-panel plot (test T, test P, external T, external P), each panel bar chart of RMSE per method (A/B/C/D) with CI error bars. Horizontal line for "best single base." Winning method annotated.

- `fig_nb03_{pipeline}_ensemble_weights.png` — for Method A and B only, show Ridge coefficients; for Method C, show which base models got picked and how many times; for Method D, show AutoGluon's internal stacking decisions.

Per `v10_figure_audit.md` these get full treatment (Okabe-Ito, JGR-MLC dim, metrics on figure).

---

## 9. v9 context transferred to v10

Lessons from v9 Ridge stacking:

- alpha at endpoint → collinearity warning (flag, continue)
- stacked test RMSE > best base RMSE (4 of 4 opx targets)
- stacked ArcPL T RMSE < best base (won by 2.65 C)

v10 expectation: Methods B and C likely improve test RMSE (where A collapsed). Method D may win either or both.

---

## 10. Package dependencies

- Method A: sklearn (already have)
- Method B: sklearn (already have)
- Method C: sklearn + custom greedy loop (add to `src/stacking.py`)
- Method D: `autogluon.tabular` (add to requirements.txt)

AutoGluon Windows install may need conda; pip wheel often works. If breaks: skip Method D, note in methods that it wasn't tested (honest disclosure).

---

## 11. Compute budget

Per pipeline × target × track:

- Method A: <1 s (Ridge fit)
- Method B: <5 s (three Ridge fits)
- Method C: 30-60 s (greedy loop)
- Method D: 5-30 min (AutoGluon train, depends on time_limit)

Total per pipeline: ~20-40 min of wall time over all 4 tracks × 2 targets × 4 methods.

Across 4 pipelines: ~2-3 h. Falls inside master plan compute estimate.

---

## 12. Deliverables

- Extended `src/stacking.py` with new functions
- `models/canonical/{pipeline}/meta_{A|B|C|D}_*_*.joblib` (up to 32 joblibs per pipeline)
- `results/{pipeline}/nb03_ensemble_winner_*_*.json`
- `fig_nb03_{pipeline}_ensemble_comparison.png` and `_weights.png`
- Updated `docs/stacking_strategy.md` with Method B, C, D descriptions and v10 per-pipeline outcome blocks
