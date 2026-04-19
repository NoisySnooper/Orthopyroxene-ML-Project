# v10 stacking propagation audit

**Status:** enforcement doc for stacking integration across all NBs
**Author:** NQTa
**Date:** 2026-04-16
**Companion to:** `v10_ensemble_methods_plan.md`

Per user concern: v9 had stacking in NB03 only. Missed stacking in every downstream scatter plot, benchmark table, SHAP analysis, etc. v10 fixes this.

This doc enumerates where stacking MUST appear in every notebook downstream of NB03.

---

## 1. Definitions

**Stacked model (pipeline P, target T, track TR):** output of `load_stacked_model(pipeline=P, target=T, track=TR)` which returns the winning ensemble method's predictor per T02 test.

For opx: one stacked model per (target × track) = 4 stacked models.
For cpx: 4 stacked models.
For twopx: 2 stacked models (single track, T and P).
For universal: 2 stacked models.

**Total: 12 stacked models** that must be referenced downstream.

---

## 2. Per-notebook stacking requirements

### 2.1 NB04 `nb04_benchmark.ipynb`

**Required stacking appearances:**

| Figure/Table | Stacking inclusion |
|---|---|
| Method benchmark bar chart (ArcPL paired) | Include `Ours {pipeline} stacked` as a bar |
| ArcPL scatter grid (T and P) | Include stacked as separate scatter panel per track |
| Method benchmark (Kd-equilibrated n=96) | Include stacked row in CSV and bar chart |
| Diagnostic encyclopedia (T and P) | Include stacked column/row in multi-panel layout |
| Model heatmap (performance matrix) | Include stacked row |
| LEPR full benchmark | Include stacked row |

**Code requirement:** every `load_canonical_model(...)` call is matched by a `load_stacked_model(...)` call for the same pipeline × target × track. Figures plotted by the same axes.

### 2.2 NB05 `nb05_generalization.ipynb`

**Required stacking appearances:**

| Analysis | Stacking inclusion |
|---|---|
| LOSO pooled RMSE | Stacked entry per (pipeline × target) |
| LOSO per-fold distribution | Stacked boxplot entry |
| Cluster-KFold pooled | Stacked entry |
| Cluster-KFold per-fold | Stacked boxplot |
| TargetBin pooled | Stacked entry |
| LeaveOneRegionOut pooled (NEW) | Stacked entry |
| Generalization figure (all 4 strategies) | Stacked bars |

**Code requirement:** LOSO needs a fresh stacked fit per fold (not just the canonical model). This means:

```python
for train_idx, test_idx in loso_splits:
    # Fit all 8 base models on train_idx
    # Compute OOF on train_idx via nested CV
    # Fit meta model on OOF
    # Predict on test_idx
```

Compute cost: 93 folds × 8 base models × ~1 s fit = ~750 s per pipeline. ~3 h across 4 pipelines. Worth it for rigor.

### 2.3 NB06 `nb06_shap_analysis.ipynb`

**Required stacking appearances:**

| Analysis | Stacking approach |
|---|---|
| Base model SHAP (tree-SHAP) | Per base model as in v9 |
| Stacked SHAP via base weighting | Weight each base's SHAP by Ridge meta coefficient |
| Stacked SHAP via kernel-SHAP on full pipeline | 100 samples, ~10 min |
| Linear-SHAP on meta model | Shows which bases contribute most per sample |

**Code requirement:** new function `src/shap_utils.py::compute_stacked_shap(meta_model, base_models, X)` that handles the Ridge weighting correctly.

For Method C (greedy) or Method D (AutoGluon), SHAP approach differs:
- Greedy: equal-weight SHAP average over selected models
- AutoGluon: use AutoGluon's built-in feature importance (no SHAP)

### 2.4 NB07 `nb07_bias_correction.ipynb` (merged with nb07b)

**Required stacking appearances:**

| Analysis | Stacking inclusion |
|---|---|
| ArcPL residual histograms | One per base model + one for stacked |
| Bias vs prediction scatter | Per base + stacked |
| Bias vs composition (T and P) | Per base + stacked |
| Probe7-style recommendation JSON | Per base + stacked |
| Composition-conditional correction fit | Fit on stacked residuals too (not just forest) |

**Output:** `nb07b_probe7_recommendation_{model}.json` per base model AND per stacked model.

### 2.5 NB08 `nb08_natural_samples.ipynb` (merged with nb08b)

**Required stacking appearances:**

| Figure | Stacking inclusion |
|---|---|
| Cross-mineral 1:1 scatter | Our stacked cpx vs our stacked opx panel |
| Stacked opx vs Agreda cpx panel | |
| Stacked cpx vs Jorgenson cpx panel | |
| Stacked twopx vs Putirka 2-px panel | |
| World map colored by stacked prediction | Separate from base-model map |
| Locality breakdown table | Stacked row per locality |
| Divergence analysis | Include stacked predictions |

### 2.6 NB09 `nb09_manuscript_compilation.ipynb`

**Required stacking appearances:**

| Table | Stacking inclusion |
|---|---|
| Table 2 model performance | Stacked row per pipeline |
| Table 3 Putirka vs ML | Stacked row |
| Table 4 validation summary | Stacked row |
| Table 5 SHAP importance | Stacked column (base-weighted SHAP) |
| Table 7 bias correction | Stacked row |

### 2.7 NB10 `nb10_extended_analyses.ipynb`

**Required stacking appearances:**

| Analysis | Stacking inclusion |
|---|---|
| Monte Carlo uncertainty | Run MC on stacked too, not just forest |
| IQR uncertainty | Stacked (via bootstrap over base models) |
| OOD score vs residual | Stacked residuals binned by OOD score |
| H2O sensitivity | Stacked predictions re-computed under H2O perturbation |
| Two-pyroxene benchmark | Stacked twopx model vs stacked opx vs stacked cpx on same samples |
| Analytical uncertainty propagation | Stacked predictions with composition noise |

### 2.8 NBF `nbF_figures.ipynb`

All NBs above already produce the figures. NBF just copies to `figures/opx/`, `figures/cpx/`, etc., with per-paper subsetting. Per-paper `CANONICAL_FIGURES` list in `config.py` includes stacked figures.

---

## 3. API extensions needed

### 3.1 `src/data.py`

```python
# NEW for v10
def load_stacked_model(
    target, track, 
    pipeline='opx',            # v10 kwarg
    models_dir=None, results_dir=None
):
    """Returns a _StackedPredictor that handles the winning ensemble method
    for the specified pipeline × target × track."""
    ...
```

Implementation detects which method won (from `nb03_ensemble_winner_*.json`) and loads the appropriate meta-model joblib. Caller doesn't need to know if Method A or C won.

### 3.2 `src/stacking.py` extensions

```python
def predict_stacked_with_cv(
    meta_model, base_models_dict, X, 
    method='ridge',            # 'ridge', 'two_level', 'greedy', 'autogluon'
    cv=5
):
    """Unified predict interface across all 4 methods."""
    ...
```

### 3.3 `src/shap_utils.py` (new)

```python
def compute_stacked_shap(
    meta_model, base_models_dict, X,
    method='ridge',
    shap_approach='base_weighted'  # 'base_weighted', 'kernel', 'linear_on_meta'
): ...
```

---

## 4. Phase B audit checks for stacking

During `scripts/v10_phase_b_nb_audit.py`:

For each notebook in NB04-NB10, verify:
1. At least one `load_stacked_model(pipeline, ...)` call per pipeline (opx, cpx, twopx, universal)
2. Every place where `load_canonical_model(...)` is called has a corresponding `load_stacked_model(...)` for the same pipeline
3. Every metric table includes a "stacked" row per pipeline
4. Every scatter grid includes a "stacked" panel per pipeline

Flag BLOCKING if stacked is missing from any of the above.

---

## 5. Test checklist per NB

Phase G acceptance: run `scripts/v10_figure_audit_checker.py` which parses each figure's underlying data and confirms stacked predictions were included. Pass criterion: every benchmark figure has stacked entry.

---

## 6. Cpx exception handling

If for cpx pipeline, no ensemble method wins T02 (all fail the criterion), then `load_stacked_model(pipeline='cpx', ...)` returns the best single base model wrapped in a dummy stack. Downstream code still works; stacked "row" in tables shows same number as best base. Honest null result.

---

## 7. Universal pipeline exception

Per `v10_universal_model_exploration.md`, universal diagnostics are ISOLATED. For NB04-NB10, universal stacking APPEARS ONLY in universal-specific figures/tables (e.g. `fig_nb08_universal_natural.png`), NOT in opx/cpx/twopx primary results.

This is the isolation rule from the user.

---

## 8. Deliverable

After Phase D-G complete, every NB contains stacked predictions alongside base predictions. User sanity check: open any downstream NB, search for "stacked" — should appear in markdown + code for every figure.
