# v10 notebooks compatibility audit

**Status:** pre-Phase-C gate
**Author:** NQTa
**Date:** 2026-04-16, updated for v10+v11
**Companion to:** `v10_master_plan.md`

Before rebuilding NB03 in its four new variants (opx, cpx, twopx, universal), every downstream notebook must read NB03 output through the documented public API — no hardcoded filenames, no hardcoded feature_set strings, no hardcoded test indices.

---

## 1. The NB03 public API — must be preserved or migrated

### Channel A: canonical model loader

```python
from src.data import (
    canonical_model_filename,
    canonical_model_spec,
    load_canonical_model,
    load_stacked_model,
    load_stacked_manifest,
)

# v10 extends signature to include pipeline
model = load_canonical_model(target='T_C', track='opx_liq', family='boosted', pipeline='opx')
```

**v10 signature change:** add `pipeline` kwarg (opx / cpx / twopx / universal). Default `pipeline='opx'` for backward compat. Update `data.py` loader path to `MODELS/canonical/{pipeline}/{filename}`.

Return types unchanged:
- `canonical_model_filename` returns `str`
- `canonical_model_spec` returns dict with keys `model_name`, `feature_set`, `filename`, `rmse_test_mean`, `rmse_test_std`, `r2_test_mean`, `r2_test_std`
- `load_canonical_model` returns fitted estimator
- `load_stacked_model` returns `_StackedPredictor` with `.predict(df)` and `.predict_base(df)`

### Channel B: JSON manifests

One per pipeline:
- `results/opx/nb03_per_family_winners.json`
- `results/cpx/nb03_per_family_winners.json`
- `results/twopx/nb03_per_family_winners.json`
- `results/universal/nb03_per_family_winners.json`

Schema: same as v9. Keys: `forest_family`, `boosted_family`, `stacked_family`. **v10 adds new families:** `elastic_family`, `mlp_family`, `catboost_family`, `lightgbm_family`, `ensemble_winner_family` (winner of 4-way ensemble test).

Per-pipeline Optuna params: `results/{pipeline}/nb03_optuna_best_params.json`

Stacking manifests: `results/{pipeline}/nb03_stacked_members_{target}_{track}.json`

### Channel C: prediction and split arrays

- `results/{pipeline}/nb03_canonical_test_predictions.npz` — keys by `(target, track, family)`, values are np.ndarray
- `data/splits/{train,test}_indices_{track}.npy` (expanded: opx_only, opx_liq, cpx_only, cpx_liq, twopx, universal)

---

## 2. Per-notebook audit

### NB01 `nb01_data_cleaning.ipynb`

**Consumes:**
- Raw ExPetDB xlsx
- Raw LEPR xlsx

**Produces:**
- v9: `opx_clean_core.parquet`, `opx_clean_full.parquet`, `opx_clean_opx_liq.parquet`, `opx_clean_opx_only.parquet`
- v10 additions: `cpx_clean_core.parquet`, `cpx_clean_cpx_liq.parquet`, `cpx_clean_cpx_only.parquet`, `twopx_clean_core.parquet`, `universal_clean_core.parquet`

**v10 audit:**
- [ ] Verify NB01 cleaning logic applies consistently to cpx (Wo > 20 cut for cpx per standard convention vs Wo < 5 for opx)
- [ ] Verify twopx pair matching logic: same experiment ID, same charge, both opx and cpx analyzed
- [ ] Verify universal training set deduplication (opx-only from one source, opx-liq from another, cpx_liq from third — ensure no sample is counted twice)

**Expected severity:** 0 BLOCKING. NB01 is expanded but not API-breaking.

### NB02 `nb02_eda_pca.ipynb`

**Consumes:** v9 processed parquets
**Produces:** `opx_clean_core_with_clusters.parquet` + EDA figures

**v10 audit:**
- [ ] Expand to cpx and twopx EDA (separate figures)
- [ ] Per-pipeline PCA and k-means
- [ ] Per-pipeline chemical_cluster column

**Expected severity:** 0 BLOCKING, 1 WARN (figure naming convention needs `{pipeline}_` prefix).

### NB03 (4 variants) — rebuild anyway

Not a compat audit target; these are the rebuilt ones.

### NB04 `nb04_benchmark.ipynb` (merged NB04 + NBM)

**Consumes:**
- `load_canonical_model(pipeline, target, track, family)` for every pipeline × target × track × family combo (5 × 2 × 2 × 5+ = ~100 loads)
- `load_stacked_model(pipeline, target, track)` per pipeline × target × track
- External models from `src/external_models.py`: Agreda, Jorgenson, Wang, Putirka, Petrelli (NEW)
- `data/splits/test_indices_*.npy` per track

**v10 audit:**
- [ ] Pipeline kwarg added to every canonical loader call
- [ ] No hardcoded filenames — all resolved through API
- [ ] Option B Kd-filter scope preserved (n=96 equilibrated subset)
- [ ] Model heatmap spec integrated (replaces standalone NBM)
- [ ] All 8 base models + 4 ensemble methods appear in every benchmark figure
- [ ] Stacking included in every scatter plot

**Known v9 issues to fix in Phase G:**
- Two parallel ArcPL benchmark scopes (n=112 paired 3-phase AND n=96 Kd-equilibrated). Pick n=96 as headline, n=112 as supplementary figure.
- `fig_nb04_arcpl_opx_liq_scatter.png` shows only forest + boosted — must include stacked + every other model
- `fig_nb04_method_benchmark_paired.png` label collisions (programmatic fix)
- `fig_nb04_diagnostic_encyclopedia_{P,T}.pdf` incomplete — must show all model families per track

**Expected severity:** 3 BLOCKING (API kwarg, scope headline pick, figure completeness), 2 WARN.

### NB05 `nb05_generalization.ipynb` (renamed from nb05_loso)

**Consumes:**
- Training data parquets per pipeline
- `config.SPLIT_SEEDS` (20 seeds)
- `load_per_family_winners(pipeline)` to know which feature sets to re-fit on folds
- `src.evaluation.loso_splits`, `cluster_kfold_splits`, `stratify_labels` — unchanged API, extended for cpx/twopx

**v10 audit:**
- [ ] LeaveOneRegionOut added (groups by tectonic_setting)
- [ ] Per-pipeline fold metrics
- [ ] PWLR feature set still canonical per T05 for opx — re-verify per pipeline

**Expected severity:** 0 BLOCKING, 1 WARN (region column must be in training data; confirm NB01 propagates it).

### NB06 `nb06_shap_analysis.ipynb`

**Consumes:**
- `load_canonical_model('T_C', track, 'forest', pipeline)` and `('P_kbar', track, 'forest', pipeline)` for tree-SHAP
- Meta ridge models for linear-SHAP
- MLP for kernel-SHAP (NEW)
- `load_opx_liq`, `load_cpx_liq`, `load_twopx`, etc.

**v10 audit:**
- [ ] Unified SHAP module `src/shap_utils.py` — add
- [ ] tree-SHAP per (pipeline × target × track) for RF/ERT/XGB/GB/CatBoost/LightGBM
- [ ] linear-SHAP on Ridge meta-model per (pipeline × target × track)
- [ ] KernelSHAP on MLP (100 samples, ~10 min each)
- [ ] Stacked SHAP: combine base-model tree-SHAP weighted by meta coefficients

**Expected severity:** 0 BLOCKING, 2 WARN (unified module needs careful testing; MLP slow).

### NB07 `nb07_bias_correction.ipynb` (merged with nb07b)

**Consumes:**
- `nb04_arcpl_*_predictions_*.csv` for residual analysis
- QRF joblibs
- Per-pipeline prediction CSVs

**v10 audit:**
- [ ] "NULL result" language updated to "NULL for T, SIGNIFICANT for P" (matches test T08)
- [ ] Composition-conditional T correction shipped per T07
- [ ] Per-pipeline probes (opx T bias, cpx T bias, twopx T bias)
- [ ] Probe7-equivalent for boosted (not just forest as in v9)

**Expected severity:** 1 BLOCKING (NB07+NB07b merge conflict resolution), 1 WARN (text update for null/significant).

### NB08 `nb08_natural_samples.ipynb` (merged with nb08b)

**Consumes:**
- `load_canonical_model` per pipeline for opx, cpx, twopx, universal models
- `src.external_models.predict_jorgenson`, `predict_agreda`, `predict_petrelli` (NEW)
- Thermobar for Putirka 2-px
- `data/natural/2024-12-SGFTFN_ORTHOPYROXENES.csv` (raw opx with lat/lon)
- Cpx natural CSV (pulled in Phase A)

**v10 audit:**
- [ ] World map integrated (not standalone NB08b)
- [ ] All 2-way comparison scatter plots added per user spec: a, b, c, d combinations
- [ ] RMSE, R², slope, intercept, 1:1 deviation on every scatter
- [ ] Locality-stratified breakdown
- [ ] Opx-cpx convergence for samples where both exist
- [ ] Twopx model inference on natural samples where both minerals available

**Expected severity:** 2 BLOCKING (data re-merge with lat/lon, new figure spec implementation), 3 WARN.

### NB09 `nb09_manuscript_compilation.ipynb`

**Consumes:** all upstream result CSVs

**v10 audit:**
- [ ] Per-paper table subsets: opx paper uses ~10 tables, cpx paper uses ~15 tables
- [ ] Stale n=204 caption in cell 22 updated
- [ ] Manuscript narrative corrections baked in (Putirka wash on T, beat on P, stacking OOD-only)

**Expected severity:** 1 BLOCKING (per-paper split), 1 WARN (n=204 fix).

### NB10 `nb10_extended_analyses.ipynb`

**Consumes:**
- QRF, IsolationForest joblibs
- Per-pipeline predictions
- Two-pyroxene benchmark (currently empty CSV)

**v10 audit:**
- [ ] Re-run NB10 two-pyroxene cell to populate empty CSV
- [ ] Per-pipeline OOD scores
- [ ] H2O sensitivity extended to cpx
- [ ] MC uncertainty propagation per pipeline

**Expected severity:** 1 BLOCKING (empty CSV), 1 WARN (per-pipeline expansion).

### NBF `nbF_figures.ipynb`

**Consumes:** all upstream outputs + `config.CANONICAL_FIGURES`

**v10 audit:**
- [ ] `CANONICAL_FIGURES` list completely rewritten per `v10_figure_audit.md`
- [ ] Per-paper subsets (opx paper figures, cpx paper figures)
- [ ] Okabe-Ito palette enforced via `src/plot_style.py`
- [ ] All JGR-MLC dimensions

**Expected severity:** 2 BLOCKING (stale list, per-paper split), 1 WARN.

---

## 3. Phase B audit script spec

`scripts/v10_phase_b_nb_audit.py`:

```python
# Per notebook, do:
1. Parse ipynb JSON
2. Extract all code cells
3. For each code cell:
   a. Grep for hardcoded .joblib filenames (flag if not resolved via canonical_model_filename)
   b. Grep for hardcoded feature_set strings ('alr', 'pwlr', 'raw') outside of loader context
   c. Grep for hardcoded test indices (numeric arrays)
   d. Check imports: all src.* modules exist
   e. For each file-write call, verify path is under {results/figures/logs} dir
4. Output to results/v10_phase_b_audit.csv:
   columns: notebook, cell_index, issue_type, excerpt, severity (BLOCKING/WARN/OK), suggested_fix
5. Summary: total BLOCKING, total WARN per notebook
```

Severity ceiling: **0 BLOCKING before Phase C starts**.

---

## 4. Fix prioritization

When Phase B audit returns results:

1. **Resolve all BLOCKING first.** These break Phase C/D/E/F.
2. **Resolve WARN before Phase G.** They don't break model fitting but do break downstream.
3. **OK items** require no action.

If >3 BLOCKING found per notebook, escalate: NB may need bigger refactor before rebuild.

---

## 5. Backward compat for pipeline kwarg

Since v10 extends `load_canonical_model` signature:

```python
# v10 signature
def load_canonical_model(target, track, family, pipeline='opx', models_dir=None, results_dir=None):
    ...
```

Default `pipeline='opx'` preserves v9 call sites. New callers explicitly pass pipeline:

```python
# Old call works in v10 (loads opx by default)
opx_model = load_canonical_model('T_C', 'opx_liq', 'boosted')

# New cpx call
cpx_model = load_canonical_model('T_C', 'cpx_liq', 'boosted', pipeline='cpx')
```

Phase B audit flags any v9 call that should be explicit pipeline='opx' for readability, even though it works by default.

---

## 6. Pre-flight runs

Before Phase B audit, run a minimal smoke test:

```python
# scripts/v10_preflight_import_smoke.py
from src import features, models, data, evaluation, stacking, optuna_search, external_models, calibration, resampling, plot_style, io_utils, geotherm
print("All imports OK")
```

If any import fails, fix before running Phase B.

---

## 7. Acceptance gate

Phase B audit passes when:
- 0 BLOCKING across all 14 notebooks
- <5 WARN per notebook OR all WARNs documented in follow-up ticket list
- Pre-flight import smoke passes
- User confirms in chat: "Phase B audit clean, proceed to Phase C"

Only then does Phase C start.
