# v10 cpx pipeline plan

**Status:** canonical spec for `nb03_cpx_baseline_models.ipynb`
**Author:** NQTa
**Date:** 2026-04-16
**Companion to:** `v10_master_plan.md`

Mirrors the opx pipeline architecture. Differences documented.

---

## 1. Scope

Two tracks:

- **cpx_only**: clinopyroxene composition alone → predict P, T
- **cpx_liq**: clinopyroxene + paired liquid composition → predict P, T

Primary benchmark: ArcPL Kd-equilibrated subset n~100 (verified during Phase D), head-to-head with Agreda-Lopez 2024 (cpx-liq), Jorgenson 2022 (cpx-only and cpx-liq), Wang 2021 (cpx-liq), Petrelli 2020, Putirka 2008 (cpx-liq eq33, eq30).

---

## 2. Training data

### 2.1 Primary source: ExPetDB

ExPetDB has cpx analyses and cpx+liq pairs. Expected:

- ~8,000 cpx analyses total
- ~4,000 cpx+liq pairs after Kd Fe-Mg filter (Putirka 2008 range 0.24-0.30 for cpx, stricter than opx)
- Wo > 20 mol% filter (pyroxene quadrilateral cpx cut, augite through diopside)
- En + Fs + Wo = 1 (normalized)

### 2.2 Secondary source: LEPR (already in repo)

`data/raw/external/LEPR_Wet_Stitched_April2023_Norm100Anhydrs.xlsx` contains cpx data alongside opx. Phase A extracts the cpx subset.

### 2.3 Verification before training

`scripts/v10_external_models_audit.py` (Phase A) confirms whether Agreda / Jorgenson / Wang / Petrelli used LEPR. If yes, our training set overlaps theirs — the "beat them" claim becomes stronger because we're training on similar data and winning via methodology.

If they used private data (unlikely per their papers), we may have smaller n. Phase D logs actual vs expected.

### 2.4 Expected final n

- cpx_only training: 6,000-9,000 (ExPetDB + LEPR cpx)
- cpx_liq training: 3,000-5,000 (subset with liquid pair)
- ArcPL cpx subset: ~200 after Kd filter

---

## 3. Features

### 3.1 Three baseline feature sets (same as opx)

- raw: oxide weight percentages (9 cpx oxides + 8 liq oxides for cpx_liq track) + engineered ratios
- alr: additive log-ratio with SiO2 denominator
- pwlr: pairwise log-ratio, every unique oxide pair

### 3.2 Cpx-specific engineered features (NEW)

From `src/cpx_features.py` (new module):

- **Jd** (jadeite) component: NaAlSi2O6, computed from Na_cat and Al_cat on M2 and tetrahedral
- **Di** (diopside): CaMgSi2O6 endmember fraction
- **Aeg** (aegirine): NaFe3+Si2O6, estimated with Fe3+/FeT = 0.15 assumption
- **CaTs** (Ca-Tschermak): CaAl2SiO6
- **CrTs** (Cr-Tschermak): CaCrAlSiO6
- **CaTi**: CaTiAl2O6
- **En** (enstatite): Mg2Si2O6 endmember fraction
- **Fs** (ferrosilite): Fe2Si2O6 endmember fraction
- **Mg_num_cpx**: Mg/(Mg+Fe_total)

These follow the Putirka 2008 recalculation scheme for thermobarometry.

Fourth feature set `cpx_components`: these structural formula components + Mg_num. Tested in T05.

### 3.3 Feature set configurations

Four feature sets per cpx pipeline:
- raw
- alr
- pwlr
- cpx_components (NEW)

Optuna searches each. Test T05 confirms which wins per (model, target, track).

---

## 4. Models

Same 8 models as opx per `v10_master_plan.md` Section 3:

| Model | Library | Notes |
|---|---|---|
| RF | sklearn | tree baseline |
| ERT | sklearn | tree baseline |
| XGB | xgboost | boosted baseline (Agreda uses this) |
| GB | sklearn HistGradientBoosting | boosted baseline |
| CatBoost | catboost | NEW; often SOTA on tabular |
| LightGBM | lightgbm | NEW |
| ElasticNet | sklearn | NEW; linear baseline |
| MLP | sklearn.MLPRegressor | NEW; NN baseline |

Ensemble methods same as opx per `v10_ensemble_methods_plan.md`:

1. Ridge stacking
2. Two-level stacking
3. Greedy ensemble (Caruana)
4. AutoGluon-Tabular

---

## 5. Optuna setup

Per pipeline:

- 8 models × 4 feature sets × 2 targets × 2 tracks = **128 Optuna studies** for cpx
- 50 trials each
- Same TPE + MedianPruner setup as v9
- Study persistence to `results/cpx/optuna_studies/`

Estimated compute at ~160 s per study (extrapolating from v9 opx where average was 160 s but cpx has larger training n so likely ~200 s):

- 128 studies × 200 s = 25,600 s = **~7 hours wall time**

---

## 6. Test protocol per T01-T12

From `v10_nb03_test_protocol.md` Section 2, applied to cpx:

### T01 cpx — Boosted is primary

Hypothesis: boosted (XGB/GB/CatBoost/LightGBM) has smallest |T bias| and lowest T RMSE on ArcPL cpx-liq.

Expected: likely passes, Agreda and Jorgenson use boosted-style models and win.

### T02 cpx — Ensemble shootout

Hypothesis: at least one ensemble method beats best base on test and external.

Expected: unknown. Cpx has better base models (higher R²), so ensemble headroom may be smaller than opx.

### T03 cpx — Resampling hurts

Hypothesis: resampling hurts in 6+ of 8 combos.

Expected: unknown. Cpx training data likely more balanced than opx (more low-P cpx experiments from MORB/arc), so resampling may have less effect either way.

### T04 cpx — N_AUG=1 wins

Hypothesis: N_AUG=1 test RMSE <= N_AUG=5 across targets and tracks.

Expected: same as opx finding, likely passes.

### T05 cpx — Feature set stability

Hypothesis: winners stable across Optuna seeds 42 and 43. Additional: compare `cpx_components` (structural) vs raw/alr/pwlr (compositional).

Expected: stable winners expected. `cpx_components` may win for models that benefit from human feature engineering; raw/alr may win for models that learn their own features.

### T06 cpx — cpx_only supplementary?

Hypothesis: cpx_only R² < 0 on ArcPL.

Expected: **likely FAILS** — Jorgenson 2022 cpx-only achieves R²=0.85 on P. Unlike opx-only, cpx-only is a viable primary model. If this test fails, cpx_only is a primary track alongside cpx_liq.

This is the most interesting scientific divergence from opx.

### T07 cpx — Composition-conditional T correction

Hypothesis: correction reduces ArcPL T RMSE by >=5 C.

Expected: unknown. Agreda already applies a piecewise bias correction; we compare ours to theirs.

### T08 cpx — P piecewise correction

Hypothesis: correction reduces test P RMSE.

Expected: likely passes; similar mechanism to opx.

### T09 cpx — CV-predict stacking

Hypothesis: CV-predict improves test stacking.

Expected: unknown.

### T10 cpx — IsolationForest OOD

Hypothesis: OOD flag correlates with residual magnitude.

Expected: unknown.

### T11 cpx — CatBoost/LightGBM beats XGB

Hypothesis: one of these beats XGB.

Expected: possible. CatBoost often beats XGB on tabular data.

### T12 cpx — MLP/ElasticNet beats trees

Hypothesis: one of these beats tree winner.

Expected: likely fails; tree models typically win on small-to-medium tabular data.

---

## 7. "Beat Agreda / Jorgenson" gate (per pipeline)

Gate (c) per user decision. To claim "beat," ALL three must hold:

1. Lower ArcPL Kd-eq RMSE (T AND P)
2. Lower LOSO cross-validation RMSE (T AND P)
3. Better natural-sample cross-mineral agreement (lower |T residual| and |P residual| on our 327-sample natural set, if sample overlap exists)

All three with 95% bootstrap CI excluding their reported numbers.

If 2 of 3: claim "competitive with." If 1 of 3: claim "matches on [specific metric]." If 0: claim "confirms their result."

Applied to each of four external cpx models: Agreda-Lopez 2024, Jorgenson 2022, Wang 2021, Petrelli 2020.

---

## 8. Diagnostics and figures

Mirrors opx figure set with cpx subdirs:

- `figures/cpx/` — primary cpx figures
- Multi-seed boxplot: `fig_nb03_cpx_multiseed_rmse.png`
- Optuna convergence: `fig_nb03_cpx_optuna_search_progress.png`
- Hyperparameter importance: `fig_nb03_cpx_optuna_hp_importance.png`
- Stacking weights and OOF correlations: `fig_nb03_cpx_stacking_*.png`
- Cpx vs opx cross-pipeline comparison: `fig_nb03_cpx_vs_opx_heatmap.png` (NEW; shows both on one figure)
- ArcPL scatter per model (all 8 bases + 4 ensembles): `fig_nb04_arcpl_cpx_scatter.png`
- Method benchmark bar chart with Agreda/Jorgenson/Wang/Petrelli: `fig_nb04_cpx_method_benchmark.png`
- Model heatmap (same schema as opx): `fig_nb04_cpx_model_heatmap.png`
- LOSO per model: `fig_nb05_cpx_loso.png`
- SHAP T and P: `fig_nb06_cpx_shap_T.png`, `fig_nb06_cpx_shap_P.png`
- Bias probes: `fig_nb07_cpx_bias_*.png`
- Cross-mineral (opx model vs cpx model on same sample): `fig_nb08_cpx_vs_opx_scatter.png`
- World map: `fig_nb08_cpx_world_map.png`

All per `v10_figure_audit.md`.

---

## 9. NB03 cpx notebook structure

Same 14-phase structure as opx per `v10_nb03_test_protocol.md` Section 9. Only differences: cpx-specific feature sets, cpx data loaders, cpx Optuna config, cpx external benchmarks.

---

## 10. Config constants (add to config.py)

```python
# Cpx-specific constants (add Phase A)
KD_CPX_LIQ_MIN = 0.24
KD_CPX_LIQ_MAX = 0.30
WO_CPX_MIN_MOL_PCT = 20.0
CPX_OXIDES = ['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeO_total', 'MnO', 'MgO', 'CaO', 'Na2O']

# Cpx pipeline feature sets
CPX_FEATURE_METHODS = ('raw', 'alr', 'pwlr', 'cpx_components')

# Cpx-specific structural components
CPX_COMPONENTS = ('Jd', 'Di', 'Aeg', 'CaTs', 'CrTs', 'CaTi', 'En', 'Fs', 'Mg_num_cpx')
```

---

## 11. Expected timeline

Per `v10_master_plan.md` Section 6 Phase D:
- 2-3 days active
- ~7 h compute
- Depends on whether ExPetDB cpx extraction in Phase A is clean

---

## 12. Deliverables

- `nb03_cpx_baseline_models.ipynb` — complete, test-first, markdown per template
- `results/cpx/` — populated with all outputs
- `models/canonical/cpx/` — 8 base × 4 feature × 2 targets × 2 tracks = up to 128 base joblibs + 4 ensemble winners per target × track
- `results/cpx/v10_nb03_test_log.csv` — all T01-T12 results
- All cpx figures per `v10_figure_audit.md`
- Updated `docs/stacking_strategy.md`, `resampling_strategy.md`, `optuna_strategy.md` with cpx-specific outcome blocks
