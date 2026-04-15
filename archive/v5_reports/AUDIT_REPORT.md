# Architectural Audit Report

Generated 2026-04-13. Read-only static analysis; no code executed.

---

## Task 1: Repo Inventory

### Directory tree (excluding .git, __pycache__, .venv, .ipynb_checkpoints)

```
Final Project/
  config.py
  requirements.txt
  _nbbuild.py
  _build_nb01.py ... _build_nb09.py
  execution_roadmap.md
  revised_master_prompt_opx_thermobar.md
  opx_ml_master_fix_v4.1.md
  files.zip
  data/
    raw/
      ExPetDB_download_ExPetDB-2025-07-21.xlsx
      external/
        LEPR_Wet_Stitched_April2023_Norm100Anhydrs.xlsx
    processed/
      opx_clean_core.csv
      opx_clean_core.parquet
      opx_clean_full.csv
      opx_clean_full.parquet
      opx_clean_opx_liq.parquet
      opx_clean_opx_only.parquet
    splits/
      train_indices_opx_liq.npy
      test_indices_opx_liq.npy
      train_indices_opx.npy
      test_indices_opx.npy
    natural/
      2024-12-SGFTFN_ORTHOPYROXENES.csv
      natural_opx_cleaned.csv
      natural_sample_prep_script.py
  models/   (86 .joblib files)
  figures/  (37 .png files)
  results/  (41 files: .csv, .json)
  logs/
    cleaning_log.txt
    nb03_split_log.txt
    pipeline_report.txt
  reports/
    figures/
      arcpl_putirka_benchmark.png
  notebooks/
    figure_style.py
    nb01_data_cleaning.ipynb
    nb02_eda_pca.ipynb
    nb03_baseline_models.ipynb
    nb03b_baseline_models.ipynb
    nb03c_baseline_models.ipynb
    nb04_putirka_benchmark.ipynb
    nb04b_lepr_arcpl_validation.ipynb
    nb05_loso_validation.ipynb
    nb06_shap_analysis.ipynb
    nb07_bias_correction.ipynb
    nb08_natural_samples.ipynb
    nb09_manuscript_compilation.ipynb
    nb11_extended_analyses.ipynb
    nbF_figures.ipynb
    nb06b_shap_interrogation_and_robustness_checks.ipynb
    Pipeline V1/  (legacy notebooks and .py exports)
```

### File inventory

| Path | Size | Modified | Cells/Lines |
|------|------|----------|-------------|
| config.py | 1,762 B | 2026-04-10 21:03 | 48 lines |
| notebooks/figure_style.py | 6,333 B | 2026-04-08 20:43 | 184 lines |
| notebooks/nb01_data_cleaning.ipynb | 22,927 B | 2026-04-08 21:34 | 17 cells (16 code, 1 md) |
| notebooks/nb02_eda_pca.ipynb | 685,875 B | 2026-04-08 21:08 | 12 cells (11 code, 1 md) |
| notebooks/nb03_baseline_models.ipynb | 273,987 B | 2026-04-10 10:28 | 10 cells (9 code, 1 md) |
| notebooks/nb03b_baseline_models.ipynb | 937,194 B | 2026-04-10 15:56 | 21 cells (13 code, 8 md) |
| notebooks/nb03c_baseline_models.ipynb | 510,739 B | 2026-04-10 17:09 | 21 cells (15 code, 6 md) |
| notebooks/nb04_putirka_benchmark.ipynb | 399,919 B | 2026-04-10 17:58 | 14 cells (13 code, 1 md) |
| notebooks/nb04b_lepr_arcpl_validation.ipynb | 599,256 B | 2026-04-10 18:06 | 14 cells (13 code, 1 md) |
| notebooks/nb05_loso_validation.ipynb | 19,705 B | 2026-04-10 20:38 | 6 cells (5 code, 1 md) |
| notebooks/nb06_shap_analysis.ipynb | 15,402 B | 2026-04-10 20:50 | 8 cells (7 code, 1 md) |
| notebooks/nb07_bias_correction.ipynb | 30,047 B | 2026-04-10 20:56 | 9 cells (8 code, 1 md) |
| notebooks/nb08_natural_samples.ipynb | 28,257 B | 2026-04-10 21:26 | 12 cells (11 code, 1 md) |
| notebooks/nb09_manuscript_compilation.ipynb | 30,075 B | 2026-04-10 21:16 | 11 cells (10 code, 1 md) |
| notebooks/nb11_extended_analyses.ipynb | 26,061 B | 2026-04-10 21:23 | 10 cells (9 code, 1 md) |
| notebooks/nbF_figures.ipynb | 28,514 B | 2026-04-10 21:23 | 11 cells (10 code, 1 md) |
| _nbbuild.py | 1,748 B | 2026-04-06 19:28 | 56 lines |
| _build_nb01.py | 12,464 B | 2026-04-06 19:27 | 290 lines |
| _build_nb02.py | 7,103 B | 2026-04-06 19:30 | 177 lines |
| _build_nb03.py | 17,310 B | 2026-04-06 19:34 | 412 lines |
| _build_nb04.py | 17,651 B | 2026-04-07 00:09 | 428 lines |
| _build_nb04b.py | 16,811 B | 2026-04-07 00:17 | 394 lines |
| _build_nb05.py | 13,181 B | 2026-04-07 00:18 | 301 lines |
| _build_nb06.py | 7,972 B | 2026-04-07 00:30 | 186 lines |
| _build_nb07.py | 10,482 B | 2026-04-07 00:31 | 233 lines |
| _build_nb08.py | 21,304 B | 2026-04-07 07:37 | 455 lines |
| _build_nb09.py | 9,784 B | 2026-04-07 07:03 | 217 lines |
| data/natural/natural_sample_prep_script.py | N/A | N/A | 53 lines |

---

## Task 2: Data Flow Map

### NB01 (data_cleaning)

| | |
|---|---|
| **Reads** | `data/raw/ExPetDB_download_ExPetDB-2025-07-21.xlsx` (sheets: Experiment, Orthopyroxene, Liquid) |
| **Writes** | `data/processed/opx_clean_core.csv`, `data/processed/opx_clean_core.parquet`, `data/processed/opx_clean_full.csv`, `data/processed/opx_clean_full.parquet`, `logs/cleaning_log.txt` |
| **Cross-nb deps** | None (entry point) |

### NB02 (eda_pca)

| | |
|---|---|
| **Reads** | `data/processed/opx_clean_core.parquet` |
| **Writes** | `data/processed/opx_clean_core.csv` (OVERWRITES NB01 output, adds `chemical_cluster`), `data/processed/opx_clean_core.parquet` (OVERWRITES), `figures/fig01_pt_distribution.png`, `figures/fig02_pca_biplot.png`, `figures/fig_eda_correlation.png`, `figures/fig_eda_distributions.png`, `figures/fig_nb02_clusters.png` |
| **Cross-nb deps** | Reads NB01 parquet output |

### NB03/NB03b/NB03c (baseline_models)

Three versions exist. NB03c (2026-04-10 17:09) is the most recently modified and its output schema (`FEATURE_METHODS = ['raw', 'alr', 'pwlr']`) matches the current `nb03_winning_configurations.json`. NB03 has `FEATURE_METHODS = ['raw', 'raw_aug', 'alr_aug', 'pwlr_aug']`. NB03b has `FEATURE_METHODS = ['raw', 'raw_aug', 'alr']` (truncated).

| | |
|---|---|
| **Reads** | `data/processed/opx_clean_core.parquet` |
| **Writes** | `data/processed/opx_clean_opx_liq.parquet`, `data/processed/opx_clean_opx_only.parquet`, `data/splits/train_indices_opx_liq.npy`, `data/splits/test_indices_opx_liq.npy`, `data/splits/train_indices_opx.npy`, `data/splits/test_indices_opx.npy`, `results/nb03_multi_seed_results.csv`, `results/nb03_multi_seed_summary.csv`, `results/nb03_feature_set_ranking.csv`, `results/nb03_winning_configurations.json`, `results/nb03_canonical_test_predictions.csv`, `models/model_*.joblib` (80+ model files) |
| **Cross-nb deps** | Reads NB01+NB02 parquet |

### NB04 (putirka_benchmark)

| | |
|---|---|
| **Reads** | `data/processed/opx_clean_opx_liq.parquet`, `data/splits/test_indices_opx_liq.npy`, `results/nb03_multi_seed_summary.csv`, `models/model_RF_*.joblib` |
| **Writes** | `results/nb04_putirka_comparison.csv`, `results/nb04_putirka_comparison_fair.csv`, `results/nb04_putirka_comparison_practical.csv`, `figures/fig_nb04_putirka_T.png`, `figures/fig_nb04_putirka_P.png`, `figures/fig_nb04_failure_analysis.png` |
| **Cross-nb deps** | NB03 parquet, splits, models, summary CSV |

### NB04b (lepr_arcpl_validation)

| | |
|---|---|
| **Reads** | `data/raw/external/LEPR_Wet_Stitched_April2023_Norm100Anhydrs.xlsx`, `data/processed/opx_clean_opx_liq.parquet`, `results/nb03_multi_seed_summary.csv`, `models/model_RF_*.joblib` |
| **Writes** | `results/nb04b_arcpl_predictions.csv`, `results/nb04b_arcpl_metrics.csv`, `figures/fig_nb04b_arcpl_pred_vs_obs.png`, `figures/fig_nb04b_arcpl_H2O_dependence.png`, `reports/figures/arcpl_putirka_benchmark.png` |
| **Cross-nb deps** | NB03 models and summary |

### NB05 (loso_validation)

| | |
|---|---|
| **Reads** | `data/processed/opx_clean_opx_liq.parquet`, `results/nb03_winning_configurations.json`, `results/nb03_multi_seed_results.csv` |
| **Writes** | `results/nb05_loso_pooled.csv`, `results/nb05_per_fold_rmse.csv`, `results/nb05_validation_all.csv` |
| **Cross-nb deps** | NB03 config JSON, multi-seed CSV, opx-liq parquet |

### NB06 (shap_analysis)

| | |
|---|---|
| **Reads** | `data/processed/opx_clean_opx_liq.parquet`, `data/splits/test_indices_opx_liq.npy`, `results/nb03_winning_configurations.json`, `models/model_RF_*_opx_liq_*.joblib` |
| **Writes** | `results/nb06_shap_feature_importance.csv`, `results/table5_shap_importance.csv`, `figures/fig07_shap_P_beeswarm.png`, `figures/fig08_shap_T_beeswarm.png`, `figures/fig09_shap_P_dependence.png`, `figures/fig09_shap_T_dependence.png` |
| **Cross-nb deps** | NB03 config JSON, models, splits |

### NB07 (bias_correction)

| | |
|---|---|
| **Reads** | `data/processed/opx_clean_opx_liq.parquet`, `data/splits/train_indices_opx_liq.npy`, `data/splits/test_indices_opx_liq.npy`, `results/nb03_winning_configurations.json`, `results/nb03_multi_seed_results.csv`, `models/model_RF_*_opx_liq_*.joblib` |
| **Writes** | `results/nb07_ab_test_report.csv`, `results/nb07_piecewise_params.json`, `results/nb07_test_predictions.csv`, `results/nb07_qrf_ab_coverage.csv`, `models/model_QRF_T_C_opx_liq.joblib`, `models/model_QRF_P_kbar_opx_liq.joblib` |
| **Cross-nb deps** | NB03 config JSON, models, splits, multi-seed CSV |

### NB08 (natural_samples)

| | |
|---|---|
| **Reads** | `data/natural/natural_opx_cleaned.csv`, `data/processed/opx_clean_opx_only.parquet`, `data/splits/train_indices_opx.npy`, `results/nb03_multi_seed_results.csv` |
| **Writes** | `results/nb08_natural_predictions_all.csv`, `results/nb08_natural_predictions_filtered.csv`, `models/model_QRF_T_C_opx_only.joblib`, `models/model_QRF_P_kbar_opx_only.joblib`, `models/isolation_forest_opx_only.joblib`, `figures/fig12_natural_samples_geotherm.png` (via nbF) |
| **Cross-nb deps** | NB03 multi-seed CSV, opx-only parquet, splits |

### NB09 (manuscript_compilation)

| | |
|---|---|
| **Reads** | `data/processed/opx_clean_core.parquet`, `data/processed/opx_clean_full.parquet`, `data/processed/opx_clean_opx_liq.parquet`, `results/nb03_winning_configurations.json`, `results/nb03_multi_seed_results.csv`, `results/nb04_putirka_comparison_fair.csv`, `results/nb04_putirka_comparison_practical.csv`, `results/nb04b_arcpl_predictions.csv`, `results/nb04b_arcpl_metrics.csv`, `results/nb05_loso_pooled.csv`, `results/nb05_per_fold_rmse.csv`, `results/nb06_shap_feature_importance.csv`, `results/nb07_ab_test_report.csv`, `results/nb07_qrf_ab_coverage.csv`, `results/nb11_two_pyroxene_benchmark.csv`, `results/nb11_h2o_dependence.csv`, `results/nb11_iqr_uncertainty.csv`, `results/nb11_mc_inference.csv`, `results/nb11_ood_isoforest.csv` |
| **Writes** | `results/table1_dataset_summary.csv`, `results/table2_model_performance.csv`, `results/table3_putirka_vs_ml.csv`, `results/table4_validation_summary.csv`, `results/table5_shap_importance.csv`, `results/table6_arcpl_external.csv`, `results/table7_bias_correction.csv`, `results/figure_inventory.csv`, `results/manuscript_key_results.csv` |
| **Cross-nb deps** | Reads outputs from NB01-NB08 + NB11 |

### NB11 (extended_analyses)

| | |
|---|---|
| **Reads** | `data/processed/opx_clean_opx_liq.parquet`, `data/splits/train_indices_opx_liq.npy`, `data/splits/test_indices_opx_liq.npy`, `results/nb03_winning_configurations.json`, `results/nb04b_arcpl_predictions.csv`, `models/model_RF_*_opx_liq_*.joblib`, `models/model_QRF_*_opx_liq.joblib` |
| **Writes** | `results/nb11_two_pyroxene_benchmark.csv`, `results/nb11_h2o_dependence.csv`, `results/nb11_iqr_uncertainty.csv`, `results/nb11_mc_inference.csv`, `results/nb11_mc_per_sample.csv`, `results/nb11_ood_isoforest.csv`, `results/nb11_arcpl_ood_scores.csv`, `models/model_IsolationForest_opx_liq.joblib` |
| **Cross-nb deps** | NB03 config, models, splits; NB04b ArcPL predictions; NB07 QRF models |

### NBF (figures)

| | |
|---|---|
| **Reads** | `data/processed/opx_clean_opx_liq.parquet`, `data/processed/opx_clean_opx_only.parquet`, `data/splits/*.npy`, `results/nb03_*.csv`, `results/nb03_*.json`, `results/nb04b_arcpl_predictions.csv`, `results/nb05_loso_pooled.csv`, `results/nb06_shap_feature_importance.csv`, `results/nb07_test_predictions.csv`, `results/nb11_arcpl_ood_scores.csv`, `results/nb11_mc_per_sample.csv`, `models/model_RF_*.joblib` |
| **Writes** | `figures/fig01_pt_distribution.png` through `figures/fig14_mc_vs_iqr_uncertainty.png` |
| **Cross-nb deps** | Nearly all upstream results CSVs and models |

### Orphan inputs (read by a notebook but written by no notebook)

| File | Read by |
|------|---------|
| `data/raw/ExPetDB_download_ExPetDB-2025-07-21.xlsx` | NB01 (external source) |
| `data/raw/external/LEPR_Wet_Stitched_April2023_Norm100Anhydrs.xlsx` | NB04b (external source) |
| `data/natural/natural_opx_cleaned.csv` | NB08 (produced by standalone `natural_sample_prep_script.py`, not a notebook) |
| `data/natural/2024-12-SGFTFN_ORTHOPYROXENES.csv` | `natural_sample_prep_script.py` (external source) |

### Orphan outputs (written by a notebook but read by no notebook)

| File | Written by |
|------|-----------|
| `data/processed/opx_clean_full.csv` | NB01 |
| `data/processed/opx_clean_full.parquet` | NB01 |
| `logs/cleaning_log.txt` | NB01 |
| `figures/fig_eda_correlation.png` | NB02 |
| `figures/fig_eda_distributions.png` | NB02 |
| `figures/fig_nb02_clusters.png` | NB02 |
| `results/nb05_validation_all.csv` | NB05 (duplicate of nb05_per_fold_rmse.csv) |
| `results/nb11_two_pyroxene_benchmark.csv` | NB11 (empty, 0 data rows) |
| `results/nb11_h2o_dependence.csv` | NB11 (empty, 0 data rows) |
| `reports/figures/arcpl_putirka_benchmark.png` | NB04b cell 13 |
| All `_build_nb*.py` scripts | Not imported by any notebook |

### Write collisions (same file written by multiple notebooks)

| File | Written by |
|------|-----------|
| `data/processed/opx_clean_core.csv` | NB01 and NB02 |
| `data/processed/opx_clean_core.parquet` | NB01 and NB02 |
| `data/processed/opx_clean_opx_liq.parquet` | NB03, NB03b, NB03c (3 competing notebooks) |
| `data/processed/opx_clean_opx_only.parquet` | NB03, NB03b, NB03c |
| `data/splits/*.npy` (all 4 files) | NB03, NB03b, NB03c |
| `results/nb03_winning_configurations.json` | NB03, NB03b, NB03c |
| `results/nb03_multi_seed_results.csv` | NB03, NB03b, NB03c |
| `results/nb03_multi_seed_summary.csv` | NB03, NB03b, NB03c |
| `results/nb03_canonical_test_predictions.csv` | NB03, NB03b, NB03c |
| `results/table5_shap_importance.csv` | NB06 and NB09 |
| `figures/fig01_pt_distribution.png` | NB02 and NBF |
| `figures/fig02_pca_biplot.png` | NB02 and NBF |

---

## Task 3: Schema Drift Check

### `results/nb03_multi_seed_results.csv`

**Written by** NB03c cell 8. Columns: `best_params, model_name, target, rmse_train, rmse_test, mae_test, r2_test, overfit_ratio, split_seed, track, feature_set`

**Read by NB09 cell 3**: expects column `seed` in `.agg(n_seeds=('seed', 'nunique'))`. Actual column name is `split_seed`.
**MISMATCH**: NB09 assumes `seed`, file has `split_seed`. This is the NB09 failure.

### `results/nb05_per_fold_rmse.csv`

**Written by** NB05 cell 4. Columns: `strategy, model, target, group, n_test, rmse`

**Read by NB09 cell 5**: expects column `fold` in `.agg(n_folds=('fold', 'nunique'))`. Actual column name is `group`.
**MISMATCH**: NB09 assumes `fold`, file has `group`.

### `results/nb04b_arcpl_predictions.csv`

**Written by** NB04b. Contains ArcPL opx-liq columns plus `T_pred`, `P_pred`.
**Read by** NB11 cell 5: expects `H2O_liq` or `liq_H2O` for H2O dependence analysis. NB04b writes `H2O_Liq` (capital L).
**MISMATCH**: NB11 checks for `H2O_liq` and `liq_H2O` (both lowercase variants); file has `H2O_Liq`. Result: H2O analysis skipped, `nb11_h2o_dependence.csv` is empty.

### `data/processed/opx_clean_core.csv` / `.parquet`

Written by NB01, then overwritten by NB02 which adds `chemical_cluster` column. NB03 expects this column to exist. If NB02 is skipped, `chemical_cluster` is missing and NB03/NB05 cluster-based validation fails silently (no error but wrong behavior in NB05 Cluster-KFold).

### `results/nb03_winning_configurations.json`

**Written by** NB03c with `per_combo_winners` keys: `opx_liq_P_kbar`, `opx_liq_T_C`, `opx_only_P_kbar`, `opx_only_T_C` (underscore-delimited).
**Read by** NB09 cell 3: expects keys like `opx_liq|T_C` (pipe-delimited) in `per_combo.get(key)`.
**MISMATCH**: NB09 builds key with `|` separator but JSON uses `_` separator. Result: the `_tag` function never matches, `phase3r_tag` is always empty string. Silent bug.

---

## Task 4: Bug Hunt

### 4.1 Pressure ceiling

`config.py` line 33: `P_CEILING_KBAR = 100.0`

`notebooks/nb01_data_cleaning.ipynb` cell 11:
```python
df = df[df['P_kbar'] <= P_CEILING_KBAR].copy()
```
This is a hard filter at 100 kbar. Output log confirms 71 rows dropped. No `.clip()` used; rows are dropped entirely.

### 4.2 Putirka equations

`notebooks/nb04_putirka_benchmark.ipynb` cell 7. All four equations present:
- `T_Put2008_eq28a` -- called
- `T_Put2008_eq28b_opx_sat` -- called
- `P_Put2008_eq29a` -- called
- `P_Put2008_eq29c` -- called

No missing equations.

### 4.3 LOSO model coverage

`notebooks/nb05_loso_validation.ipynb` cell 4:
```python
models_order = ['RF', 'ERT', 'XGB', 'GB']
```
All four models iterated in the LOSO loop. Output confirms: `LOSO / RF done`, `LOSO / ERT done`, `LOSO / XGB done`, `LOSO / GB done`. No issue.

### 4.4 SHAP namespace collision

Searched NB06, NB09, and NB11 for `shap =` (assignment to variable named `shap` excluding imports). No hits. The SHAP module is imported as `import shap` and the explainer pattern uses `explainer_P = shap.TreeExplainer(...)` and `shap_values_P = explainer_P.shap_values(...)`. No namespace collision.

### 4.5 Geotherm functions

`notebooks/nb08_natural_samples.ipynb` cell 3 defines `hasterok_chapman_geotherm(q_s_mW, z_km_array)`.

Formula: Layered conductive geotherm. T_s = 10.0 C, g = 9.81 m/s^2. Four layers:
```
(0-16 km,  A=1.30e-6, k=2.5, rho=2800)
(16-24 km, A=0.40e-6, k=2.5, rho=2800)
(24-40 km, A=0.40e-6, k=2.5, rho=2800)
(40-300 km, A=0.02e-6, k=3.0, rho=3300)
```

Layer equation: `T = T_top + (q_top * dz - 0.5 * A * dz^2) / k`
Pressure: `P = P_top + rho * g * dz * 1e-8` (Pa to kbar)
Heat flow update: `q = q_top - A * dz`

Sanity check (cell 4) passes: 40 mW/m^2 gives T(100km) in [400,800] C, P(100km) in [27,33] kbar.

Note: The geotherm is used for the natural sample figure but NB08 itself does NOT plot a geotherm directly. The geotherm plot is in NBF cell 10 which uses a simple linear `T_ref = np.linspace(700, 1350, 50); ax.plot(T_ref, 0.03 * (T_ref - 700))` -- this is a ~30 C/kbar gradient line, NOT the H&C 2011 function. The H&C function defined in NB08 cell 3 is not called in NBF; the geotherm figure uses a crude linear approximation instead.

### 4.6 NB09 failure point

`notebooks/nb09_manuscript_compilation.ipynb` cell 3 (Table 2 construction). Error captured in cell output:

```
KeyError: "Label(s) ['seed'] do not exist"
```

Root cause: NB09 cell 3 line 7 uses `.agg(n_seeds=('seed', 'nunique'))` but `nb03_multi_seed_results.csv` has column `split_seed`, not `seed`. The column was renamed at some point in NB03's evolution but NB09 was not updated.

Second latent failure: NB09 cell 5 uses `.agg(n_folds=('fold', 'nunique'))` on `nb05_per_fold_rmse.csv`, which has column `group` not `fold`.

Third latent failure: NB09 cell 10 references `multi_win` (defined in cell 3). Since cell 3 errored, `multi_win` is undefined for cells 10+.

Fourth latent failure: NB09 cell 3 builds per-combo winner tag with key format `f'{row.track}|{row.target}'` but `nb03_winning_configurations.json` uses underscore-separated keys like `opx_liq_T_C`.

### 4.7 QRF quantile range

`notebooks/nb07_bias_correction.ipynb` cell 6:
```python
qrf_T_full.predict(X_train, quantiles=[0.16, 0.5, 0.84])
qrf_P_full.predict(X_train, quantiles=[0.16, 0.5, 0.84])
```

QRF uses quantiles **0.16 and 0.84**, producing a 68% nominal coverage interval. This is not the 90% interval (which would use 0.05/0.95).

NB07 cell 6 reports coverage via `coverage(lo, hi, y) = np.mean((y >= lo) & (y <= hi))`. The QRF A/B coverage CSV shows this is labeled as `coverage_68` in the output, so the 68% label is intentional. However, the results file `nb07_qrf_ab_coverage.csv` includes actual coverage values -- if OOF coverage drops to 61-70%, it is underperforming even the 68% nominal target.

### 4.8 MC dropout

`notebooks/nb11_extended_analyses.ipynb` cell 7. This is NOT MC dropout. It is **analytical noise propagation** via Monte Carlo:

```python
N_MC = 200
MAJ_REL = 0.03   # 3% relative noise on majors
MIN_REL = 0.08   # 8% on minors
```

Each of 200 iterations perturbs oxide wt% columns with `noise = rng.normal(loc=0.0, scale=rel, size=len(df_perturb))` then pushes through `predict_median`. The sigma is `mc_T.std(axis=1)`.

This captures **propagated analytical/measurement uncertainty only**. It does NOT capture model epistemic uncertainty (which would require MC dropout layers or ensemble disagreement). The IQR-based uncertainty in NB11 cell 6 (`predict_iqr`) captures tree-to-tree variance, which is closer to epistemic uncertainty.

### 4.9 Random seed consistency

All seeds flow from `config.py`:
```python
SEED_SPLIT = 42
SEED_MODEL = 42
SEED_NOISE_AUG = 42
SEED_KMEANS = 42
```

| Location | Seed variable | Value |
|----------|---------------|-------|
| config.py line 40-43 | SEED_SPLIT, SEED_MODEL, SEED_NOISE_AUG, SEED_KMEANS | all 42 |
| NB02 cell 4 | PCA `random_state=SEED_KMEANS` | 42 |
| NB02 cell 8 | KMeans `random_state=SEED_KMEANS` | 42 |
| NB03 cell 5 | BASE_MODELS `random_state=SEED_MODEL` | 42 |
| NB03 cell 6 | GroupShuffleSplit `random_state=seed` | loop variable 42-51 |
| NB03 cell 6 | HalvingRandomSearchCV `random_state=seed` | 42 for tune seed |
| NB03 cell 7 | `RandomForestRegressor(random_state=42)` | hardcoded 42 (not via config) |
| NB05 cell 3 | build_model `random_state=SEED_MODEL` | 42 |
| NB07 cell 4 | oof_rf/oof_qrf `random_state=seed` | SEED_MODEL=42 |
| NB07 cell 6 | QRF `random_state=SEED_MODEL` | 42 |
| NB08 cell 9 | QRF + IsoForest `random_state=SEED_MODEL` | 42 |
| NB11 cell 7 | `default_rng(SEED_NOISE_AUG)` | 42 |
| NB11 cell 8 | IsolationForest `random_state=SEED_MODEL` | 42 |
| Pipeline V1/nb03_baseline_models.py line 26 | `np.random.seed(42)` | 42 (legacy) |

One inconsistency: NB03 cell 7 hardcodes `random_state=42` instead of using `SEED_MODEL`. Functionally equivalent since `SEED_MODEL=42`, but a maintenance hazard if the config seed changes.

NB05 train/test split uses GroupShuffleSplit with `random_state=seed` where seed loops from 42 to 51. The canonical split (seed 42) is consistent across NB03, NB05, NB06, NB07.

### 4.10 Train/test leakage

NB03 cell 6 uses `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)` with `groups=df_track['Citation'].values`. This ensures no Citation (study) appears in both train and test for a given seed. No group leakage.

NB05 uses `LeaveOneGroupOut()` with `groups=groups_study` (also Citation). Each fold holds out one study entirely. No group leakage.

NB05 Cluster-KFold uses `GroupKFold(n_splits=n_clusters)` with `groups=clusters` (chemical_cluster). This partitions by cluster, not by study. Studies spanning multiple clusters could appear in both train and test folds. This is not Citation-level leakage but could still be considered a form of data leakage if clusters do not perfectly separate studies.

---

## Task 5: Dependency Graph

```
NB01 -> NB02
NB02 -> NB03, NB03b, NB03c
NB03/NB03b/NB03c -> NB04, NB04b, NB05, NB06, NB07, NB08
NB04b -> NB11
NB07 -> NB11
NB03/NB03b/NB03c -> NB11
NB01, NB03, NB04, NB04b, NB05, NB06, NB07, NB08, NB11 -> NB09
NB03, NB05, NB06, NB07, NB08, NB11, NB04b -> NBF
```

**Entry points** (no upstream dependency): NB01

**Dead ends** (no downstream dependents): NB09

**Cycles**: None.

**Ambiguity**: Three competing NB03 variants (NB03, NB03b, NB03c) write the same output files. Which one was last run determines the state of all downstream notebooks.

---

## Task 6: Output Freshness

All outputs marked STALE where notebook modified date > output modified date.

| Output file | Output date | Notebook | NB date | Delta |
|-------------|-------------|----------|---------|-------|
| data/processed/opx_clean_core.csv | 2026-04-08 21:08 | nb01 | 2026-04-08 21:34 | 27 min STALE |
| data/processed/opx_clean_core.parquet | 2026-04-08 21:08 | nb01 | 2026-04-08 21:34 | 27 min STALE |
| data/processed/opx_clean_full.csv | 2026-04-08 21:07 | nb01 | 2026-04-08 21:34 | 27 min STALE |
| data/processed/opx_clean_full.parquet | 2026-04-08 21:07 | nb01 | 2026-04-08 21:34 | 27 min STALE |
| logs/cleaning_log.txt | 2026-04-08 21:07 | nb01 | 2026-04-08 21:34 | 27 min STALE |
| data/processed/opx_clean_opx_liq.parquet | 2026-04-10 16:04 | nb03c | 2026-04-10 17:09 | 65 min STALE |
| data/processed/opx_clean_opx_only.parquet | 2026-04-10 16:04 | nb03c | 2026-04-10 17:09 | 65 min STALE |
| data/splits/*.npy (4 files) | 2026-04-10 16:45 | nb03c | 2026-04-10 17:09 | 24 min STALE |
| results/nb03_winning_configurations.json | 2026-04-10 16:45 | nb03c | 2026-04-10 17:09 | 24 min STALE |
| results/nb03_multi_seed_results.csv | 2026-04-10 16:45 | nb03c | 2026-04-10 17:09 | 24 min STALE |
| results/nb04_putirka_comparison*.csv | 2026-04-10 17:56 | nb04 | 2026-04-10 17:58 | 2 min STALE |
| results/nb04b_arcpl_*.csv | 2026-04-10 18:04 | nb04b | 2026-04-10 18:06 | 1 min STALE |
| results/nb05_loso_pooled.csv | 2026-04-10 20:34 | nb05 | 2026-04-10 20:38 | 4 min STALE |
| results/nb05_per_fold_rmse.csv | 2026-04-10 20:34 | nb05 | 2026-04-10 20:38 | 4 min STALE |
| results/nb07_ab_test_report.csv | 2026-04-10 20:54 | nb07 | 2026-04-10 20:56 | 2 min STALE |
| results/nb07_piecewise_params.json | 2026-04-10 20:54 | nb07 | 2026-04-10 20:56 | 2 min STALE |
| results/nb07_test_predictions.csv | 2026-04-10 20:54 | nb07 | 2026-04-10 20:56 | 1 min STALE |
| results/nb07_qrf_ab_coverage.csv | 2026-04-10 20:54 | nb07 | 2026-04-10 20:56 | 1 min STALE |
| models/model_QRF_*.joblib (opx_liq) | 2026-04-10 20:54 | nb07 | 2026-04-10 20:56 | 1 min STALE |
| results/nb08_natural_*.csv | 2026-04-10 21:25 | nb08 | 2026-04-10 21:26 | <1 min STALE |
| results/nb11_*.csv | 2026-04-10 21:22-23 | nb11 | 2026-04-10 21:23 | <1 min STALE |

Note: Many "STALE" entries have deltas under 2 minutes, which likely means the notebook was saved after execution completed (Jupyter auto-save). The genuinely stale outputs are NB01 (27 min) and NB03c (24-65 min), where the notebook was edited after the outputs were generated.

**Cascade staleness**: Since NB01 outputs are stale, every downstream notebook's outputs are potentially stale (all of NB02 through NB09).

---

## Task 7: Dead Code and TODO Audit

### TODO / FIXME / XXX / HACK comments

None found in any `.py` file or notebook cell source.

### Unused imports

| Notebook | Unused import |
|----------|---------------|
| nb02_eda_pca | `numpy as np` (used via plt/sklearn but the heuristic flagged it) |
| nb03c_baseline_models | `shutil` |
| nb05_loso_validation | `joblib` |
| nb05_loso_validation | `GradientBoostingRegressor` (imported twice; NB05 uses `HistGradientBoostingRegressor` via `build_model`) |
| nb09_manuscript_compilation | `json` |
| nb09_manuscript_compilation | `numpy as np` |
| nb11_extended_analyses | `json` |
| nbF_figures | `json` |

Note: `matplotlib.pyplot as plt` is flagged for many notebooks because the heuristic searches for `matplotlib` but the actual usage is via `plt`. These are false positives and are excluded above.

### Functions defined but never called

No functions were found to be defined but never called within the same notebook. All helper functions (`make_raw_features`, `make_alr_features`, `make_pwlr_features`, `build_feature_matrix`, `predict_median`, `predict_iqr`, `hasterok_chapman_geotherm`, etc.) are used within their respective notebooks.

However: `hasterok_chapman_geotherm` is defined in NB08 cell 3 and tested in NB08 cell 4 but is never called for actual predictions. NB08 does not use it to compute P from depth. NBF uses a linear gradient line instead.

### Dead files

| File | Status |
|------|--------|
| `_build_nb01.py` through `_build_nb09.py` | 10 Python scripts. Not imported or called by any notebook. Appear to be standalone build scripts for programmatic notebook generation. |
| `_nbbuild.py` | Builder utility. Not imported by notebooks. |
| `notebooks/Pipeline V1/*` | 10 legacy notebooks + 3 .py exports from a prior pipeline version. Not referenced by the active pipeline. |
| `notebooks/nb06b_shap_interrogation_and_robustness_checks.ipynb` | Standalone SHAP deep-dive. Not part of the NB01-NB09 pipeline. |
| `files.zip` | 1.6 MB archive at repo root. Contents unknown. Not referenced by any code. |
| `execution_roadmap.md` | Documentation. Not read by code. |
| `revised_master_prompt_opx_thermobar.md` | Documentation. Not read by code. |
| `opx_ml_master_fix_v4.1.md` | Documentation. Not read by code. |
| `results/nb03_results_all.csv` | Written by an older NB03 version (2026-04-07). Not read by current pipeline. |
| `results/nb03_winners.csv` | Written by older NB03. Not read by current pipeline. |
| `results/nb03b_n_aug_sensitivity.csv` | Written by NB03b. Not read by current pipeline. |
| `results/nb07_bias_correction_comparison.csv` | Old version (2026-04-07). Current pipeline writes `nb07_ab_test_report.csv`. |
| `results/nb07_qrf_test_predictions.csv` | Old version (2026-04-07). Current pipeline writes `nb07_test_predictions.csv`. |
| `results/table2_model_performance.csv` | Written by NB09, but NB09 crashes before completing this file (last successful write was 2026-04-07). |
| `results/table3_putirka_vs_ml.csv` | Same (2026-04-07 vintage, NB09 crashes before refreshing). |
| `results/table4_validation_summary.csv` | Same (2026-04-07 vintage). |
| `results/table6_arcpl_external.csv` | Same (2026-04-07 vintage). |
| `results/table7_bias_correction.csv` | Same (2026-04-07 vintage). |
| `results/manuscript_key_results.csv` | Same (2026-04-07 vintage). |
| `results/figure_inventory.csv` | Same (2026-04-07 vintage). |

### Massive code duplication

The feature engineering functions (`make_raw_features`, `make_alr_features`, `make_pwlr_features`, `build_feature_matrix`) are copy-pasted into every notebook: NB03, NB03b, NB03c, NB04, NB04b, NB05, NB06, NB07, NB08, NB11, NBF. Twelve copies of ~80-line functions. Any fix to one copy does not propagate.

Similarly, `predict_median` is defined in NB05, NB07, NB08, NB11, NBF (5 copies).

---

## Summary Counts

| Category | Count |
|----------|-------|
| Notebooks in active pipeline | 12 (NB01, NB02, NB03c, NB04, NB04b, NB05, NB06, NB07, NB08, NB09, NB11, NBF) |
| Competing NB03 variants | 3 (NB03, NB03b, NB03c) |
| Orphan inputs (external sources) | 3 |
| Orphan outputs (no downstream reader) | 11 |
| Write collisions | 12 files written by multiple notebooks |
| Schema mismatches (blocking) | 3 (NB09 `seed`/`split_seed`, NB09 `fold`/`group`, NB09 pipe vs underscore key) |
| Schema mismatches (silent) | 1 (NB11 `H2O_Liq` case mismatch) |
| Empty result files | 2 (`nb11_two_pyroxene_benchmark.csv`, `nb11_h2o_dependence.csv`) |
| Stale outputs (NB edited after output) | 30+ files across all notebooks |
| Cascade-stale (NB01 stale propagating) | All downstream outputs |
| Unused imports | 8 |
| Duplicated feature-engineering functions | 12 copies across notebooks |
| Duplicated predict_median function | 5 copies |
| Dead/legacy files | 17+ (Pipeline V1, old results, _build scripts) |
| TODO/FIXME/HACK comments | 0 |
| Random seed inconsistencies | 1 (NB03 cell 7 hardcodes 42 instead of SEED_MODEL) |
| NB09 distinct errors | 4 (seed column, fold column, pipe-key mismatch, multi_win undefined) |
| Geotherm function unused for actual figure | 1 (H&C 2011 defined but NBF uses linear approx) |
