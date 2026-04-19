# v10 cleanup manifest

**Status:** canonical list for Phase A fresh-start cleanup
**Author:** NQTa
**Date:** 2026-04-16, updated for v10+v11 unified scope
**Companion to:** `v10_master_plan.md`

User approved fresh start of everything regenerable. This is the explicit archive-then-delete vs keep vs review list.

---

## 0. Prerequisites before running Phase A cleanup

1. `git status` clean (all v9 changes committed)
2. `git tag pre_v10_cleanup_YYYY_MM_DD` (explicit date); push tag
3. Verify archive target `archive/pre_v10_rebuild_YYYY_MM_DD/` has >= 1.0 GB free (expanded for v10+v11 artifacts)
4. Confirm `models/external/` subtree intact (vendor artifacts)
5. Confirm `data/natural/2024-12-SGFTFN_ORTHOPYROXENES.csv` intact (raw 78k opx for world map)

---

## 1. ARCHIVE then DELETE

Move to `archive/pre_v10_rebuild_YYYY_MM_DD/<relative_path>/`, then remove from canonical tree.

### 1.1 Results (all)

`results/` entire contents except `.gitkeep`.

Includes all v9 outputs:
- `manuscript_key_results.csv`
- `nb03_*.csv`, `nb03_*.json`, `nb03_*.npz` (~20 files)
- `nb04_*.csv` (~20 files)
- `nb04b_*.csv` (3 files)
- `nb05_*.csv` (2 files)
- `nb06_shap_feature_importance.csv`
- `nb07*.csv`, `nb07*.json` (~12 files)
- `nb08_*.csv` (2 files)
- `nb10_*.csv` (~10 files, includes empty `nb10_two_pyroxene_benchmark.csv`)
- `nbF_file_audit.csv`
- `table*.csv` (all manuscript tables)
- `figure_inventory.csv`
- `optuna_studies/*.pkl` (48 studies)

Regenerable in Phases C-H.

### 1.2 Figures (all)

`figures/` entire contents except `.gitkeep`:
- `fig01_*` through `fig14_*` (each PDF, PNG, TXT = 42 files)
- `fig_eda_*` (4 files)
- `fig_nb02_*`
- `fig_nb03*` (~20 files)
- `fig_nb04_*` (~15 files)
- `fig_nb05_generalization.*`
- `fig_nb06_*` (~5 files)
- `fig_nb07b_*` (~5 files)
- `fig_nb08_twopx_1to1.*`

All regenerable.

### 1.3 Models (root level only, ~37 joblibs)

`models/*.joblib` at root level:
- `meta_ridge_*.joblib` (4 stacked meta)
- `model_ERT_*.joblib`, `model_GB_*.joblib`, `model_RF_*.joblib`, `model_XGB_*.joblib`
- `model_IsolationForest_opx_liq.joblib`
- `model_QRF_*.joblib`
- `model_P_kbar_*_forest.joblib`, `model_P_kbar_*_boosted.joblib`, `model_T_C_*_forest.joblib`, `model_T_C_*_boosted.joblib`
- `model_P_kbar_*_forest_resampled.joblib` (8 resampled ablations)
- `model_RF_*_H2O.joblib` (2 H2O-engineered variants)

**DO NOT TOUCH:** `models/external/` (Agreda onnx + joblib + json; vendor) and `models/canonical/` (empty scaffold — will be populated with v10 subdirs).

### 1.4 Executed notebooks

`notebooks/executed/` entire contents. Papermill regenerates in downstream phases.

### 1.5 Logs

`logs/` entire contents except `.gitkeep`:
- `cleaning_log.txt`
- `nb*_papermill.log` (11 files)
- `nb03_search.log`, `optuna_overnight_launch.log`, `recovery_run.log`, `run_phase2_optuna.log`, `run_phase35_to_312.log`
- `pipeline_health.txt`

### 1.6 Root-level v9 artifacts

- `nb03_optuna_best_params.json` at repo root (duplicate of `results/` copy)
- `nb03c_training.log` at repo root
- `nb03_search.log` at repo root (if present)

### 1.7 Stale runners

- `run_all.py` (references deleted notebooks)
- `run_from.py` (references nb04b which no longer exists as source)
- `wipe_old_files_v7.py` (one-time script, already executed)

### 1.8 Stale runner

- `run_all_v7.py` at repo root — rename to `run_all_v10.py` in Phase A rather than archive (keeps git history cleaner).

---

## 2. KEEP (do not touch)

### 2.1 Source code

`src/` (10 modules):
- `__init__.py`, `calibration.py`, `data.py`, `evaluation.py`, `external_models.py`, `features.py`, `geotherm.py`, `io_utils.py`, `models.py`, `optuna_search.py`, `plot_style.py`, `resampling.py`, `stacking.py`

**Modifications during Phases C-F** (not cleanup):
- Expand `models.py` with CatBoost, LightGBM, ElasticNet, MLP
- Expand `external_models.py` with Petrelli 2020, AutoGluon
- Expand `evaluation.py` with LeaveOneRegionOut
- Expand `optuna_search.py` with search spaces for new models
- Fix `stacking.py` per test T09 (CV-predict at test time)
- Add `cpx_features.py`, `twopx_features.py`, `universal_features.py`, `shap_utils.py`, `world_map.py`
- Create `src/ablations/` subdirectory for failed-test code paths

### 2.2 Data

**KEEP ALL:**
- `data/raw/ExPetDB_download_ExPetDB-2025-07-21.xlsx` (~10 MB)
- `data/raw/external/LEPR_Wet_Stitched_April2023_Norm100Anhydrs.xlsx` (~2.2 MB)
- `data/processed/*.parquet` and `*.csv` (7 files, NB01/NB02 output)
- `data/splits/*.npy` (4 files, canonical splits)
- `data/external/agreda_lopez_2024/` (vendor, 137 MB)
- `data/external/thermobar_examples/Thermobar/` (vendor, 1.1 GB, required by NB04)
- `data/external/thermobar_examples/_INDEX.csv`
- `data/natural/2024-12-SGFTFN_ORTHOPYROXENES.csv` (78 MB, raw GEOROC opx with lat/lon — KEY for world map)
- `data/natural/natural_opx_cleaned.csv` (53k samples, cleaned but missing lat/lon — will re-merge in Phase H)
- `data/natural/natural_sample_prep_script.py`

### 2.3 Models external (vendor)

`models/external/`:
- `agreda_cpx_P.{joblib,json,onnx}` (3 files)
- `agreda_cpx_T.{joblib,json,onnx}`
- `agreda_cpx_liq_P.{joblib,json,onnx}`
- `agreda_cpx_liq_T.{joblib,json,onnx}`

### 2.4 Configuration

- `config.py` — **update in Phase A** to remove stale `CANONICAL_FIGURES` entries, add cpx+twopx+universal constants, add new model hyperparameter grids
- `requirements.txt` — **update in Phase A**: add catboost, lightgbm, autogluon, folium, cartopy, plotly
- `.gitignore` — check if archive/ is excluded from git to avoid bloat

### 2.5 Notebooks (source only)

All 11 v9 source notebooks KEPT as editing source. During Phase C-I they get rebuilt:

- `nb01_data_cleaning.ipynb` — expand Phase A for cpx+twopx outputs
- `nb02_eda_pca.ipynb` — expand for cpx+twopx PCA
- `nb03_baseline_models.ipynb` — **split into 4 new NBs** (opx, cpx, twopx, universal)
- `nb04_putirka_benchmark.ipynb` — merged with model heatmap (NBM spec folded in) and renamed to `nb04_benchmark.ipynb`
- `nb05_loso_validation.ipynb` — rename to `nb05_generalization.ipynb`, add LeaveOneRegionOut
- `nb06_shap_analysis.ipynb` — expanded SHAP
- `nb07_bias_correction.ipynb` and `nb07b_arcpl_bias_probe.ipynb` — **merge** into `nb07_bias_correction.ipynb`
- `nb08_natural_twopx.ipynb` — **rename** to `nb08_natural_samples.ipynb` and incorporate world map (NB08b spec folded in)
- `nb09_manuscript_compilation.ipynb`
- `nbF_figures.ipynb`

Also NEW notebook files to create in Phase A-F:
- `nb03_opx_baseline_models.ipynb`
- `nb03_cpx_baseline_models.ipynb`
- `nb03_twopx_baseline_models.ipynb`
- `nb03_universal_exploration.ipynb`
- `nb10_extended_analyses.ipynb` (if it doesn't exist; v9 ran it implicitly)

### 2.6 Documentation (all KEEP)

`docs/`:
- `codebase_consistency_audit_optionB.md`
- `optionB_pre_implementation_prediction.md`
- `optuna_strategy.md` — v10 update block appended in Phase A
- `putirka_inconsistency_audit.md`
- `putirka_kd_filter_lookup.md`
- `resampling_strategy.md` — v10 update block appended
- `stacking_strategy.md` — v10 update block appended
- `v9_archive_plan.md`, `v9_deletion_plan.md`, `v9_inventory_report.md` (historical, KEEP)
- `v9_outcomes.md` (if it exists — KEEP, historical)
- **New v10 docs** (this doc plus 12 others) — all KEEP

### 2.7 Scripts

`scripts/audit_*` (4 scripts) — KEEP, audit trail.

All v5-v8 one-shot patches (44 files) — already archived in v9 — LEAVE ARCHIVED.

**New v10 scripts to create in Phase A:**
- `scripts/v10_phase_a_cleanup.py`
- `scripts/v10_phase_b_nb_audit.py`
- `scripts/v10_pull_external_training_data.py`
- `scripts/v10_pull_georoc_cpx.py`
- `scripts/v10_world_map_static.py`
- `scripts/v10_world_map_interactive.py`
- `scripts/v10_audit_notebook_markdown.py`
- `scripts/v10_figure_audit_checker.py`

---

## 3. REVIEW before deletion

User decision required on each.

### 3.1 `data/natural/2024-12-SGFTFN_ORTHOPYROXENES.csv` (78 MB)

Raw GEOROC opx with lat/lon and full metadata. Previously flagged as redundant since cleaned version exists, BUT the cleaned version stripped lat/lon.

**Recommendation: KEEP.** Required for world map re-merge in Phase H.

**Decision: KEEP (updated from v9 recommendation).**

### 3.2 `data/natural/natural_opx_cleaned.csv` (10 MB)

Cleaned 53k opx, missing lat/lon.

**Recommendation: KEEP** as intermediate; Phase H will re-merge lat/lon from raw.

### 3.3 `data/external/agreda_lopez_2024/repo/.git/`

Nested git repo inside Agreda-Lopez code directory.

**Recommendation: DELETE** the nested `.git/` only. Keep source code.

**User decision required: [Y/N]**

### 3.4 `build_smoke/` directory

16 stub files for nonexistent notebooks.

**Recommendation: DELETE.**

**User decision required: [Y/N]**

### 3.5 `__pycache__/` at root and `src/__pycache__/`

Bytecode caches.

**Recommendation: DELETE.**

**User decision required: [Y/N]**

### 3.6 `extract_results.py` at repo root (18 KB)

v7/v8 review summary generator.

**Recommendation: ARCHIVE.**

**User decision required: [Y/N]**

### 3.7 `app_extract_inventory.txt` at repo root

App deployment manifest.

**Recommendation: ARCHIVE.**

**User decision required: [Y/N]**

---

## 4. NEW artifacts Phase A creates

After cleanup + initial setup:

```
data/
  natural/
    2024-XX-GEOROC_CLINOPYROXENES.csv   <- Phase A pull (if GEOROC has it)
    natural_cpx_cleaned.csv             <- Phase A, cleaned cpx
  external/
    jorgenson_2022/                     <- Phase A pull if available
      training_data.csv or placeholder
    petrelli_2020/                      <- Phase A pull if available
  hashes.json                           <- Phase A, SHA256 of every data file

models/
  canonical/
    opx/                                <- Phase A scaffolding (empty)
    cpx/
    twopx/
    universal/
  ablation/
    resampled/

scripts/
  v10_phase_a_cleanup.py                <- new
  v10_phase_b_nb_audit.py               <- new
  v10_pull_external_training_data.py    <- new
  v10_pull_georoc_cpx.py                <- new
  [5 other v10 scripts]

manuscripts/
  opx_2026/                             <- new scaffold
    figures/
    tables/
    text/
    arxiv_submission/
  cpx_2026/                             <- new scaffold
    figures/
    tables/
    text/
    arxiv_submission/
```

---

## 5. Phase A cleanup execution script

Proposed `scripts/v10_phase_a_cleanup.py`:

```
1. Check git status clean -> abort if not
2. Check git tag pre_v10_cleanup_YYYY_MM_DD exists -> abort if not
3. Check free space >= 1.0 GB -> abort if not
4. Create archive/pre_v10_rebuild_YYYY_MM_DD/ directory
5. For each path in ARCHIVE-THEN-DELETE list:
   - copy path -> archive target
   - verify SHA256 match
   - delete original
6. For each REVIEW item user approved:
   - apply action (DELETE or ARCHIVE)
7. Create new directory scaffolds (manuscripts/, models/canonical/*/, models/ablation/, src/ablations/)
8. Rename run_all_v7.py to run_all_v10.py
9. Generate data/hashes.json with SHA256 of every file in data/
10. Write logs/v10_phase_a_cleanup_log.txt with every action
11. Final directory sanity check: confirm src/, data/, models/external/ intact
12. Print summary
```

Script is idempotent: re-running after partial completion picks up where it left off.

---

## 6. Post-cleanup expected directory state

After `scripts/v10_phase_a_cleanup.py` runs successfully:

```
Final Project/
  .git/                                <- tagged pre_v10_cleanup_*
  .venv/
  archive/
    pre_v10_rebuild_YYYY_MM_DD/        <- v9 artifacts preserved
    v7_preparation_20260414_164844/    <- older archives untouched
  config.py                            <- edited
  data/
    raw/                               <- intact
    processed/                         <- intact (NB01/NB02 output still valid)
    splits/                            <- intact
    external/                          <- intact; nested .git deleted
      jorgenson_2022/                  <- placeholder; populated by external training pull
    natural/                           <- intact
    hashes.json                        <- NEW
  docs/                                <- 13 v10 docs + v9 historical
  figures/
    .gitkeep                           <- empty otherwise
    opx/                               <- NEW scaffold
    cpx/
    twopx/
    universal/
  logs/
    .gitkeep                           <- empty otherwise
    v10_phase_a_cleanup_log.txt        <- NEW
  manuscripts/                         <- NEW
    opx_2026/
    cpx_2026/
  models/
    canonical/                         <- NEW subdirs
      opx/, cpx/, twopx/, universal/
    external/                          <- intact
    ablation/
      resampled/                       <- empty scaffold
  notebooks/                           <- 11 source NBs
                                       <- no executed/ subdir
  README.md                            <- updated
  PROJECT_OVERVIEW.md                  <- updated
  requirements.txt                     <- expanded for new models
  results/
    .gitkeep                           <- empty otherwise
  run_all_v10.py                       <- renamed
  scripts/
    audit_*                            <- v9 kept
    v10_*.py                           <- 8 new scripts
  src/
    [all v9 modules intact]
    ablations/                         <- NEW empty subdir
```

---

## 7. Rollback

If Phase C encounters a problem and user wants to revert to v9:

1. `git checkout pre_v10_cleanup_YYYY_MM_DD`
2. Copy from `archive/pre_v10_rebuild_YYYY_MM_DD/` back to original paths
3. Confirm `logs/pipeline_health.txt` reproduces 23/23 PASS

Archive is the safety net. Do not delete `archive/pre_v10_rebuild_YYYY_MM_DD/` until Phase K completes and opx paper is submitted.

---

## 8. Expected Phase A runtime

- Cleanup script: 5-10 min (mostly I/O)
- Hash generation: 1-2 min
- External data audit (`v10_pull_external_training_data.py`): 10-30 min depending on downloads
- GEOROC cpx pull: 5-15 min
- Dir scaffolding: < 1 min

Total Phase A active+compute: 30-60 min of attention + script runtime.
