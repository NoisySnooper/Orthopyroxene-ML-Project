# v9 Rebuild Archive Plan

**Archived 2026-04-15 as part of v9 clean-slate rebuild. Contents represent the v8 final state. See `docs/codebase_consistency_audit_optionB.md` for the inconsistency review that motivated the rebuild.**

**Target archive root:** `archive/pre_v9_rebuild_2026_04_15/`

Existing `archive/` subdirs (`pipeline_v1_legacy/`, `v5_reports/`, `v7_preparation_20260414_164844/`) are left in place. The new v9 archive sits alongside them.

---

## Subdirs to create

```
archive/pre_v9_rebuild_2026_04_15/
    notebooks_executed/
    results/
    figures/
    figures_archive/
    models/
    logs/
    logs_archive/
    scripts_obsolete_patches/
    root_orphans/
```

---

## Source to destination map

### 1. `notebooks_executed/` (12 files, ~29 MB)

```
notebooks/executed/nb01_data_cleaning_executed.ipynb            -> archive/pre_v9_rebuild_2026_04_15/notebooks_executed/nb01_data_cleaning_executed.ipynb
notebooks/executed/nb02_eda_pca_executed.ipynb                  -> archive/pre_v9_rebuild_2026_04_15/notebooks_executed/nb02_eda_pca_executed.ipynb
notebooks/executed/nb03_baseline_models_executed.ipynb          -> archive/pre_v9_rebuild_2026_04_15/notebooks_executed/nb03_baseline_models_executed.ipynb
notebooks/executed/nb04_putirka_benchmark_executed.ipynb        -> archive/pre_v9_rebuild_2026_04_15/notebooks_executed/nb04_putirka_benchmark_executed.ipynb
notebooks/executed/nb04b_lepr_arcpl_validation_executed.ipynb   -> archive/pre_v9_rebuild_2026_04_15/notebooks_executed/nb04b_lepr_arcpl_validation_executed.ipynb
notebooks/executed/nb05_loso_validation_executed.ipynb          -> archive/pre_v9_rebuild_2026_04_15/notebooks_executed/nb05_loso_validation_executed.ipynb
notebooks/executed/nb06_shap_analysis_executed.ipynb            -> archive/pre_v9_rebuild_2026_04_15/notebooks_executed/nb06_shap_analysis_executed.ipynb
notebooks/executed/nb07_bias_correction_executed.ipynb          -> archive/pre_v9_rebuild_2026_04_15/notebooks_executed/nb07_bias_correction_executed.ipynb
notebooks/executed/nb07b_arcpl_bias_probe_executed.ipynb        -> archive/pre_v9_rebuild_2026_04_15/notebooks_executed/nb07b_arcpl_bias_probe_executed.ipynb
notebooks/executed/nb08_natural_twopx_executed.ipynb            -> archive/pre_v9_rebuild_2026_04_15/notebooks_executed/nb08_natural_twopx_executed.ipynb
notebooks/executed/nb09_manuscript_compilation_executed.ipynb   -> archive/pre_v9_rebuild_2026_04_15/notebooks_executed/nb09_manuscript_compilation_executed.ipynb
notebooks/executed/nbF_figures_executed.ipynb                   -> archive/pre_v9_rebuild_2026_04_15/notebooks_executed/nbF_figures_executed.ipynb
```

### 2. `results/` (79 files, ~3.5 MB)

All files in `results/` except `.gitkeep`. Glob pattern:
```
results/*.csv          -> archive/pre_v9_rebuild_2026_04_15/results/
results/*.json         -> archive/pre_v9_rebuild_2026_04_15/results/
```

This captures: nb03_*, nb04_*, nb04b_*, nb05_*, nb06_*, nb07_*, nb07b_*, nb08_*, nb10_*, nb10b_*, nb11_*, nbF_*, optionB_preflight_metrics.csv, putirka_path_A_vs_B_metrics.csv, figure_inventory.csv, manuscript_key_results.csv, table1..table11*.csv.

### 3. `figures/` (top-level) and `figures_archive/`

All content files at `figures/` root (~64 files, ~23 MB):
```
figures/*.pdf          -> archive/pre_v9_rebuild_2026_04_15/figures/
figures/*.png          -> archive/pre_v9_rebuild_2026_04_15/figures/
figures/*.txt          -> archive/pre_v9_rebuild_2026_04_15/figures/
```

`figures/archive/` contents (2 files):
```
figures/archive/fig_nb10b_two_pyroxene.pdf   -> archive/pre_v9_rebuild_2026_04_15/figures_archive/fig_nb10b_two_pyroxene.pdf
figures/archive/fig_nb10b_two_pyroxene.png   -> archive/pre_v9_rebuild_2026_04_15/figures_archive/fig_nb10b_two_pyroxene.png
```

### 4. `models/` (37 root-level joblibs, ~325 MB)

All `.joblib` files directly under `models/` (not under `models/external/` or `models/canonical/`):
```
models/isolation_forest_opx_only.joblib         -> archive/pre_v9_rebuild_2026_04_15/models/
models/model_ERT_*.joblib                       -> archive/pre_v9_rebuild_2026_04_15/models/
models/model_GB_*.joblib                        -> archive/pre_v9_rebuild_2026_04_15/models/
models/model_IsolationForest_opx_liq.joblib     -> archive/pre_v9_rebuild_2026_04_15/models/
models/model_P_kbar_*.joblib                    -> archive/pre_v9_rebuild_2026_04_15/models/
models/model_QRF_*.joblib                       -> archive/pre_v9_rebuild_2026_04_15/models/
models/model_RF_*.joblib                        -> archive/pre_v9_rebuild_2026_04_15/models/
models/model_T_C_*.joblib                       -> archive/pre_v9_rebuild_2026_04_15/models/
models/model_XGB_*.joblib                       -> archive/pre_v9_rebuild_2026_04_15/models/
```

**`models/external/` stays in place** (external cpx reference models, KEEP-CANONICAL).
**`models/canonical/` stays in place** (empty scaffold dir).

### 5. `logs/` (47 files) and `logs_archive/`

All content in `logs/` root except `.gitkeep`:
```
logs/*.log             -> archive/pre_v9_rebuild_2026_04_15/logs/
logs/*.md              -> archive/pre_v9_rebuild_2026_04_15/logs/
logs/*.json            -> archive/pre_v9_rebuild_2026_04_15/logs/
logs/*.txt             -> archive/pre_v9_rebuild_2026_04_15/logs/
```

`logs/archive/` contents (4 files):
```
logs/archive/FAILURE_nb04_putirka_benchmark.log             -> archive/pre_v9_rebuild_2026_04_15/logs_archive/
logs/archive/FAILURE_nb09_manuscript_compilation.log        -> archive/pre_v9_rebuild_2026_04_15/logs_archive/
logs/archive/FAILURE_nb10_extended_analyses.log             -> archive/pre_v9_rebuild_2026_04_15/logs_archive/
logs/archive/FAILURE_nbF_figures.log                        -> archive/pre_v9_rebuild_2026_04_15/logs_archive/
```

### 6. `scripts_obsolete_patches/` (44 files, ~540 KB)

All scripts in `scripts/` EXCEPT the 4 audit scripts (audit_optionB_prediction.py, audit_putirka_paths.py, audit_structure.py, audit_thermobar_kd_api.py which are KEEP-DOC):

```
scripts/append_nb03_ceiling.py                  -> archive/pre_v9_rebuild_2026_04_15/scripts_obsolete_patches/
scripts/apply_v7_patches.py                     -> ...
scripts/apply_v7_phase_naming_fixes.py          -> ...
scripts/build_nb04_three_way.py                 -> ...
scripts/build_nb08_v6.py                        -> ...
scripts/build_nb10b.py                          -> ...
scripts/build_nb11.py                           -> ...
scripts/merge_nb03.py                           -> ...
scripts/merge_nb06.py                           -> ...
scripts/merge_nb10_into_nb09.py                 -> ...
scripts/overhaul_nb02.py                        -> ...
scripts/patch_nb03_cell27_ceiling.py            -> ...
scripts/patch_nb03_naug_subset.py               -> ...
scripts/patch_nb04_cell18_plotstyle.py          -> ...
scripts/patch_nb04_encyclopedia.py              -> ...
scripts/patch_nb04_shim.py                      -> ...
scripts/patch_nb04_three_panel_benchmark.py     -> ...
scripts/patch_nb04_three_way_two_family.py      -> ...
scripts/patch_nb04_upgrade_shim.py              -> ...
scripts/patch_nb04_v7_consolidation.py          -> ...
scripts/patch_nb04b_v5.py                       -> ...
scripts/patch_nb05_generalization_fig.py        -> ...
scripts/patch_nb06_shap_additivity.py           -> ...
scripts/patch_nb07_per_target_feature_set.py    -> ...
scripts/patch_nb08_per_family.py                -> ...
scripts/patch_nb09_per_target_features.py       -> ...
scripts/patch_nbF_fig14_per_target.py           -> ...
scripts/patch_nbF_per_combo_features.py         -> ...
scripts/patch_nb_docs_template.py               -> ...
scripts/patch_v8_nb04_fixes.py                  -> ...
scripts/patch_v8_nb04_putirka_vs_ml_fig.py      -> ...
scripts/patch_v8_nb06_canonical_names.py        -> ...
scripts/patch_v8_nb09_fixes.py                  -> ...
scripts/rebuild_nb03_v7.py                      -> ...
scripts/reorder_nb03.py                         -> ...
scripts/rewire_nb04_per_family.py               -> ...
scripts/rewire_win_feat_to_per_family.py        -> ...
scripts/rewrite_notebook_imports.py             -> ...
scripts/run_ceiling_smoke.py                    -> ...
scripts/run_nb03_phase38.py                     -> ...
scripts/smoke_test_imports.py                   -> ...
scripts/verify_phase_naming.py                  -> ...
scripts/verify_v6.py                            -> ...
scripts/verify_v7.py                            -> ...
```

(All destinations under `archive/pre_v9_rebuild_2026_04_15/scripts_obsolete_patches/` with the same filename.)

### 7. `root_orphans/` (7 files, ~160 KB)

```
nb03_search.log                -> archive/pre_v9_rebuild_2026_04_15/root_orphans/nb03_search.log
nb03c_frozen_params.json       -> archive/pre_v9_rebuild_2026_04_15/root_orphans/nb03c_frozen_params.json
nb03c_training.log             -> archive/pre_v9_rebuild_2026_04_15/root_orphans/nb03c_training.log
app_extract_inventory.txt      -> archive/pre_v9_rebuild_2026_04_15/root_orphans/app_extract_inventory.txt
run_all.py                     -> archive/pre_v9_rebuild_2026_04_15/root_orphans/run_all.py
run_from.py                    -> archive/pre_v9_rebuild_2026_04_15/root_orphans/run_from.py
extract_results.py             -> archive/pre_v9_rebuild_2026_04_15/root_orphans/extract_results.py
```

### 8. `examples_archive/` (1 file, 4 KB) — user decision 2026-04-15

```
examples/test_sample_for_app_validation.csv  -> archive/pre_v9_rebuild_2026_04_15/examples_archive/test_sample_for_app_validation.csv
```

After moving, `examples/` directory will be empty; it can be removed or kept as an empty scaffold. The rebuild plan does not reference `examples/`.

---

## Estimated archive size

| Subdir | Size |
|---|---|
| notebooks_executed/ | 29 MB |
| results/ | 3.5 MB |
| figures/ + figures_archive/ | 23 MB |
| models/ | 325 MB |
| logs/ + logs_archive/ | 1 MB |
| scripts_obsolete_patches/ | 540 KB |
| root_orphans/ | 170 KB |
| **TOTAL** | **~382 MB** |

---

## Phase 1 execution order (for Phase 1 prompt)

1. Create archive subdirs listed above.
2. Move files in this order (safest first, most aggressive last):
   - `notebooks/executed/*` (nothing depends on these)
   - `results/*` (nothing depends on these during rebuild)
   - `figures/*` and `figures/archive/*`
   - `logs/*` and `logs/archive/*`
   - `models/*.joblib` (at root; leave `models/external/` and `models/canonical/` in place)
   - `scripts/` obsolete patches (44 files)
   - Root orphans (8 files)
3. Verify: `notebooks/executed/`, `results/`, `figures/`, `logs/`, and `models/` (root) each contain only `.gitkeep`.
4. Re-add `.gitkeep` files if any were moved inadvertently.

---

## What does NOT move

- `notebooks/nb*.ipynb` (11 source notebooks)
- `src/*.py`
- `config.py`, `requirements.txt`, `run_all_v7.py`, `README.md`, `PROJECT_OVERVIEW.md`
- `docs/*.md` (including the three new v9 docs)
- `scripts/audit_*.py` (4 audit scripts, KEEP-DOC)
- `data/` (all subdirs, KEEP-CANONICAL except deletion items covered in deletion plan)
- `models/external/` (external reference models)
- `models/canonical/` (empty scaffold)
- `reports/` (scaffold)
- `examples/` (pending REVIEW-MANUAL outcome)
- `archive/` existing subdirs (they already represent historical state)

---

## Post-archive verification checklist

- [ ] `notebooks/executed/` is empty (only `.gitkeep`)
- [ ] `results/` is empty (only `.gitkeep`)
- [ ] `figures/` contains no PDF/PNG/TXT at root (only `.gitkeep`)
- [ ] `figures/archive/` is empty or removed
- [ ] `models/` root contains no `*.joblib` (only `.gitkeep`, subdirs `external/` and `canonical/`)
- [ ] `logs/` root contains no logs (only `.gitkeep`)
- [ ] `scripts/` contains only 4 audit_*.py files + the `__init__` if any
- [ ] None of the 8 root orphan files are present at repo root
- [ ] `archive/pre_v9_rebuild_2026_04_15/` exists with expected subdir structure
- [ ] Archived file count matches source file count
