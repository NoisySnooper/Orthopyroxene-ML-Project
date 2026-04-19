# v9 Rebuild Inventory Report

**Generated:** 2026-04-15
**Scope:** Read-only audit of entire repo (excluding `.venv/`, `.git/`, `archive/`)
**Purpose:** Classify every file as KEEP-CANONICAL, KEEP-DOC, ARCHIVE, DELETE, or REVIEW-MANUAL ahead of Phase 1 clean-slate execution.

---

## 1. Directory-level summary

| Directory | Size | File count | Classification |
|---|---|---|---|
| `notebooks/` | 29 MB | 11 source + 12 executed (subdir) | source KEEP, executed ARCHIVE |
| `src/` | 166 KB | 10 modules + `__pycache__` | modules KEEP, cache DELETE |
| `scripts/` | 540 KB | 48 scripts | 4 KEEP-DOC, 44 ARCHIVE |
| `docs/` | 44 KB | 4 methodology docs | all KEEP-CANONICAL |
| `results/` | 3.5 MB | 79 CSV/JSON + `.gitkeep` | ARCHIVE all |
| `figures/` | 23 MB | 64 PDF/PNG/TXT + archive subdir | ARCHIVE all |
| `models/` | 325 MB | 37 root joblibs + external subdir + empty canonical subdir | root ARCHIVE, external KEEP |
| `logs/` | 1.0 MB | 47 logs + archive subdir | ARCHIVE all |
| `data/raw/` | 12 MB | 2 xlsx | KEEP-CANONICAL |
| `data/processed/` | 4.6 MB | 7 files | KEEP-CANONICAL |
| `data/external/` | 1.2 GB | agreda (137 MB), thermobar (1.1 GB), jorgenson (empty) | KEEP-CANONICAL (big) |
| `data/natural/` | 108 MB | 3 files (raw CSV + cleaned CSV + prep script) | 2 KEEP, 1 REVIEW |
| `data/splits/` | 24 KB | 4 npy + `.gitkeep` | KEEP-CANONICAL |
| `archive/` | 23 MB | existing archive subdirs | leave in place |
| `build_smoke/` | 96 KB | 16 stub tests | DELETE (reference nonexistent notebooks) |
| `examples/` | 8 KB | 1 CSV | REVIEW (app-related) |
| `reports/` | 0 | empty `.gitkeep` scaffold | KEEP scaffold |
| Root orphans | ~160 KB | logs, params, runners, utilities | mixed ARCHIVE / DELETE |

**Grand totals:** 1.75 GB overall. KEEP-CANONICAL ~1.25 GB, KEEP-DOC ~50 KB, ARCHIVE ~350 MB, DELETE ~150 MB.

---

## 2. File classification

### KEEP-CANONICAL (source of truth, do not move or touch)

**Root:**
- `README.md`
- `PROJECT_OVERVIEW.md`
- `config.py`
- `requirements.txt`
- `run_all_v7.py` (current canonical pipeline runner)

**`notebooks/`** (11 source notebooks):
- `nb01_data_cleaning.ipynb`
- `nb02_eda_pca.ipynb`
- `nb03_baseline_models.ipynb`
- `nb04_putirka_benchmark.ipynb`
- `nb05_loso_validation.ipynb`
- `nb06_shap_analysis.ipynb`
- `nb07_bias_correction.ipynb`
- `nb07b_arcpl_bias_probe.ipynb`
- `nb08_natural_twopx.ipynb`
- `nb09_manuscript_compilation.ipynb`
- `nbF_figures.ipynb`

**`src/`** (10 modules):
- `__init__.py`, `calibration.py`, `data.py`, `evaluation.py`, `external_models.py`, `features.py`, `geotherm.py`, `io_utils.py`, `models.py`, `plot_style.py`

**`docs/`** (4 existing methodology/audit docs):
- `codebase_consistency_audit_optionB.md`
- `optionB_pre_implementation_prediction.md`
- `putirka_inconsistency_audit.md`
- `putirka_kd_filter_lookup.md`

**`data/raw/`:**
- `ExPetDB_download_ExPetDB-2025-07-21.xlsx` (10 MB)
- `external/LEPR_Wet_Stitched_April2023_Norm100Anhydrs.xlsx` (2.2 MB)

**`data/processed/`** (7 files, all regeneratable from raw but expensive to recompute, keep as canonical):
- `opx_clean_core.csv`, `opx_clean_core.parquet`, `opx_clean_core_with_clusters.parquet`
- `opx_clean_full.csv`, `opx_clean_full.parquet`
- `opx_clean_opx_liq.parquet`, `opx_clean_opx_only.parquet`

**`data/splits/`** (4 frozen train/test index files):
- `test_indices_opx.npy`, `test_indices_opx_liq.npy`
- `train_indices_opx.npy`, `train_indices_opx_liq.npy`

**`data/external/`:**
- `agreda_lopez_2024/repo/` (137 MB, external ML model repo)
- `thermobar_examples/Thermobar/` (1.1 GB, Thermobar repo clone — source of ArcPL benchmark data)
- `thermobar_examples/_INDEX.csv`
- `jorgenson_2022/` (empty placeholder dir)

**`data/natural/`:**
- `natural_opx_cleaned.csv` (10 MB, imported via `config.LIN2023_NATURAL`)
- `natural_sample_prep_script.py` (reproducibility)

**`models/external/`** (12 files, external cpx models for nb04 benchmark):
- `agreda_cpx_P.{joblib,json,onnx}`, `agreda_cpx_T.{joblib,json,onnx}`
- `agreda_cpx_liq_P.{joblib,json,onnx}`, `agreda_cpx_liq_T.{joblib,json,onnx}`

**`models/canonical/`** (empty dir, keep as scaffold)

**`.gitkeep` files** (preserve directory structure): `results/.gitkeep`, `figures/.gitkeep`, `models/.gitkeep`, `logs/.gitkeep`, `data/processed/.gitkeep`, `data/splits/.gitkeep`, `reports/figures/.gitkeep`

### KEEP-DOC (audit trail, one-line justification each)

- `scripts/audit_optionB_prediction.py` — audit of Option B Kd-filter prediction pipeline
- `scripts/audit_putirka_paths.py` — audit of Putirka path A vs path B consistency
- `scripts/audit_structure.py` — generic repo structure audit
- `scripts/audit_thermobar_kd_api.py` — audit of Thermobar Kd API behavior

Plus the three v9 planning docs being produced in this phase:
- `docs/v9_inventory_report.md` (this file)
- `docs/v9_archive_plan.md`
- `docs/v9_deletion_plan.md`

### ARCHIVE (move to `archive/pre_v9_rebuild_2026_04_15/`, will regenerate)

**`notebooks/executed/`** — all 12 files (will regenerate via papermill in Phase 2-4):
- `nb01_...executed.ipynb`, `nb02_...executed.ipynb`, `nb03_...executed.ipynb`, `nb04_...executed.ipynb`, `nb04b_lepr_arcpl_validation_executed.ipynb` (orphan from deleted notebook), `nb05_...`, `nb06_...`, `nb07_...`, `nb07b_...`, `nb08_...`, `nb09_...`, `nbF_...`

**`results/`** — all 79 files (will regenerate): nb03_*, nb04_*, nb04b_*, nb05_*, nb06_*, nb07_*, nb07b_*, nb08_*, nb10_*, nb10b_*, nb11_*, nbF_*, optionB_preflight_metrics.csv, putirka_path_A_vs_B_metrics.csv, figure_inventory.csv, manuscript_key_results.csv, table1..table11*.csv

**`figures/`** — all content files (will regenerate): fig01..fig14_*.{pdf,png,txt}, fig_eda_*.png, fig_nb02_*, fig_nb03*, fig_nb04_*, fig_nb04b_*, fig_nb05_*, fig_nb06_*, fig_nb07b_*, fig_nb08_*, fig_nb11_*. Plus `figures/archive/` contents.

**`models/`** (root level, 37 joblibs, ~325 MB, will retrain):
All `model_*.joblib` and `isolation_forest_opx_only.joblib` at `models/` root.

**`logs/`** — all content (47 files including pipeline runs, FAILURE logs, cleaning log, codebase_audit.md, plus `logs/archive/`).

**`scripts/` one-shot patches and builders** (44 files already applied to canonical notebooks):
- `append_nb03_ceiling.py`
- `apply_v7_patches.py`, `apply_v7_phase_naming_fixes.py`
- `build_nb04_three_way.py`, `build_nb08_v6.py`, `build_nb10b.py`, `build_nb11.py`
- `merge_nb03.py`, `merge_nb06.py`, `merge_nb10_into_nb09.py`
- `overhaul_nb02.py`
- `patch_nb03_cell27_ceiling.py`, `patch_nb03_naug_subset.py`
- `patch_nb04_cell18_plotstyle.py`, `patch_nb04_encyclopedia.py`, `patch_nb04_shim.py`, `patch_nb04_three_panel_benchmark.py`, `patch_nb04_three_way_two_family.py`, `patch_nb04_upgrade_shim.py`, `patch_nb04_v7_consolidation.py`
- `patch_nb04b_v5.py`
- `patch_nb05_generalization_fig.py`
- `patch_nb06_shap_additivity.py`
- `patch_nb07_per_target_feature_set.py`
- `patch_nb08_per_family.py`
- `patch_nb09_per_target_features.py`
- `patch_nbF_fig14_per_target.py`, `patch_nbF_per_combo_features.py`
- `patch_nb_docs_template.py`
- `patch_v8_nb04_fixes.py`, `patch_v8_nb04_putirka_vs_ml_fig.py`, `patch_v8_nb06_canonical_names.py`, `patch_v8_nb09_fixes.py`
- `rebuild_nb03_v7.py`
- `reorder_nb03.py`
- `rewire_nb04_per_family.py`, `rewire_win_feat_to_per_family.py`
- `rewrite_notebook_imports.py`
- `run_ceiling_smoke.py`, `run_nb03_phase38.py`
- `smoke_test_imports.py`
- `verify_phase_naming.py`, `verify_v6.py`, `verify_v7.py`

**Root orphans (archive):**
- `nb03_search.log`, `nb03c_frozen_params.json`, `nb03c_training.log` (training artifacts from prior NB03 runs)
- `app_extract_inventory.txt` (app-deployment inventory)
- `run_all.py` (**stale**: references nb10/nb10b/nb11/nb04b/nb08_natural_samples, none exist)
- `run_from.py` (**partially stale**: references nb04b_lepr_arcpl_validation which was deleted)
- `extract_results.py` (v7/v8 review-summary generator, may be reused post-v9 but not needed for rebuild)

### DELETE (permanent, one-line justification each)

- `build_smoke/` — 16 stub files for notebooks that no longer exist (nb04b, nb08_natural_samples, nb10, nb10b, nb11). All import paths reference deleted notebooks. Stale.
- `wipe_old_files_v7.py` (root) — one-time wipe script, already executed.
- `__pycache__/` (root) — bytecode cache.
- `src/__pycache__/` — bytecode cache.
- `data/external/agreda_lopez_2024/repo/.DS_Store` — macOS filesystem noise.
- `data/external/agreda_lopez_2024/repo/.git/` — nested git repo from cloned Agreda-Lopez code; bloats repo size. Source code itself stays.

### REVIEW-MANUAL (user decides before Phase 1)

1. **`data/natural/2024-12-SGFTFN_ORTHOPYROXENES.csv`** (102 MB raw CSV) — cleaned version `natural_opx_cleaned.csv` and prep script are both retained in canonical. Raw is redundant if the cleaning is reproducible. **Recommendation: DELETE** to reclaim 102 MB.

2. **`data/natural/natural_opx_cleaned.csv`** — imported as `LIN2023_NATURAL` in every current notebook but not actually read anywhere (confirmed via grep). File is 10 MB. **Recommendation: KEEP** (low cost, import would fail without it).

3. **`examples/test_sample_for_app_validation.csv`** (4 KB) — app-deployment test fixture. No mention in v9 plan. **Recommendation: KEEP** (trivial size; likely used for future Streamlit app).

4. **`extract_results.py`** (root, 18 KB) — used during v7/v8 to produce review summaries. Not invoked by any notebook or v9 plan. **Recommendation: ARCHIVE** (may revive for manuscript reporting).

5. **`app_extract_inventory.txt`** (root, 20 KB) — app-deployment file manifest. Not invoked during pipeline. **Recommendation: ARCHIVE**.

6. **`data/external/jorgenson_2022/`** — empty placeholder dir referenced by config? (Not found in config.py). **Recommendation: KEEP** as scaffold (size is zero).

7. **`data/external/thermobar_examples/Thermobar/`** — 1.1 GB vendored Thermobar repo. Source of ArcPL benchmark data used by nb04. Bulk of repo size. **Recommendation: KEEP-CANONICAL** (needed for nb04, alternatives would require reworking data pipeline).

8. **`logs/codebase_audit.md`** (26 KB) — historical codebase audit from v7 era. Content informative. **Recommendation: ARCHIVE** per default logs rule.

9. **`archive/` existing subdirs** (`pipeline_v1_legacy/`, `v5_reports/`, `v7_preparation_20260414_164844/`) — prior archive snapshots from earlier rebuilds. **Recommendation: LEAVE IN PLACE** (don't nest an archive inside an archive; they already represent historical state).

---

## 3. Risk register

| # | Risk | Status | Action |
|---|---|---|---|
| 1 | Plan states `data/natural/` has missing Table S*.xlsx + lin_2023 CSV inputs and NB08 will break | **FALSE ALARM** | No current notebook reads from `data/natural/`. `natural_opx_cleaned.csv` exists (10 MB). NB08 uses Thermobar's AllMatches dataset instead. |
| 2 | `run_all.py` references 5 notebooks that don't exist (nb04b, nb08_natural_samples, nb10, nb10b, nb11) | CONFIRMED | Archive it. `run_all_v7.py` is the canonical runner. |
| 3 | `run_from.py` references nb04b_lepr_arcpl_validation (deleted) | CONFIRMED | Archive it. |
| 4 | `config.py` references `WINNING_CONFIG_FILE` as deprecated (pre-v7) | COSMETIC | No action; obsolete constant kept for history. |
| 5 | `data/external/thermobar_examples/Thermobar/` is 1.1 GB (64% of repo size) | CONFIRMED | Keep (required by nb04). User should be aware if pushing to GitHub. |
| 6 | `data/external/agreda_lopez_2024/repo/.git/` is a nested git repo inside the parent repo | CONFIRMED | DELETE the nested `.git/` dir to reclaim space and avoid submodule confusion. |
| 7 | `models/canonical/` is an empty directory; plan writes joblibs to `models/` root | COSMETIC | No action; keep as scaffold. |
| 8 | `data/external/jorgenson_2022/` is empty | NOT USED | Keep; scaffold. |
| 9 | Scripts list is bloated with 44 one-shot patches from v5 through v8 | CONFIRMED | Archive under `scripts_obsolete_patches/`. |
| 10 | `notebooks/executed/nb04b_lepr_arcpl_validation_executed.ipynb` present but no source | CONFIRMED | Archive with other executed files; source was absorbed into nb04 parts 3-5. |

---

## 4. Summary

- Total files surveyed: ~400 (excluding venv/archive/git)
- KEEP-CANONICAL: ~70 files, ~1.25 GB
- KEEP-DOC: 7 files, ~60 KB
- ARCHIVE: ~280 files, ~355 MB
- DELETE: ~20 files/dirs, ~150 MB
- REVIEW-MANUAL: 9 items (recommendations given above)

**The plan's premise that NB08 will break because data/natural/ inputs are missing is incorrect.** The cascade risk in Phase 3 for NB08 is low; NB08 reads Thermobar's AllMatches dataset from the vendored thermobar_examples, not from data/natural/.

**Next step:** user reviews this report plus `v9_archive_plan.md` and `v9_deletion_plan.md`, then approves Phase 1 execution.
